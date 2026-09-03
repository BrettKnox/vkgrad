"""Does cooperative matrix survive the toolchain, and does the driver honour it?

Two layers can silently drop matrix acceleration, and only one of them is
visible without vendor tooling:

  1. GLSL -> SPIR-V. Checked here by disassembling with `spirv-dis` and counting
     OpCooperativeMatrix* instructions. Free, needs no GPU, works anywhere the
     Vulkan SDK is installed.
  2. SPIR-V -> RDNA3 ISA, done by the driver. If the driver lowers coopmat to
     v_fmac_f32 instead of v_wmma_f32_16x16x16_f16, the matrix units are never
     touched and no amount of tiling will help. Only Radeon GPU Analyzer can
     show this, and it is a separate AMD download.

Layer 1 matters most for the transposed variants. The whole backward pass is
transposed matmuls (dW = X^T dY, dX = dY W^T), and bench/predict.py found that
the traffic model predicts forward matmuls well (rho +0.76) but fails on exactly
those transposed shapes. A toolchain quietly degrading transposed operands would
produce that signature.

  python -m bench.isa
  python -m bench.isa --rga "C:/Program Files/RGA/rga.exe"
"""

import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile import compile_glsl  # noqa: E402
from kernels import matmul_glsl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "runs", "isa")

VARIANTS = {
    "forward": dict(),
    "trans_a dW": dict(trans_a=True),
    "trans_b dX": dict(trans_b=True),
    "batched": dict(batched=True),
    "swizzled": dict(group_m=2),
}

_OPS = re.compile(
    r"OpCooperativeMatrixMulAddKHR|OpCooperativeMatrixLoadKHR|"
    r"OpCooperativeMatrixStoreKHR|OpTypeCooperativeMatrixKHR")

# What a real RDNA3 matrix instruction looks like, versus the scalar fallback
# the driver emits when it declines to use the matrix units.
_WMMA = re.compile(r"v_wmma_\w+")
_SCALAR_FMA = re.compile(r"v_fmac_f32|v_fma_f32|v_dot2\w*")


def find_spirv_dis(explicit=None):
    if explicit:
        return explicit
    sdk = os.environ.get("VULKAN_SDK")
    names = ["spirv-dis.exe", "spirv-dis"]
    roots = ([sdk] if sdk else []) + sorted(
        glob.glob("C:/VulkanSDK/*"), reverse=True)
    for r in roots:
        for n in names:
            p = os.path.join(r, "Bin", n)
            if os.path.exists(p):
                return p
    from shutil import which
    return which("spirv-dis")


def probe_spirv(dis):
    """Count matrix ops per variant. Identical counts mean no variant was
    silently degraded relative to the others."""
    os.makedirs(OUT, exist_ok=True)
    print(f"{'variant':<12}{'bytes':>8}{'mulAdd':>8}{'load':>6}{'store':>7}"
          f"{'types':>7}   verdict")
    print("-" * 72)
    rows = {}
    for name, kw in VARIANTS.items():
        src = matmul_glsl(sg=4, wm=2, wn=4, **kw)
        spv = compile_glsl(src, name="isa_" + name.replace(" ", "_"))
        path = os.path.join(OUT, name.replace(" ", "_") + ".spv")
        with open(path, "wb") as f:
            f.write(spv)
        ops = _OPS.findall(
            subprocess.run([dis, path], capture_output=True, text=True).stdout)
        mul = ops.count("OpCooperativeMatrixMulAddKHR")
        rows[name] = (path, mul)
        verdict = "coopmat present" if mul else "NO MATRIX OPS"
        print(f"{name:<12}{len(spv):>8}{mul:>8}"
              f"{ops.count('OpCooperativeMatrixLoadKHR'):>6}"
              f"{ops.count('OpCooperativeMatrixStoreKHR'):>7}"
              f"{ops.count('OpTypeCooperativeMatrixKHR'):>7}   {verdict}")

    muls = {m for _, m in rows.values()}
    print()
    if len(muls) == 1 and muls != {0}:
        print("  Every variant emits the same matrix ops, so the transposed and")
        print("  batched paths are not degraded at the SPIR-V level. Whatever")
        print("  costs them happens below this layer.")
    elif 0 in muls:
        print("  At least one variant lost its matrix ops during compilation.")
        print("  That is the bug; look no further down the stack.")
    return rows


def probe_isa(rga, rows, arch="gfx1103"):
    """Compile SPIR-V to RDNA3 ISA offline and look for real WMMA.

    This is the half that decides whether the matrix units are used at all.
    """
    print(f"\n{'variant':<12}{'v_wmma':>8}{'scalar fma':>12}{'vgpr':>7}"
          f"{'lds':>8}   verdict")
    print("-" * 62)
    for name, (spv, _) in rows.items():
        tag = name.replace(" ", "_")
        d = os.path.join(OUT, f"{tag}.{arch}.isa")
        a = os.path.join(OUT, f"{tag}.{arch}.csv")
        r = subprocess.run(
            [rga, "-s", "vk-spv-offline", "-c", arch,
             "--isa", d, "-a", a, "--comp", spv],
            capture_output=True, text=True)
        text = ""
        for f in glob.glob(os.path.join(OUT, f"*{tag}*")):
            if f.endswith((".isa", ".txt")) or ".isa" in os.path.basename(f):
                with open(f, errors="ignore") as fh:
                    text += fh.read()
        if not text:
            msg = (r.stderr or r.stdout).strip().replace("\n", " ")
            print(f"{name:<12}   RGA produced no ISA: {msg[:70]}")
            continue
        w, s = len(_WMMA.findall(text)), len(_SCALAR_FMA.findall(text))
        stat = {}
        for f in glob.glob(os.path.join(OUT, f"*{tag}*.csv")):
            with open(f, errors="ignore") as fh:
                head, *body = fh.read().splitlines()
            if body:
                stat = dict(zip([c.strip() for c in head.split(",")],
                                [v.strip() for v in body[0].split(",")]))
        vgpr = stat.get("USED_VGPRs", "?")
        lds = stat.get("USED_LDS_BYTES", "?")
        # Spills mean the kernel ran out of registers and is round-tripping
        # through memory, which on a bandwidth-bound machine is the difference
        # between a fast kernel and a hopeless one.
        spill = int(stat.get("VGPR_SPILLS", 0) or 0)
        scratch = int(stat.get("SCRATCH_MEM", 0) or 0)
        if not w:
            verdict = "EMULATED -- matrix units never touched"
        elif spill or scratch:
            verdict = f"matrix units used, but SPILLING ({spill} vgpr)"
        else:
            verdict = "matrix units used, no spills"
        print(f"{name:<12}{w:>8}{s:>12}{vgpr:>7}{lds:>8}   {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirv-dis", default=None)
    ap.add_argument("--rga", default=None,
                    help="path to Radeon GPU Analyzer, for the driver-lowering half")
    ap.add_argument("--arch", default="gfx1103")
    args = ap.parse_args()

    dis = find_spirv_dis(args.spirv_dis)
    if not dis:
        raise SystemExit("spirv-dis not found; pass --spirv-dis or set VULKAN_SDK")
    rows = probe_spirv(dis)

    if args.rga:
        probe_isa(args.rga, rows, args.arch)
    else:
        print("\n  Pass --rga to settle the other half. SPIR-V carrying matrix")
        print("  ops only proves the compiler emitted them; whether the driver")
        print("  lowers them to v_wmma_f32_16x16x16_f16 rather than emulating")
        print("  with v_fmac_f32 needs Radeon GPU Analyzer.")


if __name__ == "__main__":
    main()
