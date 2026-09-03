"""Matmul roofline sweep, plus the comparison that decides whether any of this
is worth doing: the iGPU against the CPU sharing its die and its DRAM.

  python -m bench.matmul_sweep --quick
  python -m bench.matmul_sweep --json out.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import autotune_matmul, matmul_reference  # noqa: E402
from vk import Device  # noqa: E402


def cpu_matmul_tflops(size, reps=3):
    """numpy on OpenBLAS, all 8 cores. The thing the GPU has to beat."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    a @ b  # warm up threads
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return 2 * size ** 3 / best / 1e12, best


def verify(dev, mm, size):
    """A tuned config that is fast and wrong is worse than useless."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((size, size)).astype(np.float16)
    b = rng.standard_normal((size, size)).astype(np.float16)
    ba = dev.buffer(a.size * 2, "shared")
    bb = dev.buffer(b.size * 2, "shared")
    bc = dev.buffer(size * size * 4, "cached")
    try:
        ba.array(np.float16, a.shape)[:] = a
        bb.array(np.float16, b.shape)[:] = b
        ba.flush()
        bb.flush()
        mm(ba, bb, bc, size, size, size)
        bc.invalidate()
        got = bc.array(np.float32, (size, size))
        ref = matmul_reference(a, b)
        return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6))
    finally:
        ba.destroy()
        bb.destroy()
        bc.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--peak", type=float, default=13.71,
                    help="measured WMMA peak TFLOPS, from bench.roofline")
    args = ap.parse_args()

    sizes = [256, 512] if args.quick else [256, 512, 1024, 2048, 4096]
    dev = Device()
    print(dev)
    core = dev.shader_core_props()
    if core:
        print(f"  {core['cus']} CUs, {core['vgprs_per_simd']} VGPRs/SIMD, "
              f"max {core['waves_per_simd']} waves/SIMD")
    print(f"\n{'size':>6} {'GPU TFLOPS':>11} {'% peak':>7} {'CPU TFLOPS':>11} "
          f"{'speedup':>8} {'config':>22} {'rel err':>9}")

    rows = []
    try:
        for size in sizes:
            mm, best = autotune_matmul(dev, size, size, size, use_cache=False)
            try:
                err = verify(dev, mm, size)
                cpu_tf, _ = cpu_matmul_tflops(size)
                sg, wm, wn = best["config"]
                cfg = f"sg{sg} wm{wm} wn{wn} lds{int(best['lds'])}"
                row = {"size": size, "gpu_tflops": best["tflops"], "cpu_tflops": cpu_tf,
                       "speedup": best["tflops"] / cpu_tf, "config": best["config"],
                       "lds": best["lds"], "tile": best["tile"], "vgpr": best["vgpr"],
                       "pct_peak": 100 * best["tflops"] / args.peak,
                       "rel_err": err, "pruned": best["pruned"],
                       "evaluated": best["evaluated"]}
                rows.append(row)
                print(f"{size:6d} {best['tflops']:11.3f} {row['pct_peak']:6.1f}% "
                      f"{cpu_tf:11.3f} {row['speedup']:7.2f}x {cfg:>22} {err:9.1e}")
            finally:
                mm.destroy()

        print("\nautotuner: per size, evaluated/pruned = " +
              ", ".join(f"{r['size']}:{r['evaluated']}/{r['pruned']}" for r in rows))
    finally:
        dev.destroy()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev.name, "rows": rows, "peak_tflops": args.peak}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
