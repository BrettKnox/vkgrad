"""Phase 1 measurements. These numbers decide the framework's design.

Run on a QUIET machine: DDR5 bandwidth is shared with the CPU, so a browser
playing video will show up in the results.

  python -m bench.roofline --quick    # smoke test, safe any time
  python -m bench.roofline            # real numbers, wants an idle box
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile import compile_glsl          # noqa: E402
from vk import Device, VkError            # noqa: E402

# Grid-stride so the workgroup count stays far below maxComputeWorkGroupCount
# and each dispatch stays well inside the ~2s TDR window.
COPY_SRC = """#version 450
layout(local_size_x = 256) in;
layout(binding = 0) readonly  buffer A { vec4 a[]; };
layout(binding = 1) writeonly buffer B { vec4 b[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < p.n; i += stride) b[i] = a[i];
}
"""

READ_SRC = """#version 450
layout(local_size_x = 256) in;
layout(binding = 0) readonly buffer A { vec4 a[]; };
layout(binding = 1) buffer B { vec4 b[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    vec4 acc = vec4(0.0);
    for (uint i = gl_GlobalInvocationID.x; i < p.n; i += stride) acc += a[i];
    // Never true, but the compiler can't prove it: keeps the loads alive.
    if (acc.x == 1234.5678) b[gl_GlobalInvocationID.x] = acc;
}
"""

FMA_SRC = """#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer O { float o[]; };
layout(push_constant) uniform P { uint iters; } p;
void main() {
    float base = float(gl_GlobalInvocationID.x) * 1e-6;
    float a0 = base, a1 = base + 1.0, a2 = base + 2.0, a3 = base + 3.0;
    float a4 = base + 4.0, a5 = base + 5.0, a6 = base + 6.0, a7 = base + 7.0;
    const float k1 = 1.0000001, k2 = 1e-7;
    for (uint i = 0u; i < p.iters; ++i) {
        a0 = fma(a0, k1, k2); a1 = fma(a1, k1, k2);
        a2 = fma(a2, k1, k2); a3 = fma(a3, k1, k2);
        a4 = fma(a4, k1, k2); a5 = fma(a5, k1, k2);
        a6 = fma(a6, k1, k2); a7 = fma(a7, k1, k2);
    }
    o[gl_GlobalInvocationID.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"""

# Four independent accumulators: one chain would measure WMMA latency, not
# throughput.
WMMA_SRC = """#version 450
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_16bit_storage : enable

layout(local_size_x = 32) in;
layout(binding = 0) buffer A { float16_t a[]; };
layout(binding = 1) buffer C { float c[]; };
layout(push_constant) uniform P { uint iters; } p;

#define MA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

void main() {
    MA ma; MB mb;
    coopMatLoad(ma, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(mb, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    MC c0 = MC(0.0), c1 = MC(0.0), c2 = MC(0.0), c3 = MC(0.0);
    for (uint i = 0u; i < p.iters; ++i) {
        c0 = coopMatMulAdd(ma, mb, c0);
        c1 = coopMatMulAdd(ma, mb, c1);
        c2 = coopMatMulAdd(ma, mb, c2);
        c3 = coopMatMulAdd(ma, mb, c3);
    }
    MC sum = c0 + c1 + c2 + c3;
    coopMatStore(sum, c, 256u * gl_WorkGroupID.x, 16, gl_CooperativeMatrixLayoutRowMajor);
}
"""

EMPTY_SRC = """#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer O { uint o[]; };
void main() { if (gl_GlobalInvocationID.x == 0xffffffffu) o[0] = 1u; }
"""


def _best(fn, reps=5):
    """Best of N. Bandwidth is noisy upward; the floor is the real number."""
    return min(fn() for _ in range(reps))


def bench_bandwidth(dev, mb, groups=2048, reps=5):
    """Copy bandwidth per memory kind, and read-only bandwidth for the fast one."""
    copy_k = dev.kernel(compile_glsl(COPY_SRC, name="copy"), 2, 4, name="copy")
    read_k = dev.kernel(compile_glsl(READ_SRC, name="read"), 2, 4, name="read")
    out = {}

    for kind in ("shared", "cached", "device", "bar"):
        # Two buffers must fit the heap; the BAR heap is only 256 MiB.
        try:
            budget = dev.kind_heap_bytes(kind) * 0.4
        except VkError as e:
            out[kind] = {"error": str(e)}
            continue
        n_vec4 = int(min(mb << 20, budget)) // 16
        nbytes = n_vec4 * 16
        push = struct.pack("I", n_vec4)
        try:
            src = dev.buffer(nbytes, kind)
            dst = dev.buffer(nbytes, kind)
        except VkError as e:
            out[kind] = {"error": str(e)}
            continue
        try:
            t = _best(lambda: dev.run(copy_k, [src, dst], groups, push), reps)
            # A copy moves the buffer twice: one read + one write.
            out[kind] = {"copy_gbs": 2 * nbytes / t / 1e9, "copy_ms": t * 1e3,
                         "mib": nbytes >> 20}
            t = _best(lambda: dev.run(read_k, [src, dst], groups, push), reps)
            out[kind]["read_gbs"] = nbytes / t / 1e9
        finally:
            src.destroy()
            dst.destroy()

    copy_k.destroy()
    read_k.destroy()
    return out


def bench_staging(dev, mb, reps=5):
    """What a discrete-GPU-style upload costs here: host buffer -> device-local."""
    nbytes = mb << 20
    host = dev.buffer(nbytes, "shared")
    devbuf = dev.buffer(nbytes, "device")
    try:
        t = _best(lambda: dev.copy(host, devbuf, nbytes), reps)
        return {"upload_gbs": nbytes / t / 1e9, "upload_ms": t * 1e3, "mb": mb}
    finally:
        host.destroy()
        devbuf.destroy()


def bench_dispatch(dev, n=2000):
    """Per-dispatch and per-submit CPU+GPU overhead: the fusion cost model's
    constant term."""
    k = dev.kernel(compile_glsl(EMPTY_SRC, name="empty"), 1, name="empty")
    buf = dev.buffer(256, "shared")
    try:
        dev.run(k, [buf], 1, repeat=10)  # warm up
        t_batch = min(dev.run(k, [buf], 1, repeat=n) for _ in range(3))
        t0 = time.perf_counter()
        for _ in range(200):
            dev.run(k, [buf], 1, repeat=1)
        t_submit = (time.perf_counter() - t0) / 200
        return {"per_dispatch_us": t_batch / n * 1e6, "per_submit_us": t_submit * 1e6}
    finally:
        k.destroy()
        buf.destroy()


def bench_fp32(dev, iters, groups=1536, reps=5):
    k = dev.kernel(compile_glsl(FMA_SRC, name="fma"), 1, 4, name="fma")
    buf = dev.buffer(groups * 256 * 4, "shared")
    try:
        push = struct.pack("I", iters)
        t = _best(lambda: dev.run(k, [buf], groups, push), reps)
        flops = groups * 256 * iters * 8 * 2  # 8 FMAs, 2 FLOP each
        return {"tflops": flops / t / 1e12, "ms": t * 1e3}
    finally:
        k.destroy()
        buf.destroy()


def bench_wmma(dev, iters, groups=1536, reps=5):
    if not dev.coop_matrix_configs():
        return {"error": "no cooperative matrix support"}
    k = dev.kernel(compile_glsl(WMMA_SRC, name="wmma"), 2, 4, subgroup_size=32, name="wmma")
    a = dev.buffer(16 * 16 * 2, "shared")
    c = dev.buffer(groups * 256 * 4, "shared")
    try:
        a.array(np.float16)[:] = np.float16(1.0)
        a.flush()
        push = struct.pack("I", iters)
        t = _best(lambda: dev.run(k, [a, c], groups, push), reps)
        # 4 MulAdds per iteration, each 16x16x16 MACs = 2*16^3 FLOP.
        flops = groups * iters * 4 * 2 * 16 ** 3
        return {"tflops": flops / t / 1e12, "ms": t * 1e3}
    finally:
        k.destroy()
        a.destroy()
        c.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes; safe to run while the machine is busy")
    ap.add_argument("--json", default=None, help="write raw results here")
    args = ap.parse_args()

    mb = 16 if args.quick else 256
    fma_iters = 2000 if args.quick else 20000
    wmma_iters = 500 if args.quick else 5000
    reps = 2 if args.quick else 7

    dev = Device()
    print(dev)
    heaps = dev.heap_report()
    print("heaps: " + ", ".join(f"{h['gib']:.2f}GiB{'*' if h['device_local'] else ''}"
                                for h in heaps) + "   (* = device-local)")
    print(f"mode: {'quick' if args.quick else 'FULL'}  buffer={mb}MiB  reps={reps}\n")

    results = {"device": dev.name, "quick": args.quick, "buffer_mb": mb}
    try:
        print("[1] memory bandwidth by kind")
        bw = bench_bandwidth(dev, mb, reps=reps)
        results["bandwidth"] = bw
        for kind, v in bw.items():
            if "error" in v:
                print(f"    {kind:7s} unavailable: {v['error']}")
            else:
                print(f"    {kind:7s} copy {v['copy_gbs']:7.2f} GB/s   "
                      f"read {v['read_gbs']:7.2f} GB/s   ({v['mib']} MiB)")

        print("\n[2] staging upload (the cost zero-copy avoids)")
        st = bench_staging(dev, mb, reps=reps)
        results["staging"] = st
        print(f"    host->device {st['upload_gbs']:.2f} GB/s "
              f"({st['upload_ms']:.2f} ms for {st['mb']} MiB)")

        print("\n[3] dispatch overhead")
        d = bench_dispatch(dev, n=500 if args.quick else 2000)
        results["dispatch"] = d
        print(f"    per dispatch (batched) {d['per_dispatch_us']:.2f} us")
        print(f"    per submit+fence       {d['per_submit_us']:.2f} us")

        print("\n[4] fp32 vector peak")
        f = bench_fp32(dev, fma_iters, reps=reps)
        results["fp32"] = f
        print(f"    {f['tflops']:.2f} TFLOPS")

        print("\n[5] f16 WMMA peak (cooperative matrix)")
        w = bench_wmma(dev, wmma_iters, reps=reps)
        results["wmma"] = w
        if "error" in w:
            print(f"    {w['error']}")
        else:
            print(f"    {w['tflops']:.2f} TFLOPS")

        if "error" not in w and bw.get("shared", {}).get("read_gbs"):
            ai = w["tflops"] * 1e12 / (bw["shared"]["read_gbs"] * 1e9)
            results["ridge_point_flop_per_byte"] = ai
            print(f"\n  ridge point: {ai:.0f} FLOP/byte")
            print("  below this arithmetic intensity a kernel is memory-bound,")
            print("  which on this machine is nearly everything -> fuse aggressively.")
    finally:
        dev.destroy()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
