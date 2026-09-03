"""Is DRAM traffic *causal* for step time, or merely correlated with it?

bench/traffic.py shows the transformer step's effective traffic sits on the
measured DRAM ceiling. That is consistent with being bandwidth-bound, but it
does not prove that removing a byte removes time: a step could sit near the
roofline and still be limited by something else.

This is the controlled version. A ballast kernel that does nothing but read N
MiB is inserted into the recorded training step, and the step is timed for
several N. If the machine is genuinely bandwidth-bound, step time must rise
linearly with N at a slope of exactly 1/bandwidth, and the fitted slope should
reproduce the independently measured DRAM read bandwidth.

The fitted slope is also directly useful: it is the exchange rate between bytes
saved and time saved, which says whether any given traffic optimisation is
worth implementing.

  python -m bench.ballast --tune
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW  # noqa: E402
from compile import compile_glsl  # noqa: E402
from kernels import warmup  # noqa: E402
from transformer import GPT, TCtx  # noqa: E402
from vk import Device  # noqa: E402

PEAK_READ_GBS = 79.75

# Pure streaming read. The conditional write is never taken but the compiler
# cannot prove it, so the loads survive optimisation.
BALLAST_SRC = """#version 450
layout(local_size_x = 256) in;
layout(binding = 0) readonly buffer A { vec4 a[]; };
layout(binding = 1) buffer B { vec4 b[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    vec4 acc = vec4(0.0);
    for (uint i = gl_GlobalInvocationID.x; i < p.n; i += stride) acc += a[i];
    if (acc.x == 1234.5678) b[gl_GlobalInvocationID.x] = acc;
}
"""


def build_step(dev, model, opt, idb, tgb, ballast_kernel, ballast_bufs, mib):
    """One graph per ballast level over the *same* model and buffers.

    Recording several graphs from one model keeps the descriptor sets shared
    (they are cached per buffer tuple) and, more importantly, makes the only
    difference between the variants the ballast itself.
    """
    g = dev.graph(f"step_{mib}")
    model.record(idb, tgb, g)
    opt.record(g)
    if mib:
        n_vec4 = (mib << 20) // 16
        g.record(ballast_kernel, ballast_bufs, 2048, struct.pack("I", n_vec4))
    g.finish()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmodel", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--levels", default="0,64,128,256,384,512")
    args = ap.parse_args()

    B, T, D, H, L, V = 16, 128, args.dmodel, args.heads, args.layers, 96
    levels = [int(x) for x in args.levels.split(",")]

    dev = Device()
    warmup(dev)
    ctx = TCtx(dev, tune=args.tune)

    big = max(levels)
    src = ctx.buf((big << 20) if big else (1 << 20), "device")
    dst = ctx.buf(1 << 20, "device")
    bk = dev.kernel(compile_glsl(BALLAST_SRC, name="ballast"), 2, 4, name="ballast")

    print(f"model d={D} h={H} L={L}, ballast levels {levels} MiB\n")
    print(f"{'ballast MiB':>12}{'step ms':>10}{'delta ms':>10}{'implied GB/s':>14}")
    print("-" * 46)

    # Build every graph first, then time them interleaved: comparisons on this
    # hardware are only trustworthy when the variants alternate within one run.
    model = GPT(ctx, B, T, D, H, L, V)
    opt = AdamW(ctx, model.params(), lr=3e-4)
    rows = B * T
    idb = ctx.buf(rows * 4, "shared")
    tgb = ctx.buf(rows * 4, "shared")
    idb.array(np.uint32, (rows,))[:] = np.random.randint(0, V, rows)
    tgb.array(np.uint32, (rows,))[:] = np.random.randint(0, V, rows)
    idb.flush()
    tgb.flush()

    graphs = [(mib, build_step(dev, model, opt, idb, tgb, bk, [src, dst], mib))
              for mib in levels]

    for _, g in graphs:
        for _ in range(3):
            g.submit()

    samples = {mib: [] for mib, _ in graphs}
    for _ in range(9):
        for mib, g in graphs:
            samples[mib].append(g.submit())
    times = {mib: min(v) for mib, v in samples.items()}

    base = times[levels[0]]
    xs, ys = [], []
    for mib in levels:
        d_ms = (times[mib] - base) * 1e3
        gbs = ((mib << 20) / (times[mib] - base) / 1e9) if mib and times[mib] > base else float("nan")
        print(f"{mib:>12}{times[mib] * 1e3:>10.2f}{d_ms:>10.2f}"
              f"{gbs:>14.1f}" if mib else
              f"{mib:>12}{times[mib] * 1e3:>10.2f}{0.0:>10.2f}{'-':>14}")
        xs.append(mib << 20)
        ys.append(times[mib])

    # Least squares slope of time vs bytes.
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    slope = num / den
    fitted_gbs = 1 / slope / 1e9
    print("-" * 46)
    print(f"\n  fitted marginal bandwidth: {fitted_gbs:.1f} GB/s")
    print(f"  independently measured   : {PEAK_READ_GBS:.1f} GB/s")
    ratio = fitted_gbs / PEAK_READ_GBS
    print(f"  ratio                    : {ratio:.2f}\n")
    if 0.75 <= ratio <= 1.3:
        print("  Added bytes cost time at the full DRAM rate, so the step has no")
        print("  spare bandwidth: traffic is causal and every byte removed from the")
        print("  step is time removed at ~1/{:.0f} GB/s.".format(fitted_gbs))
    elif ratio > 1.3:
        print("  Added bytes cost LESS time than the DRAM rate implies, so the step")
        print("  has spare bandwidth and is partly bound by something else.")
    else:
        print("  Added bytes cost MORE time than the DRAM rate implies: the ballast")
        print("  is interfering beyond its bandwidth (cache pollution, or the extra")
        print("  dispatch is not overlapping).")

    ctx.destroy()
    dev.destroy()


if __name__ == "__main__":
    main()
