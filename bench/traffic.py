"""Is the transformer step actually bandwidth-bound?

RESEARCH.md claims the flat GPU-vs-CPU speedup is caused by both processors
sharing one memory controller. That is inferred from the shape of the speedup
curve, not measured. This measures it: total the bytes each dispatch moves,
divide by the step time, and compare against the DRAM bandwidth the same
machine actually achieves.

Two bounds are reported because neither is exactly right on its own:

  compulsory  every buffer touched once. A lower bound: assumes perfect reuse.
  amplified   every matmul tile re-reads its operands from DRAM. An upper
              bound: assumes the L2 catches nothing.

The truth sits between them. If even the *compulsory* figure implies a
bandwidth close to the measured ceiling, the step is bandwidth-bound and there
is little headroom. If the *amplified* figure implies far less than the
ceiling, the step is latency- or occupancy-bound and there is headroom.
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW  # noqa: E402
from kernels import warmup  # noqa: E402
from transformer import GPT, TCtx  # noqa: E402
from vk import Device  # noqa: E402

PEAK_READ_GBS = 79.75   # measured, bench/roofline.py
MARGINAL_GBS = 85.1     # measured causally, bench/ballast.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmodel", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    B, T, D, H, L, V = args.batch, args.seq, args.dmodel, args.heads, args.layers, 96
    dev = Device()
    ctx = TCtx(dev, tune=args.tune)
    model = GPT(ctx, B, T, D, H, L, V)
    opt = AdamW(ctx, model.params(), lr=3e-4)

    rows = B * T
    idb = ctx.buf(rows * 4, "shared")
    tgb = ctx.buf(rows * 4, "shared")
    idb.array(np.uint32, (rows,))[:] = np.random.randint(0, V, rows)
    tgb.array(np.uint32, (rows,))[:] = np.random.randint(0, V, rows)
    idb.flush()
    tgb.flush()

    g = dev.graph("step")
    model.record(idb, tgb, g)
    opt.record(g)
    g.finish()

    warmup(dev)
    for _ in range(3):
        g.submit()
    step = min(g.submit() for _ in range(7))

    comp = sum(t[1] for t in g.traffic)
    ampl = sum(t[2] for t in g.traffic)

    print(f"model d={D} h={H} L={L} batch={B} seq={T}, {model.n_params():,} params")
    print(f"step {step * 1e3:.2f} ms over {g.n_dispatch} dispatches\n")

    print(f"  compulsory traffic  {comp / 2**20:8.1f} MiB  -> "
          f"{comp / step / 1e9:6.1f} GB/s  ({100 * comp / step / 1e9 / PEAK_READ_GBS:.0f}% of peak)")
    print(f"  amplified traffic   {ampl / 2**20:8.1f} MiB  -> "
          f"{ampl / step / 1e9:6.1f} GB/s  ({100 * ampl / step / 1e9 / PEAK_READ_GBS:.0f}% of peak)")
    print(f"  measured DRAM ceiling {PEAK_READ_GBS:.1f} GB/s\n")

    # bench/ballast.py established causally that injected bytes cost time at
    # 85.1 GB/s, so the amplified byte count can be turned straight into a
    # predicted step time and checked against the clock.
    pred = ampl / (MARGINAL_GBS * 1e9)
    err = 100 * (pred - step) / step
    print(f"  predicted step from traffic alone: {pred * 1e3:6.2f} ms "
          f"({err:+.1f}% vs measured {step * 1e3:.2f} ms)\n")

    by = collections.defaultdict(lambda: [0, 0, 0])
    for name, c, a in g.traffic:
        kind = "matmul" if name.startswith("mm") else name
        by[kind][0] += 1
        by[kind][1] += c
        by[kind][2] += a
    print("  by kernel, amplified bytes:")
    for kind, (n, c, a) in sorted(by.items(), key=lambda kv: -kv[1][2])[:10]:
        print(f"    {kind:18s} x{n:3d}  {a / 2**20:8.1f} MiB  ({100 * a / ampl:4.1f}%)")

    lo = comp / step / 1e9
    hi = ampl / step / 1e9
    print()
    if hi < 0.4 * PEAK_READ_GBS:
        print("  VERDICT: NOT bandwidth-bound. Even assuming zero cache reuse the step")
        print("  moves too few bytes to explain its duration, so it is latency or")
        print("  occupancy bound and there is real headroom left.")
    elif 0.8 * PEAK_READ_GBS <= hi <= 1.25 * PEAK_READ_GBS:
        print(f"  VERDICT: bandwidth-bound. The amplified bound lands at {hi:.0f} GB/s "
              f"against a\n  measured ceiling of {PEAK_READ_GBS:.0f} GB/s, so effective "
              "traffic is close to the\n  no-reuse bound and the step is running at the "
              "DRAM limit. The lever is\n  fewer bytes (wider tiles, fewer intermediates), "
              "not more FLOPs.")
    elif lo > 0.7 * PEAK_READ_GBS:
        print("  VERDICT: bandwidth-bound even at the optimistic bound. Little headroom.")
    else:
        print("  VERDICT: inconclusive between the bounds; cache behaviour decides it.")

    ctx.destroy()
    dev.destroy()


if __name__ == "__main__":
    main()
