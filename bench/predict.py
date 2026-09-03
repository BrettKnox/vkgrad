"""Can an analytic cost model replace the empirical autotuner?

If the matmul is bandwidth-bound, the config that moves the fewest bytes should
be the fastest, and tile selection becomes arithmetic instead of a benchmark
sweep. This tests that directly: for every candidate config on a shape, compute
predicted DRAM traffic, measure actual throughput, and ask two questions.

  1. Does the minimum-traffic config match the measured fastest one?
  2. Do predicted and measured orderings agree (Spearman rho over all configs)?

It also tests the obvious correction, since minimising traffic favours large
tiles and large tiles can starve the GPU of workgroups: filter to configs with
at least N workgroups per CU first, then minimise traffic.

  python -m bench.predict
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import CONFIGS, Matmul, warmup  # noqa: E402
from vk import Device  # noqa: E402

SHAPES = [
    (2048, 768, 192, False, False, "qkv forward"),
    (1024, 1024, 1024, False, False, "square"),
    (2048, 192, 768, False, True, "dX proj"),
    (2048, 96, 192, False, False, "head forward"),
    (192, 768, 2048, True, False, "dW qkv"),
    (192, 192, 2048, True, False, "dW proj"),
]


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def sweep(dev, m, n, k, ta, tb):
    a = dev.buffer(m * k * 2, "device")
    b = dev.buffer(k * n * 2, "device")
    c = dev.buffer(m * n * 4, "device")
    rows = []
    try:
        for sg, wm, wn, lds in CONFIGS:
            if lds and (ta or tb):
                continue
            try:
                mm = Matmul(dev, sg, wm, wn, ta, tb, lds=lds)
            except ValueError:
                continue
            if not mm.fits(m, n, k):
                mm.destroy()
                continue
            groups = (m // mm.bm) * (n // mm.bn)
            if groups < 12:
                mm.destroy()
                continue
            mm(a, b, c, m, n, k)
            t = min(mm(a, b, c, m, n, k, repeat=3) / 3 for _ in range(2))
            rows.append(dict(traffic=mm.traffic_bytes(m, n, k), time=t,
                             tflops=2 * m * n * k / t / 1e12, groups=groups,
                             tile=(mm.bm, mm.bn), lds=lds))
            mm.destroy()
    finally:
        for x in (a, b, c):
            x.destroy()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = Device()
    # Mandatory: GPU clocks ramp ~2.7x over the first ~30 dispatches, which
    # would make every config measured early look worse than it is.
    warmup(dev)
    cus = (dev.shader_core_props() or {}).get("cus", 12)
    print(f"{dev.name}, {cus} CUs\n")
    print("% of best achieved by picking min-traffic among configs with >= N workgroups\n")
    head = "shape".ljust(24) + "".join(f"{x:>10}" for x in
                                       ("traffic", f">={cus}", f">={4 * cus}",
                                        f">={8 * cus}", "rho", "best"))
    print(head)
    print("-" * len(head))

    out = []
    for m, n, k, ta, tb, label in SHAPES:
        rows = sweep(dev, m, n, k, ta, tb)
        if not rows:
            continue
        best = max(r["tflops"] for r in rows)
        pct = []
        for thr in (0, cus, 4 * cus, 8 * cus):
            cand = [r for r in rows if r["groups"] >= thr] or rows
            pick = min(cand, key=lambda r: r["traffic"])
            pct.append(100 * pick["tflops"] / best)
        rho = spearman([r["traffic"] for r in rows], [r["time"] for r in rows])
        name = f"{m}x{n}x{k} {label}"
        print(name.ljust(24) + "".join(f"{v:9.0f}%" for v in pct)
              + f"{rho:+10.2f}{best:9.2f}T")
        out.append(dict(shape=[m, n, k], trans_a=ta, trans_b=tb, label=label,
                        pct=pct, rho=rho, best_tflops=best, n_configs=len(rows)))

    print("\nReading: 100% means the analytic model picked the measured optimum.")
    print("rho near +1 means predicted and measured orderings agree; near 0 means")
    print("traffic carries no information about which config wins.")
    dev.destroy()

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
