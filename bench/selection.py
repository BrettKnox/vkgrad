"""Does the autotuner's selection criterion actually matter?

`autotune_matmul` scores each candidate by its *fastest* observed time
(best-of-N). With residual measurement noise of ~1.28x even after warmup, a max
over ~85 candidates is biased toward whichever config got a lucky sample, not
whichever is genuinely fastest. That was listed as a limitation without being
tested. This tests it.

Method, which avoids re-running the tuner hundreds of times:

  1. Measure every candidate config N times (warmed up).
  2. Ground truth = median of all N samples per config, the config with the
     best ground-truth median is the true optimum.
  3. Bootstrap: repeatedly draw k samples per config from the collected pool,
     apply each selection rule, and record the *ground-truth* quality of the
     config it selected.

Regret is 1 - (ground truth of picked) / (ground truth of true best), so 0%
means the rule found the optimum.

  python -m bench.selection
"""

import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import CONFIGS, Matmul, warmup  # noqa: E402
from vk import Device  # noqa: E402

SHAPES = [
    (1024, 1024, 1024, False, "square"),
    (2048, 768, 192, False, "qkv forward"),
    (192, 768, 2048, True, "dW qkv"),
    (192, 192, 2048, True, "dW proj"),
]


def collect(dev, m, n, k, ta, samples):
    """Timing samples per config. One measurement pass, reused for every rule."""
    a = dev.buffer(m * k * 2, "device")
    b = dev.buffer(k * n * 2, "device")
    c = dev.buffer(m * n * 4, "device")
    pool = []
    try:
        cands = []
        for sg, wm, wn, lds in CONFIGS:
            if lds and ta:
                continue
            try:
                mm = Matmul(dev, sg, wm, wn, ta, False, lds=lds)
            except ValueError:
                continue
            if not mm.fits(m, n, k) or (m // mm.bm) * (n // mm.bn) < 12:
                mm.destroy()
                continue
            cands.append(mm)
        for mm in cands:
            mm(a, b, c, m, n, k)  # touch once so the first sample isn't cold
        # Round-robin rather than all samples of one config together, so any
        # residual drift hits every config equally instead of the last ones.
        times = {i: [] for i in range(len(cands))}
        for _ in range(samples):
            for i, mm in enumerate(cands):
                t = mm(a, b, c, m, n, k, repeat=3) / 3
                times[i].append(2 * m * n * k / t / 1e12)
        for i, mm in enumerate(cands):
            pool.append(((mm.bm, mm.bn, mm.lds), times[i]))
            mm.destroy()
    finally:
        for x in (a, b, c):
            x.destroy()
    return pool


def bootstrap(pool, k, trials, rng):
    """Expected regret of best-of-k vs median-of-k, over random subsamples."""
    truth = [statistics.median(v) for _, v in pool]
    best_truth = max(truth)
    regret = {"best-of-k": [], "median-of-k": []}
    for _ in range(trials):
        draws = [rng.choice(v, size=k, replace=True) for _, v in pool]
        for name, fn in (("best-of-k", np.max), ("median-of-k", np.median)):
            scores = [fn(d) for d in draws]
            picked = int(np.argmax(scores))
            regret[name].append(1 - truth[picked] / best_truth)
    return {n: 100 * statistics.mean(v) for n, v in regret.items()}, best_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=9)
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    dev = Device()
    warmup(dev)
    print(f"{dev.name}: {args.samples} samples per config, "
          f"{args.trials} bootstrap trials\n")
    print(f"{'shape':<26}{'configs':>8}{'noise':>8}"
          f"{'best-of-3':>11}{'median-of-3':>13}{'best-of-9':>11}")
    print("-" * 77)

    for m, n, k, ta, label in SHAPES:
        pool = collect(dev, m, n, k, ta, args.samples)
        if len(pool) < 4:
            continue
        # Per-config spread, as a sanity check on how noisy this shape is.
        noise = statistics.median([max(v) / min(v) for _, v in pool])
        r3, best = bootstrap(pool, 3, args.trials, rng)
        r9, _ = bootstrap(pool, args.samples, args.trials, rng)
        name = f"{m}x{n}x{k} {label}"
        print(f"{name:<26}{len(pool):>8}{noise:>7.2f}x"
              f"{r3['best-of-k']:>10.1f}%{r3['median-of-k']:>12.1f}%"
              f"{r9['best-of-k']:>10.1f}%")

    print("\nRegret is how far below the true optimum the selected config lands,")
    print("averaged over bootstrap trials. Lower is better; 0% means it always")
    print("found the best config.")
    dev.destroy()


if __name__ == "__main__":
    main()
