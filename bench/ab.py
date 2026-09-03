"""Interleaved A/B measurement, and the noise floor that makes it necessary.

Nine of the corrections in RESULTS.md come from comparing two configurations
measured in separate runs. That mistake was made five times, twice after writing
sections warning against it, so this file exists to make the correct method the
easy one and to put a number on why it matters.

Two things live here:

  `interleaved(variants)`  alternates the variants within one process and
                           returns per-variant medians. Use this for every
                           comparison.

  `--noise`                measures the actual run-to-run spread on this
                           machine, which sets the smallest effect that a
                           cross-run comparison could ever resolve.

  python -m bench.ab --noise
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import Matmul, warmup  # noqa: E402
from vk import Device  # noqa: E402


def interleaved(variants, reps=11, warmups=3):
    """Median time per variant, alternating them within one process.

    `variants` maps a name to a zero-argument callable that performs one unit of
    work and returns its elapsed seconds. Alternating means any drift over the
    measurement window hits every variant equally instead of accumulating along
    whichever one is measured last.
    """
    for fn in variants.values():
        for _ in range(warmups):
            fn()
    samples = {k: [] for k in variants}
    for _ in range(reps):
        for name, fn in variants.items():
            samples[name].append(fn())
    return {k: statistics.median(v) for k, v in samples.items()}, samples


def _one_measurement(size=1024, reps=9):
    """A fixed, representative unit of work: one tuned matmul."""
    dev = Device()
    warmup(dev)
    a = dev.buffer(size * size * 2, "device")
    b = dev.buffer(size * size * 2, "device")
    c = dev.buffer(size * size * 4, "device")
    mm = Matmul(dev, 4, 2, 4)
    try:
        for _ in range(3):
            mm(a, b, c, size, size, size)
        ts = [mm(a, b, c, size, size, size) for _ in range(reps)]
        return statistics.median(ts), min(ts)
    finally:
        mm.destroy()
        for x in (a, b, c):
            x.destroy()
        dev.destroy()


def measure_noise(processes=7):
    """Compare within-run spread against across-run spread.

    The gap between them is the entire reason interleaving matters: an effect
    smaller than the across-run spread cannot be resolved by comparing two runs,
    no matter how carefully each is measured.
    """
    print(f"running {processes} separate processes, identical work in each\n")
    here = os.path.abspath(__file__)
    med, mins = [], []
    for i in range(processes):
        out = subprocess.run(
            [sys.executable, here, "--single"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(here)))
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(f"  process {i}: failed\n{out.stdout[-300:]}{out.stderr[-300:]}")
            continue
        d = json.loads(line[-1])
        med.append(d["median"])
        mins.append(d["min"])
        print(f"  process {i + 1}: median {d['median'] * 1e3:7.3f} ms   "
              f"best {d['min'] * 1e3:7.3f} ms")

    if len(med) < 3:
        print("\ntoo few successful processes to summarise")
        return

    within = _one_measurement()[0]
    print()
    print(f"  within one process, median          {within * 1e3:7.3f} ms")
    print(f"  across processes, median of medians "
          f"{statistics.median(med) * 1e3:7.3f} ms")
    spread = max(med) / min(med)
    spread_min = max(mins) / min(mins)
    print()
    print(f"  across-run spread (medians): {spread:.2f}x "
          f"({100 * (spread - 1):.0f}%)")
    print(f"  across-run spread (bests)  : {spread_min:.2f}x "
          f"({100 * (spread_min - 1):.0f}%)")
    print(f"  across-run stdev / mean    : "
          f"{100 * statistics.stdev(med) / statistics.mean(med):.1f}%")
    print()
    print("  Any effect smaller than the across-run spread cannot be resolved")
    print("  by comparing two separate runs. Effects that size must be measured")
    print("  interleaved within one process.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="internal: one measurement, prints JSON")
    ap.add_argument("--processes", type=int, default=7)
    args = ap.parse_args()

    if args.single:
        m, lo = _one_measurement()
        print(json.dumps({"median": m, "min": lo}))
        return
    if args.noise:
        measure_noise(args.processes)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
