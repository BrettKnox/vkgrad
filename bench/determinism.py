"""Does the autotuner pick the same configuration twice, and a good one?

Section 47 claimed the tuner was nondeterministic: the same sweep in three
processes chose three configurations spanning 1.28x. That claim does not
survive this file. Scoring a pick by re-measuring it in its own process carries
1.1-1.2x of cross-process noise on these shapes, so 1.28x proved nothing. The
5% noise floor from bench/ab.py was measured on one shape with one config and
does not generalise, which is the same single-configuration mistake this
project has now made ten times.

So selection is measured against a *shared* ground truth instead. Every
configuration any run picked is measured once, interleaved, in a single
process. Each run's pick is then scored by that one table. Cross-run spread now
reflects only which configuration was chosen, because the measurement is
common to all of them.

Two defects are under test, both instances of comparing things measured under
different conditions:

  1. Stage two scored the swizzled variants against a stage-one number for
     group_m=0 that was minutes old and taken under a different protocol.
  2. Stage one gave every candidate the same repeat *count*, so per-submit
     overhead contaminated each measurement in inverse proportion to kernel
     speed, systematically penalising the fastest tiles.

VKGRAD_LEGACY_TUNE=1 restores both, so the fix is measured against the unfixed
tuner in alternating processes rather than against a remembered number.

  python -m bench.determinism --rounds 3
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels import Matmul, autotune_matmul, warmup  # noqa: E402
from vk import Device  # noqa: E402

SHAPES = [
    (2048, 768, 192, False, "qkv forward"),
    (1024, 1024, 1024, False, "square"),
    (192, 768, 2048, True, "dW qkv"),
]

# Deliberately not kernels._repeat_for: that function is one of the things under
# test and it changes behaviour under VKGRAD_LEGACY_TUNE.
_BUDGET_S = 0.02


def _repeat(t_one):
    return max(1, int(_BUDGET_S / max(t_one, 1e-6)))


def run_one():
    """Tune every shape from scratch and report what was picked. Prints JSON."""
    dev = Device()
    warmup(dev)
    out = {}
    try:
        for m, n, k, ta, label in SHAPES:
            mm, best = autotune_matmul(dev, m, n, k, trans_a=ta, use_cache=False)
            out[label] = {"pick": [mm.sg, mm.wm, mm.wn, int(mm.lds), mm.group_m],
                          "claimed": best["tflops"]}
            mm.destroy()
    finally:
        dev.destroy()
    print(json.dumps(out))


def score(picks_path):
    """Measure every picked configuration once, interleaved. Prints JSON.

    One process, one warmup, round-robin across configurations, median of many.
    Whatever bias remains is then identical for every configuration, which is
    the only property the scoring actually needs.
    """
    picks = json.load(open(picks_path))
    dev = Device()
    warmup(dev)
    out = {}
    try:
        for m, n, k, ta, label in SHAPES:
            want = [tuple(p) for p in picks.get(label, [])]
            if not want:
                continue
            a = dev.buffer(m * k * 2, "device")
            b = dev.buffer(k * n * 2, "device")
            c = dev.buffer(m * n * 4, "device")
            built = []
            try:
                for sg, wm, wn, lds, gm in want:
                    mm = Matmul(dev, sg, wm, wn, ta, False, lds=bool(lds),
                                group_m=gm)
                    built.append(((sg, wm, wn, lds, gm), mm))
                reps = {}
                for kk, mm in built:
                    reps[kk] = _repeat(mm(a, b, c, m, n, k))
                samples = {kk: [] for kk, _ in built}
                for _ in range(15):
                    for kk, mm in built:
                        r = reps[kk]
                        samples[kk].append(
                            2 * m * n * k / (mm(a, b, c, m, n, k, repeat=r) / r)
                            / 1e12)
                out[label] = {"|".join(map(str, kk)): statistics.median(v)
                              for kk, v in samples.items()}
            finally:
                for _, mm in built:
                    mm.destroy()
                for x in (a, b, c):
                    x.destroy()
    finally:
        dev.destroy()
    print(json.dumps(out))


def key_of(pick):
    return "|".join(map(str, pick))


def show(pick):
    sg, wm, wn, lds, gm = pick
    return f"sg{sg} wm{wm} wn{wn} lds{lds} g{gm}"


def summarise(name, runs, truth):
    print(f"\n  {name}")
    for _, _, _, _, label in SHAPES:
        rows = [r[label] for r in runs if label in r]
        tt = truth.get(label, {})
        if not rows or not tt:
            continue
        best = max(tt.values())
        got = [tt.get(key_of(r["pick"])) for r in rows]
        got = [g for g in got if g is not None]
        if not got:
            continue
        uniq = {key_of(r["pick"]) for r in rows}
        regret = 100 * (1 - statistics.mean(got) / best)
        worst = 100 * (1 - min(got) / best)
        print(f"    {label:<14} {len(uniq)} distinct in {len(rows)}   "
              f"picked {min(got):.2f}-{max(got):.2f} of {best:.2f} TFLOPS   "
              f"regret {regret:4.1f}% mean, {worst:4.1f}% worst")
        # A tuner that overstates its winner is a different failure from one
        # that picks badly, and only a shared re-measurement separates them.
        ratio = statistics.median(
            [r["claimed"] / tt[key_of(r["pick"])] for r in rows
             if key_of(r["pick"]) in tt])
        print(f"    {'':<14} claimed / actual {ratio:.2f}x   "
              f"{', '.join(sorted(show(r['pick']) for r in rows))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--single", action="store_true", help="internal")
    ap.add_argument("--score", help="internal: path to picks json")
    args = ap.parse_args()

    if args.single:
        run_one()
        return
    if args.score:
        score(args.score)
        return

    here = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(here))
    arms = [("fixed", {}), ("legacy (both defects)", {"VKGRAD_LEGACY_TUNE": "1"})]

    print(f"{args.rounds} rounds, arms alternating within each round\n")
    results = {name: [] for name, _ in arms}
    for i in range(args.rounds):
        # Alternate within a round rather than finishing one arm first, so drift
        # over the session hits both arms equally. Same reason the tuner itself
        # round-robins.
        for name, env in arms:
            e = dict(os.environ)
            e.update(env)
            p = subprocess.run([sys.executable, here, "--single"],
                               capture_output=True, text=True, cwd=root, env=e)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(f"  round {i + 1} {name}: FAILED\n"
                      f"{p.stdout[-400:]}{p.stderr[-400:]}")
                continue
            got = json.loads(line[-1])
            results[name].append(got)
            print(f"  round {i + 1} {name:<22} " + "  ".join(
                f"{lab}={show(got[lab]['pick'])}" for *_, lab in SHAPES
                if lab in got))

    # Every configuration anyone picked, measured once, in one process.
    picks = {}
    for runs in results.values():
        for r in runs:
            for label, v in r.items():
                picks.setdefault(label, [])
                if v["pick"] not in picks[label]:
                    picks[label].append(v["pick"])
    path = os.path.join(root, "bench", "runs", "picks.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(picks, f)
    n = sum(len(v) for v in picks.values())
    print(f"\n  scoring {n} distinct configurations against one shared "
          f"measurement")
    p = subprocess.run([sys.executable, here, "--score", path],
                       capture_output=True, text=True, cwd=root)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not line:
        print(f"  scoring FAILED\n{p.stdout[-400:]}{p.stderr[-400:]}")
        return
    truth = json.loads(line[-1])

    for name, _ in arms:
        summarise(name, results[name], truth)

    print("\n  Regret is how far below the best configuration anyone found the")
    print("  tuner's pick lands, scored by one shared measurement, so it")
    print("  reflects selection alone and not the noise of re-measuring.")


if __name__ == "__main__":
    main()
