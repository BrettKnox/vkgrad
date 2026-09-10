"""Run the headline MNIST benchmark N times and write the spread, not one lucky number.

RESULTS.md section 14 quoted a single run: 97.69% and 5.09x. A single run on a laptop
iGPU is a sample, not a measurement, and re-running it gave 4.75x. This writes
bench/mnist-repro.json so the README can cite a file instead of a memory.

    python bench/mnist_repro.py --runs 5
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent

ACC = re.compile(r"epoch (\d+)/(\d+)\s+loss ([\d.]+)\s+test acc ([\d.]+)%")
WALL = re.compile(r"wall ([\d.]+)s\s+best GPU step ([\d.]+) ms\s+([\d,]+) samples/s")
CPU = re.compile(r"CPU \(numpy/OpenBLAS\) step ([\d.]+) ms\s+([\d,]+) samples/s")
SPEED = re.compile(r"GPU speedup on a full training step: ([\d.]+)x")


def one(epochs):
    out = subprocess.run([sys.executable, "-m", "examples.mnist", "--epochs", str(epochs)],
                         cwd=ROOT, capture_output=True, text=True).stdout
    epochs_seen = [(int(a), float(loss), float(acc)) for a, _, loss, acc in ACC.findall(out)]
    w, cpu, sp = WALL.search(out), CPU.search(out), SPEED.search(out)
    if not (epochs_seen and w and cpu and sp):
        raise SystemExit("could not parse examples.mnist output:\n" + out[-1500:])
    return {
        "final_accuracy_pct": epochs_seen[-1][2],
        "final_loss": epochs_seen[-1][1],
        "wall_s": float(w.group(1)),
        "gpu_step_ms": float(w.group(2)),
        "cpu_step_ms": float(cpu.group(1)),
        "speedup_x": float(sp.group(1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()

    runs = []
    for i in range(a.runs):
        r = one(a.epochs)
        runs.append(r)
        print("run %d/%d  acc %.2f%%  speedup %.2fx  gpu step %.3f ms"
              % (i + 1, a.runs, r["final_accuracy_pct"], r["speedup_x"], r["gpu_step_ms"]))

    def spread(k):
        v = sorted(x[k] for x in runs)
        return {"min": v[0], "median": median(v), "max": v[-1]}

    result = {
        "command": "python -m examples.mnist --epochs %d" % a.epochs,
        "runs": a.runs,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accuracy_pct": spread("final_accuracy_pct"),
        "speedup_x": spread("speedup_x"),
        "gpu_step_ms": spread("gpu_step_ms"),
        "cpu_step_ms": spread("cpu_step_ms"),
        "each": runs,
    }
    # bench/runs/ is gitignored, so an artifact written there cannot be cited by the
    # README: that is exactly how section 14 ended up unciteable. This one lives where
    # the repo actually carries it.
    out = ROOT / "bench" / "mnist-repro.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: result[k] for k in
                             ("accuracy_pct", "speedup_x", "gpu_step_ms")}, indent=1))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
