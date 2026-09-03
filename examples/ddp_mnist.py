"""Data-parallel MNIST across N Vulkan devices sharing gradients via host memory.

The point is not speed on one physical GPU: two logical devices on one chip
share the same hardware, so wall time cannot improve. The point is that the
mechanism is correct and vendor-neutral. If the replicas converge exactly like
single-device training while only gradients cross between them, then the same
code runs on a machine with an AMD iGPU and an NVIDIA card, with no NCCL.

  python -m examples.ddp_mnist --devices 2 --steps 200
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW, Ctx, MLP  # noqa: E402
from dataparallel import HostArena, Replica  # noqa: E402
from examples.mnist import load_mnist, prep, synthetic  # noqa: E402
from vk import Device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128, help="global batch")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-pad", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    xtr, ytr, xte, yte = synthetic() if args.synthetic else load_mnist()
    xtr_p = prep(xtr, args.in_pad)
    n_cls = 10
    nd = args.devices
    shard = args.batch // nd
    assert shard * nd == args.batch, "global batch must divide by device count"

    def build(dev):
        ctx = Ctx(dev)
        model = MLP(ctx, shard, [args.in_pad, args.hidden], n_cls, pad_classes=16)
        opt = AdamW(ctx, model.params(), lr=args.lr)
        xbuf = ctx.buf(shard * args.in_pad * 2, "shared")
        ybuf = ctx.buf(shard * 4, "shared")
        return ctx, model, opt, xbuf, ybuf

    devs = [Device() for _ in range(nd)]
    print(devs[0])
    print(f"{nd} logical devices, global batch {args.batch} = {nd} x {shard}")

    probe_ctx = Ctx(devs[0])
    probe = MLP(probe_ctx, shard, [args.in_pad, args.hidden], n_cls, pad_classes=16)
    n_params = sum(p.n for p in probe.params())
    probe_ctx.destroy()

    arena = HostArena(n_params)
    print(f"shared gradient arena: {n_params:,} floats "
          f"({arena.nbytes / 2**20:.1f} MiB of host memory, imported by all {nd})")

    reps = [Replica(d, build, arena) for d in devs]
    # Replicas must start from identical weights: same init, so verify not assume.
    w0 = [r.params[0].numpy().copy() for r in reps]
    assert all(np.array_equal(w0[0], w) for w in w0[1:]), "replicas differ at init"
    print("replicas identical at init                    OK\n")

    views = [(r.xbuf.array(np.float16, (shard, args.in_pad)),
              r.ybuf.array(np.uint32, (shard,))) for r in reps]

    rng = np.random.default_rng(0)
    n = len(xtr_p)
    t0 = time.perf_counter()
    losses = []
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n, args.batch)
        reps[0].zero_arena()
        tot = 0.0
        for r, (xv, yv), s in zip(reps, views, range(nd)):
            sl = idx[s * shard:(s + 1) * shard]
            xv[:] = xtr_p[sl]
            yv[:] = ytr[sl]
            tot += r.model.forward_backward(r.xbuf, r.ybuf)
            r.push_gradients()
        for r in reps:
            r.step_from_arena(args.lr, n_workers=nd)
        losses.append(tot / nd)
        if step % max(args.steps // 8, 1) == 0:
            print(f"  step {step:5d}  loss {np.mean(losses[-25:]):.4f}")
    wall = time.perf_counter() - t0

    # The claim: replicas never diverged, so only gradients ever crossed.
    drift = 0.0
    for i, r in enumerate(reps[1:], 1):
        for a, b in zip(reps[0].params, r.params):
            drift = max(drift, float(np.abs(a.numpy() - b.numpy()).max()))
    print(f"\n  wall {wall:.1f}s   loss {losses[0]:.4f} -> "
          f"{np.mean(losses[-25:]):.4f}")
    print(f"  max weight drift between replicas: {drift:.3e}")
    assert drift == 0.0, "replicas diverged; the exchange is not bit-exact"
    print("  replicas bit-identical after training       OK")
    print(f"  bytes transferred between devices: 0 "
          f"({arena.nbytes / 2**20:.1f} MiB shared in place)")

    for r in reps:
        r.destroy()
    for d in devs:
        d.destroy()


if __name__ == "__main__":
    main()
