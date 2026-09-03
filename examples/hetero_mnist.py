"""Heterogeneous data parallelism: the GPU and the CPU train the same model.

The shared gradient arena is ordinary host memory, so a CPU worker needs no
plumbing at all to join: numpy adds into the same array the GPU atomically adds
into. Weights live in host-visible memory, so the CPU reads them directly too.
No copies anywhere, in either direction.

This is a sharper test of the vendor-neutral collective than several logical
GPUs, because the two workers here genuinely differ: different instruction sets,
different memory paths, different arithmetic. If pooling works across that gap,
the AMD-plus-NVIDIA case is a smaller step, not a bigger one.

The outcome is not obvious. The GPU is already saturating DRAM, and the CPU
competes for the same controller, so the extra worker may cost more in
contention than it contributes in arithmetic. That is the measurement.

  python -m examples.hetero_mnist --steps 300
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW, Ctx, MLP  # noqa: E402
from dataparallel import HostArena, make_reduce_kernels  # noqa: E402
from examples.mnist import load_mnist, prep, synthetic  # noqa: E402
from vk import Device  # noqa: E402

GELU_C = np.sqrt(2.0 / np.pi)


def cpu_backward(x, y, W1, b1, W2, b2, n_cls):
    """Same 2-layer GELU MLP as the GPU model, in f32 numpy.

    Returns gradients in the same parameter order the GPU replica uses, so they
    can be summed into the shared arena without translation.
    """
    B = x.shape[0]
    z1 = x @ W1 + b1
    t = np.tanh(GELU_C * (z1 + 0.044715 * z1 ** 3))
    h = 0.5 * z1 * (1 + t)
    logits = h @ W2 + b2

    valid = logits[:, :n_cls]
    mx = valid.max(1, keepdims=True)
    e = np.exp(valid - mx)
    s = e.sum(1, keepdims=True)
    loss = float((np.log(s)[:, 0] + mx[:, 0] - valid[np.arange(B), y]).mean())

    dlog = np.zeros_like(logits)
    p = e / s
    p[np.arange(B), y] -= 1.0
    dlog[:, :n_cls] = p / B

    dW2 = h.T @ dlog
    db2 = dlog.sum(0)
    dh = dlog @ W2.T
    sech2 = 1 - t * t
    dz1 = dh * (0.5 * (1 + t) + 0.5 * z1 * sech2 * GELU_C * (1 + 3 * 0.044715 * z1 ** 2))
    dW1 = x.T @ dz1
    db1 = dz1.sum(0)
    return loss, [dW1, db1, dW2, db2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256, help="global batch")
    ap.add_argument("--cpu-frac", type=float, default=0.25,
                    help="fraction of the global batch the CPU worker takes")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-pad", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    xtr, ytr, _, _ = synthetic() if args.synthetic else load_mnist()
    xtr_p = prep(xtr, args.in_pad)
    xtr_f32 = xtr_p.astype(np.float32)
    n_cls = 10

    def run(cpu_frac):
        """One training run at a given CPU share. Returns samples/s and loss."""
        cpu_n = int(args.batch * cpu_frac) // 16 * 16
        gpu_n = args.batch - cpu_n
        dev = Device()
        ctx = Ctx(dev)
        model = MLP(ctx, gpu_n, [args.in_pad, args.hidden], n_cls, pad_classes=16)
        opt = AdamW(ctx, model.params(), lr=args.lr)
        params = list(model.params())
        K = make_reduce_kernels(dev)

        n_par = sum(p.n for p in params)
        arena = HostArena(n_par)
        abuf = arena.buffer_for(dev)
        offs, off = [], 0
        for p in params:
            offs.append(off)
            off += p.n

        xbuf = ctx.buf(gpu_n * args.in_pad * 2, "shared")
        ybuf = ctx.buf(gpu_n * 4, "shared")
        xv = xbuf.array(np.float16, (gpu_n, args.in_pad))
        yv = ybuf.array(np.uint32, (gpu_n,))

        # The GPU side must be ONE submit, not ~23 eager dispatches. Every
        # ctypes call takes the GIL, and a numpy worker thread holds it in 5 ms
        # slices, so an eager GPU loop and a CPU worker starve each other. A
        # recorded graph collapses the step to a single call that blocks inside
        # vkWaitForFences with the GIL released, which is what lets the two
        # actually run at the same time.
        gfwd = dev.graph("fwd")
        K["zero"]([abuf], off, graph=gfwd)
        tape = []
        logits = model.logits(xbuf, tape, graph=gfwd)
        ctx.K["softmax_ce"]([logits, ybuf, model.dlog16, model.dlog32, model.loss],
                            gpu_n, model.pad, model.n_classes, graph=gfwd)
        g16, g32 = model.dlog16, model.dlog32
        for fn in reversed(tape):
            g16, g32 = fn(g16, g32, graph=gfwd)
        for p_, o_ in zip(params, offs):
            K["push"]([p_.g32, abuf], p_.n, o_, graph=gfwd)
        gfwd.finish()

        gopt = dev.graph("opt")
        for p_, o_ in zip(params, offs):
            K["pull"]([abuf, p_.g32], p_.n, o_, graph=gopt)
            ctx.K["adamw"]([p_.w32, p_.g32, p_.m, p_.v, p_.w16, opt.hp], p_.n,
                           0.0 if p_.no_decay else opt.wd, graph=gopt)
        gopt.finish()

        rng = np.random.default_rng(0)
        n = len(xtr_p)
        losses = []
        cpu_out = {}

        # A PERSISTENT worker, not a thread per step. OpenBLAS builds its
        # thread team per calling thread, so spawning a fresh thread each step
        # pays that setup every time and turns a 0.8 ms shard into 18 ms.
        work = threading.Event()
        done = threading.Event()
        stop = [False]
        job = [None]

        def worker():
            while True:
                work.wait()
                work.clear()
                if stop[0]:
                    return
                sl = job[0]
                W1 = params[0].numpy(); b1 = params[1].numpy()
                W2 = params[2].numpy(); b2 = params[3].numpy()
                loss, grads = cpu_backward(xtr_f32[sl], ytr[sl], W1, b1, W2, b2, n_cls)
                cpu_out["loss"] = loss
                cpu_out["grads"] = grads
                done.set()

        wt = None
        if cpu_n:
            wt = threading.Thread(target=worker, daemon=True)
            wt.start()
            # warm the BLAS thread team before timing starts
            job[0] = np.arange(cpu_n)
            work.set(); done.wait(); done.clear()

        for _ in range(5):
            idx = rng.integers(0, n, gpu_n)
            xv[:] = xtr_p[idx]; yv[:] = ytr[idx]
            gfwd.submit(); opt.advance(args.lr); gopt.submit()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            idx = rng.integers(0, n, args.batch)
            gsl, csl = idx[:gpu_n], idx[gpu_n:]
            xv[:] = xtr_p[gsl]; yv[:] = ytr[gsl]

            if cpu_n:
                job[0] = csl
                work.set()

            gfwd.submit()          # blocks with the GIL released
            gl = model.read_loss()

            tot = gl
            if cpu_n:
                done.wait(); done.clear()
                for g, o_, p_ in zip(cpu_out["grads"], offs, params):
                    arena.view[o_:o_ + p_.n] += g.reshape(-1)
                tot = (gl * gpu_n + cpu_out["loss"] * cpu_n) / args.batch

            opt.advance(args.lr)
            gopt.submit()
            losses.append(tot)
        wall = time.perf_counter() - t0

        if wt is not None:
            stop[0] = True
            work.set()
            wt.join(timeout=1.0)
        abuf.destroy()
        for kk in K.values():
            kk.destroy()
        ctx.destroy()
        dev.destroy()
        return args.batch * args.steps / wall, float(np.mean(losses[-50:])), gpu_n, cpu_n

    print("heterogeneous data parallelism: one GPU + the CPU on the same die\n")
    print(f"{'cpu share':>10}{'split':>14}{'samples/s':>12}{'loss':>9}{'vs GPU only':>13}")
    print("-" * 58)
    base = None
    for frac in (0.0, 0.125, 0.25, 0.375, 0.5):
        sps, loss, gn, cn = run(frac)
        if base is None:
            base = sps
        print(f"{frac:10.3f}{f'{gn}+{cn}':>14}{sps:12,.0f}{loss:9.4f}{sps / base:12.2f}x")

    print("\nBoth workers add into the same host array. Zero bytes are copied")
    print("between them in either direction.")


if __name__ == "__main__":
    main()
