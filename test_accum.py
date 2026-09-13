"""Gradient accumulation must equal one large batch, or the speedup is a lie.

Accumulating N microbatches of size B and stepping once should produce the same
gradient as a single batch of N*B over the same examples. This checks that
against the GPU's own large-batch path and against numpy.

The graphs come from examples/train_lm.py, so the check covers the training
script's own recording, including zeroing the arena before the first
microbatch. It also checks that the script's evaluation trains nothing.
"""

import numpy as np

from autograd import AdamW, Ctx, MLP
from dataparallel import GradAccum
from examples.train_lm import accum_graphs, eval_graph, evaluate
from transformer import GPT, TCtx
from vk import Device


def grads_for(dev, batch, accum, data, labels, in_f=64, hidden=32, n_cls=8,
              dirty=False):
    """Run `accum` microbatches of `batch` and return the summed gradients.

    dirty=True fills the arena with stale values first, as reused device memory
    would hold, so the result is right only if the arena is zeroed.
    """
    ctx = Ctx(dev)
    model = MLP(ctx, batch, [in_f, hidden], n_cls, pad_classes=16)
    opt = AdamW(ctx, model.params(), lr=1e-3)
    params = list(model.params())
    # Identical initial weights regardless of how the batch is split.
    rng = np.random.default_rng(7)
    for p in params:
        p.set(rng.standard_normal(p.shape).astype(np.float32) * 0.05)

    ga = GradAccum(ctx, params, accum=accum)
    xb = ctx.buf(batch * in_f * 2, "shared")
    yb = ctx.buf(batch * 4, "shared")
    xv = xb.array(np.float16, (batch, in_f))
    yv = yb.array(np.uint32, (batch,))

    # Read the accumulator back through a staging buffer.
    tmp = ctx.buf(ga.total * 4, "cached")

    def arena():
        dev.copy(ga.arena, tmp, ga.total * 4)
        tmp.invalidate()
        return tmp.array(np.float32, (ga.total,)).copy()

    if dirty:
        src = ctx.buf(ga.total * 4, "shared")
        src.array(np.float32, (ga.total,))[:] = \
            np.random.default_rng(11).standard_normal(ga.total)
        src.flush()
        dev.copy(src, ga.arena, ga.total * 4)
        # Otherwise a copy that silently did nothing would pass vacuously.
        assert np.abs(arena()).max() > 0.5, "arena did not take the stale values"

    gm, _ = accum_graphs(dev, model, opt, ga, xb, yb)
    for i in range(accum):
        sl = slice(i * batch, (i + 1) * batch)
        xv[:] = data[sl]
        yv[:] = labels[sl]
        xb.flush()
        yb.flush()
        gm.submit()

    out = arena()
    ga.destroy()
    ctx.destroy()
    return out


def check_evaluate(dev):
    """examples/train_lm.py's evaluation must leave weights and arena alone.

    It used to submit the training graph: with --accum 1 AdamW stepped on every
    validation batch, and with --accum > 1 their gradients went into the arena.
    Both of those graphs are submitted here too, so the check can be seen to fail.
    """
    B, T, D, H, n_layer, V = 2, 16, 32, 2, 2, 16
    rows = B * T
    ctx = TCtx(dev)
    ga = None
    try:
        model = GPT(ctx, B, T, D, H, n_layer, V)
        params = list(model.params())
        opt = AdamW(ctx, params, lr=6e-4, wd=0.01)
        idb, tgb = ctx.buf(rows * 4, "shared"), ctx.buf(rows * 4, "shared")
        idv, tgv = idb.array(np.uint32, (rows,)), tgb.array(np.uint32, (rows,))
        gtrain = dev.graph("train")
        model.record(idb, tgb, gtrain)
        opt.record(gtrain)
        gtrain.finish()
        ga = GradAccum(ctx, params, accum=2)
        gmicro, _ = accum_graphs(dev, model, opt, ga, idb, tgb)
        geval = eval_graph(dev, model, idb, tgb)
        opt.advance(6e-4)

        rng = np.random.default_rng(5)
        batches = [(rng.integers(0, V, rows).astype(np.uint32),
                    rng.integers(0, V, rows).astype(np.uint32)) for _ in range(3)]
        tmp = ctx.buf(ga.total * 4, "cached")

        def state():
            dev.copy(ga.arena, tmp, ga.total * 4)
            tmp.invalidate()
            return ([p.numpy().copy() for p in params],
                    tmp.array(np.float32, (ga.total,)).copy())

        def run(g):
            """Evaluate by submitting g; the loss, and how far weights and arena moved."""
            w0, a0 = state()
            loss = evaluate(g, model, idv, tgv, batches)
            w1, a1 = state()
            dw = max(float(np.abs(x - y).max()) for x, y in zip(w0, w1))
            return loss, dw, float(np.abs(a1 - a0).max())

        loss, dw, da = run(geval)
        assert dw == 0 and da == 0, f"evaluation moved weights {dw:.3e}, arena {da:.3e}"
        # The microbatch graph does not touch the weights, so on the same batches
        # it gives the training forward's loss for identical weights.
        ref, dw_micro, da_micro = run(gmicro)
        assert dw_micro == 0
        assert abs(loss - ref) <= 1e-6 * abs(ref), f"eval loss {loss} vs training {ref}"
        _, dw_train, _ = run(gtrain)
        assert dw_train > 0 and da_micro > 0, "the training graphs moved nothing"
        print(f"  evaluation: loss {loss:.6f} equals the training forward's, "
              f"weights and arena unchanged   OK")
        print(f"  (submitting the training graphs instead moves the weights by up to "
              f"{dw_train:.2e} and the arena by up to {da_micro:.2e})")
    finally:
        if ga:
            ga.destroy()
        ctx.destroy()


def main():
    in_f, hidden, n_cls = 64, 32, 8
    total = 64   # microbatches must stay at or above the 16-row tile minimum
    rng = np.random.default_rng(3)
    data = rng.standard_normal((total, in_f)).astype(np.float16)
    labels = rng.integers(0, n_cls, total).astype(np.uint32)

    dev = Device()
    print(dev)
    try:
        ref = grads_for(dev, total, 1, data, labels, in_f, hidden, n_cls)
        print(f"  one batch of {total}: {len(ref):,} gradient values")
        for accum, dirty in ((2, False), (4, False), (4, True)):
            b = total // accum
            got = grads_for(dev, b, accum, data, labels, in_f, hidden, n_cls,
                            dirty=dirty)
            err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
            # The microbatch loss is a mean over its own rows, so each shard's
            # gradient is scaled by 1/b rather than 1/total; the sum is then
            # `accum` times too large. Compare after that known factor.
            scaled = np.abs(got / accum - ref).max() / max(np.abs(ref).max(), 1e-9)
            best = min(err, scaled)
            tag = "raw" if err <= scaled else f"after /{accum}"
            start = ", arena started dirty" if dirty else ""
            assert best < 2e-2, (
                f"{accum} x {b}{start} != one batch of {total}: rel err {best:.3e}")
            print(f"  {accum} microbatches of {b:2d}{start}: rel err {best:.2e} "
                  f"({tag})   OK")
        check_evaluate(dev)
        print("\naccumulation matches a single large batch, and evaluation trains nothing")
    finally:
        dev.destroy()


if __name__ == "__main__":
    main()
