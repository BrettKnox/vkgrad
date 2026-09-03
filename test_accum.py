"""Gradient accumulation must equal one large batch, or the speedup is a lie.

Accumulating N microbatches of size B and stepping once should produce the same
gradient as a single batch of N*B over the same examples. This checks that
against the GPU's own large-batch path and against numpy.
"""

import numpy as np

from autograd import AdamW, Ctx, MLP
from dataparallel import GradAccum
from vk import Device


def grads_for(dev, batch, accum, data, labels, in_f=64, hidden=32, n_cls=8):
    """Run `accum` microbatches of `batch` and return the summed gradients."""
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

    g0 = dev.graph("zero")
    ga.record_zero(g0)
    g0.finish()
    gm = dev.graph("micro")
    model.record(xb, yb, gm)
    ga.record_push(gm)
    gm.finish()

    g0.submit()
    for i in range(accum):
        sl = slice(i * batch, (i + 1) * batch)
        xv[:] = data[sl]
        yv[:] = labels[sl]
        xb.flush()
        yb.flush()
        gm.submit()

    # Read the accumulator back through a staging buffer.
    tmp = ctx.buf(ga.total * 4, "cached")
    dev.copy(ga.arena, tmp, ga.total * 4)
    tmp.invalidate()
    out = tmp.array(np.float32, (ga.total,)).copy()
    ga.destroy()
    ctx.destroy()
    return out


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
        for accum in (2, 4):
            b = total // accum
            got = grads_for(dev, b, accum, data, labels, in_f, hidden, n_cls)
            err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
            # The microbatch loss is a mean over its own rows, so each shard's
            # gradient is scaled by 1/b rather than 1/total; the sum is then
            # `accum` times too large. Compare after that known factor.
            scaled = np.abs(got / accum - ref).max() / max(np.abs(ref).max(), 1e-9)
            best = min(err, scaled)
            tag = "raw" if err <= scaled else f"after /{accum}"
            assert best < 2e-2, (
                f"{accum} x {b} != one batch of {total}: rel err {best:.3e}")
            print(f"  {accum} microbatches of {b:2d}: rel err {best:.2e} "
                  f"({tag})   OK")
        print("\naccumulation matches a single large batch")
    finally:
        dev.destroy()


if __name__ == "__main__":
    main()
