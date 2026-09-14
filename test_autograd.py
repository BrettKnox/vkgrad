"""Gradient checks. Tiny model, small dispatches: safe on a busy machine."""

import numpy as np

from autograd import MLP, AdamW, Ctx
from vk import Device

GELU_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x ** 3)))


def dgelu(x):
    t = np.tanh(GELU_C * (x + 0.044715 * x ** 3))
    sech2 = 1.0 - t * t
    return 0.5 * (1 + t) + 0.5 * x * sech2 * GELU_C * (1 + 3 * 0.044715 * x * x)


def numpy_forward_backward(x, W1, b1, W2, b2, y, n_cls):
    """Independent reference for the same 2-layer model, entirely in f32."""
    B = x.shape[0]
    z1 = x @ W1 + b1
    h = gelu(z1)
    logits = h @ W2 + b2

    valid = logits[:, :n_cls]
    mx = valid.max(1, keepdims=True)
    s = np.exp(valid - mx).sum(1, keepdims=True)
    logZ = mx + np.log(s)
    loss = float((logZ[:, 0] - valid[np.arange(B), y]).mean())

    dlogits = np.zeros_like(logits)
    probs = np.exp(valid - logZ)
    onehot = np.zeros_like(valid)
    onehot[np.arange(B), y] = 1.0
    dlogits[:, :n_cls] = (probs - onehot) / B

    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dh = dlogits @ W2.T
    dz1 = dh * dgelu(z1)
    dW1 = x.T @ dz1
    db1 = dz1.sum(0)
    return loss, dW1, db1, dW2, db2


def relerr(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-8))


def build(dev, batch=16, n_in=32, n_hid=16, n_cls=4, pad=16, seed=0):
    ctx = Ctx(dev)
    model = MLP(ctx, batch, [n_in, n_hid], n_cls, pad_classes=pad)
    rng = np.random.default_rng(seed)

    # Inputs are f16 on the GPU; use f16-exact values so the reference sees
    # exactly the same numbers and precision is not what we are testing.
    x = rng.standard_normal((batch, n_in)).astype(np.float16).astype(np.float32)
    y = rng.integers(0, n_cls, batch).astype(np.uint32)
    # Linear initialises from hash(name), which Python randomises per process,
    # so redraw every parameter from the seed: a failure then reproduces at
    # any PYTHONHASHSEED. Biases are nonzero so the forward bias add is tested.
    for p in model.params():
        std = np.sqrt(2.0 / p.shape[0]) if len(p.shape) == 2 else 0.1
        p.set(rng.standard_normal(p.shape).astype(np.float32) * std)

    xb = ctx.buf(batch * n_in * 2, "shared")
    xb.array(np.float16, (batch, n_in))[:] = x.astype(np.float16)
    xb.flush()
    yb = ctx.buf(batch * 4, "shared")
    yb.array(np.uint32, (batch,))[:] = y
    yb.flush()
    return ctx, model, xb, yb, x, y


def test_gradients(dev):
    batch, n_in, n_hid, n_cls, pad = 16, 32, 16, 4, 16
    ctx, model, xb, yb, x, y = build(dev, batch, n_in, n_hid, n_cls, pad)
    try:
        fc0, head = model.layers
        W1, b1 = fc0.W.numpy().copy(), fc0.b.numpy().copy()
        W2, b2 = head.W.numpy().copy(), head.b.numpy().copy()

        loss = model.forward_backward(xb, yb)
        ref_loss, dW1, db1, dW2, db2 = numpy_forward_backward(
            x, W1, b1, W2, b2, y, n_cls)

        assert abs(loss - ref_loss) < 2e-3, f"loss {loss} vs reference {ref_loss}"
        print(f"  loss           {loss:.6f}  vs numpy {ref_loss:.6f}   OK")

        # f16 activations feed the dW matmuls, so a few 1e-3 of relative error
        # is expected and correct; a wrong formula shows up orders above this.
        for got, ref, name in ((head.W.grad_numpy(), dW2, "dW head"),
                               (head.b.grad_numpy(), db2, "db head"),
                               (fc0.W.grad_numpy(), dW1, "dW fc0"),
                               (fc0.b.grad_numpy(), db1, "db fc0")):
            e = relerr(got, ref)
            assert e < 2e-2, f"{name}: relative error {e:.3e}"
            print(f"  {name:14s} rel err {e:.2e}   OK")
    finally:
        ctx.destroy()


def test_finite_differences(dev, seed=3):
    """Perturb a weight, watch the loss, compare with the GPU's gradient.

    eps is large because the loss comes through f16 activations (at 0.02 the
    rounding more than doubles the worst error), and a central difference at
    0.05 is off by O(eps^2) in absolute terms: relative to a near-zero gradient
    that is unbounded, so 4 head.W entries picked at random failed for about 2%
    of inits. Each tensor's two largest entries are checked instead, ranked by the
    float64 numpy reference rather than by the GPU's own gradient, so a bug that
    zeroes or shrinks a gradient cannot steer the check away from it. The
    verdict compares the GPU gradient with the finite difference only; the
    reference picks the entries and scales the error.
    """
    batch, n_in, n_hid, n_cls, pad = 16, 32, 16, 4, 16
    ctx, model, xb, yb, x, y = build(dev, batch, n_in, n_hid, n_cls, pad, seed=seed)
    try:
        params = list(model.params())      # fc0.W, fc0.b, head.W, head.b
        w64 = [p.numpy().astype(np.float64) for p in params]
        refs = numpy_forward_backward(x.astype(np.float64), *w64, y, n_cls)[1:]
        model.forward_backward(xb, yb)
        # Read them all now: every perturbed forward_backward overwrites them.
        grads = [p.grad_numpy() for p in params]
        eps = 0.05
        worst = 0.0
        for p, ref, grad in zip(params, refs, grads):
            w = p.numpy().copy()
            for flat in np.argsort(-np.abs(ref).ravel())[:2]:
                i = np.unravel_index(flat, ref.shape)
                wp = w.copy(); wp[i] += eps
                p.set(wp)
                lp = model.forward_backward(xb, yb)
                wm = w.copy(); wm[i] -= eps
                p.set(wm)
                lm = model.forward_backward(xb, yb)
                p.set(w)
                fd = (lp - lm) / (2 * eps)
                e = abs(fd - float(grad[i])) / abs(ref[i])
                worst = max(worst, e)
                assert e < 0.05, (f"{p.name}{tuple(int(k) for k in i)}: finite diff "
                                  f"{fd:.6f} vs grad {grad[i]:.6f}, rel err {e:.3e}")
        print(f"  finite differences (8 entries) worst rel err {worst:.2e}   OK")
        return worst
    finally:
        ctx.destroy()


def test_adamw_decreases_loss(dev):
    """The optimiser must actually optimise: overfit one fixed batch."""
    batch, n_in, n_hid, n_cls, pad = 32, 64, 32, 8, 16
    ctx, model, xb, yb, _, _ = build(dev, batch, n_in, n_hid, n_cls, pad, seed=5)
    try:
        opt = AdamW(ctx, model.params(), lr=3e-3)
        first = model.forward_backward(xb, yb)
        for _ in range(200):
            model.forward_backward(xb, yb)
            opt.step()
        last = model.forward_backward(xb, yb)
        assert last < first * 0.2, f"loss {first:.4f} -> {last:.4f}, not learning"
        print(f"  AdamW overfit 1 batch: {first:.4f} -> {last:.4f}   OK")
    finally:
        ctx.destroy()


def main():
    dev = Device()
    print(dev)
    try:
        test_gradients(dev)
        test_finite_differences(dev)
        test_adamw_decreases_loss(dev)
        print("\nall autograd checks passed")
    finally:
        dev.destroy()


if __name__ == "__main__":
    main()
