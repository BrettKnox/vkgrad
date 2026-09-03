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


def test_finite_differences(dev):
    """Independent of the numpy reference: perturb a weight, watch the loss."""
    batch, n_in, n_hid, n_cls, pad = 16, 32, 16, 4, 16
    ctx, model, xb, yb, x, y = build(dev, batch, n_in, n_hid, n_cls, pad, seed=3)
    try:
        head = model.layers[-1]
        model.forward_backward(xb, yb)
        grad = head.W.grad_numpy()
        W = head.W.numpy().copy()

        rng = np.random.default_rng(1)
        picks = [(int(rng.integers(0, n_hid)), int(rng.integers(0, n_cls)))
                 for _ in range(4)]
        eps = 0.05  # large: the forward runs through f16 activations
        worst = 0.0
        for (i, j) in picks:
            Wp = W.copy(); Wp[i, j] += eps
            head.W.set(Wp)
            lp = model.forward_backward(xb, yb)
            Wm = W.copy(); Wm[i, j] -= eps
            head.W.set(Wm)
            lm = model.forward_backward(xb, yb)
            head.W.set(W)
            fd = (lp - lm) / (2 * eps)
            e = abs(fd - grad[i, j]) / max(abs(grad[i, j]), 1e-4)
            worst = max(worst, e)
            assert e < 0.05, f"W[{i},{j}]: finite diff {fd:.6f} vs grad {grad[i, j]:.6f}"
        print(f"  finite differences (4 entries) worst rel err {worst:.2e}   OK")
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
