"""Train an MLP on MNIST on the GPU, and race it against the same model on the CPU.

  python -m examples.mnist --synthetic     # no data needed, checks the pipeline
  python -m examples.mnist                 # needs data/ (see --help)

Input features are padded 784 -> 1024 and classes 10 -> 16 so every matmul
dimension is tile-friendly. The padding is zeros and the padded logits are
masked in the softmax, so it changes nothing numerically.
"""

import argparse
import gzip
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW, Ctx, MLP, TrainStep  # noqa: E402
from vk import Device  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8).reshape(n, rows * cols)
        if magic == 2049:
            return np.frombuffer(f.read(), np.uint8)
        raise ValueError(f"bad magic {magic} in {path}")


def load_mnist():
    missing = [v for v in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, v))]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} MNIST file(s) in {DATA_DIR}: {missing}\n"
            "Download the four idx.gz files from "
            "https://storage.googleapis.com/cvdf-datasets/mnist/ into that folder, "
            "or run with --synthetic.")
    d = {k: _read_idx(os.path.join(DATA_DIR, v)) for k, v in FILES.items()}
    return (d["train_x"], d["train_y"].astype(np.uint32),
            d["test_x"], d["test_y"].astype(np.uint32))


def synthetic(n_train=12800, n_test=2560, n_cls=10, seed=0):
    """Linearly separable-ish clusters. Exercises the whole pipeline without
    a download; accuracy on it is not a meaningful model result."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_cls, 784)) * 2.0

    def make(n):
        y = rng.integers(0, n_cls, n)
        x = centers[y] + rng.standard_normal((n, 784))
        x = np.clip((x - x.min()) / (x.max() - x.min()) * 255, 0, 255)
        return x.astype(np.uint8), y.astype(np.uint32)

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte


def prep(x, in_pad):
    """uint8 pixels -> zero-padded, normalised f16 in one shot."""
    out = np.zeros((x.shape[0], in_pad), np.float16)
    out[:, :x.shape[1]] = (x.astype(np.float32) / 255.0).astype(np.float16)
    return out


# ---------------------------------------------------------------- CPU baseline

GELU_C = np.sqrt(2.0 / np.pi)


def cpu_train_steps(xb, yb, sizes, n_cls, steps=20, lr=1e-3):
    """The same model in numpy/OpenBLAS. Times a full fwd+bwd+update step so
    the comparison is a training step, not a bare matmul."""
    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((sizes[0], sizes[1])) * np.sqrt(2 / sizes[0])).astype(np.float32)
    b1 = np.zeros(sizes[1], np.float32)
    W2 = (rng.standard_normal((sizes[1], n_cls)) * np.sqrt(2 / sizes[1])).astype(np.float32)
    b2 = np.zeros(n_cls, np.float32)
    x = xb.astype(np.float32)
    B = x.shape[0]
    idx = np.arange(B)

    best = float("inf")
    for _ in range(steps):
        t0 = time.perf_counter()
        z1 = x @ W1 + b1
        t = np.tanh(GELU_C * (z1 + 0.044715 * z1 ** 3))
        h = 0.5 * z1 * (1 + t)
        logits = h @ W2 + b2
        mx = logits.max(1, keepdims=True)
        e = np.exp(logits - mx)
        p = e / e.sum(1, keepdims=True)
        dlog = p.copy()
        dlog[idx, yb] -= 1.0
        dlog /= B
        dW2 = h.T @ dlog
        db2 = dlog.sum(0)
        dh = dlog @ W2.T
        sech2 = 1 - t * t
        dz1 = dh * (0.5 * (1 + t) + 0.5 * z1 * sech2 * GELU_C * (1 + 3 * 0.044715 * z1 ** 2))
        dW1 = x.T @ dz1
        db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="skip the dataset")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-pad", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tune", action="store_true", help="autotune each matmul shape")
    args = ap.parse_args()

    if args.synthetic:
        xtr, ytr, xte, yte = synthetic()
        print("data: SYNTHETIC (pipeline check only, accuracy is not meaningful)")
    else:
        xtr, ytr, xte, yte = load_mnist()
        print(f"data: MNIST  train {xtr.shape[0]}  test {xte.shape[0]}")

    n_cls = 10
    xtr_p, xte_p = prep(xtr, args.in_pad), prep(xte, args.in_pad)

    dev = Device()
    print(dev)
    ctx = Ctx(dev, tune=args.tune)
    B = args.batch
    model = MLP(ctx, B, [args.in_pad, args.hidden], n_cls, pad_classes=16)
    opt = AdamW(ctx, model.params(), lr=args.lr)

    xbuf = ctx.buf(B * args.in_pad * 2, "shared")
    ybuf = ctx.buf(B * 4, "shared")
    xview = xbuf.array(np.float16, (B, args.in_pad))
    yview = ybuf.array(np.uint32, (B,))

    train_step = TrainStep(model, opt, xbuf, ybuf)
    n_steps = (len(xtr_p) // B) * args.epochs
    print(f"model: {args.in_pad} -> {args.hidden} -> {n_cls}   batch {B}   "
          f"{n_steps} steps")
    print(f"training step recorded as {train_step.n_dispatch} dispatches in "
          f"1 command buffer\n")

    rng = np.random.default_rng(0)
    step = 0
    t_start = time.perf_counter()
    gpu_step_best = float("inf")
    for epoch in range(args.epochs):
        order = rng.permutation(len(xtr_p) // B * B).reshape(-1, B)
        run_loss, nb = 0.0, 0
        for batch_idx in order:
            # Zero-copy: writing into the mapped array IS the upload.
            xview[:] = xtr_p[batch_idx]
            yview[:] = ytr[batch_idx]
            t0 = time.perf_counter()
            loss, _ = train_step(args.lr)
            gpu_step_best = min(gpu_step_best, time.perf_counter() - t0)
            run_loss += loss
            nb += 1
            step += 1
        # Evaluate.
        correct = 0
        for i in range(0, len(xte_p) // B * B, B):
            xview[:] = xte_p[i:i + B]
            correct += int((model.predict(xbuf) == yte[i:i + B]).sum())
        acc = correct / (len(xte_p) // B * B)
        print(f"  epoch {epoch + 1}/{args.epochs}  loss {run_loss / nb:.4f}  "
              f"test acc {acc * 100:.2f}%")
    wall = time.perf_counter() - t_start

    print(f"\n  wall {wall:.1f}s   best GPU step {gpu_step_best * 1e3:.3f} ms   "
          f"{B / gpu_step_best:,.0f} samples/s")

    cpu_step = cpu_train_steps(xtr_p[:B, :args.in_pad], ytr[:B],
                               [args.in_pad, args.hidden], n_cls)
    print(f"  CPU (numpy/OpenBLAS) step {cpu_step * 1e3:.3f} ms   "
          f"{B / cpu_step:,.0f} samples/s")
    print(f"  GPU speedup on a full training step: {cpu_step / gpu_step_best:.2f}x")

    ctx.destroy()
    dev.destroy()
    return acc


if __name__ == "__main__":
    main()
