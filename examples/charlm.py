"""Train a small character-level transformer, and race it against the CPU.

The corpus is local text: by default the Python standard library's own source,
which is plentiful, highly structured, and already on any machine that can run
this. Point --file at anything else to use it instead.

  python -m examples.charlm --steps 300
  python -m examples.charlm --file some.txt --steps 2000
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW  # noqa: E402
from transformer import GPT, TCtx  # noqa: E402
from vk import Device  # noqa: E402


def local_corpus(max_bytes=2_000_000):
    """Concatenate stdlib sources until we have enough text."""
    root = os.path.dirname(os.__file__)
    files = sorted(glob.glob(os.path.join(root, "*.py")))
    out = []
    total = 0
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
        except OSError:
            continue
        out.append(t)
        total += len(t)
        if total >= max_bytes:
            break
    if not out:
        raise RuntimeError("no local text found; pass --file")
    return "".join(out)[:max_bytes]


def encode(text, max_vocab=96):
    """Char vocabulary, rarest characters folded into a single unknown slot."""
    chars, counts = np.unique(np.frombuffer(text.encode("utf-8", "ignore"),
                                            np.uint8), return_counts=True)
    keep = chars[np.argsort(-counts)][:max_vocab - 1]
    table = np.full(256, max_vocab - 1, np.uint32)
    for i, c in enumerate(np.sort(keep)):
        table[c] = i
    data = table[np.frombuffer(text.encode("utf-8", "ignore"), np.uint8)]
    itos = {int(table[c]): chr(c) for c in np.sort(keep)}
    return data.astype(np.uint32), max_vocab, itos


def cpu_step_time(B, T, D, H, n_layer, vocab, reps=3):
    """One forward+backward of the same architecture in numpy, for scale.

    Only the dominant matmuls and attention are modelled; this is a floor on
    the CPU cost, which makes the comparison conservative in the CPU's favour.
    """
    rows = B * T
    hd = D // H
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, D)).astype(np.float32)
    Wq = rng.standard_normal((D, 3 * D)).astype(np.float32)
    Wp = rng.standard_normal((D, D)).astype(np.float32)
    W1 = rng.standard_normal((D, 4 * D)).astype(np.float32)
    W2 = rng.standard_normal((4 * D, D)).astype(np.float32)
    Wh = rng.standard_normal((D, vocab)).astype(np.float32)

    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n_layer):
            qkv = x @ Wq
            q = qkv[:, :D].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = qkv[:, D:2 * D].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            v = qkv[:, 2 * D:].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(0, 2, 1, 3).reshape(rows, D)
            a = a @ Wp
            h = a @ W1
            h = h @ W2
            # Backward is roughly two more matmuls per forward matmul.
            _ = h.T @ x
            _ = h @ W2.T
        _ = x @ Wh
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dmodel", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--tune", action="store_true",
                    help="autotune each matmul shape (slow first run, cached after)")
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"corpus: {args.file}  {len(text):,} chars")
    else:
        text = local_corpus()
        print(f"corpus: local Python stdlib source  {len(text):,} chars")

    data, vocab, itos = encode(text)
    B, T, D, H, L = args.batch, args.seq, args.dmodel, args.heads, args.layers
    assert D % H == 0 and (D // H) % 16 == 0, "head dim must be a multiple of 16"

    dev = Device()
    print(dev)
    ctx = TCtx(dev, tune=args.tune)
    model = GPT(ctx, B, T, D, H, L, vocab)
    opt = AdamW(ctx, model.params(), lr=args.lr, wd=0.01)

    rows = B * T
    idb = ctx.buf(rows * 4, "shared")
    tgb = ctx.buf(rows * 4, "shared")
    idv = idb.array(np.uint32, (rows,))
    tgv = tgb.array(np.uint32, (rows,))

    graph = dev.graph("train")
    model.record(idb, tgb, graph)
    opt.record(graph)
    graph.finish()

    print(f"model: d={D} heads={H} layers={L} seq={T} batch={B} vocab={vocab}")
    print(f"params: {model.n_params():,}")
    print(f"step recorded as {graph.n_dispatch} dispatches in 1 command buffer\n")

    rng = np.random.default_rng(0)
    n = len(data) - T - 1
    best_step = float("inf")
    t_start = time.perf_counter()
    losses = []
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, n, B)
        idx = starts[:, None] + np.arange(T)[None, :]
        idv[:] = data[idx].reshape(-1)
        tgv[:] = data[idx + 1].reshape(-1)
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        opt.advance(lr)
        dt = graph.submit()
        best_step = min(best_step, dt)
        losses.append(model.read_loss())
        if step % max(args.steps // 10, 1) == 0:
            recent = float(np.mean(losses[-50:]))
            print(f"  step {step:5d}/{args.steps}  loss {recent:.4f}  "
                  f"({dt * 1e3:.2f} ms)")
    wall = time.perf_counter() - t_start

    tok_per_step = rows
    print(f"\n  wall {wall:.1f}s   best step {best_step * 1e3:.2f} ms   "
          f"{tok_per_step / best_step:,.0f} tokens/s")
    print(f"  first loss {losses[0]:.4f}  ->  last50 {np.mean(losses[-50:]):.4f}")

    cpu = cpu_step_time(B, T, D, H, L, vocab)
    print(f"  CPU (numpy/OpenBLAS, matmuls only) {cpu * 1e3:.2f} ms   "
          f"{tok_per_step / cpu:,.0f} tokens/s")
    print(f"  GPU speedup: {cpu / best_step:.2f}x (conservative: the CPU "
          f"figure omits layernorm, softmax and the optimiser)")

    ctx.destroy()
    dev.destroy()


if __name__ == "__main__":
    main()
