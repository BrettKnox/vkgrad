"""Train a character-level language model to a real loss curve.

Everything else in this repo measures step times and extrapolates. This runs a
model for a bounded wall-clock budget and records what actually happened:
throughput over time, training loss, and held-out validation loss.

The corpus is local text (Python's own installed sources by default), so no
download is required. Checkpoints and the log are written continuously, so a
run can be stopped and resumed, and a long run survives a reboot.

  python -m examples.train_lm --minutes 20
  python -m examples.train_lm --dmodel 384 --layers 6 --minutes 150
  python -m examples.train_lm --resume runs/lm_384x6
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import AdamW  # noqa: E402
from dataparallel import GradAccum  # noqa: E402
from kernels import warmup  # noqa: E402
from transformer import GPT, TCtx  # noqa: E402
from vk import Device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_corpus(cache, max_mib=48):
    """Concatenate local Python sources into a byte corpus, cached on disk."""
    if os.path.exists(cache):
        return np.load(cache)
    root = os.path.dirname(os.__file__)
    parts, total = [], 0
    for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        try:
            with open(f, "rb") as fh:
                parts.append(fh.read())
        except OSError:
            continue
        total += len(parts[-1])
        if total >= max_mib << 20:
            break
    raw = b"".join(parts)
    data = np.frombuffer(raw, np.uint8)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, data)
    return data


def encode(data, vocab):
    """Map the most frequent bytes to ids, everything else to one unknown id."""
    counts = np.bincount(data, minlength=256)
    keep = np.argsort(-counts)[:vocab - 1]
    table = np.full(256, vocab - 1, np.uint16)
    for i, b in enumerate(np.sort(keep)):
        table[b] = i
    return table[data], {int(table[b]): chr(b) for b in np.sort(keep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmodel", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8, help="microbatch size")
    ap.add_argument("--accum", type=int, default=1,
                    help="microbatches per optimiser step; helps only models near "
                         "the memory ceiling, and is slower below it")
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    tag = args.resume or os.path.join(
        ROOT, "runs", f"lm_{args.dmodel}x{args.layers}")
    os.makedirs(tag, exist_ok=True)
    logpath = os.path.join(tag, "log.jsonl")
    ckpt = os.path.join(tag, "ckpt.npz")

    data = build_corpus(os.path.join(ROOT, "data", "pycorpus.npy"))
    tokens, itos = encode(data, args.vocab)
    split = int(len(tokens) * 0.98)
    train, val = tokens[:split], tokens[split:]
    print(f"corpus {len(tokens):,} tokens  train {len(train):,}  val {len(val):,}")

    dev = Device()
    warmup(dev)
    ctx = TCtx(dev)
    B, T = args.batch, args.seq
    model = GPT(ctx, B, T, args.dmodel, args.heads, args.layers, args.vocab)
    opt = AdamW(ctx, model.params(), lr=args.lr, wd=0.01)
    params = list(model.params())
    n_par = model.n_params()
    print(dev)
    print(f"model d={args.dmodel} L={args.layers} H={args.heads} "
          f"seq={T} batch={B}: {n_par:,} params")

    rows = B * T
    idb = ctx.buf(rows * 4, "shared")
    tgb = ctx.buf(rows * 4, "shared")
    idv = idb.array(np.uint32, (rows,))
    tgv = tgb.array(np.uint32, (rows,))

    accum = max(args.accum, 1)
    ga = None
    gmicro = gstep = None
    if accum == 1:
        g = dev.graph("train")
        model.record(idb, tgb, g)
        opt.record(g)
        g.finish()
        print(f"step recorded as {g.n_dispatch} dispatches")
    else:
        # Many small microbatches beat one large batch on this hardware:
        # throughput falls as batch grows, because the machine is
        # bandwidth-bound and activation traffic scales with batch.
        ga = GradAccum(ctx, params, accum=accum)
        gmicro = dev.graph("micro")
        model.record(idb, tgb, gmicro)
        ga.record_push(gmicro)
        gmicro.finish()
        gstep = dev.graph("step")
        ga.record_apply(gstep, opt)
        ga.record_zero(gstep)
        gstep.finish()
        g = gmicro
        print(f"microbatch {gmicro.n_dispatch} dispatches x{accum}, "
              f"step {gstep.n_dispatch}, effective batch {B * accum}")
    print()

    start_step = 0
    if args.resume and os.path.exists(ckpt):
        z = np.load(ckpt)
        for i, p in enumerate(params):
            p.set(z[f"p{i}"].reshape(p.shape))
        start_step = int(z["step"])
        print(f"resumed from step {start_step}\n")

    rng = np.random.default_rng(1234)

    def batch_from(arr):
        starts = rng.integers(0, len(arr) - T - 1, B)
        idx = starts[:, None] + np.arange(T)[None, :]
        return arr[idx].reshape(-1).astype(np.uint32), \
            arr[idx + 1].reshape(-1).astype(np.uint32)

    def evaluate(n_batches=20):
        tot = 0.0
        for _ in range(n_batches):
            x, y = batch_from(val)
            idv[:] = x
            tgv[:] = y
            g.submit()
            tot += model.read_loss()
        return tot / n_batches

    def train_step(lr):
        """One optimiser step, over `accum` microbatches."""
        if accum == 1:
            x, y = batch_from(train)
            idv[:] = x
            tgv[:] = y
            opt.advance(lr)
            g.submit()
            return model.read_loss(), rows
        tot = 0.0
        for _ in range(accum):
            x, y = batch_from(train)
            idv[:] = x
            tgv[:] = y
            gmicro.submit()
            tot += model.read_loss()
        opt.advance(lr)
        gstep.submit()
        return tot / accum, rows * accum

    budget = args.minutes * 60
    t0 = time.perf_counter()
    step = start_step
    tokens_done = 0
    losses = []
    logf = open(logpath, "a", encoding="utf-8")
    print(f"{'step':>7}{'loss':>9}{'val':>9}{'tok/s':>10}{'elapsed':>10}{'ETA':>9}")
    try:
        while time.perf_counter() - t0 < budget:
            step += 1
            lr = args.lr * min(1.0, step / max(args.warmup_steps, 1))
            loss, ntok = train_step(lr)
            losses.append(loss)
            tokens_done += ntok

            if step % args.eval_every == 0:
                el = time.perf_counter() - t0
                tps = tokens_done / el
                vl = evaluate()
                tr = float(np.mean(losses[-args.eval_every:]))
                left = max(budget - el, 0)
                print(f"{step:7d}{tr:9.4f}{vl:9.4f}{tps:10,.0f}"
                      f"{el / 60:9.1f}m{left / 60:8.1f}m")
                logf.write(json.dumps({"step": step, "train": tr, "val": vl,
                                       "tokens": tokens_done, "elapsed": el,
                                       "tok_per_s": tps}) + "\n")
                logf.flush()
                np.savez(ckpt, step=step,
                         **{f"p{i}": p.numpy() for i, p in enumerate(params)})
    except KeyboardInterrupt:
        print("\ninterrupted; checkpoint written")
    finally:
        logf.close()

    el = time.perf_counter() - t0
    print(f"\n  {step - start_step:,} steps, {tokens_done:,} tokens in "
          f"{el / 60:.1f} min = {tokens_done / el:,.0f} tokens/s")
    if losses:
        print(f"  loss {losses[0]:.4f} -> {np.mean(losses[-100:]):.4f}   "
              f"val {evaluate():.4f}")
    chinchilla = 20 * n_par
    print(f"  Chinchilla-optimal for {n_par:,} params is {chinchilla:,} tokens: "
          f"{chinchilla / (tokens_done / el) / 3600:.1f} h at this rate")
    print(f"  checkpoint: {ckpt}")

    ctx.destroy()
    dev.destroy()


if __name__ == "__main__":
    main()
