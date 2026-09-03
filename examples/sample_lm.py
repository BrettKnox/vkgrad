"""Generate text from a checkpoint trained by examples/train_lm.py.

A validation loss is abstract. What the model actually writes is not, and it is
the only check that catches a model which is optimising something other than
what you think.

  python -m examples.sample_lm --run runs/lm_384x6 --tokens 400
  python -m examples.sample_lm --prompt "def " --temperature 0.6
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_lm import build_corpus, encode  # noqa: E402
from transformer import GPT, TCtx  # noqa: E402
from vk import Device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run directory with ckpt.npz")
    ap.add_argument("--dmodel", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--prompt", default="def ")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = args.run or os.path.join(ROOT, "runs", f"lm_{args.dmodel}x{args.layers}")
    ckpt = os.path.join(run, "ckpt.npz")
    if not os.path.exists(ckpt):
        found = glob.glob(os.path.join(ROOT, "runs", "*", "ckpt.npz"))
        raise SystemExit(f"no checkpoint at {ckpt}\navailable: {found}")

    # Same vocabulary the training run built, derived from the same corpus.
    data = build_corpus(os.path.join(ROOT, "data", "pycorpus.npy"))
    _, itos = encode(data, args.vocab)
    stoi = {c: i for i, c in itos.items()}
    unk = args.vocab - 1

    dev = Device()
    ctx = TCtx(dev)
    T = args.seq
    # Batch of 1: generation is inherently serial, and rows = T still satisfies
    # the tile constraints.
    model = GPT(ctx, 1, T, args.dmodel, args.heads, args.layers, args.vocab)
    params = list(model.params())
    z = np.load(ckpt)
    for i, p in enumerate(params):
        p.set(z[f"p{i}"].reshape(p.shape))
    print(f"{dev.name}: {model.n_params():,} params, checkpoint step {int(z['step']):,}\n")

    idb = ctx.buf(T * 4, "shared")
    idv = idb.array(np.uint32, (T,))
    logits_host = ctx.buf(T * model.pad_vocab * 4, "cached")

    # Record the forward pass once: generation runs it per token, and an eager
    # forward would spend more time submitting than computing.
    g = dev.graph("sample")
    logits = model.forward(idb, graph=g)
    g.finish()

    # Prime with real corpus text. Left-padding the window with the unknown
    # token puts the model far outside its training distribution and it simply
    # continues the padding: the first attempt produced 300 '?' in a row, which
    # looked like a broken model and was a broken prompt.
    tok_all, _ = encode(data, args.vocab)
    pick = np.random.default_rng(args.seed)
    start = int(pick.integers(0, len(tok_all) - T - 1))
    prompt_ids = [stoi.get(c, unk) for c in args.prompt]
    ctxt = list(tok_all[start:start + T - len(prompt_ids)]) + prompt_ids
    primed = "".join(itos.get(int(t), "?")
                     for t in ctxt[:len(ctxt) - len(prompt_ids)])
    rng = np.random.default_rng(args.seed + 1)
    out = list(args.prompt)
    print("--- priming context, last 200 chars (real corpus text) ---")
    print(primed[-200:])
    print("--- model continues from here ---")

    for _ in range(args.tokens):
        window = ctxt[-T:]
        pad = T - len(window)
        idv[:] = np.array([unk] * pad + window, np.uint32)
        idb.flush()
        g.submit()
        dev.copy(logits, logits_host, T * model.pad_vocab * 4)
        logits_host.invalidate()
        row = logits_host.array(np.float32, (T, model.pad_vocab))[-1, :args.vocab]

        row = row.astype(np.float64) / max(args.temperature, 1e-3)
        if args.top_k:
            cut = np.partition(row, -args.top_k)[-args.top_k]
            row = np.where(row < cut, -np.inf, row)
        p = np.exp(row - row.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        ctxt.append(nxt)
        out.append(itos.get(nxt, "?"))

    print("".join(out))
    ctx.destroy()
    dev.destroy()


if __name__ == "__main__":
    main()
