"""GPT-2 on vkgrad against an independent numpy GPT-2 read from the same file.

The reference follows Hugging Face's GPT2Model: Conv1D is x @ weight + bias
with weight (in, out), c_attn splits into q, k, v in that order, heads are
contiguous hd-wide slices, gelu_new, LayerNorm eps 1e-5, logits = h @ wte^T.
It shares only the file reader with hf_gpt2.py: no vkgrad kernel, and not the
name mapping in hf_gpt2.load.

  python test_hf_gpt2.py
"""

import os
import time

import numpy as np

import hf_gpt2
from transformer import TCtx
from vk import Device

T = 128
PARAGRAPH = (
    "The library opens at nine in the morning and closes at six in the evening. "
    "On weekdays, students fill the reading room long before lunch, and the quiet "
    "is broken only by the sound of pages turning. Visitors who want to borrow a "
    "book need a card, which the front desk can issue in a few minutes. Most of the "
    "collection is on the second floor, but maps, newspapers and old photographs "
    "are kept in a separate room downstairs.")
PROMPT = "The history of the steam engine begins"
N_NEW = 20


def ref_logits(W, H, ids, eps=1e-5):
    ids = np.asarray(ids)
    n = len(ids)
    D = W["wte.weight"].shape[1]
    hd = D // H

    def ln(x, name):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * W[name + ".weight"] + W[name + ".bias"]

    def conv1d(x, name):
        return x @ W[name + ".weight"] + W[name + ".bias"]

    def gelu_new(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    future = np.triu(np.ones((n, n), bool), 1)
    x = W["wte.weight"][ids] + W["wpe.weight"][:n]
    layer = 0
    while f"h.{layer}.ln_1.weight" in W:
        p = f"h.{layer}."
        q, k, v = np.split(conv1d(ln(x, p + "ln_1"), p + "attn.c_attn"), 3, axis=-1)
        q, k, v = (t.reshape(n, H, hd).transpose(1, 0, 2) for t in (q, k, v))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
        s[:, future] = -np.inf
        e = np.exp(s - s.max(-1, keepdims=True))
        a = (e / e.sum(-1, keepdims=True)) @ v
        x = x + conv1d(a.transpose(1, 0, 2).reshape(n, D), p + "attn.c_proj")
        x = x + conv1d(gelu_new(conv1d(ln(x, p + "ln_2"), p + "mlp.c_fc")),
                       p + "mlp.c_proj")
        layer += 1
    return ln(x, "ln_f") @ W["wte.weight"].T


def perplexity(logits, ids):
    lg = logits[:-1].astype(np.float64)
    mx = lg.max(-1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(-1))
    return float(np.exp((lse - lg[np.arange(len(lg)), np.asarray(ids)[1:]]).mean()))


def compare(name, got, ref):
    diff = np.abs(got - ref)
    agree = got.argmax(-1) == ref.argmax(-1)
    print(f"  {name}: max |logit diff| {diff.max():.4f}, mean {diff.mean():.5f}, "
          f"reference logits in [{ref.min():.1f}, {ref.max():.1f}]")
    print(f"  {name}: argmax agrees at {agree.sum()}/{len(agree)} positions"
          + ("" if agree.all() else f", differs at {np.flatnonzero(~agree).tolist()}"))
    return float(diff.max()), agree


def greedy(next_logits, ids, n):
    ids = list(ids)
    for _ in range(n):
        ids.append(int(next_logits(ids).argmax()))
    return ids[-n:]


def main():
    enc = hf_gpt2.encoding()
    cfg = hf_gpt2.config(hf_gpt2.GPT2)
    W = hf_gpt2.gpt2_tensors(os.path.join(hf_gpt2.GPT2, "model.safetensors"))
    para = enc.encode(PARAGRAPH)
    prompt = enc.encode(PROMPT)
    assert len(para) <= T and len(prompt) + N_NEW <= T

    dev = Device()
    print(dev)
    try:
        ctx = TCtx(dev)
        try:
            model = hf_gpt2.build(ctx, 1, T)
            print(f"  openai-community/gpt2, tied head: {model.n_params():,} params at T={T}")
            fwd = hf_gpt2.Forward(model)

            got = fwd([para])[0]
            t0 = time.perf_counter()
            ref = ref_logits(W, cfg["H"], para)
            print(f"  paragraph: {len(para)} tokens (numpy reference {time.perf_counter() - t0:.2f} s)")
            dmax, agree = compare("paragraph", got, ref)
            p_got, p_ref = perplexity(got, para), perplexity(ref, para)
            print(f"  perplexity over {len(para) - 1} predictions: "
                  f"vkgrad {p_got:.4f}, numpy {p_ref:.4f}")
            assert dmax < 1.0, dmax
            assert agree.mean() >= 0.98, agree.mean()
            assert abs(p_got - p_ref) / p_ref < 5e-3, (p_got, p_ref)
            assert p_ref < 100, f"GPT-2 should read plain English far better: {p_ref}"

            # vkgrad pads on the right to T; numpy runs the exact prefix.
            g_got = greedy(lambda s: fwd([s])[0, -1], prompt, N_NEW)
            g_ref = greedy(lambda s: ref_logits(W, cfg["H"], s)[-1], prompt, N_NEW)
            print(f"  greedy vkgrad: {enc.decode(prompt + g_got)!r}")
            print(f"  greedy numpy : {enc.decode(prompt + g_ref)!r}")
            assert g_got == g_ref, (g_got, g_ref)

            # attn.c_proj is square, so only the logits can catch a transpose.
            for b in model.blocks:
                b.attn.proj.W.set(b.attn.proj.W.numpy().T.copy())
            bad = fwd([para])[0]
            bmax, _ = compare("c_proj transposed", bad, ref)
            p_bad = perplexity(bad, para)
            print(f"  c_proj transposed: perplexity {p_bad:.1f}")
            assert bmax > 10 * dmax and p_bad > 1.5 * p_ref, (bmax, p_bad)
        finally:
            ctx.destroy()

        # The paper's model through the same loader. Its tokenizer is not GPT-2's,
        # so the input is arbitrary ids; this checks the mapping, not the text.
        cpath = os.path.join(hf_gpt2.CODEGPT, "model.safetensors")
        ccfg = hf_gpt2.config(hf_gpt2.CODEGPT)
        CW = hf_gpt2.gpt2_tensors(cpath)
        ids = np.random.default_rng(0).integers(0, ccfg["vocab"], 64)
        ctx = TCtx(dev)
        try:
            model = hf_gpt2.build(ctx, 1, 64, hf_gpt2.CODEGPT)
            print(f"\n  microsoft/CodeGPT-small-py, tied head: {model.n_params():,} params")
            dmax, agree = compare("CodeGPT", hf_gpt2.Forward(model)([ids])[0],
                                  ref_logits(CW, ccfg["H"], ids))
            assert dmax < 1.0 and agree.mean() >= 0.98, (dmax, agree.mean())
        finally:
            ctx.destroy()
        print("\nGPT-2 checks passed")
    finally:
        dev.destroy()


if __name__ == "__main__":
    main()
