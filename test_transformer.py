"""Full transformer forward and backward against an independent numpy model.

Tiny shapes so it runs in seconds and stays safe on a busy machine.
"""

import numpy as np

from transformer import GPT, TCtx
from vk import Device

GELU_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x ** 3)))


def dgelu(x):
    t = np.tanh(GELU_C * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    r = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * r
    return xh * g + b, xh, r[..., 0]


def layernorm_bwd(dy, xh, r, g, D):
    dxh = dy * g
    dx = r[:, None] * (dxh - dxh.mean(-1, keepdims=True)
                       - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, (dy * xh).sum(0), dy.sum(0)


def np_model(P, ids, targets, B, T, D, H, n_layer, vocab, pad_vocab):
    """Forward and backward for the same architecture, in f32 numpy."""
    hd = D // H
    rows = B * T
    scale = 1.0 / np.sqrt(hd)
    cache = {}

    x = P["tok"][ids] + np.tile(P["pos"], (B, 1))
    cache["x0"] = x
    for l in range(n_layer):
        p = P[f"b{l}"]
        h, xh1, r1 = layernorm(x, p["ln1.g"], p["ln1.b"])
        qkv = h @ p["qkv.W"] + p["qkv.b"]
        q, k, v = (qkv[:, i * D:(i + 1) * D].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
                   for i in range(3))
        s = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = np.tril(np.ones((T, T), bool))
        s = np.where(mask, s, -1e30)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s) * mask
        pr = e / e.sum(-1, keepdims=True)
        ao = pr @ v
        ao_m = ao.transpose(0, 2, 1, 3).reshape(rows, D)
        a = ao_m @ p["proj.W"] + p["proj.b"]
        res1 = x + a

        h2, xh2, r2 = layernorm(res1, p["ln2.g"], p["ln2.b"])
        z1 = h2 @ p["fc1.W"] + p["fc1.b"]
        hh = gelu(z1)
        m = hh @ p["fc2.W"] + p["fc2.b"]
        x = res1 + m
        cache[l] = dict(h=h, xh1=xh1, r1=r1, qkv=qkv, q=q, k=k, v=v, pr=pr, ao=ao,
                        ao_m=ao_m, res1=res1, h2=h2, xh2=xh2, r2=r2, z1=z1, hh=hh,
                        x_in=cache["x0"] if l == 0 else cache[l - 1]["x_out"])
        cache[l]["x_out"] = x

    hf, xhf, rf = layernorm(x, P["lnf.g"], P["lnf.b"])
    logits = hf @ P["head.W"] + P["head.b"]

    valid = logits[:, :vocab]
    mx = valid.max(1, keepdims=True)
    lse = mx + np.log(np.exp(valid - mx).sum(1, keepdims=True))
    loss = float((lse[:, 0] - valid[np.arange(rows), targets]).mean())

    dlogits = np.zeros_like(logits)
    probs = np.exp(valid - lse)
    oh = np.zeros_like(valid)
    oh[np.arange(rows), targets] = 1.0
    dlogits[:, :vocab] = (probs - oh) / rows

    G = {}
    G["head.W"] = hf.T @ dlogits
    G["head.b"] = dlogits.sum(0)
    dhf = dlogits @ P["head.W"].T
    d, G["lnf.g"], G["lnf.b"] = layernorm_bwd(dhf, xhf, rf, P["lnf.g"], D)

    for l in reversed(range(n_layer)):
        p, c = P[f"b{l}"], cache[l]
        g = {}
        g["fc2.W"] = c["hh"].T @ d
        g["fc2.b"] = d.sum(0)
        dhh = d @ p["fc2.W"].T
        dz1 = dhh * dgelu(c["z1"])
        g["fc1.W"] = c["h2"].T @ dz1
        g["fc1.b"] = dz1.sum(0)
        dh2 = dz1 @ p["fc1.W"].T
        dres1_mlp, g["ln2.g"], g["ln2.b"] = layernorm_bwd(
            dh2, c["xh2"], c["r2"], p["ln2.g"], D)
        dres1 = d + dres1_mlp

        g["proj.W"] = c["ao_m"].T @ dres1
        g["proj.b"] = dres1.sum(0)
        dao_m = dres1 @ p["proj.W"].T
        dao = dao_m.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        dv = c["pr"].transpose(0, 1, 3, 2) @ dao
        dp = dao @ c["v"].transpose(0, 1, 3, 2)
        dot = (dp * c["pr"]).sum(-1, keepdims=True)
        ds = c["pr"] * (dp - dot) * scale
        dq = ds @ c["k"]
        dk = ds.transpose(0, 1, 3, 2) @ c["q"]

        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(rows, D)

        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=1)
        g["qkv.W"] = c["h"].T @ dqkv
        g["qkv.b"] = dqkv.sum(0)
        dh = dqkv @ p["qkv.W"].T
        dx_attn, g["ln1.g"], g["ln1.b"] = layernorm_bwd(
            dh, c["xh1"], c["r1"], p["ln1.g"], D)
        d = dres1 + dx_attn
        G[f"b{l}"] = g

    G["pos"] = d.reshape(B, T, D).sum(0)
    G["tok"] = np.zeros_like(P["tok"])
    np.add.at(G["tok"], ids, d)
    return loss, G


def extract(model, n_layer):
    P = {"tok": model.tok.numpy().copy(), "pos": model.pos.numpy().copy(),
         "lnf.g": model.lnf.g.numpy().copy(), "lnf.b": model.lnf.b.numpy().copy(),
         "head.W": model.head.W.numpy().copy(), "head.b": model.head.b.numpy().copy()}
    for i, b in enumerate(model.blocks):
        P[f"b{i}"] = {
            "ln1.g": b.ln1.g.numpy().copy(), "ln1.b": b.ln1.b.numpy().copy(),
            "ln2.g": b.ln2.g.numpy().copy(), "ln2.b": b.ln2.b.numpy().copy(),
            "qkv.W": b.attn.qkv.W.numpy().copy(), "qkv.b": b.attn.qkv.b.numpy().copy(),
            "proj.W": b.attn.proj.W.numpy().copy(), "proj.b": b.attn.proj.b.numpy().copy(),
            "fc1.W": b.fc1.W.numpy().copy(), "fc1.b": b.fc1.b.numpy().copy(),
            "fc2.W": b.fc2.W.numpy().copy(), "fc2.b": b.fc2.b.numpy().copy()}
    return P


def relerr(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-8))


def main():
    B, T, D, H, n_layer, vocab = 2, 16, 32, 2, 2, 16
    dev = Device()
    print(dev)
    ctx = TCtx(dev)
    try:
        model = GPT(ctx, B, T, D, H, n_layer, vocab)
        rows = B * T
        rng = np.random.default_rng(0)
        ids = rng.integers(0, vocab, rows).astype(np.uint32)
        tgt = rng.integers(0, vocab, rows).astype(np.uint32)

        idb = ctx.buf(rows * 4, "shared")
        idb.array(np.uint32, (rows,))[:] = ids
        idb.flush()
        tgb = ctx.buf(rows * 4, "shared")
        tgb.array(np.uint32, (rows,))[:] = tgt
        tgb.flush()

        P = extract(model, n_layer)
        graph = dev.graph("test")
        model.record(idb, tgb, graph)
        graph.finish()
        print(f"  recorded {graph.n_dispatch} dispatches, "
              f"{model.n_params():,} params")
        graph.submit()
        loss = model.read_loss()

        ref_loss, G = np_model(P, ids, tgt, B, T, D, H, n_layer, vocab,
                               model.pad_vocab)
        e = abs(loss - ref_loss) / abs(ref_loss)
        assert e < 5e-3, f"loss {loss} vs numpy {ref_loss} (rel {e:.2e})"
        print(f"  loss {loss:.6f} vs numpy {ref_loss:.6f}   rel {e:.1e}   OK")

        checks = [("head.W", model.head.W, G["head.W"]),
                  ("head.b", model.head.b, G["head.b"]),
                  ("lnf.g", model.lnf.g, G["lnf.g"]),
                  ("lnf.b", model.lnf.b, G["lnf.b"]),
                  ("pos", model.pos, G["pos"]),
                  ("tok", model.tok, G["tok"][:model.pad_vocab])]
        for i, b in enumerate(model.blocks):
            g = G[f"b{i}"]
            checks += [(f"b{i}.fc2.W", b.fc2.W, g["fc2.W"]),
                       (f"b{i}.fc1.W", b.fc1.W, g["fc1.W"]),
                       (f"b{i}.ln2.g", b.ln2.g, g["ln2.g"]),
                       (f"b{i}.proj.W", b.attn.proj.W, g["proj.W"]),
                       (f"b{i}.qkv.W", b.attn.qkv.W, g["qkv.W"]),
                       (f"b{i}.qkv.b", b.attn.qkv.b, g["qkv.b"]),
                       (f"b{i}.ln1.g", b.ln1.g, g["ln1.g"])]

        worst = 0.0
        bad = []
        for name, param, ref in checks:
            got = param.grad_numpy()
            if ref.shape != got.shape:
                ref = ref.reshape(got.shape)
            e = relerr(got, ref)
            worst = max(worst, e)
            flag = "OK " if e < 3e-2 else "BAD"
            if e >= 3e-2:
                bad.append((name, e))
            print(f"  {name:14s} rel err {e:.2e}  {flag}")
        assert not bad, f"gradients wrong: {bad}"
        print(f"\n  all {len(checks)} gradient tensors match, worst {worst:.2e}")
        print("transformer checks passed")
    finally:
        ctx.destroy()
        dev.destroy()


if __name__ == "__main__":
    main()
