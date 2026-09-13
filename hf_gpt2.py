"""Hugging Face GPT-2 checkpoints loaded into vkgrad's GPT, with numpy only.

A .safetensors file is an 8-byte little-endian header length, a JSON header of
{name: {dtype, shape, data_offsets}}, then raw little-endian tensor bytes.

Layout, checked against the weights rather than assumed: HF's Conv1D computes
x @ weight + bias with weight stored (in, out), and vkgrad's Dense computes
x @ W + b with W stored (in_f, out_f). Same layout, so no transpose. Every
Conv1D except attn.c_proj is non-square, so the shape check catches a wrong
transpose there; c_proj is 768 x 768, and test_hf_gpt2.py shows the logit
check fails if it is transposed.

  python hf_gpt2.py      download the checkpoints to D:/cadlm-data/models
"""

import hashlib
import json
import os
import struct

import numpy as np

MODELS = "D:/cadlm-data/models"
GPT2 = os.path.join(MODELS, "openai-community-gpt2")
GPT2_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
# The model Text-to-CadQuery fine-tuned. main ships only pytorch_model.bin,
# flax and h5; the safetensors file is SFconvertbot's unmerged refs/pr/1.
CODEGPT = os.path.join(MODELS, "microsoft-CodeGPT-small-py")
CODEGPT_REV = "3751264d4b325a132e57f09527e1f7ac44ae370a"  # refs/pr/1

_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4",
           "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "?"}


def read_safetensors(path):
    """{name: read-only array} over a memory map of the file."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > min(size - 8, 100 << 20):
            raise ValueError(f"{path}: header length {n} does not fit a {size}-byte file")
        header = json.loads(f.read(n))
    body = np.memmap(path, np.uint8, "r", offset=8 + n)
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        if t["dtype"] not in _DTYPES:
            raise ValueError(f"{path}: {name} has unsupported dtype {t['dtype']}")
        dt = np.dtype(_DTYPES[t["dtype"]])
        a, b = t["data_offsets"]
        count = int(np.prod(t["shape"]))
        if not 0 <= a <= b <= body.size or b - a != count * dt.itemsize:
            raise ValueError(f"{path}: {name} bytes {a}..{b} do not match "
                             f"{t['shape']} {t['dtype']}")
        out[name] = np.frombuffer(body, dt, count, a).reshape(t["shape"])
    return out


def gpt2_tensors(path):
    """GPT-2 tensors under HF GPT2Model names (wte.weight, h.0.ln_1.weight, ...).

    Strips the 'transformer.' prefix some checkpoints carry (CodeGPT-small-py)
    and drops attn.bias / attn.masked_bias, which are causal-mask buffers.
    """
    W = {}
    for k, v in read_safetensors(path).items():
        k = k.removeprefix("transformer.")
        if not k.endswith((".attn.bias", ".attn.masked_bias")):
            W[k] = v
    head = W.pop("lm_head.weight", None)
    if head is not None and not np.array_equal(head, W["wte.weight"]):
        raise ValueError(f"{path}: untied lm_head.weight is not supported")
    return W


def config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    # What vkgrad's kernels implement: tanh GELU, LayerNorm eps 1e-5.
    if (c["model_type"], c["activation_function"], c["layer_norm_epsilon"]) != \
            ("gpt2", "gelu_new", 1e-5):
        raise ValueError(f"{model_dir}: not a GPT-2 config vkgrad implements")
    return dict(D=c["n_embd"], H=c["n_head"], n_layer=c["n_layer"],
                vocab=c["vocab_size"], n_ctx=c["n_positions"])


def load(model, W):
    """Copy GPT-2 tensors into a vkgrad GPT of the matching shape."""
    V, D = W["wte.weight"].shape
    if (V, D, W["h.0.ln_1.weight"].shape[0]) != (model.vocab, model.D, model.D) \
            or f"h.{len(model.blocks)}.ln_1.weight" in W \
            or f"h.{len(model.blocks) - 1}.ln_1.weight" not in W \
            or model.T > W["wpe.weight"].shape[0]:
        raise ValueError("checkpoint does not match the model shape")
    todo = set(W)

    def put(param, name, arr=None):
        a = W[name] if arr is None else arr
        if a.shape != param.shape:
            raise ValueError(f"{name}: {a.shape} into {param.name} {param.shape}")
        param.set(a)
        todo.discard(name)

    # Padding rows are never looked up, and their logits are masked out of the
    # softmax, so they stay zero.
    tok = np.zeros((model.pad_vocab, D), np.float32)
    tok[:V] = W["wte.weight"]
    put(model.tok, "wte.weight", tok)
    put(model.pos, "wpe.weight", W["wpe.weight"][:model.T])
    for i, b in enumerate(model.blocks):
        for param, name in ((b.ln1.g, "ln_1.weight"), (b.ln1.b, "ln_1.bias"),
                            (b.attn.qkv.W, "attn.c_attn.weight"),
                            (b.attn.qkv.b, "attn.c_attn.bias"),
                            (b.attn.proj.W, "attn.c_proj.weight"),
                            (b.attn.proj.b, "attn.c_proj.bias"),
                            (b.ln2.g, "ln_2.weight"), (b.ln2.b, "ln_2.bias"),
                            (b.fc1.W, "mlp.c_fc.weight"), (b.fc1.b, "mlp.c_fc.bias"),
                            (b.fc2.W, "mlp.c_proj.weight"), (b.fc2.b, "mlp.c_proj.bias")):
            put(param, f"h.{i}.{name}")
    put(model.lnf.g, "ln_f.weight")
    put(model.lnf.b, "ln_f.bias")
    if not model.tie:
        # GPT-2 ties the head to wte; untied, the head starts as a copy.
        head = np.zeros((D, model.pad_vocab), np.float32)
        head[:, :V] = W["wte.weight"].T
        model.head.W.set(head)
        model.head.b.set(np.zeros(model.pad_vocab, np.float32))
    if todo:
        raise ValueError(f"unmapped tensors: {sorted(todo)}")


def build(ctx, B, T, model_dir=GPT2, tie=True):
    """A vkgrad GPT with the checkpoint's weights."""
    from transformer import GPT
    c = config(model_dir)
    model = GPT(ctx, B, T, c["D"], c["H"], c["n_layer"], c["vocab"], tie=tie)
    load(model, gpt2_tensors(os.path.join(model_dir, "model.safetensors")))
    return model


class Forward:
    """The forward pass recorded once, with logits copied back to the host.

    vkgrad has no KV cache, so every call runs all T positions.
    """

    def __init__(self, model):
        ctx = model.ctx
        self.model = model
        self.ids = ctx.buf(model.rows * 4, "shared")
        self.host = ctx.buf(model.rows * model.pad_vocab * 4, "cached")
        self.graph = ctx.dev.graph("forward")
        self.out = model.forward(self.ids, graph=self.graph)
        self.graph.finish()
        self.submit_s = self.copy_s = 0.0

    def __call__(self, ids, pad=50256, last=False):
        """ids (B, n), n <= T -> logits (B, n, vocab), or (B, vocab) at position
        n-1 with last=True. The pad goes on the right: causal attention keeps
        it out of every real position, and positions stay what the checkpoint
        was trained with."""
        m = self.model
        ids = np.asarray(ids)
        a = np.full((m.B, m.T), pad, np.uint32)
        a[:, :ids.shape[1]] = ids
        self.ids.array(np.uint32, (m.rows,))[:] = a.reshape(-1)
        self.ids.flush()
        self.submit_s = self.graph.submit()
        self.copy_s = m.ctx.dev.copy(self.out, self.host, m.rows * m.pad_vocab * 4)
        self.host.invalidate()
        n = ids.shape[1]
        rows = slice(n - 1, n) if last else slice(0, n)
        out = self.host.array(np.float32, (m.B, m.T, m.pad_vocab))[:, rows, :m.vocab]
        return out[:, 0].copy() if last else out.copy()


def encoding():
    """tiktoken.get_encoding("gpt2"), with tiktoken's cache seeded from the HF
    repo's merges.txt and vocab.json (byte-identical to OpenAI's vocab.bpe and
    encoder.json: same sha256), so tiktoken never fetches them itself."""
    import tempfile
    import tiktoken
    cache = (os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
             or os.path.join(tempfile.gettempdir(), "data-gym-cache"))
    base = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/"
    for url, src, sha in (
            (base + "vocab.bpe", "merges.txt",
             "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
            (base + "encoder.json", "vocab.json",
             "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783")):
        dst = os.path.join(cache, hashlib.sha1(url.encode()).hexdigest())
        if os.path.exists(dst):
            continue
        with open(os.path.join(GPT2, src), "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{src} does not match tiktoken's expected hash")
        os.makedirs(cache, exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
    return tiktoken.get_encoding("gpt2")


def fetch():
    from huggingface_hub import hf_hub_download
    for f in ("config.json", "vocab.json", "merges.txt", "model.safetensors"):
        print(hf_hub_download("openai-community/gpt2", f, revision=GPT2_REV, local_dir=GPT2))
    for f in ("config.json", "model.safetensors"):
        print(hf_hub_download("microsoft/CodeGPT-small-py", f, revision=CODEGPT_REV,
                              local_dir=CODEGPT))


if __name__ == "__main__":
    fetch()
