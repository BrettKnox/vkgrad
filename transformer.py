"""A decoder-only transformer on the Vulkan runtime.

Layout conventions, fixed throughout:

  residual stream   (B*T, D)      f32   the single source of truth
  matmul operands   f16                 forced by cooperative matrix
  matmul outputs    f32                 f32 accumulate
  attention heads   (B*H, T, hd)  f16   batched matmul over gl_WorkGroupID.z

Every dimension is a multiple of 16 so the cooperative matrix tiling applies
without a masked edge path; the vocabulary is padded up and the padding is
masked inside the softmax.
"""

import numpy as np

from autograd import Ctx, Param
from tkernels import make_transformer_kernels

NCHUNK = 32  # row chunks for the column-sum reduction


class TCtx(Ctx):
    """Ctx plus the transformer kernels."""

    def __init__(self, dev, tune=False, scalar_only=None):
        super().__init__(dev, tune, scalar_only)
        self.TK = make_transformer_kernels(dev)

    def col_sum(self, src, dst, ncol, nrow, graph=None, f16=True):
        """Column sums of a gradient. Reads f16 by default: every tensor whose
        column sum is needed already exists in f16 for the matmuls, so the f32
        copy is pure traffic."""
        self.TK["col_sum_zero"]([dst], ncol, graph=graph)
        k = "col_sum_chunk16" if f16 else "col_sum_chunk"
        self.TK[k]([src, dst], NCHUNK * ncol, ncol, nrow, NCHUNK, graph=graph)

    def destroy(self):
        for k in self.TK.values():
            k.destroy()
        super().destroy()


class LayerNorm:
    def __init__(self, ctx, rows, D, name=""):
        self.ctx, self.rows, self.D = ctx, rows, D
        self.g = Param(ctx, (D,), np.ones(D, np.float32), name + ".g", no_decay=True)
        self.b = Param(ctx, (D,), np.zeros(D, np.float32), name + ".b", no_decay=True)
        self.y16 = ctx.buf(rows * D * 2)
        self.mu = ctx.buf(rows * 4)
        self.rstd = ctx.buf(rows * 4)
        self.dx = ctx.buf(rows * D * 4)
        self.dyxh = ctx.buf(rows * D * 2)
        self.dy16 = ctx.buf(rows * D * 2)

    def forward(self, x32, graph=None):
        self.x32 = x32
        self.ctx.TK["layernorm"](
            [x32, self.g.w32, self.b.w32, self.y16, self.mu, self.rstd],
            self.rows * self.ctx.dev.row_subgroup_size, self.D, graph=graph)
        return self.y16

    def backward(self, dy32, graph=None):
        c = self.ctx
        c.TK["layernorm_bwd"](
            [dy32, self.x32, self.g.w32, self.mu, self.rstd, self.dx,
             self.dyxh, self.dy16],
            self.rows * self.ctx.dev.row_subgroup_size, self.D, graph=graph)
        c.col_sum(self.dyxh, self.g.g32, self.D, self.rows, graph=graph)
        c.col_sum(self.dy16, self.b.g32, self.D, self.rows, graph=graph)
        return self.dx


class Dense:
    """Plain y = x @ W + b with f16 in and f32 out. No activation, no fusion
    choices: the caller decides what happens to the output."""

    def __init__(self, ctx, rows, in_f, out_f, name="", scale=None):
        self.ctx, self.rows, self.in_f, self.out_f = ctx, rows, in_f, out_f
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        s = scale if scale is not None else np.sqrt(2.0 / in_f)
        w = (rng.standard_normal((in_f, out_f)) * s).astype(np.float32)
        self.W = Param(ctx, (in_f, out_f), w, name + ".W")
        self.b = Param(ctx, (out_f,), np.zeros(out_f, np.float32), name + ".b",
                       no_decay=True)
        self.acc = ctx.buf(rows * out_f * 4)
        self.dx32 = ctx.buf(rows * in_f * 4)

    def forward(self, x16, graph=None):
        self.x16 = x16
        self.ctx.matmul(self.rows, self.out_f, self.in_f)(
            x16, self.W.w16, self.acc, self.rows, self.out_f, self.in_f, graph=graph)
        return self.acc

    def backward(self, dy16, need_dx=True, graph=None):
        c = self.ctx
        c.matmul(self.in_f, self.out_f, self.rows, trans_a=True)(
            self.x16, dy16, self.W.g32, self.in_f, self.out_f, self.rows, graph=graph)
        c.col_sum(dy16, self.b.g32, self.out_f, self.rows, graph=graph)
        if not need_dx:
            return None
        c.matmul(self.rows, self.in_f, self.out_f, trans_b=True)(
            dy16, self.W.w16, self.dx32, self.rows, self.in_f, self.out_f, graph=graph)
        return self.dx32


class Attention:
    def __init__(self, ctx, B, T, D, H, name=""):
        self.ctx, self.B, self.T, self.D, self.H = ctx, B, T, D, H
        self.hd = D // H
        self.rows = B * T
        self.nbh = B * H
        self.scale = 1.0 / np.sqrt(self.hd)

        self.qkv = Dense(ctx, self.rows, D, 3 * D, name + ".qkv")
        self.proj = Dense(ctx, self.rows, D, D, name + ".proj",
                          scale=np.sqrt(2.0 / D) * 0.5)

        head = self.nbh * T * self.hd
        att = self.nbh * T * T
        self.q16 = ctx.buf(head * 2)
        self.k16 = ctx.buf(head * 2)
        self.v16 = ctx.buf(head * 2)
        self.scores = ctx.buf(att * 4)
        self.p16 = ctx.buf(att * 2)
        self.ao32 = ctx.buf(head * 4)
        self.ao16 = ctx.buf(self.rows * D * 2)

        self.dao16 = ctx.buf(head * 2)
        self.dp32 = ctx.buf(att * 4)
        self.ds16 = ctx.buf(att * 2)
        self.dq32 = ctx.buf(head * 4)
        self.dk32 = ctx.buf(head * 4)
        self.dv32 = ctx.buf(head * 4)
        self.dqkv16 = ctx.buf(self.rows * 3 * D * 2)

    def forward(self, x16, graph=None):
        c, TK = self.ctx, self.ctx.TK
        T, hd, D, H = self.T, self.hd, self.D, self.H
        qkv = self.qkv.forward(x16, graph=graph)
        TK["split_qkv"]([qkv, self.q16, self.k16, self.v16],
                        self.rows * D, D, T, H, hd, graph=graph)
        # scores = Q @ K^T, one batched dispatch over every (batch, head).
        c.matmul(T, T, hd, trans_b=True, batched=True, nbatch=self.nbh)(
            self.q16, self.k16, self.scores, T, T, hd, graph=graph,
            nbatch=self.nbh, strides=(T * hd, T * hd, T * T))
        TK["attn_softmax"]([self.scores, self.p16],
                           self.nbh * T * c.dev.row_subgroup_size, T, self.scale, graph=graph)
        c.matmul(T, hd, T, batched=True, nbatch=self.nbh)(
            self.p16, self.v16, self.ao32, T, hd, T, graph=graph,
            nbatch=self.nbh, strides=(T * T, T * hd, T * hd))
        TK["merge_heads"]([self.ao32, self.ao16], self.rows * D, D, T, H, hd,
                          graph=graph)
        return self.proj.forward(self.ao16, graph=graph)

    def backward(self, dy16, graph=None):
        c, TK = self.ctx, self.ctx.TK
        T, hd, D, H = self.T, self.hd, self.D, self.H
        dao = self.proj.backward(dy16, need_dx=True, graph=graph)
        TK["split_heads"]([dao, self.dao16], self.rows * D, D, T, H, hd, graph=graph)

        # dV = P^T @ dOut
        c.matmul(T, hd, T, trans_a=True, batched=True, nbatch=self.nbh)(
            self.p16, self.dao16, self.dv32, T, hd, T, graph=graph,
            nbatch=self.nbh, strides=(T * T, T * hd, T * hd))
        # dP = dOut @ V^T
        c.matmul(T, T, hd, trans_b=True, batched=True, nbatch=self.nbh)(
            self.dao16, self.v16, self.dp32, T, T, hd, graph=graph,
            nbatch=self.nbh, strides=(T * hd, T * hd, T * T))
        TK["attn_softmax_bwd"]([self.dp32, self.p16, self.ds16],
                               self.nbh * T * c.dev.row_subgroup_size, T, self.scale, graph=graph)
        # dQ = dS @ K, dK = dS^T @ Q
        c.matmul(T, hd, T, batched=True, nbatch=self.nbh)(
            self.ds16, self.k16, self.dq32, T, hd, T, graph=graph,
            nbatch=self.nbh, strides=(T * T, T * hd, T * hd))
        c.matmul(T, hd, T, trans_a=True, batched=True, nbatch=self.nbh)(
            self.ds16, self.q16, self.dk32, T, hd, T, graph=graph,
            nbatch=self.nbh, strides=(T * T, T * hd, T * hd))

        TK["merge_qkv_grad"]([self.dq32, self.dk32, self.dv32, self.dqkv16],
                             self.rows * D, D, T, H, hd, graph=graph)
        return self.qkv.backward(self.dqkv16, need_dx=True, graph=graph)


class Block:
    def __init__(self, ctx, B, T, D, H, name=""):
        self.ctx = ctx
        self.rows = B * T
        self.D = D
        self.ln1 = LayerNorm(ctx, self.rows, D, name + ".ln1")
        self.attn = Attention(ctx, B, T, D, H, name + ".attn")
        self.ln2 = LayerNorm(ctx, self.rows, D, name + ".ln2")
        self.fc1 = Dense(ctx, self.rows, D, 4 * D, name + ".fc1")
        self.fc2 = Dense(ctx, self.rows, 4 * D, D, name + ".fc2",
                         scale=np.sqrt(2.0 / (4 * D)) * 0.5)
        self.res1 = ctx.buf(self.rows * D * 4)
        self.res2 = ctx.buf(self.rows * D * 4)
        self.z1 = ctx.buf(self.rows * 4 * D * 4)
        self.h16 = ctx.buf(self.rows * 4 * D * 2)
        self.dh16 = ctx.buf(self.rows * 4 * D * 2)
        self.dres1 = ctx.buf(self.rows * D * 4)
        self.dx = ctx.buf(self.rows * D * 4)
        self.attn16 = ctx.buf(self.rows * D * 2)
        self.mlp16 = ctx.buf(self.rows * D * 2)

    def forward(self, x32, graph=None):
        c, TK, K = self.ctx, self.ctx.TK, self.ctx.K
        n = self.rows * self.D
        self.x32 = x32
        h = self.ln1.forward(x32, graph=graph)
        a = self.attn.forward(h, graph=graph)
        TK["residual"]([a, self.attn.proj.b.w32, x32, self.res1], n, self.D, graph=graph)

        h2 = self.ln2.forward(self.res1, graph=graph)
        acc = self.fc1.forward(h2, graph=graph)
        K["bias_gelu"]([acc, self.fc1.b.w32, self.z1, self.h16],
                       self.rows * 4 * self.D, 4 * self.D, graph=graph)
        m = self.fc2.forward(self.h16, graph=graph)
        TK["residual"]([m, self.fc2.b.w32, self.res1, self.res2], n, self.D, graph=graph)
        return self.res2

    def backward(self, d32, graph=None):
        c, TK, K = self.ctx, self.ctx.TK, self.ctx.K
        n = self.rows * self.D
        # MLP branch. The residual sends the same gradient down both paths.
        TK["to16"]([d32, self.mlp16], n, graph=graph)
        dh = self.fc2.backward(self.mlp16, need_dx=True, graph=graph)
        K["bias_gelu_bwd16"]([dh, self.z1, self.dh16],
                             self.rows * 4 * self.D, graph=graph)
        dln2 = self.fc1.backward(self.dh16, need_dx=True, graph=graph)
        dres1_mlp = self.ln2.backward(dln2, graph=graph)
        TK["add"]([d32, dres1_mlp, self.dres1], n, graph=graph)

        # Attention branch.
        TK["to16"]([self.dres1, self.attn16], n, graph=graph)
        dln1 = self.attn.backward(self.attn16, graph=graph)
        dx_attn = self.ln1.backward(dln1, graph=graph)
        TK["add"]([self.dres1, dx_attn, self.dx], n, graph=graph)
        return self.dx

    def params(self):
        for p in (self.ln1.g, self.ln1.b, self.ln2.g, self.ln2.b,
                  self.attn.qkv.W, self.attn.qkv.b,
                  self.attn.proj.W, self.attn.proj.b,
                  self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b):
            yield p


class GPT:
    def __init__(self, ctx, B, T, D, H, n_layer, vocab, pad_vocab=None):
        self.ctx, self.B, self.T, self.D = ctx, B, T, D
        self.rows = B * T
        self.vocab = vocab
        self.pad_vocab = pad_vocab or ((vocab + 15) // 16) * 16
        rng = np.random.default_rng(1234)

        self.tok = Param(ctx, (self.pad_vocab, D),
                         (rng.standard_normal((self.pad_vocab, D)) * 0.02).astype(np.float32),
                         "tok")
        self.pos = Param(ctx, (T, D),
                         (rng.standard_normal((T, D)) * 0.02).astype(np.float32), "pos")
        self.x0 = ctx.buf(self.rows * D * 4)
        self.blocks = [Block(ctx, B, T, D, H, f"b{i}") for i in range(n_layer)]
        self.lnf = LayerNorm(ctx, self.rows, D, "lnf")
        self.head = Dense(ctx, self.rows, D, self.pad_vocab, "head", scale=0.02)

        self.dlog16 = ctx.buf(self.rows * self.pad_vocab * 2)
        self.dlog32 = ctx.buf(self.rows * self.pad_vocab * 4)
        self.loss = ctx.buf(self.rows * 4, "cached")

    def params(self):
        yield self.tok
        yield self.pos
        for b in self.blocks:
            yield from b.params()
        yield self.lnf.g
        yield self.lnf.b
        yield self.head.W
        yield self.head.b

    def n_params(self):
        return sum(p.n for p in self.params())

    def forward(self, ids_buf, graph=None):
        TK = self.ctx.TK
        TK["embed"]([self.tok.w32, self.pos.w32, ids_buf, self.x0],
                    self.rows * self.D, self.D, self.T, graph=graph)
        x = self.x0
        for b in self.blocks:
            x = b.forward(x, graph=graph)
        h = self.lnf.forward(x, graph=graph)
        return self.head.forward(h, graph=graph)

    def record(self, ids_buf, targets_buf, graph):
        K, TK = self.ctx.K, self.ctx.TK
        logits = self.forward(ids_buf, graph=graph)
        K["softmax_ce"]([logits, targets_buf, self.dlog16, self.dlog32, self.loss],
                        self.rows, self.pad_vocab, self.vocab, graph=graph)
        d = self.head.backward(self.dlog16, need_dx=True, graph=graph)
        d = self.lnf.backward(d, graph=graph)
        for b in reversed(self.blocks):
            d = b.backward(d, graph=graph)
        # Embedding gradients are a scatter-add, so the buffers must start at
        # zero every step.
        TK["col_sum_zero"]([self.tok.g32], self.tok.n, graph=graph)
        TK["col_sum_zero"]([self.pos.g32], self.pos.n, graph=graph)
        TK["embed_bwd"]([d, ids_buf, self.tok.g32, self.pos.g32],
                        self.rows * self.D, self.D, self.T, graph=graph)

    def read_loss(self):
        self.loss.invalidate()
        return float(self.loss.array(np.float32, (self.rows,)).mean())
