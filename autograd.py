"""Tape-based autograd over the Vulkan kernels.

Tape nodes are coarse on purpose. A node is "linear + bias + GELU", not
"matmul" then "add" then "gelu", because on this hardware every node boundary
is a DRAM round trip. Fusing at the node level *is* the optimisation; there is
no separate fusion pass afterwards.

Mixed precision is forced by the hardware: cooperative matrix takes f16 inputs
and accumulates in f32, so activations and the weight mirrors are f16 while
master weights, gradients and optimiser state stay f32.
"""

import numpy as np

from kernels import CONFIGS, TILE, Matmul, make_kernels
from vk import VkError


def pick_config(m, n, k, trans_a=False, trans_b=False, min_groups=12):
    """Largest tile that divides the shape, chosen without benchmarking.

    autotune_matmul is better but costs a compile-and-run sweep per shape.
    This is what a first run uses before the tuner has seen the model.
    """
    best = None
    for sg, wm, wn, lds in CONFIGS:
        if lds and (trans_a or trans_b):
            continue
        bm, bn = sg * wm * TILE, wn * TILE
        if m % bm or n % bn or k % (32 if lds else TILE):
            continue
        groups = (m // bm) * (n // bn)
        score = (groups >= min_groups, bm * bn, groups)
        if best is None or score > best[0]:
            best = (score, (sg, wm, wn, lds))
    if best is None:
        raise VkError(f"no matmul tile fits {m}x{n}x{k}")
    return best[1]


class Ctx:
    """Owns the device, the compiled kernels, and a matmul cache keyed by shape."""

    def __init__(self, dev, tune=False):
        self.dev = dev
        self.K = make_kernels(dev)
        self.tune = tune
        self._mm = {}
        self._owned = []

    def matmul(self, m, n, k, trans_a=False, trans_b=False, batched=False, nbatch=1):
        key = (m, n, k, trans_a, trans_b, batched, nbatch)
        if key not in self._mm:
            if self.tune:
                from kernels import autotune_matmul
                mm, _ = autotune_matmul(self.dev, m, n, k, trans_a, trans_b,
                                        batched=batched, nbatch=nbatch)
            else:
                # Batched dispatches multiply the workgroup count by the batch
                # dimension, so a small per-matrix grid is fine there.
                sg, wm, wn, lds = pick_config(m, n, k, trans_a, trans_b,
                                              min_groups=1 if batched else 12)
                if batched:
                    lds = False
                mm = Matmul(self.dev, sg, wm, wn, trans_a, trans_b, lds=lds,
                            batched=batched)
            self._mm[key] = mm
        return self._mm[key]

    def buf(self, nbytes, kind="device"):
        b = self.dev.buffer(nbytes, kind)
        self._owned.append(b)
        return b

    def zeros(self, n_floats):
        b = self.buf(n_floats * 4, "device")
        self.K["zero"]([b], n_floats)
        return b

    def destroy(self):
        for m in self._mm.values():
            m.destroy()
        for k in self.K.values():
            k.destroy()
        for b in self._owned:
            b.destroy()
        self._mm.clear()


class Param:
    """f32 master, f16 mirror for the matmul, f32 grad, Adam moments.

    The mirror is refreshed inside the fused optimiser step, so keeping it
    costs no extra pass over the parameters.
    """

    def __init__(self, ctx, shape, init=None, name="", no_decay=False):
        self.ctx = ctx
        self.shape = shape
        self.n = int(np.prod(shape))
        self.name = name
        self.no_decay = no_decay
        # host-CACHED, not host-coherent-only: the CPU reads master weights
        # (checkpointing, gradient checks, and any CPU worker in a
        # heterogeneous job), and CPU reads from write-combined memory run at
        # 0.32 GB/s against 23.30 GB/s cached, a factor of 73. The GPU pays
        # about 2% on its own reads for that.
        self.w32 = ctx.buf(self.n * 4, "cached")
        self.w16 = ctx.buf(self.n * 2, "device")
        self.g32 = ctx.buf(self.n * 4, "device")
        self.m = ctx.zeros(self.n)
        self.v = ctx.zeros(self.n)
        w = self.w32.array(np.float32, shape)
        w[:] = 0 if init is None else init
        self.w32.flush()
        self.sync16()

    def sync16(self):
        self.ctx.K["cast16"]([self.w32, self.w16], self.n)

    def numpy(self):
        self.w32.invalidate()
        return self.w32.array(np.float32, self.shape)

    def set(self, arr):
        self.w32.array(np.float32, self.shape)[:] = arr
        self.w32.flush()
        self.sync16()

    def grad_numpy(self):
        """Copy the gradient back for checking. Only used by tests."""
        tmp = self.ctx.buf(self.n * 4, "cached")
        self.ctx.dev.copy(self.g32, tmp, self.n * 4)
        tmp.invalidate()
        return tmp.array(np.float32, self.shape).copy()


class Linear:
    """y = gelu(x @ W + b), or the same without the activation.

    Forward: one matmul plus one fused bias+activation+narrow kernel.
    Backward: one fused activation-backward, then a matmul for dW, a tiny
    reduction for db, and a matmul for dX when an earlier layer needs it.
    """

    def __init__(self, ctx, batch, in_f, out_f, activation=True, need_dx=True, name=""):
        self.ctx, self.name = ctx, name
        self.batch, self.in_f, self.out_f = batch, in_f, out_f
        self.activation = activation
        self.need_dx = need_dx
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        w = (rng.standard_normal((in_f, out_f)) * np.sqrt(2.0 / in_f)).astype(np.float32)
        self.W = Param(ctx, (in_f, out_f), w, name + ".W")
        self.b = Param(ctx, (out_f,), np.zeros(out_f, np.float32), name + ".b",
                       no_decay=True)

        self.acc = ctx.buf(batch * out_f * 4)    # raw matmul output, f32
        self.z = ctx.buf(batch * out_f * 4)      # pre-activation, kept for backward
        self.y16 = ctx.buf(batch * out_f * 2)    # activation, f16, feeds next matmul
        self.dz16 = ctx.buf(batch * out_f * 2)
        self.dz32 = ctx.buf(batch * out_f * 4)
        self.dx32 = ctx.buf(batch * in_f * 4) if need_dx else None
        self.x16 = None

    def forward(self, x16, tape, graph=None):
        c, K = self.ctx, self.ctx.K
        self.x16 = x16
        c.matmul(self.batch, self.out_f, self.in_f)(
            x16, self.W.w16, self.acc, self.batch, self.out_f, self.in_f, graph=graph)
        n = self.batch * self.out_f
        if self.activation:
            K["bias_gelu"]([self.acc, self.b.w32, self.z, self.y16], n, self.out_f,
                           graph=graph)
            out = self.y16
        else:
            K["bias"]([self.acc, self.b.w32, self.z], n, self.out_f, graph=graph)
            out = self.z
        tape.append(self.backward)
        return out

    def backward(self, dout16, dout32, graph=None):
        """dout16/dout32 are dLoss/dOutput in both precisions: f16 for the
        matmuls, f32 for the bias reduction and the activation derivative.

        Returns (None, dx32); the next layer down narrows to f16 itself inside
        its fused activation-backward kernel, so no separate cast is needed.
        """
        c, K = self.ctx, self.ctx.K
        n = self.batch * self.out_f
        if self.activation:
            K["bias_gelu_bwd"]([dout32, self.z, self.dz16, self.dz32], n, graph=graph)
            g16, g32 = self.dz16, self.dz32
        else:
            g16, g32 = dout16, dout32

        # dW = X^T @ dZ. X is stored batch x in_f, which is K x M, so trans_a.
        c.matmul(self.in_f, self.out_f, self.batch, trans_a=True)(
            self.x16, g16, self.W.g32, self.in_f, self.out_f, self.batch, graph=graph)
        # db = column sums of dZ.
        K["col_sum"]([g32, self.b.g32], self.out_f, self.batch, graph=graph)

        if not self.need_dx:
            return None, None
        # dX = dZ @ W^T. W is stored in_f x out_f, which is N x K, so trans_b.
        c.matmul(self.batch, self.in_f, self.out_f, trans_b=True)(
            g16, self.W.w16, self.dx32, self.batch, self.in_f, self.out_f, graph=graph)
        return None, self.dx32


class MLP:
    """Hidden layers use GELU; the head is linear and feeds softmax + CE.

    n_classes is padded up to `pad_classes` so the head's matmul has a tile
    friendly N. The padded logits are masked inside the softmax kernel, so they
    contribute neither to the loss nor to the gradient.
    """

    def __init__(self, ctx, batch, sizes, n_classes, pad_classes=16):
        assert pad_classes >= n_classes and pad_classes % TILE == 0
        self.ctx, self.batch = ctx, batch
        self.n_classes, self.pad = n_classes, pad_classes
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(ctx, batch, sizes[i], sizes[i + 1],
                                      activation=True, need_dx=(i > 0), name=f"fc{i}"))
        self.layers.append(Linear(ctx, batch, sizes[-1], pad_classes,
                                  activation=False, need_dx=True, name="head"))

        self.dlog16 = ctx.buf(batch * pad_classes * 2)
        self.dlog32 = ctx.buf(batch * pad_classes * 4)
        self.loss = ctx.buf(batch * 4, "cached")

    def params(self):
        for l in self.layers:
            yield l.W
            yield l.b

    def logits(self, x16, tape=None, graph=None):
        h = x16
        for l in self.layers:
            h = l.forward(h, tape if tape is not None else [], graph=graph)
        return h

    def record(self, x16, labels_buf, graph):
        """Record forward and backward into a graph. Read the loss from
        read_loss() after the graph has run."""
        K = self.ctx.K
        tape = []
        logits = self.logits(x16, tape, graph=graph)
        K["softmax_ce"]([logits, labels_buf, self.dlog16, self.dlog32, self.loss],
                        self.batch, self.pad, self.n_classes, graph=graph)
        g16, g32 = self.dlog16, self.dlog32
        for fn in reversed(tape):
            g16, g32 = fn(g16, g32, graph=graph)

    def read_loss(self):
        self.loss.invalidate()
        return float(self.loss.array(np.float32, (self.batch,)).mean())

    def forward_backward(self, x16, labels_buf):
        """Eager: one submit per kernel. Correct but launch-bound. The tests
        use this; training should use TrainStep."""
        K = self.ctx.K
        tape = []
        logits = self.logits(x16, tape)
        K["softmax_ce"]([logits, labels_buf, self.dlog16, self.dlog32, self.loss],
                        self.batch, self.pad, self.n_classes)
        loss = self.read_loss()
        g16, g32 = self.dlog16, self.dlog32
        for fn in reversed(tape):
            g16, g32 = fn(g16, g32)
        return loss

    def predict(self, x16):
        logits = self.logits(x16)
        tmp = self.ctx.buf(self.batch * self.pad * 4, "cached")
        self.ctx.dev.copy(logits, tmp, self.batch * self.pad * 4)
        tmp.invalidate()
        out = tmp.array(np.float32, (self.batch, self.pad))[:, :self.n_classes]
        return out.argmax(1).copy()


class AdamW:
    """One fused kernel per parameter: reads w, g, m, v and writes w, m, v and
    the f16 mirror. Unfused this is four passes over every parameter, which at
    this ridge point can cost as much as the forward pass."""

    def __init__(self, ctx, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.ctx, self.params = ctx, list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.t = 0
        # Step-varying scalars live in a mapped buffer rather than push
        # constants, so a recorded command buffer stays valid as they change.
        self.hp = ctx.buf(8 * 4, "shared")
        self.hpv = self.hp.array(np.float32)
        self._set_hp(lr)

    def _set_hp(self, lr):
        bc1 = 1 - self.b1 ** max(self.t, 1)
        bc2 = 1 - self.b2 ** max(self.t, 1)
        self.hpv[:6] = [lr, self.b1, self.b2, bc1, bc2, self.eps]
        self.hp.flush()

    def advance(self, lr=None):
        """Roll the schedule forward. On unified memory this is just a write to
        memory the GPU already sees."""
        self.t += 1
        self._set_hp(self.lr if lr is None else lr)

    def record(self, graph):
        k = self.ctx.K["adamw"]
        for p in self.params:
            wd = 0.0 if p.no_decay else self.wd
            k([p.w32, p.g32, p.m, p.v, p.w16, self.hp], p.n, wd, graph=graph)

    def step(self, lr=None):
        """Eager: one submit per parameter."""
        self.advance(lr)
        k = self.ctx.K["adamw"]
        for p in self.params:
            wd = 0.0 if p.no_decay else self.wd
            k([p.w32, p.g32, p.m, p.v, p.w16, self.hp], p.n, wd)


class TrainStep:
    """The whole step recorded once and replayed: forward, backward, optimiser.

    This is where the gap between a 1.7 us batched dispatch and a 127 us submit
    gets collected. Everything that varies per step is either data in mapped
    memory (the batch, the labels) or a scalar in the optimiser's hyperparameter
    buffer, so the command buffer is never re-recorded.
    """

    def __init__(self, model, opt, xbuf, ybuf):
        self.model, self.opt = model, opt
        self.graph = model.ctx.dev.graph("train_step")
        model.record(xbuf, ybuf, self.graph)
        opt.record(self.graph)
        self.graph.finish()
        self.n_dispatch = self.graph.n_dispatch

    def __call__(self, lr=None):
        self.opt.advance(lr)
        dt = self.graph.submit()
        return self.model.read_loss(), dt
