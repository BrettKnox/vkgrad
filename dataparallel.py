"""Data-parallel training with gradient exchange through shared host memory.

Multi-GPU training is where vendor lock-in bites hardest. Portable kernels do
not help if the collective is NCCL, which is NVIDIA-only. This is the same
pattern as DDP, built on `VK_EXT_external_memory_host` instead:

  * weights are *replicated*, one copy per device, never exchanged
  * every device computes gradients for its own shard of the batch
  * each device atomically adds its gradients into one shared host arena
  * every device then applies the identical optimiser step from that arena

Because the replicas start identical and apply identical updates, they stay
identical, so only gradients ever cross between devices. And because the arena
is ordinary host memory imported by each device, the crossing needs no vendor
interconnect, no common driver, and no copy: on unified memory it is the same
DRAM the GPU already reads.

The arena is a flat concatenation of every parameter's gradient, so the whole
exchange is two kernels: one zeroing pass and one atomic add per device.
"""

import numpy as np

from kernels import Elementwise
from vk import Device


def make_reduce_kernels(dev):
    K = {}
    # Atomically accumulate a device's local gradient into the shared arena.
    # This IS the all-reduce: on unified memory there is nothing to transfer,
    # the workers are adding into the same bytes.
    #
    # Built only where the device has float atomics. Raspberry Pi's v3dv and
    # several mobile drivers do not, and building this unconditionally took
    # single-device gradient accumulation down with it -- GradAccum has one
    # writer, never dispatches `push`, and was failing at construction over a
    # kernel it does not use.
    if getattr(dev, "has_atomic_float", True):
        K["push"] = Elementwise(
            dev, "grad_push",
            [("g", "f32", "readonly"), ("arena", "f32", "")],
            """
            atomicAdd(arena[p.off + i], g[i]);
            """,
            push=[("off", "uint")],
            extensions=["GL_EXT_shader_atomic_float"])

    # Non-atomic accumulate, for a single writer. Gradient accumulation on one
    # device has no concurrent writers, and at GPT-2 scale a step issues ~162
    # million atomics, which is slow enough to trip the 2 s TDR watchdog and
    # lose the device. A plain read-modify-write is both correct here and much
    # faster.
    K["add"] = Elementwise(
        dev, "grad_add",
        [("g", "f32", "readonly"), ("arena", "f32", "")],
        "arena[p.off + i] += g[i];",
        push=[("off", "uint")])

    K["zero"] = Elementwise(
        dev, "arena_zero", [("arena", "f32", "writeonly")], "arena[i] = 0.0;")

    # Read one parameter's summed gradient back out of the arena, scaled.
    #
    # The scale matters and is easy to get wrong. Each worker's loss is a mean
    # over its OWN rows, so summing W workers (or W microbatches) gives a
    # gradient W times larger than the same examples in one batch. Without the
    # 1/W the effective learning rate is silently multiplied by W.
    K["pull"] = Elementwise(
        dev, "grad_pull",
        [("arena", "f32", "readonly"), ("g", "f32", "writeonly")],
        "g[i] = arena[p.off + i] * p.scale;",
        push=[("off", "uint"), ("scale", "float")])
    return K


class HostArena:
    """One host allocation, importable by any number of devices.

    Alignment must satisfy every participating device's
    minImportedHostPointerAlignment; 4096 covers the ones seen so far.
    """

    def __init__(self, n_floats, alignment=4096):
        import ctypes as C
        self.n = n_floats
        # Both the pointer AND the imported size must be multiples of the
        # device's minImportedHostPointerAlignment, or vkAllocateMemory fails
        # with VK_ERROR_INVALID_EXTERNAL_HANDLE.
        nbytes = ((n_floats * 4 + alignment - 1) // alignment) * alignment
        self._raw = (C.c_ubyte * (nbytes + alignment))()
        addr = C.addressof(self._raw)
        self.ptr = addr + ((-addr) % alignment)
        self.nbytes = nbytes
        self.view = np.frombuffer((C.c_float * (nbytes // 4)).from_address(self.ptr),
                                  dtype=np.float32)

    def buffer_for(self, dev):
        return dev.import_host_buffer(self.ptr, self.nbytes)


class Replica:
    """One device's copy of the model, plus its slot in the exchange."""

    def __init__(self, dev, build, arena):
        self.dev = dev
        self.ctx, self.model, self.opt, self.xbuf, self.ybuf = build(dev)
        self.K = make_reduce_kernels(dev)
        self.arena_buf = arena.buffer_for(dev)
        self.params = list(self.model.params())
        offs, off = [], 0
        for p in self.params:
            offs.append(off)
            off += p.n
        self.offsets = offs
        self.total = off
        assert off <= arena.n, f"arena too small: need {off}, have {arena.n}"

    def push_gradients(self, graph=None):
        if "push" not in self.K:
            raise VkError(
                "multi-device gradient exchange needs VK_EXT_shader_atomic_float, "
                "which this device lacks. Single-device accumulation (GradAccum) "
                "still works; it uses the non-atomic path.")
        for p, off in zip(self.params, self.offsets):
            self.K["push"]([p.g32, self.arena_buf], p.n, off, graph=graph)

    def zero_arena(self, graph=None):
        self.K["zero"]([self.arena_buf], self.total, graph=graph)

    def step_from_arena(self, lr, n_workers=1):
        """Apply AdamW using the shared summed gradient rather than the local
        one. Identical inputs on every replica keep the weights in lockstep.

        n_workers scales the sum back to a mean, so the effective learning rate
        does not grow with the device count.
        """
        self.opt.advance(lr)
        k = self.ctx.K["adamw"]
        scale = 1.0 / max(n_workers, 1)
        for p, off in zip(self.params, self.offsets):
            self.K["pull"]([self.arena_buf, p.g32], p.n, off, scale)
            k([p.w32, p.g32, p.m, p.v, p.w16, self.opt.hp], p.n,
              0.0 if p.no_decay else self.opt.wd)

    def destroy(self):
        for k in self.K.values():
            k.destroy()
        self.arena_buf.destroy()
        self.ctx.destroy()


class GradAccum:
    """Gradient accumulation over microbatches, on one device.

    This is the same primitive as the multi-device exchange above, with the
    workers separated in time rather than across devices: each microbatch adds
    its gradients into an arena, and the optimiser steps once from the sum.

    It matters more here than on a discrete GPU. Throughput on this hardware
    *falls* as batch grows (1,117 to 668 tokens/s from batch 2 to 8 at GPT-2
    scale) because the machine is bandwidth-bound and activation traffic scales
    with batch. So a large effective batch is cheaper as many small microbatches
    than as one large batch, which is the opposite of the usual advice.
    """

    def __init__(self, ctx, params, accum=1):
        self.ctx, self.params, self.accum = ctx, list(params), accum
        self.K = make_reduce_kernels(ctx.dev)
        offs, off = [], 0
        for p in self.params:
            offs.append(off)
            off += p.n
        self.offsets, self.total = offs, off
        # Device-local: nothing outside this device reads it.
        self.arena = ctx.buf(off * 4, "device")

    def record_push(self, graph):
        """Add this microbatch's gradients into the accumulator.

        Uses the non-atomic add: one device, one writer, and the atomic version
        is slow enough at scale to lose the device to the TDR watchdog.
        """
        for p, o in zip(self.params, self.offsets):
            self.K["add"]([p.g32, self.arena], p.n, o, graph=graph)

    def record_zero(self, graph):
        self.K["zero"]([self.arena], self.total, graph=graph)

    def record_apply(self, graph, opt, scale=None):
        """Move the summed gradient back and take one optimiser step.

        `scale` defaults to 1/accum so the result matches a single batch of
        accum times the size.
        """
        sc = (1.0 / self.accum) if scale is None else scale
        k = self.ctx.K["adamw"]
        for p, o in zip(self.params, self.offsets):
            self.K["pull"]([self.arena, p.g32], p.n, o, sc, graph=graph)
            k([p.w32, p.g32, p.m, p.v, p.w16, opt.hp], p.n,
              0.0 if p.no_decay else opt.wd, graph=graph)

    def destroy(self):
        for k in self.K.values():
            k.destroy()
