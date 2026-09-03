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
    K["push"] = Elementwise(
        dev, "grad_push",
        [("g", "f32", "readonly"), ("arena", "f32", "")],
        """
        atomicAdd(arena[p.off + i], g[i]);
        """,
        push=[("off", "uint")],
        extensions=["GL_EXT_shader_atomic_float"])

    K["zero"] = Elementwise(
        dev, "arena_zero", [("arena", "f32", "writeonly")], "arena[i] = 0.0;")

    # Read one parameter's summed gradient back out of the arena.
    K["pull"] = Elementwise(
        dev, "grad_pull",
        [("arena", "f32", "readonly"), ("g", "f32", "writeonly")],
        "g[i] = arena[p.off + i];",
        push=[("off", "uint")])
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
        for p, off in zip(self.params, self.offsets):
            self.K["push"]([p.g32, self.arena_buf], p.n, off, graph=graph)

    def zero_arena(self, graph=None):
        self.K["zero"]([self.arena_buf], self.total, graph=graph)

    def step_from_arena(self, lr):
        """Apply AdamW using the shared summed gradient rather than the local
        one. Identical inputs on every replica keep the weights in lockstep."""
        self.opt.advance(lr)
        k = self.ctx.K["adamw"]
        for p, off in zip(self.params, self.offsets):
            # The arena holds every parameter's gradient contiguously, so the
            # optimiser reads a slice of it. Offsetting inside the kernel would
            # need a second push constant; instead each parameter's gradient is
            # copied back, which is one pass over the parameters.
            self.K["pull"]([self.arena_buf, p.g32], p.n, off)
            k([p.w32, p.g32, p.m, p.v, p.w16, self.opt.hp], p.n,
              0.0 if p.no_decay else self.opt.wd)

    def destroy(self):
        for k in self.K.values():
            k.destroy()
        self.arena_buf.destroy()
        self.ctx.destroy()
