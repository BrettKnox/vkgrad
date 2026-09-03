"""Two independent Vulkan devices operating on one shared host allocation.

This is the plumbing a cross-vendor multi-GPU training job needs.

Multi-GPU training today runs on NCCL, which is NVIDIA-only, and that is one of
the practical lock-ins: even if your kernels were portable, your gradient
exchange is not. `VK_EXT_external_memory_host` offers a way around it. Ordinary
host memory can be imported by several independent VkDevices at once, so plain
system RAM becomes a shared arena between GPUs. Nothing in that requires the
devices to come from the same vendor, share a driver, or sit on a common
interconnect.

On a unified-memory APU the shared arena is literally the same DRAM the GPU
already reads, so exchange costs nothing beyond the read. On a discrete card it
is the host side of that card's PCIe path.

These checks run two VkDevices from the one physical GPU available here. That
validates the mechanism (independent devices, independent queues, one shared
allocation, visible writes both ways) without claiming to have tested two
vendors, which would need hardware this machine does not have.
"""

import ctypes as C
import struct

import numpy as np

from compile import compile_glsl
from vk import Device, VkError

ADD_SRC = """#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer X { float x[]; };
layout(push_constant) uniform P { uint n; float v; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.n) x[i] += p.v;
}
"""


def aligned_host_alloc(nbytes, alignment):
    """A raw host allocation the GPU is allowed to import. Kept alive by the
    returned object; numpy views it without copying."""
    raw = (C.c_ubyte * (nbytes + alignment))()
    addr = C.addressof(raw)
    off = (-addr) % alignment
    ptr = addr + off
    view = np.frombuffer((C.c_ubyte * nbytes).from_address(ptr), dtype=np.uint8)
    return raw, ptr, view


def test_import_roundtrip(dev):
    align = dev.host_pointer_alignment()
    assert align, "device cannot import host memory"
    n = 4096
    keep, ptr, _ = aligned_host_alloc(n * 4, align)
    arr = np.frombuffer((C.c_float * n).from_address(ptr), dtype=np.float32)
    arr[:] = np.arange(n, dtype=np.float32)

    buf = dev.import_host_buffer(ptr, n * 4)
    k = dev.kernel(compile_glsl(ADD_SRC, name="addv"), 1, push_size=8, name="addv")
    try:
        dev.run(k, [buf], (n + 255) // 256, struct.pack("=If", n, 1.5))
        expect = np.arange(n, dtype=np.float32) + 1.5
        assert np.array_equal(arr, expect), "GPU write not visible in host memory"
        print(f"  host alloc imported, GPU wrote through it   OK  (align {align})")
    finally:
        k.destroy()
        buf.destroy()
    del keep


def test_two_devices_one_allocation():
    """The actual claim: two independent devices, one allocation, both see it."""
    a = Device()
    b = Device()
    try:
        align = max(a.host_pointer_alignment(), b.host_pointer_alignment())
        n = 8192
        keep, ptr, _ = aligned_host_alloc(n * 4, align)
        arr = np.frombuffer((C.c_float * n).from_address(ptr), dtype=np.float32)
        arr[:] = 0.0

        buf_a = a.import_host_buffer(ptr, n * 4)
        buf_b = b.import_host_buffer(ptr, n * 4)
        assert a.dev.value != b.dev.value, "expected two distinct VkDevices"

        ka = a.kernel(compile_glsl(ADD_SRC, name="addv"), 1, push_size=8, name="addv_a")
        kb = b.kernel(compile_glsl(ADD_SRC, name="addv"), 1, push_size=8, name="addv_b")
        try:
            # Device A contributes 3, device B contributes 4, through the same
            # bytes, with no copy and no transfer between them.
            a.run(ka, [buf_a], (n + 255) // 256, struct.pack("=If", n, 3.0))
            b.run(kb, [buf_b], (n + 255) // 256, struct.pack("=If", n, 4.0))
            assert np.all(arr == 7.0), f"expected 7.0 everywhere, got {arr.min()}..{arr.max()}"
            print("  two VkDevices sharing one allocation          OK  (3 + 4 = 7)")
        finally:
            ka.destroy()
            kb.destroy()
            buf_a.destroy()
            buf_b.destroy()
        del keep
    finally:
        a.destroy()
        b.destroy()


def test_gradient_exchange():
    """Data-parallel gradient averaging with no copies at all.

    Each device reduces its own shard into the shared arena. On unified memory
    the 'all-reduce' is just both devices adding into the same bytes, which is
    the cheapest possible collective: zero transfers.
    """
    a = Device()
    b = Device()
    try:
        align = max(a.host_pointer_alignment(), b.host_pointer_alignment())
        n = 65536
        keep, ptr, _ = aligned_host_alloc(n * 4, align)
        grad = np.frombuffer((C.c_float * n).from_address(ptr), dtype=np.float32)
        grad[:] = 0.0

        ba = a.import_host_buffer(ptr, n * 4)
        bb = b.import_host_buffer(ptr, n * 4)
        ka = a.kernel(compile_glsl(ADD_SRC, name="addv"), 1, push_size=8, name="ga")
        kb = b.kernel(compile_glsl(ADD_SRC, name="addv"), 1, push_size=8, name="gb")
        try:
            groups = (n + 255) // 256
            # Two workers, gradients 0.25 and 0.75, summed in place.
            a.run(ka, [ba], groups, struct.pack("=If", n, 0.25))
            b.run(kb, [bb], groups, struct.pack("=If", n, 0.75))
            assert np.allclose(grad, 1.0), f"sum wrong: {grad.min()}..{grad.max()}"
            print(f"  2-worker gradient sum over shared memory     OK  "
                  f"({n * 4 >> 10} KiB, 0 bytes transferred)")
        finally:
            ka.destroy()
            kb.destroy()
            ba.destroy()
            bb.destroy()
        del keep
    finally:
        a.destroy()
        b.destroy()


def main():
    dev = Device()
    print(dev)
    if "VK_EXT_external_memory_host" not in dev.extensions:
        print("  VK_EXT_external_memory_host unsupported; skipping")
        dev.destroy()
        return
    try:
        test_import_roundtrip(dev)
    finally:
        dev.destroy()
    test_two_devices_one_allocation()
    test_gradient_exchange()
    print("\nshared-memory checks passed")


if __name__ == "__main__":
    main()
