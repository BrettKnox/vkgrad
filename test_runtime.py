"""Runtime self-checks. Small dispatches only: safe to run on a busy machine.

Run: python test_runtime.py   (or: VKGRAD_VALIDATE=1 python test_runtime.py)
"""

import struct

import numpy as np

from compile import compile_glsl
from vk import Device

DOUBLE_SRC = """#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer X { float x[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.n) x[i] = x[i] * 2.0 + 1.0;
}
"""

# One subgroup, one 16x16x16 WMMA tile. Proves the matrix cores actually run.
COOP_SRC = """#version 450
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_16bit_storage : enable

layout(local_size_x = 32) in;
layout(binding = 0) buffer A { float16_t a[]; };
layout(binding = 1) buffer B { float16_t b[]; };
layout(binding = 2) buffer C { float      c[]; };

void main() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mc =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopMatLoad(ma, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(mb, b, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    mc = coopMatMulAdd(ma, mb, mc);
    coopMatStore(mc, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
"""

# Reads its own subgroup size back, so we can prove wave32 was really granted
# rather than trusting the pipeline creation to have honoured the request.
SUBGROUP_SRC = """#version 450
#extension GL_KHR_shader_subgroup_basic : enable
layout(local_size_x = {size}) in;
layout(binding = 0) buffer O {{ uint o[]; }};
void main() {{ if (gl_LocalInvocationID.x == 0) o[0] = gl_SubgroupSize; }}
"""


def test_zero_copy_roundtrip(dev):
    """A tensor is a numpy array and a GPU buffer at once. No transfers."""
    n = 4096
    buf = dev.buffer(n * 4, "shared")
    x = buf.array(np.float32, (n,))
    x[:] = np.arange(n, dtype=np.float32)
    expected = x * 2.0 + 1.0
    buf.flush()

    k = dev.kernel(compile_glsl(DOUBLE_SRC, name="double"), n_buffers=1,
                   push_size=4, name="double")
    dev.run(k, [buf], groups=(n + 255) // 256, push=struct.pack("I", n))
    buf.invalidate()

    # Same memory, no download step.
    assert np.array_equal(x, expected), "zero-copy roundtrip mismatch"
    k.destroy()
    buf.destroy()
    print("  zero-copy roundtrip           OK")


def test_wave32(dev):
    if not dev.can_set_subgroup_size:
        print(f"  required subgroup size      unavailable, kernels use "
              f"native width {dev.subgroup_size_native}")
        return
    lo, hi, stages = dev.subgroup_size_range()
    assert stages & 0x20, "requiredSubgroupSize not supported for compute stage"
    assert lo <= 32 <= hi, f"32 outside subgroup range [{lo},{hi}]"
    buf = dev.buffer(4, "cached")
    for want in (32, 64):
        if not lo <= want <= hi:
            continue
        # REQUIRE_FULL_SUBGROUPS: workgroup size must be a multiple of the
        # requested subgroup size, so the shader is generated to match.
        spv = compile_glsl(SUBGROUP_SRC.format(size=want), name=f"subgroup{want}")
        k = dev.kernel(spv, n_buffers=1, subgroup_size=want, name=f"wave{want}")
        dev.run(k, [buf], groups=1)
        buf.invalidate()
        got = int(buf.array(np.uint32, (1,))[0])
        assert got == want, f"asked for subgroup size {want}, shader saw {got}"
        k.destroy()
        print(f"  required subgroup size {want}      OK")
    buf.destroy()


def test_coop_matrix(dev):
    """16x16x16 f16 x f16 -> f32 on the matrix cores, checked against numpy."""
    if not dev.has_coop_matrix:
        print("  cooperative matrix          unavailable, scalar path in use")
        return
    cfgs = [c for c in dev.coop_matrix_configs()
            if c["A"] == "f16" and c["C"] == "f32" and c["scope"] == "subgroup"]
    assert cfgs, "no f16->f32 subgroup cooperative matrix config"

    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 16)).astype(np.float16)
    b = rng.standard_normal((16, 16)).astype(np.float16)

    ba = dev.buffer(16 * 16 * 2, "shared")
    bb = dev.buffer(16 * 16 * 2, "shared")
    bc = dev.buffer(16 * 16 * 4, "cached")
    ba.array(np.float16, (16, 16))[:] = a
    bb.array(np.float16, (16, 16))[:] = b
    ba.flush()
    bb.flush()

    k = dev.kernel(compile_glsl(COOP_SRC, name="coop"), n_buffers=3,
                   subgroup_size=32, name="coop16")
    dev.run(k, [ba, bb, bc], groups=1)
    bc.invalidate()
    got = bc.array(np.float32, (16, 16)).copy()

    ref = a.astype(np.float32) @ b.astype(np.float32)
    err = np.abs(got - ref).max()
    # f16 inputs, f32 accumulate: error is input rounding only.
    assert err < 2e-2, f"WMMA result off by {err}\ngot\n{got}\nref\n{ref}"
    print(f"  cooperative matrix 16x16x16    OK  (max abs err {err:.2e})")
    for o in (k, ba, bb, bc):
        o.destroy()


def test_memory_kinds(dev):
    """Every memory kind must allocate and, if host-visible, round-trip."""
    for kind in ("device", "shared", "cached", "bar"):
        b = dev.buffer(1 << 16, kind)
        try:
            if b.host_visible:
                b.array(np.uint32)[:16] = np.arange(16, dtype=np.uint32)
                assert np.array_equal(b.array(np.uint32)[:16], np.arange(16))
        finally:
            b.destroy()
    print("  memory kinds                  OK")


def main():
    dev = Device()
    print(dev)
    print(f"  heaps: " + ", ".join(f"{h['gib']:.2f} GiB"
                                   f"{' (device-local)' if h['device_local'] else ''}"
                                   for h in dev.heap_report()))
    try:
        test_memory_kinds(dev)
        test_zero_copy_roundtrip(dev)
        test_wave32(dev)
        test_coop_matrix(dev)
        print("all runtime checks passed")
    finally:
        dev.destroy()


if __name__ == "__main__":
    main()
