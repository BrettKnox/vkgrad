"""What does int8 cooperative matrix actually buy on this hardware?

int8 matrix units are the most widely available accelerator primitive in
consumer silicon: mobile NPUs, integrated graphics, older discrete cards and
every vendor's low-end parts have them, while f16 tensor units are scarcer and
bf16 scarcer still. If int8 training were viable it would run on hardware with
no CUDA and no realistic path to it.

Two things decide whether that is worth pursuing, and both are cheap to measure:

  1. Issue rate. If i8 matmul is no faster than f16 per operation, the only
     gain is halved operand bytes.
  2. Operand bytes. On a bandwidth-bound machine, halving operand width is
     worth roughly half of whatever share operand traffic holds.

  python -m bench.int8peak
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile import compile_glsl  # noqa: E402
from kernels import warmup  # noqa: E402
from vk import Device  # noqa: E402

HEAD = """#version 450
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_8bit_storage : enable
"""

BODY = """
layout(local_size_x = 32) in;
layout(binding = 0) buffer A {{ {at} a[]; }};
layout(binding = 1) buffer C {{ {ct} c[]; }};
layout(push_constant) uniform P {{ uint iters; }} p;

#define MA coopmat<{at}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MB coopmat<{at}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MC coopmat<{ct}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

void main() {{
    MA ma; MB mb;
    coopMatLoad(ma, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(mb, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    MC c0 = MC({zero}), c1 = MC({zero}), c2 = MC({zero}), c3 = MC({zero});
    for (uint i = 0u; i < p.iters; ++i) {{
        c0 = coopMatMulAdd(ma, mb, c0);
        c1 = coopMatMulAdd(ma, mb, c1);
        c2 = coopMatMulAdd(ma, mb, c2);
        c3 = coopMatMulAdd(ma, mb, c3);
    }}
    MC s = c0 + c1 + c2 + c3;
    coopMatStore(s, c, 256u * gl_WorkGroupID.x, 16, gl_CooperativeMatrixLayoutRowMajor);
}}
"""

VARIANTS = [
    ("f16 x f16 -> f32", "float16_t", "float", "0.0", 2, 4),
    ("f16 x f16 -> f16", "float16_t", "float16_t", "0.0", 2, 2),
    ("i8  x i8  -> i32", "int8_t", "int32_t", "0", 1, 4),
    ("u8  x u8  -> i32", "uint8_t", "int32_t", "0", 1, 4),
]


def main():
    groups, iters = 1536, 5000
    dev = Device()
    warmup(dev)
    print(f"{dev.name}\n")
    print("cooperative matrix ISSUE RATE by operand type")
    print("(pure register loop, no memory traffic: measures the matrix units only)\n")
    base = None
    for label, at, ct, zero, asz, csz in VARIANTS:
        src = HEAD + BODY.format(at=at, ct=ct, zero=zero)
        try:
            spv = compile_glsl(src, name="peak_" + at + "_" + ct)
        except Exception as e:
            print(f"  {label}   COMPILE FAILED: {str(e).splitlines()[1][:70]}")
            continue
        k = dev.kernel(spv, 2, 4, subgroup_size=32, name="peak")
        a = dev.buffer(256 * asz + 512, "shared")
        c = dev.buffer(groups * 256 * csz, "device")
        a.array(np.uint8)[:] = 1
        a.flush()
        push = struct.pack("I", iters)
        try:
            dev.run(k, [a, c], groups, push)
            t = min(dev.run(k, [a, c], groups, push) for _ in range(7))
            ops = groups * iters * 4 * 2 * 16 ** 3
            rate = ops / t / 1e12
            if base is None:
                base = rate
            print(f"  {label}   {rate:6.2f} TOPS   {rate / base:.2f}x vs f16")
        finally:
            k.destroy()
            a.destroy()
            c.destroy()

    print("\nInterpretation: if int8 shows no issue-rate advantage, its only")
    print("benefit is halved operand bytes, which on a bandwidth-bound machine")
    print("is worth about half of whatever share operand traffic holds.")
    dev.destroy()


if __name__ == "__main__":
    main()
