"""A matmul for GPUs with no matrix units at all.

Everything else in this project runs on `VK_KHR_cooperative_matrix`, which needs
AMD RDNA3+, NVIDIA Turing+, or Intel Arc. That excludes Polaris, Vega, Pascal,
Maxwell, every Intel integrated GPU before Arc, Adreno, Mali and the Raspberry
Pi: most of the GPUs that actually exist. A training stack that only runs on
recent high-end silicon does not make commodity hardware more useful.

This is the classic LDS-tiled, register-blocked scalar matmul, generated the
same way as the cooperative-matrix one. It needs nothing beyond Vulkan 1.1 plus
16-bit storage *in principle* -- the GLSL compiles cleanly at target_env
vulkan1.1 -- but compile.py emits a 1.3 target today, so this has never been run
against a 1.1 driver. It can drop to fp32 operands where even 16-bit storage is
missing, though nothing currently selects that path.
Slower than the matrix units, obviously. The point is that it runs at all.

Layout and semantics match `kernels.matmul_glsl` exactly, including both
backward transposes, so it is a drop-in substitute and the same tests cover it.
"""

import struct

from compile import compile_glsl
from vk import VkError

HEAD16 = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_16bit_storage : enable
"""

HEAD32 = """#version 450
"""


def matmul_scalar_glsl(bm=64, bn=64, bk=16, tm=4, tn=4, trans_a=False,
                       trans_b=False, f16=True, batched=False):
    """C[MxN] = A @ B, fp32 accumulate, no matrix instructions.

    Each workgroup computes a BM x BN tile; each invocation keeps a TM x TN
    block of accumulators in registers. Operands are staged through LDS so the
    inner product loop reads no global memory.
    """
    if bm % tm or bn % tn:
        raise ValueError("tile must divide by the per-thread block")
    threads = (bm // tm) * (bn // tn)
    if threads > 1024:
        raise ValueError(f"{threads} threads exceeds the 1024 workgroup limit")
    if (bm * bk) % threads or (bk * bn) % threads:
        raise ValueError("staging loop needs threads to divide the tile")

    ty = "float16_t" if f16 else "float"
    src = [HEAD16 if f16 else HEAD32]
    src.append(f"layout(local_size_x = {threads}) in;")
    src.append(f"layout(binding = 0) readonly  buffer BufA {{ {ty} A[]; }};")
    src.append(f"layout(binding = 1) readonly  buffer BufB {{ {ty} B[]; }};")
    src.append("layout(binding = 2) writeonly buffer BufC { float Cm[]; };")
    if batched:
        src.append("layout(push_constant) uniform P { uint M; uint N; uint K; "
                   "uint sa; uint sb; uint sc; } p;")
    else:
        src.append("layout(push_constant) uniform P { uint M; uint N; uint K; } p;")
    src.append(f"shared {ty} As[{bm * bk}];")
    src.append(f"shared {ty} Bs[{bk * bn}];")
    src.append("")
    src.append("void main() {")
    src.append("    uint tid = gl_LocalInvocationIndex;")
    src.append(f"    uint row0 = gl_WorkGroupID.x * {bm}u;")
    src.append(f"    uint col0 = gl_WorkGroupID.y * {bn}u;")
    src.append(f"    uint trow = (tid / {bn // tn}u) * {tm}u;")
    src.append(f"    uint tcol = (tid % {bn // tn}u) * {tn}u;")
    if batched:
        src.append("    uint oa = gl_WorkGroupID.z * p.sa;")
        src.append("    uint ob = gl_WorkGroupID.z * p.sb;")
        src.append("    uint oc = gl_WorkGroupID.z * p.sc;")
    oa, ob, oc = ("oa + ", "ob + ", "oc + ") if batched else ("", "", "")
    for i in range(tm):
        for j in range(tn):
            src.append(f"    float c{i}_{j} = 0.0;")
    src.append("")
    src.append(f"    for (uint k0 = 0u; k0 < p.K; k0 += {bk}u) {{")

    # Stage A into LDS as [BM][BK]. The transposed operand differs only in how
    # the global index is formed.
    per_a = (bm * bk) // threads
    src.append(f"        for (uint q = 0u; q < {per_a}u; ++q) {{")
    src.append(f"            uint e = tid + q * {threads}u;")
    src.append(f"            uint r = e / {bk}u, c = e % {bk}u;")
    if trans_a:
        src.append(f"            As[e] = A[{oa}(k0 + c) * p.M + row0 + r];")
    else:
        src.append(f"            As[e] = A[{oa}(row0 + r) * p.K + k0 + c];")
    src.append("        }")

    per_b = (bk * bn) // threads
    src.append(f"        for (uint q = 0u; q < {per_b}u; ++q) {{")
    src.append(f"            uint e = tid + q * {threads}u;")
    src.append(f"            uint r = e / {bn}u, c = e % {bn}u;")
    if trans_b:
        src.append(f"            Bs[e] = B[{ob}(col0 + c) * p.K + k0 + r];")
    else:
        src.append(f"            Bs[e] = B[{ob}(k0 + r) * p.N + col0 + c];")
    src.append("        }")
    src.append("        memoryBarrierShared();")
    src.append("        barrier();")

    # Inner product, fully unrolled over the register block.
    src.append(f"        for (uint kk = 0u; kk < {bk}u; ++kk) {{")
    for i in range(tm):
        src.append(f"            float a{i} = float(As[(trow + {i}u) * {bk}u + kk]);")
    for j in range(tn):
        src.append(f"            float b{j} = float(Bs[kk * {bn}u + tcol + {j}u]);")
    for i in range(tm):
        for j in range(tn):
            src.append(f"            c{i}_{j} = fma(a{i}, b{j}, c{i}_{j});")
    src.append("        }")
    src.append("        memoryBarrierShared();")
    src.append("        barrier();")
    src.append("    }")
    src.append("")
    for i in range(tm):
        for j in range(tn):
            src.append(f"    Cm[{oc}(row0 + trow + {i}u) * p.N + col0 + tcol + {j}u] "
                       f"= c{i}_{j};")
    src.append("}")
    return "\n".join(src)


class ScalarMatmul:
    """Same interface as kernels.Matmul, so callers cannot tell them apart."""

    def __init__(self, dev, bm=64, bn=64, bk=16, tm=4, tn=4, trans_a=False,
                 trans_b=False, f16=True, capture_stats=False, batched=False):
        self.dev = dev
        self.bm, self.bn, self.bk = bm, bn, bk
        self.tm, self.tn = tm, tn
        self.trans_a, self.trans_b = trans_a, trans_b
        self.f16 = f16
        self.lds = True
        self.batched = batched
        self.acc16 = False
        self.group_m = 0
        self.src = matmul_scalar_glsl(bm, bn, bk, tm, tn, trans_a, trans_b,
                                      f16, batched)
        self.name = (f"mms_{bm}x{bn}x{bk}_t{tm}{tn}"
                     f"{'_ta' if trans_a else ''}{'_tb' if trans_b else ''}"
                     f"{'' if f16 else '_f32'}{'_bat' if batched else ''}")
        spv = compile_glsl(self.src, name=self.name)
        self.push_size = 24 if batched else 12
        self.kernel = dev.kernel(spv, n_buffers=3, push_size=self.push_size,
                                 name=self.name,
                                 capture_stats=capture_stats)

    def fits(self, m, n, k):
        return m % self.bm == 0 and n % self.bn == 0 and k % self.bk == 0

    def traffic_bytes(self, m, n, k, nbatch=1):
        w = 2 if self.f16 else 4
        return (m * k * w * (n // self.bn) + k * n * w * (m // self.bm)
                + m * n * 4)

    def flops(self, m, n, k):
        return 2 * m * n * k

    def __call__(self, a, b, c, m, n, k, repeat=1, graph=None, nbatch=1,
                 strides=(0, 0, 0)):
        if not self.fits(m, n, k):
            raise VkError(f"{self.name}: {m}x{n}x{k} not a multiple of "
                          f"({self.bm}, {self.bn}, {self.bk})")
        if self.batched:
            push = struct.pack("IIIIII", m, n, k, *strides)
            groups = (m // self.bm, n // self.bn, nbatch)
        else:
            push = struct.pack("III", m, n, k)
            groups = (m // self.bm, n // self.bn, 1)
        if graph is not None:
            graph.record(self.kernel, [a, b, c], groups, push,
                         bytes_hint=self.traffic_bytes(m, n, k))
            return 0.0
        return self.dev.run(self.kernel, [a, b, c], groups, push, repeat=repeat)

    def destroy(self):
        self.kernel.destroy()


# Tile shapes worth trying, ordered so the first that fits is a decent default.
SCALAR_CONFIGS = [
    (64, 64, 16, 4, 4),
    (64, 64, 16, 4, 2),
    (128, 64, 16, 8, 4),
    (64, 32, 16, 4, 2),
    (32, 32, 16, 2, 2),
    (32, 32, 16, 4, 4),
    (16, 16, 16, 2, 2),
]


def pick_scalar(dev, m, n, k, trans_a=False, trans_b=False, f16=True,
                batched=False, min_groups=12):
    """First tile that divides the shape and fills the GPU reasonably."""
    best = None
    for bm, bn, bk, tm, tn in SCALAR_CONFIGS:
        if m % bm or n % bn or k % bk:
            continue
        groups = (m // bm) * (n // bn)
        score = (groups >= min_groups, bm * bn, groups)
        if best is None or score > best[0]:
            best = (score, (bm, bn, bk, tm, tn))
    if best is None:
        raise VkError(f"no scalar tile fits {m}x{n}x{k}")
    bm, bn, bk, tm, tn = best[1]
    return ScalarMatmul(dev, bm, bn, bk, tm, tn, trans_a, trans_b, f16,
                        batched=batched)
