"""Kernel generators. Python emits GLSL text; glslc turns it into SPIR-V.

The matmul is emitted fully unrolled rather than relying on a shader compiler's
unroller: we are a code generator already, so the tile shape is a Python loop
and what reaches glslc is straight-line code.
"""

import json
import os
import struct
import time

import numpy as np

from compile import compile_glsl
from vk import VkError

TILE = 16  # the only cooperative matrix shape AMD RDNA3 exposes: 16x16x16

_HEADER = """#version 450
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_16bit_storage : enable
"""

_ROW = "gl_CooperativeMatrixLayoutRowMajor"
_COL = "gl_CooperativeMatrixLayoutColumnMajor"


def matmul_glsl(sg=4, wm=2, wn=2, trans_a=False, trans_b=False, batched=False,
                acc16=False, group_m=0):
    """C[MxN] = A @ B with f16 inputs and f32 accumulation.

    sg  subgroups per workgroup (each is one wave32)
    wm  16x16 tiles per subgroup along M
    wn  16x16 tiles per subgroup along N

    Workgroup covers BM x BN where BM = sg*wm*16, BN = wn*16.

    trans_a: A is stored K x M, so a row of A is a column read (used by dW = X^T @ dY)
    trans_b: B is stored N x K, so B is read column-major   (used by dX = dY @ W^T)

    Both transposes are layout flips on the cooperative matrix load, not a
    materialised transpose in memory. That matters: on a machine with a ridge
    point of ~200 FLOP/byte, writing a transposed copy would cost more than the
    matmul that consumes it.
    """
    bm = sg * wm * TILE
    bn = wn * TILE
    src = [_HEADER]
    src.append(f"layout(local_size_x = {sg * 32}) in;")
    src.append("layout(binding = 0) readonly  buffer BufA { float16_t A[]; };")
    src.append("layout(binding = 1) readonly  buffer BufB { float16_t B[]; };")
    src.append("layout(binding = 2) writeonly buffer BufC { %s Cm[]; };"
               % ("float16_t" if acc16 else "float"))
    if batched:
        # gl_WorkGroupID.z indexes the batch; strides let one dispatch cover
        # every (batch, head) pair of an attention block.
        src.append("layout(push_constant) uniform P { uint M; uint N; uint K; "
                   "uint sa; uint sb; uint sc; } p;")
    else:
        src.append("layout(push_constant) uniform P { uint M; uint N; uint K; } p;")
    src.append("")
    src.append("#define MA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>")
    src.append("#define MB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>")
    src.append("#define MC coopmat<%s, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>"
               % ("float16_t" if acc16 else "float"))
    src.append("")
    src.append("void main() {")
    if group_m:
        # L2-aware workgroup ordering.
        #
        # With the natural 2D grid, consecutive workgroups sweep one axis of C
        # and every one of them pulls a fresh operand block from DRAM. Grouping
        # the launch so that a band of `group_m` row-blocks is finished across
        # all column-blocks before moving on keeps that band's A rows, and all
        # of B, resident in L2 across the whole band. Same work, same tiles,
        # different visit order.
        #
        # This is the Triton/CUTLASS grouped rasterisation, expressed against a
        # flat 1D dispatch so the driver's launch order is the traversal order.
        src.append(f"    uint num_m = (p.M + {bm}u - 1u) / {bm}u;")
        src.append(f"    uint num_n = (p.N + {bn}u - 1u) / {bn}u;")
        src.append("    uint pid = gl_WorkGroupID.x;")
        src.append(f"    uint in_group = {group_m}u * num_n;")
        src.append("    uint gid = pid / in_group;")
        src.append(f"    uint first_m = gid * {group_m}u;")
        src.append(f"    uint gsize = min(num_m - first_m, {group_m}u);")
        src.append("    uint pid_m = first_m + ((pid % in_group) % gsize);")
        src.append("    uint pid_n = (pid % in_group) / gsize;")
        src.append(f"    uint row0 = pid_m * {bm}u + gl_SubgroupID * {wm * TILE}u;")
        src.append(f"    uint col0 = pid_n * {bn}u;")
    else:
        src.append(f"    uint row0 = gl_WorkGroupID.x * {bm}u + gl_SubgroupID * {wm * TILE}u;")
        src.append(f"    uint col0 = gl_WorkGroupID.y * {bn}u;")
    if batched:
        src.append("    uint oa = gl_WorkGroupID.z * p.sa;")
        src.append("    uint ob = gl_WorkGroupID.z * p.sb;")
        src.append("    uint oc = gl_WorkGroupID.z * p.sc;")
    oa, ob, oc = ("oa + ", "ob + ", "oc + ") if batched else ("", "", "")

    for i in range(wm):
        for j in range(wn):
            src.append(f"    MC c{i}_{j} = MC(0.0);")

    src.append("    for (uint k = 0u; k < p.K; k += 16u) {")

    # A fragment: normally M-major with row stride K; transposed it lives as
    # K x M, so the same tile is a column-major read with stride M.
    for i in range(wm):
        if trans_a:
            off = f"{oa}k * p.M + row0 + {i * TILE}u"
            src.append(f"        MA a{i}; coopMatLoad(a{i}, A, {off}, p.M, {_COL});")
        else:
            off = f"{oa}(row0 + {i * TILE}u) * p.K + k"
            src.append(f"        MA a{i}; coopMatLoad(a{i}, A, {off}, p.K, {_ROW});")

    for j in range(wn):
        if trans_b:
            off = f"{ob}(col0 + {j * TILE}u) * p.K + k"
            src.append(f"        MB b{j}; coopMatLoad(b{j}, B, {off}, p.K, {_COL});")
        else:
            off = f"{ob}k * p.N + col0 + {j * TILE}u"
            src.append(f"        MB b{j}; coopMatLoad(b{j}, B, {off}, p.N, {_ROW});")

    for i in range(wm):
        for j in range(wn):
            src.append(f"        c{i}_{j} = coopMatMulAdd(a{i}, b{j}, c{i}_{j});")
    src.append("    }")

    for i in range(wm):
        for j in range(wn):
            off = f"{oc}(row0 + {i * TILE}u) * p.N + col0 + {j * TILE}u"
            src.append(f"    coopMatStore(c{i}_{j}, Cm, {off}, p.N, {_ROW});")
    src.append("}")
    return "\n".join(src)


def matmul_lds_glsl(sg=4, wm=2, wn=2, bk=32):
    """Same matmul, but staging both operands through LDS first.

    Two things change versus the direct-from-global version:

    1. B is read from DRAM once per workgroup instead of once per subgroup.
       The direct kernel has every subgroup load the same B columns, so B
       traffic is multiplied by `sg`.
    2. Global reads become contiguous f16vec4 (8 byte) loads instead of the
       strided 32-byte row reads a cooperative-matrix load issues.

    Whether that actually pays is an empirical question on an APU, which is
    what bench/matmul_sweep.py answers.
    """
    bm = sg * wm * TILE
    bn = wn * TILE
    threads = sg * 32
    lds_bytes = (bm * bk + bk * bn) * 2
    if lds_bytes > 65536:
        raise ValueError(f"LDS {lds_bytes} B over 64 KiB for sg={sg} wm={wm} wn={wn} bk={bk}")
    if bk % TILE:
        raise ValueError("bk must be a multiple of 16")

    src = [_HEADER]
    src.append(f"layout(local_size_x = {threads}) in;")
    # f16vec4 loads: 8 contiguous bytes per lane, which coalesces properly.
    src.append("layout(binding = 0) readonly  buffer BufA { f16vec4 A[]; };")
    src.append("layout(binding = 1) readonly  buffer BufB { f16vec4 B[]; };")
    src.append("layout(binding = 2) writeonly buffer BufC { float Cm[]; };")
    src.append("layout(push_constant) uniform P { uint M; uint N; uint K; } p;")
    src.append("")
    src.append(f"shared float16_t As[{bm * bk}];")
    src.append(f"shared float16_t Bs[{bk * bn}];")
    src.append("")
    src.append("#define MA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>")
    src.append("#define MB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>")
    src.append("#define MC coopmat<float,     gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>")
    src.append("")
    src.append("void main() {")
    src.append("    uint tid = gl_LocalInvocationIndex;")
    src.append(f"    uint row0 = gl_WorkGroupID.x * {bm}u;")
    src.append(f"    uint col0 = gl_WorkGroupID.y * {bn}u;")
    src.append(f"    uint sgrow = gl_SubgroupID * {wm * TILE}u;")
    for i in range(wm):
        for j in range(wn):
            src.append(f"    MC c{i}_{j} = MC(0.0);")
    src.append("")
    src.append(f"    for (uint k0 = 0u; k0 < p.K; k0 += {bk}u) {{")

    # Stage A: BM x BK, row-major in LDS.
    src.append(f"        for (uint i = tid; i < {bm * bk // 4}u; i += {threads}u) {{")
    src.append(f"            uint fo = i * 4u;")
    src.append(f"            uint r = fo / {bk}u, c = fo % {bk}u;")
    src.append("            f16vec4 v = A[((row0 + r) * p.K + k0 + c) >> 2];")
    src.append("            As[fo] = v.x; As[fo+1u] = v.y;")
    src.append("            As[fo+2u] = v.z; As[fo+3u] = v.w;")
    src.append("        }")
    # Stage B: BK x BN, row-major in LDS.
    src.append(f"        for (uint i = tid; i < {bk * bn // 4}u; i += {threads}u) {{")
    src.append(f"            uint fo = i * 4u;")
    src.append(f"            uint r = fo / {bn}u, c = fo % {bn}u;")
    src.append("            f16vec4 v = B[((k0 + r) * p.N + col0 + c) >> 2];")
    src.append("            Bs[fo] = v.x; Bs[fo+1u] = v.y;")
    src.append("            Bs[fo+2u] = v.z; Bs[fo+3u] = v.w;")
    src.append("        }")
    src.append("        memoryBarrierShared();")
    src.append("        barrier();")

    for kk in range(0, bk, TILE):
        for i in range(wm):
            off = f"(sgrow + {i * TILE}u) * {bk}u + {kk}u"
            src.append(f"        MA a{kk}_{i}; coopMatLoad(a{kk}_{i}, As, {off}, {bk}u, {_ROW});")
        for j in range(wn):
            off = f"{kk * bn + j * TILE}u"
            src.append(f"        MB b{kk}_{j}; coopMatLoad(b{kk}_{j}, Bs, {off}, {bn}u, {_ROW});")
        for i in range(wm):
            for j in range(wn):
                src.append(f"        c{i}_{j} = coopMatMulAdd(a{kk}_{i}, b{kk}_{j}, c{i}_{j});")

    src.append("        memoryBarrierShared();")
    src.append("        barrier();")
    src.append("    }")

    for i in range(wm):
        for j in range(wn):
            off = f"(row0 + sgrow + {i * TILE}u) * p.N + col0 + {j * TILE}u"
            src.append(f"    coopMatStore(c{i}_{j}, Cm, {off}, p.N, {_ROW});")
    src.append("}")
    return "\n".join(src)


class Matmul:
    """One compiled matmul configuration."""

    def __init__(self, dev, sg=4, wm=2, wn=2, trans_a=False, trans_b=False,
                 capture_stats=False, lds=False, bk=32, batched=False,
                 acc16=False, group_m=0):
        self.dev = dev
        self.sg, self.wm, self.wn = sg, wm, wn
        self.trans_a, self.trans_b = trans_a, trans_b
        self.lds, self.bk, self.batched = lds, bk, batched
        self.acc16 = acc16
        self.group_m = group_m
        self.bm = sg * wm * TILE
        self.bn = wn * TILE
        if lds:
            if trans_a or trans_b or batched:
                raise ValueError("LDS variant has no transposed or batched path")
            self.src = matmul_lds_glsl(sg, wm, wn, bk)
            name = f"mmlds_sg{sg}_wm{wm}_wn{wn}_bk{bk}"
        else:
            self.src = matmul_glsl(sg, wm, wn, trans_a, trans_b, batched, acc16,
                                   group_m)
            name = (f"mm_sg{sg}_wm{wm}_wn{wn}"
                    f"{'_ta' if trans_a else ''}{'_tb' if trans_b else ''}"
                    f"{'_bat' if batched else ''}{'_a16' if acc16 else ''}"
                    f"{'_g' + str(group_m) if group_m else ''}")
        self.name = name
        self.push_size = 24 if batched else 12
        spv = compile_glsl(self.src, name=name)
        self.kernel = dev.kernel(spv, n_buffers=3, push_size=self.push_size,
                                 subgroup_size=32, name=name,
                                 capture_stats=capture_stats)

    def fits(self, m, n, k):
        kstep = self.bk if self.lds else TILE
        return m % self.bm == 0 and n % self.bn == 0 and k % kstep == 0

    def __call__(self, a, b, c, m, n, k, repeat=1, graph=None,
                 nbatch=1, strides=(0, 0, 0)):
        # ponytail: tile-aligned shapes only. Ragged edges are handled by
        # padding the tensor allocation, not by a masked slow path in the
        # kernel; add a masked epilogue only if padding ever costs real memory.
        if not self.fits(m, n, k):
            raise VkError(f"{self.name}: shape {m}x{n}x{k} not a multiple of "
                          f"({self.bm}, {self.bn}, {TILE})")
        # Swizzled kernels take a flat 1D grid so that launch order and
        # traversal order are the same thing.
        nx = (m // self.bm) * (n // self.bn) if self.group_m else m // self.bm
        ny = 1 if self.group_m else n // self.bn
        if self.batched:
            push = struct.pack("IIIIII", m, n, k, *strides)
            groups = (nx, ny, nbatch)
        else:
            push = struct.pack("III", m, n, k)
            groups = (nx, ny, 1)
        if graph is not None:
            graph.record(self.kernel, [a, b, c], groups, push,
                         bytes_hint=self.traffic_bytes(m, n, k, nbatch))
            # Operand re-reads and output writes have completely different
            # fixes (bigger tiles vs narrower dtype), so keep them apart.
            nb_ = nbatch if self.batched else 1
            getattr(graph, "mm_split", []).append(
                (nb_ * (m * k * 2 * (n // self.bn) + k * n * 2 * (m // self.bm)),
                 nb_ * m * n * (2 if self.acc16 else 4)))
            return 0.0
        return self.dev.run(self.kernel, [a, b, c], groups, push, repeat=repeat)

    def traffic_bytes(self, m, n, k, nbatch=1):
        """Upper bound on DRAM traffic: every tile re-reads its A row-block and
        B column-block from scratch. Real traffic is lower because L2 catches
        some of the re-reads, so this brackets the truth from above while
        `compulsory` (each tensor once) brackets it from below.
        """
        nb = nbatch if self.batched else 1
        reads_a = m * k * 2 * (n // self.bn)
        reads_b = k * n * 2 * (m // self.bm)
        writes_c = m * n * (2 if self.acc16 else 4)
        return nb * (reads_a + reads_b + writes_c)

    def flops(self, m, n, k):
        return 2 * m * n * k

    def destroy(self):
        self.kernel.destroy()


# Configurations worth trying. wm*wn accumulators at 8 VGPRs each, plus
# fragments, is what sets register pressure. Wide tiles (large wn) matter most:
# global traffic is M*K*(N/BN) + K*N*(M/BM), so BN and BM are the only knobs
# that reduce it.
# 3 and 6 are here because real models have dimensions that are not powers of
# two: a 192-wide model with 4 heads has head dim 48, and with only power-of-two
# tile counts the widest tile that divides 48 is 16. Allowing wn=3 makes a
# 48-wide tile reachable, which covers the whole dimension in one tile.
CONFIGS = [(sg, wm, wn, lds)
           for sg in (1, 2, 4, 8)
           for wm in (1, 2, 3, 4)
           for wn in (1, 2, 3, 4, 6, 8)
           for lds in (False, True)
           if sg * 32 <= 1024 and wm * wn <= 16]

_TUNE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".autotune.json")


def _load_cache():
    if os.path.exists(_TUNE_CACHE):
        with open(_TUNE_CACHE) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    tmp = _TUNE_CACHE + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, _TUNE_CACHE)


_WARMED = set()


def warmup(dev, seconds=5.0, force=False):
    """Run a sustained load until the GPU clocks settle.

    Without this, benchmarking is invalid. Throughput on identical work climbs
    monotonically from ~1.1 to ~3.1 TFLOPS over the first ~30 dispatches, a
    2.7x spread, because the GPU starts at idle clocks. Anything measured early
    in a sweep looks far worse than the same thing measured late, which biases
    an autotuner toward whatever it happens to evaluate last.

    After a sustained warmup the spread over 30 repeats drops to 1.28x with no
    trend. Once per process is enough.
    """
    if not force and id(dev) in _WARMED:
        return 0.0
    m = n = k = 512
    a = dev.buffer(m * k * 2, "device")
    b = dev.buffer(k * n * 2, "device")
    c = dev.buffer(m * n * 4, "device")
    mm = Matmul(dev, 4, 2, 4)
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            mm(a, b, c, m, n, k, repeat=10)
    finally:
        mm.destroy()
        for x in (a, b, c):
            x.destroy()
    _WARMED.add(id(dev))
    return time.perf_counter() - t0


def occupancy(dev, stats, workgroup_threads, min_waves=4):
    """Waves resident per SIMD for a compiled pipeline, from driver-reported
    register and LDS usage. Returns (waves, reason) where reason is None if the
    config is worth benchmarking.

    This is the pruning signal: a config that cannot keep enough waves in
    flight will never hide DRAM latency on a machine whose ridge point is
    ~200 FLOP/byte, so there is no point running it.
    """
    core = dev.shader_core_props()
    vgpr = stats.get("numUsedVgprs", 0)
    lds = stats.get("ldsUsageSizeInBytes", 0)
    lds_max = stats.get("ldsSizePerLocalWorkGroup", 65536)
    if not core or not vgpr:
        return None, None  # unknown: benchmark it rather than guess

    gran = max(core["vgpr_granularity"], 1)
    alloc = ((vgpr + gran - 1) // gran) * gran
    waves = min(core["waves_per_simd"], core["vgprs_per_simd"] // max(alloc, 1))

    if lds and lds_max:
        wgs_per_cu = max(lds_max // lds, 1)
        waves_from_lds = wgs_per_cu * (workgroup_threads // 32) // core["simd_per_cu"]
        waves = min(waves, max(waves_from_lds, 1))

    if waves < min_waves:
        return waves, f"only {waves} waves/SIMD (vgpr={vgpr})"
    return waves, None


def lds_only_best(best):
    """The LDS variant has no swizzled path, so skip stage two when it wins."""
    return best.get("lds", False)


def autotune_matmul(dev, m, n, k, trans_a=False, trans_b=False, reps=3,
                    verbose=False, use_cache=True, batched=False, nbatch=1):
    """Pick the fastest tile configuration for one shape.

    Two-stage: compile every candidate and read back its register usage from
    the driver, drop the ones that can't reach useful occupancy, then benchmark
    only the survivors. Compiling is far cheaper than running, so the pruning
    pass costs almost nothing and removes most of the search space.
    """
    key = (f"{dev.name}|{m}x{n}x{k}|ta{int(trans_a)}tb{int(trans_b)}"
           f"{'|bat' + str(nbatch) if batched else ''}")
    cache = _load_cache() if use_cache else {}
    if key in cache:
        e = cache[key]
        return Matmul(dev, *e["config"], trans_a=trans_a, trans_b=trans_b,
                      lds=e.get("lds", False), batched=batched,
                      group_m=e.get("group_m", 0)), e

    warmup(dev)
    strides = (m * k, k * n, m * n)
    nb = nbatch if batched else 1
    call = dict(nbatch=nb, strides=strides) if batched else {}
    a = dev.buffer(nb * m * k * 2, "device")
    b = dev.buffer(nb * k * n * 2, "device")
    c = dev.buffer(nb * m * n * 4, "device")

    candidates, pruned = [], []
    try:
        for sg, wm, wn, lds in CONFIGS:
            if lds and (trans_a or trans_b or batched):
                continue  # no transposed or batched LDS path
            try:
                mm = Matmul(dev, sg, wm, wn, trans_a, trans_b, capture_stats=True,
                            lds=lds, batched=batched)
            except ValueError:
                continue  # LDS budget exceeded
            if not mm.fits(m, n, k):
                mm.destroy()
                continue
            # Enough workgroups to fill 12 CUs; otherwise most of the GPU idles.
            # A batched dispatch multiplies the grid by the batch dimension.
            if (m // mm.bm) * (n // mm.bn) * nb < 12:
                mm.destroy()
                continue
            stats = mm.kernel.statistics()
            vgpr = stats.get("numUsedVgprs", 0)
            occ, reason = occupancy(dev, stats, sg * 32)
            if reason:
                pruned.append((sg, wm, wn, lds, reason))
                if verbose:
                    print(f"    sg={sg} wm={wm} wn={wn:2d} lds={int(lds)}  pruned: {reason}")
                mm.destroy()
                continue
            candidates.append((mm, vgpr, occ))

        if not candidates:
            raise VkError(f"no matmul config fits {m}x{n}x{k}")

        # Round-robin the measurements rather than finishing one config before
        # starting the next. warmup() removes the big clock ramp, but residual
        # drift over a sweep is still correlated with evaluation order, and
        # measuring each config to completion in turn bakes that drift into the
        # comparison. Interleaving spreads it evenly across candidates.
        warm = {}
        for mm, _, _ in candidates:
            warm[id(mm)] = mm(a, b, c, m, n, k, **call)
        best_t = {id(mm): float("inf") for mm, _, _ in candidates}
        for _ in range(3):
            for mm, _, _ in candidates:
                # Keep any single submit well under the ~2s TDR window: a slow
                # config at 4096^3 can take half a second per dispatch.
                r = max(1, min(reps, int(0.2 / max(warm[id(mm)], 1e-6))))
                t = mm(a, b, c, m, n, k, repeat=r, **call) / r
                best_t[id(mm)] = min(best_t[id(mm)], t)

        results = []
        for mm, vgpr, occ in candidates:
            t = best_t[id(mm)]
            tf = mm.flops(m, n, k) * nb / t / 1e12
            results.append({"config": [mm.sg, mm.wm, mm.wn], "lds": mm.lds, "tflops": tf,
                            "ms": t * 1e3, "vgpr": vgpr, "occupancy": occ,
                            "tile": [mm.bm, mm.bn]})
            if verbose:
                print(f"    sg={mm.sg} wm={mm.wm} wn={mm.wn:2d} lds={int(mm.lds)} "
                      f"tile {mm.bm:3d}x{mm.bn:3d}  "
                      f"{tf:7.3f} TFLOPS  vgpr={vgpr or '?'} waves={occ or '?'}")

        results.sort(key=lambda r: -r["tflops"])
        best = results[0]

        # Second stage: for the winning tile only, sweep the workgroup
        # ordering. Folding group_m into the main search would quadruple the
        # candidate count for a knob that is nearly orthogonal to tile shape.
        if not lds_only_best(best):
            sgb, wmb, wnb = best["config"]
            gbest = (best["tflops"], 0)
            for gm in (2, 4, 8):
                try:
                    gmm = Matmul(dev, sgb, wmb, wnb, trans_a, trans_b, lds=False,
                                 batched=batched, group_m=gm)
                except ValueError:
                    continue
                if not gmm.fits(m, n, k):
                    gmm.destroy()
                    continue
                gmm(a, b, c, m, n, k, **call)
                tg = min(gmm(a, b, c, m, n, k, repeat=3, **call) / 3 for _ in range(3))
                tf = gmm.flops(m, n, k) * nb / tg / 1e12
                if verbose:
                    print(f"    group_m={gm:2d} on winning tile   {tf:7.3f} TFLOPS")
                if tf > gbest[0]:
                    gbest = (tf, gm)
                gmm.destroy()
            if gbest[1]:
                best = dict(best, tflops=gbest[0], group_m=gbest[1], lds=False)
                if verbose:
                    print(f"    -> group_m={gbest[1]} wins, {gbest[0]:.3f} TFLOPS")
        best["pruned"] = len(pruned)
        best["evaluated"] = len(results)
        if verbose and pruned:
            print(f"    pruned {len(pruned)} configs on register pressure "
                  f"without running them")

        for mm, _, _ in candidates:
            mm.destroy()

        if use_cache:
            cache[key] = best
            _save_cache(cache)
        return Matmul(dev, *best["config"], trans_a=trans_a, trans_b=trans_b,
                      lds=best["lds"], batched=batched,
                      group_m=best.get("group_m", 0)), best
    finally:
        a.destroy()
        b.destroy()
        c.destroy()


def matmul_reference(a, b, trans_a=False, trans_b=False):
    """numpy reference matching the kernel's accumulation: f16 in, f32 out."""
    a32 = (a.T if trans_a else a).astype(np.float32)
    b32 = (b.T if trans_b else b).astype(np.float32)
    return a32 @ b32


# --------------------------------------------------------------------------
# Elementwise fusion
#
# There is no separate "fusion pass". A fused kernel is one GLSL body touching
# several buffers, so fusing means writing one body instead of two kernels.
# On a machine whose ridge point is ~205 FLOP/byte that is the entire
# optimisation story: every kernel boundary is a round trip to DRAM.

_DTYPES = {"f32": ("float", 4), "f16": ("float16_t", 2),
           "u32": ("uint", 4), "i32": ("int", 4)}

_PRELUDE = """
const float GELU_C = 0.7978845608028654;   // sqrt(2/pi)
float gelu(float x) {
    float x3 = x * x * x;
    return 0.5 * x * (1.0 + tanh(GELU_C * (x + 0.044715 * x3)));
}
float dgelu(float x) {
    float x3 = x * x * x;
    float t = tanh(GELU_C * (x + 0.044715 * x3));
    float sech2 = 1.0 - t * t;
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * GELU_C * (1.0 + 3.0 * 0.044715 * x * x);
}
"""


def elementwise_glsl(bufs, body, push=(), local=256, prelude=True, extensions=()):
    """Generate a flat elementwise kernel.

    bufs: [(name, dtype, access)] where access is 'readonly'/'writeonly'/''
    push: extra push-constant fields after `n`, as [(name, glsl_type)]
    body: GLSL statements; `i` is the element index, buffers are named arrays.
    """
    src = ["#version 450",
           "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable",
           "#extension GL_EXT_shader_16bit_storage : enable"]
    src += [f"#extension {e} : require" for e in extensions]
    src.append(f"layout(local_size_x = {local}) in;")
    for b, (name, dtype, access) in enumerate(bufs):
        gl, _ = _DTYPES[dtype]
        acc = f"{access} " if access else ""
        src.append(f"layout(binding = {b}) {acc}buffer B{b} {{ {gl} {name}[]; }};")
    fields = "".join(f" {t} {n};" for n, t in push)
    src.append(f"layout(push_constant) uniform P {{ uint n;{fields} }} p;")
    if prelude:
        src.append(_PRELUDE)
    src.append("void main() {")
    src.append("    uint i = gl_GlobalInvocationID.x;")
    src.append("    if (i >= p.n) return;")
    for line in body.strip().splitlines():
        src.append("    " + line.strip())
    src.append("}")
    return "\n".join(src)


class Elementwise:
    """A compiled elementwise kernel plus its dispatch arithmetic."""

    def __init__(self, dev, name, bufs, body, push=(), local=256, extensions=(),
                 subgroup_size=None, width=None):
        # SUBWu in a body stands for the negotiated subgroup width. It cannot be
        # a format placeholder: GLSL bodies contain braces and '%'.
        if width is not None:
            body = body.replace("SUBWu", f"{width}u")
        self.dev = dev
        self.name = name
        self.local = local
        self.push_fmt = "I" + "".join("f" if t == "float" else "I" for _, t in push)
        src = elementwise_glsl(bufs, body, push, local, extensions=extensions)
        self.src = src
        self.kernel = dev.kernel(compile_glsl(src, name=name), n_buffers=len(bufs),
                                 push_size=struct.calcsize("=" + self.push_fmt),
                                 subgroup_size=subgroup_size, name=name)

    def __call__(self, buffers, n, *args, repeat=1, graph=None):
        push = struct.pack("=" + self.push_fmt, n, *args)
        groups = (n + self.local - 1) // self.local
        if graph is not None:
            graph.record(self.kernel, buffers, groups, push)
            return 0.0
        return self.dev.run(self.kernel, buffers, groups, push, repeat=repeat)

    def destroy(self):
        self.kernel.destroy()


def make_kernels(dev):
    """Every elementwise kernel the training loop needs, compiled once.

    Each one is deliberately fused: the bias add, the activation, and the f32
    to f16 narrowing all happen in a single pass over the data rather than
    three passes with two DRAM round trips between them.
    """
    K = {}

    # Forward: z = acc + bias; y = f16(gelu(z)); keep z (f32) for backward.
    K["bias_gelu"] = Elementwise(
        dev, "bias_gelu",
        [("acc", "f32", "readonly"), ("bias", "f32", "readonly"),
         ("z", "f32", "writeonly"), ("y", "f16", "writeonly")],
        """
        float v = acc[i] + bias[i % p.ncol];
        z[i] = v;
        y[i] = float16_t(gelu(v));
        """,
        push=[("ncol", "uint")])

    # Backward through bias+GELU: dz = dy * gelu'(z), narrowed for the next matmul.
    K["bias_gelu_bwd"] = Elementwise(
        dev, "bias_gelu_bwd",
        [("dy", "f32", "readonly"), ("z", "f32", "readonly"),
         ("dz", "f16", "writeonly"), ("dz32", "f32", "writeonly")],
        """
        float g = dy[i] * dgelu(z[i]);
        dz[i] = float16_t(g);
        dz32[i] = g;
        """)

    # f16-only variant: the bias column sum can read the f16 gradient the
    # matmuls already need, so the f32 duplicate is dead weight.
    K["bias_gelu_bwd16"] = Elementwise(
        dev, "bias_gelu_bwd16",
        [("dy", "f32", "readonly"), ("z", "f32", "readonly"),
         ("dz", "f16", "writeonly")],
        """
        dz[i] = float16_t(dy[i] * dgelu(z[i]));
        """)

    # z = acc + bias, no activation (output layer).
    K["bias"] = Elementwise(
        dev, "bias",
        [("acc", "f32", "readonly"), ("bias", "f32", "readonly"),
         ("z", "f32", "writeonly")],
        """
        z[i] = acc[i] + bias[i % p.ncol];
        """,
        push=[("ncol", "uint")])

    # Column sums for the bias gradient. One thread per column, loop the rows:
    # nrow is the batch size, so this is a tiny kernel next to the matmuls.
    K["col_sum"] = Elementwise(
        dev, "col_sum",
        [("g", "f32", "readonly"), ("sums", "f32", "writeonly")],
        """
        float s = 0.0;
        for (uint r = 0u; r < p.nrow; ++r) s += g[r * p.n + i];
        sums[i] = s;
        """,
        push=[("nrow", "uint")])

    # Softmax + cross entropy, forward and backward in one pass.
    # One thread per row. Rows are tiny (num classes), and doing it in one
    # kernel avoids materialising the probabilities at all.
    K["softmax_ce"] = Elementwise(
        dev, "softmax_ce",
        [("logits", "f32", "readonly"), ("labels", "u32", "readonly"),
         ("dl16", "f16", "writeonly"), ("dl32", "f32", "writeonly"),
         ("loss", "f32", "writeonly")],
        """
        uint base = i * p.stride;
        float mx = -1e30;
        for (uint c = 0u; c < p.ncls; ++c) mx = max(mx, logits[base + c]);
        float s = 0.0;
        for (uint c = 0u; c < p.ncls; ++c) s += exp(logits[base + c] - mx);
        float logZ = mx + log(s);
        uint y = labels[i];
        loss[i] = logZ - logits[base + y];
        float inv = 1.0 / float(p.n);
        for (uint c = 0u; c < p.stride; ++c) {
            float gv = 0.0;
            if (c < p.ncls) {
                float pr = exp(logits[base + c] - logZ);
                gv = (pr - (c == y ? 1.0 : 0.0)) * inv;
            }
            dl16[base + c] = float16_t(gv);
            dl32[base + c] = gv;
        }
        """,
        push=[("stride", "uint"), ("ncls", "uint")])

    # AdamW, fused: one pass reads w, g, m, v and writes w, m, v, and the f16
    # mirror the matmul consumes. Unfused this would be four passes over every
    # parameter, which at this ridge point can cost as much as the forward.
    K["adamw"] = Elementwise(
        dev, "adamw",
        [("w", "f32", ""), ("g", "f32", "readonly"), ("m", "f32", ""),
         ("v", "f32", ""), ("w16", "f16", "writeonly"), ("hp", "f32", "readonly")],
        """
        float lr = hp[0], b1 = hp[1], b2 = hp[2];
        float bc1 = hp[3], bc2 = hp[4], eps = hp[5];
        float gi = g[i];
        float mi = b1 * m[i] + (1.0 - b1) * gi;
        float vi = b2 * v[i] + (1.0 - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float mh = mi / bc1;
        float vh = vi / bc2;
        float wi = w[i];
        wi -= lr * (mh / (sqrt(vh) + eps) + p.wd * wi);
        w[i] = wi;
        w16[i] = float16_t(wi);
        """,
        push=[("wd", "float")])

    K["zero"] = Elementwise(
        dev, "zero", [("dst", "f32", "writeonly")], "dst[i] = 0.0;")

    # f32 -> f16 narrowing, for tensors that must feed a matmul.
    K["cast16"] = Elementwise(
        dev, "cast16",
        [("src", "f32", "readonly"), ("dst", "f16", "writeonly")],
        "dst[i] = float16_t(src[i]);")

    return K
