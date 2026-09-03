"""Transformer-specific kernels: LayerNorm, causal attention softmax, the
head permutes, embeddings, and residual adds.

Same principle as the MLP kernels: each one does as much work per pass over the
data as it can. LayerNorm emits the f16 the next matmul needs; the attention
softmax applies the scale, the causal mask, the max subtraction and the
normalisation in one sweep; residual adds fold the f16 narrowing in.

Row-wise kernels use one invocation per row. Rows here are d_model (256) or
sequence length (128), so a row fits comfortably in a single thread's loop and
this avoids a second reduction pass.
"""

from kernels import Elementwise


def make_transformer_kernels(dev):
    K = {}
    # Row kernels stride a row across one subgroup's lanes, so the stride must
    # equal the real subgroup width. dev.row_subgroup_size is 32 where the
    # device will grant it and the native width otherwise.
    W = dev.row_subgroup_size
    RS = dev.row_subgroup_size if dev.can_set_subgroup_size else None

    # ---- LayerNorm ------------------------------------------------------
    # One invocation per row. Saves mu and rstd so backward need not recompute.
    # One subgroup per row, for the same coalescing reason as the attention
    # softmax: a thread-per-row version has adjacent lanes reading D*4 bytes
    # apart. Dispatch n = rows * 32.
    SUBGROUP = ["GL_KHR_shader_subgroup_basic", "GL_KHR_shader_subgroup_arithmetic"]

    K["layernorm"] = Elementwise(
        dev, "layernorm",
        [("x", "f32", "readonly"), ("gamma", "f32", "readonly"),
         ("beta", "f32", "readonly"), ("y16", "f16", "writeonly"),
         ("mu", "f32", "writeonly"), ("rstd", "f32", "writeonly")],
        """
        uint row = gl_WorkGroupID.x;
        uint lane = gl_SubgroupInvocationID;
        uint base = row * p.D;
        float s = 0.0;
        for (uint d = lane; d < p.D; d += SUBWu) s += x[base + d];
        float m = subgroupAdd(s) / float(p.D);
        float v = 0.0;
        for (uint d = lane; d < p.D; d += SUBWu) { float t = x[base + d] - m; v += t * t; }
        float r = inversesqrt(subgroupAdd(v) / float(p.D) + 1e-5);
        if (lane == 0u) { mu[row] = m; rstd[row] = r; }
        for (uint d = lane; d < p.D; d += SUBWu) {
            float xh = (x[base + d] - m) * r;
            y16[base + d] = float16_t(xh * gamma[d] + beta[d]);
        }
        """,
        push=[("D", "uint")], local=W, subgroup_size=RS, extensions=SUBGROUP,
        width=W)

    # dx needs both row means; dgamma/dbeta are column sums done separately,
    # so this also emits dy*xhat for that reduction.
    K["layernorm_bwd"] = Elementwise(
        dev, "layernorm_bwd",
        [("dy", "f32", "readonly"), ("x", "f32", "readonly"),
         ("gamma", "f32", "readonly"), ("mu", "f32", "readonly"),
         ("rstd", "f32", "readonly"), ("dx", "f32", "writeonly"),
         ("dyxh", "f16", "writeonly"), ("dy16", "f16", "writeonly")],
        """
        uint row = gl_WorkGroupID.x;
        uint lane = gl_SubgroupInvocationID;
        uint base = row * p.D;
        float m = mu[row], r = rstd[row];
        float sum_dxh = 0.0, sum_dxh_xh = 0.0;
        for (uint d = lane; d < p.D; d += SUBWu) {
            float xh = (x[base + d] - m) * r;
            float dxh = dy[base + d] * gamma[d];
            sum_dxh += dxh;
            sum_dxh_xh += dxh * xh;
            dyxh[base + d] = float16_t(dy[base + d] * xh);
            dy16[base + d] = float16_t(dy[base + d]);
        }
        sum_dxh = subgroupAdd(sum_dxh);
        sum_dxh_xh = subgroupAdd(sum_dxh_xh);
        float inv = 1.0 / float(p.D);
        for (uint d = lane; d < p.D; d += SUBWu) {
            float xh = (x[base + d] - m) * r;
            float dxh = dy[base + d] * gamma[d];
            dx[base + d] = r * (dxh - sum_dxh * inv - xh * sum_dxh_xh * inv);
        }
        """,
        push=[("D", "uint")], local=W, subgroup_size=RS, extensions=SUBGROUP,
        width=W)

    # ---- Attention softmax ----------------------------------------------
    # One invocation per (batch*head*query) row. Scale, causal mask, max
    # subtraction and normalisation in a single sweep; emits f16 for the P@V
    # matmul and f32 for the backward pass.
    # One subgroup per attention row, not one thread per row.
    #
    # The obvious version gives each invocation a whole row, which makes
    # adjacent lanes read addresses T*4 bytes apart: every access is its own
    # cache line and the kernel becomes the most expensive thing in the step
    # (measured: 3.3 ms forward, 4.4 ms backward, together 31 of a 47 ms step).
    # Striding the row across the lanes of one wave makes the reads contiguous
    # and the reductions become subgroup ops.
    #
    # Dispatch convention: n = rows * 32, local_size 32, so one workgroup and
    # one wave32 per row, indexed by gl_WorkGroupID.x.
    K["attn_softmax"] = Elementwise(
        dev, "attn_softmax",
        [("s", "f32", "readonly"), ("p16", "f16", "writeonly")],
        """
        uint row = gl_WorkGroupID.x;
        uint lane = gl_SubgroupInvocationID;
        uint base = row * p.T;
        uint q = row % p.T;
        float mx = -1e30;
        for (uint j = lane; j <= q; j += SUBWu) mx = max(mx, s[base + j] * p.scale);
        mx = subgroupMax(mx);
        float sum = 0.0;
        for (uint j = lane; j <= q; j += SUBWu) sum += exp(s[base + j] * p.scale - mx);
        sum = subgroupAdd(sum);
        float inv = 1.0 / sum;
        for (uint j = lane; j < p.T; j += SUBWu) {
            float v = 0.0;
            if (j <= q) v = exp(s[base + j] * p.scale - mx) * inv;
            p16[base + j] = float16_t(v);
        }
        """,
        push=[("T", "uint"), ("scale", "float")],
        local=W, subgroup_size=RS, width=W,
        extensions=["GL_KHR_shader_subgroup_basic", "GL_KHR_shader_subgroup_arithmetic"])

    # dS = P * (dP - sum_j dP_j P_j), same scale, same mask.
    # P is re-read in f16: attention probabilities live in [0, 1], where f16
    # carries ~1e-3 relative error, and not keeping an f32 copy removes a 4 MB
    # write and a 4 MB read per layer per step.
    K["attn_softmax_bwd"] = Elementwise(
        dev, "attn_softmax_bwd",
        [("dp", "f32", "readonly"), ("p16", "f16", "readonly"),
         ("ds16", "f16", "writeonly")],
        """
        uint row = gl_WorkGroupID.x;
        uint lane = gl_SubgroupInvocationID;
        uint base = row * p.T;
        uint q = row % p.T;
        float dot = 0.0;
        for (uint j = lane; j <= q; j += SUBWu)
            dot += dp[base + j] * float(p16[base + j]);
        dot = subgroupAdd(dot);
        for (uint j = lane; j < p.T; j += SUBWu) {
            float v = 0.0;
            if (j <= q) v = float(p16[base + j]) * (dp[base + j] - dot) * p.scale;
            ds16[base + j] = float16_t(v);
        }
        """,
        push=[("T", "uint"), ("scale", "float")],
        local=W, subgroup_size=RS, width=W,
        extensions=["GL_KHR_shader_subgroup_basic", "GL_KHR_shader_subgroup_arithmetic"])

    # ---- Head permutes ---------------------------------------------------
    # (B*T, 3D) -> three (B*H, T, hd) f16 tensors. Fused: one pass over the
    # fused QKV projection produces all three attention operands.
    K["split_qkv"] = Elementwise(
        dev, "split_qkv",
        [("qkv", "f32", "readonly"), ("q16", "f16", "writeonly"),
         ("k16", "f16", "writeonly"), ("v16", "f16", "writeonly")],
        """
        uint d = i % p.D;
        uint bt = i / p.D;
        uint t = bt % p.T;
        uint b = bt / p.T;
        uint h = d / p.hd;
        uint j = d % p.hd;
        uint dst = ((b * p.H + h) * p.T + t) * p.hd + j;
        uint src = bt * 3u * p.D + d;
        q16[dst] = float16_t(qkv[src]);
        k16[dst] = float16_t(qkv[src + p.D]);
        v16[dst] = float16_t(qkv[src + 2u * p.D]);
        """,
        push=[("D", "uint"), ("T", "uint"), ("H", "uint"), ("hd", "uint")])

    K["merge_qkv_grad"] = Elementwise(
        dev, "merge_qkv_grad",
        [("dq", "f32", "readonly"), ("dk", "f32", "readonly"),
         ("dv", "f32", "readonly"), ("dqkv16", "f16", "writeonly")],
        """
        uint d = i % p.D;
        uint bt = i / p.D;
        uint t = bt % p.T;
        uint b = bt / p.T;
        uint h = d / p.hd;
        uint j = d % p.hd;
        uint src = ((b * p.H + h) * p.T + t) * p.hd + j;
        uint dst = bt * 3u * p.D + d;
        dqkv16[dst] = float16_t(dq[src]);
        dqkv16[dst + p.D] = float16_t(dk[src]);
        dqkv16[dst + 2u * p.D] = float16_t(dv[src]);
        """,
        push=[("D", "uint"), ("T", "uint"), ("H", "uint"), ("hd", "uint")])

    # Column sums with real parallelism. The naive one-thread-per-column
    # version runs D threads over B*T rows, which is 8 waves on a 12 CU GPU and
    # ends up dominating the step. This splits the rows into chunks and lets
    # atomics combine them: NCHUNK x D threads instead of D.
    K["col_sum_zero"] = Elementwise(
        dev, "col_sum_zero", [("dst", "f32", "writeonly")], "dst[i] = 0.0;")

    K["col_sum_chunk"] = Elementwise(
        dev, "col_sum_chunk",
        [("g", "f32", "readonly"), ("sums", "f32", "")],
        """
        uint col = i % p.ncol;
        uint chunk = i / p.ncol;
        float s = 0.0;
        for (uint r = chunk; r < p.nrow; r += p.nchunk) s += g[r * p.ncol + col];
        atomicAdd(sums[col], s);
        """,
        push=[("ncol", "uint"), ("nrow", "uint"), ("nchunk", "uint")],
        extensions=["GL_EXT_shader_atomic_float"])

    # Same reduction over an f16 gradient. Every tensor whose column sum we
    # need already exists in f16 for the matmuls, so reading f16 here removes
    # both the f32 duplicate write and half of this kernel's read traffic.
    # Accumulation stays in f32, so only the inputs lose precision.
    K["col_sum_chunk16"] = Elementwise(
        dev, "col_sum_chunk16",
        [("g", "f16", "readonly"), ("sums", "f32", "")],
        """
        uint col = i % p.ncol;
        uint chunk = i / p.ncol;
        float s = 0.0;
        for (uint r = chunk; r < p.nrow; r += p.nchunk)
            s += float(g[r * p.ncol + col]);
        atomicAdd(sums[col], s);
        """,
        push=[("ncol", "uint"), ("nrow", "uint"), ("nchunk", "uint")],
        extensions=["GL_EXT_shader_atomic_float"])

    # (B*H, T, hd) f32 -> (B*T, D) f16, the attention output back to the
    # residual stream layout, narrowed for the projection matmul.
    K["merge_heads"] = Elementwise(
        dev, "merge_heads",
        [("src", "f32", "readonly"), ("y16", "f16", "writeonly")],
        """
        uint d = i % p.D;
        uint bt = i / p.D;
        uint t = bt % p.T;
        uint b = bt / p.T;
        uint h = d / p.hd;
        uint j = d % p.hd;
        y16[i] = float16_t(src[((b * p.H + h) * p.T + t) * p.hd + j]);
        """,
        push=[("D", "uint"), ("T", "uint"), ("H", "uint"), ("hd", "uint")])

    K["split_heads"] = Elementwise(
        dev, "split_heads",
        [("src", "f32", "readonly"), ("dst16", "f16", "writeonly")],
        """
        uint d = i % p.D;
        uint bt = i / p.D;
        uint t = bt % p.T;
        uint b = bt / p.T;
        uint h = d / p.hd;
        uint j = d % p.hd;
        dst16[((b * p.H + h) * p.T + t) * p.hd + j] = float16_t(src[i]);
        """,
        push=[("D", "uint"), ("T", "uint"), ("H", "uint"), ("hd", "uint")])

    # ---- Embeddings ------------------------------------------------------
    K["embed"] = Elementwise(
        dev, "embed",
        [("tok", "f32", "readonly"), ("pos", "f32", "readonly"),
         ("ids", "u32", "readonly"), ("x", "f32", "writeonly")],
        """
        uint d = i % p.D;
        uint bt = i / p.D;
        uint t = bt % p.T;
        x[i] = tok[ids[bt] * p.D + d] + pos[t * p.D + d];
        """,
        push=[("D", "uint"), ("T", "uint")])

    # Scatter-add: many positions in a batch share a vocabulary row, so this
    # needs atomics. VK_EXT_shader_atomic_float, checked at device creation.
    K["embed_bwd"] = Elementwise(
        dev, "embed_bwd",
        [("dx", "f32", "readonly"), ("ids", "u32", "readonly"),
         ("dtok", "f32", ""), ("dpos", "f32", "")],
        """
        uint d = i % p.D;
        uint bt = i / p.D;
        uint t = bt % p.T;
        float g = dx[i];
        atomicAdd(dtok[ids[bt] * p.D + d], g);
        atomicAdd(dpos[t * p.D + d], g);
        """,
        push=[("D", "uint"), ("T", "uint")],
        extensions=["GL_EXT_shader_atomic_float"])

    # ---- Residual --------------------------------------------------------
    # acc + bias broadcast + residual, in one pass over the stream.
    K["residual"] = Elementwise(
        dev, "residual",
        [("acc", "f32", "readonly"), ("bias", "f32", "readonly"),
         ("res", "f32", "readonly"), ("out32", "f32", "writeonly")],
        """
        out32[i] = res[i] + acc[i] + bias[i % p.D];
        """,
        push=[("D", "uint")])

    K["add"] = Elementwise(
        dev, "add",
        [("a", "f32", "readonly"), ("b", "f32", "readonly"),
         ("out32", "f32", "writeonly")],
        "out32[i] = a[i] + b[i];")

    K["to16"] = Elementwise(
        dev, "to16",
        [("src", "f32", "readonly"), ("dst", "f16", "writeonly")],
        "dst[i] = float16_t(src[i]);")

    return K
