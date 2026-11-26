// Y[t] = W.X[t] for the rows of a verify chunk, reading each weight once. Not
// a standalone source: it follows packed_gguf_codebooks.metal,
// packed_gguf_dequant.metal and packed_gguf_gemv.metal, whose block decoders
// and PackedShape it uses.
//
// A row keeps the lane map and the fma chain of row_<type>, then one simd_sum,
// and owns its accumulator, so its bits are the single-row GEMV's whether the
// tile holds one row or nine. One kernel serves every row count: the tile is
// never swapped for a narrower variant.
// Kernel shape adapted from k_ink2v_q5k_rows in internal/compute/cuda/inkling2_verify_rows.cuh.

struct Ink2VRowsShape {
    // Activation row t is x + t*xStride; its output is y + t*yStride.
    uint tokens, xStride, yStride;
};

#define INK2V_ROWS_TILE 9

#define INK2V_ROWS_ARGS \
    device const uchar *raw [[buffer(0)]], \
    device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], \
    constant PackedShape &shape [[buffer(3)]], \
    constant Ink2VRowsShape &batch [[buffer(4)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

// group.x is a tile of four weight rows (one per SIMD group), group.y a tile
// of INK2V_ROWS_TILE activation rows.
#define INK2V_ROWS_KERNEL(TYPE, BLOCK) \
kernel void ink2v_rows_##TYPE(INK2V_ROWS_ARGS) { \
    const uint row = group.x * 4 + simd; \
    if (row >= shape.rows) return; \
    const uint t0 = group.y * INK2V_ROWS_TILE; \
    const uint count = min(uint(INK2V_ROWS_TILE), batch.tokens - t0); \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + \
            size_t(row) * (shape.cols / BLOCK); \
    float4 acc[INK2V_ROWS_TILE]; \
    for (uint p = 0; p < INK2V_ROWS_TILE; p++) acc[p] = float4(0.0f); \
    for (uint g = lane; g < shape.cols / 16; g += 32) { \
        float4x4 values; \
        dequantize_##TYPE(w + g / (BLOCK/16), g % (BLOCK/16), values); \
        for (uint p = 0; p < count; p++) { \
            device const float4 *v = (device const float4 *)(x + size_t(t0 + p) * batch.xStride + 16*g); \
            for (int i = 0; i < 4; i++) acc[p] = fma(values[i], v[i], acc[p]); \
        } \
    } \
    for (uint p = 0; p < count; p++) { \
        float total = simd_sum(acc[p].x + acc[p].y + acc[p].z + acc[p].w); \
        if (lane == 0) y[size_t(t0 + p) * batch.yStride + row] = total; \
    } \
}

#define INK2V_SCALAR_ROWS_KERNEL(NAME, CT) \
kernel void ink2v_rows_##NAME(INK2V_ROWS_ARGS) { \
    const uint row = group.x * 4 + simd; \
    if (row >= shape.rows) return; \
    const uint t0 = group.y * INK2V_ROWS_TILE; \
    const uint count = min(uint(INK2V_ROWS_TILE), batch.tokens - t0); \
    device const CT *w = (device const CT *)(raw + shape.byte_offset) + size_t(row) * shape.cols; \
    float acc[INK2V_ROWS_TILE]; \
    for (uint p = 0; p < INK2V_ROWS_TILE; p++) acc[p] = 0.0f; \
    for (uint col = lane; col < shape.cols; col += 32) { \
        float value = decode_scalar(w[col]); \
        for (uint p = 0; p < count; p++) \
            acc[p] = fma(value, x[size_t(t0 + p) * batch.xStride + col], acc[p]); \
    } \
    for (uint p = 0; p < count; p++) { \
        float total = simd_sum(acc[p]); \
        if (lane == 0) y[size_t(t0 + p) * batch.yStride + row] = total; \
    } \
}

INK2V_ROWS_KERNEL(q8_0, 32)
INK2V_ROWS_KERNEL(q3_K, 256)
INK2V_ROWS_KERNEL(q4_K, 256)
INK2V_ROWS_KERNEL(q5_K, 256)
INK2V_ROWS_KERNEL(q6_K, 256)
INK2V_ROWS_KERNEL(iq2_xs, 256)
INK2V_ROWS_KERNEL(iq2_s, 256)
INK2V_ROWS_KERNEL(iq3_xxs, 256)
INK2V_ROWS_KERNEL(iq3_s, 256)
INK2V_ROWS_KERNEL(iq4_nl, 32)
INK2V_ROWS_KERNEL(nvfp4, 64)
INK2V_SCALAR_ROWS_KERNEL(f32, float)
INK2V_SCALAR_ROWS_KERNEL(f16, half)
INK2V_SCALAR_ROWS_KERNEL(bf16, ushort)

// The rows of a short chunk, two weight rows a SIMD group: a g dequantizes
// both rows once, then each activation row of a pair of them is loaded once
// and fed to both, which halves the activation loads and the dequantizing of
// ink2v_rows_ (2.0x at 8 rows, M3 Pro). Wider tiles lose to register
// pressure. A (weight row, activation row) pair still keeps row_<type>'s lane
// map and fma chain, so its bits are those of the single-row GEMV.
#define INK2V_PAIR_R 2
#define INK2V_PAIR_P 2

// acc[r][p] += W[row0+r] . x(p) over this lane's g; xrow(p) is activation row p
// of the tile, p < count.
#define INK2V_PAIR_SWEEP(TYPE, BLOCK, XROW) \
    for (uint g = lane; g < shape.cols / 16; g += 32) { \
        float4x4 values[INK2V_PAIR_R]; \
        _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
            dequantize_##TYPE(w + size_t(min(r, nr - 1)) * rowBlocks + g / (BLOCK/16), g % (BLOCK/16), values[r]); \
        _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) \
            if (p < count) { \
                device const float4 *xv = (device const float4 *)(XROW(p) + 16*g); \
                _Pragma("unroll") for (int i = 0; i < 4; i++) { \
                    const float4 v = xv[i]; \
                    _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) acc[r][p] = fma(values[r][i], v, acc[r][p]); \
                } \
            } \
    }

#define INK2V_PAIR_ROWS_KERNEL(TYPE, BLOCK) \
kernel void ink2v_pair_rows_##TYPE(INK2V_ROWS_ARGS) { \
    const uint row0 = (group.x * 4 + simd) * INK2V_PAIR_R; \
    if (row0 >= shape.rows) return; \
    const uint nr = min(uint(INK2V_PAIR_R), shape.rows - row0), rowBlocks = shape.cols / BLOCK; \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + size_t(row0) * rowBlocks; \
    for (uint t0 = 0; t0 < batch.tokens; t0 += INK2V_PAIR_P) { \
        const uint count = min(uint(INK2V_PAIR_P), batch.tokens - t0); \
        float4 acc[INK2V_PAIR_R][INK2V_PAIR_P]; \
        _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
            _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) acc[r][p] = float4(0.0f); \
        INK2V_PAIR_SWEEP(TYPE, BLOCK, INK2V_PAIR_TOKEN_ROW) \
        _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
            _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) \
                if (r < nr && p < count) { \
                    float total = simd_sum(acc[r][p].x + acc[r][p].y + acc[r][p].z + acc[r][p].w); \
                    if (lane == 0) y[size_t(t0 + p) * batch.yStride + row0 + r] = total; \
                } \
    } \
}
#define INK2V_PAIR_TOKEN_ROW(p) (x + size_t(t0 + (p)) * batch.xStride)

#define INK2V_SCALAR_PAIR_ROWS_KERNEL(NAME, CT) \
kernel void ink2v_pair_rows_##NAME(INK2V_ROWS_ARGS) { \
    const uint row0 = (group.x * 4 + simd) * INK2V_PAIR_R; \
    if (row0 >= shape.rows) return; \
    const uint nr = min(uint(INK2V_PAIR_R), shape.rows - row0); \
    device const CT *w = (device const CT *)(raw + shape.byte_offset) + size_t(row0) * shape.cols; \
    for (uint t0 = 0; t0 < batch.tokens; t0 += INK2V_PAIR_P) { \
        const uint count = min(uint(INK2V_PAIR_P), batch.tokens - t0); \
        float acc[INK2V_PAIR_R][INK2V_PAIR_P]; \
        _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
            _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) acc[r][p] = 0.0f; \
        for (uint col = lane; col < shape.cols; col += 32) { \
            float value[INK2V_PAIR_R]; \
            _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
                value[r] = decode_scalar(w[size_t(min(r, nr - 1)) * shape.cols + col]); \
            _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) \
                if (p < count) { \
                    const float v = x[size_t(t0 + p) * batch.xStride + col]; \
                    _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) acc[r][p] = fma(value[r], v, acc[r][p]); \
                } \
        } \
        _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
            _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) \
                if (r < nr && p < count) { \
                    float total = simd_sum(acc[r][p]); \
                    if (lane == 0) y[size_t(t0 + p) * batch.yStride + row0 + r] = total; \
                } \
    } \
}

INK2V_PAIR_ROWS_KERNEL(q8_0, 32)
INK2V_PAIR_ROWS_KERNEL(q3_K, 256)
INK2V_PAIR_ROWS_KERNEL(q4_K, 256)
INK2V_PAIR_ROWS_KERNEL(q5_K, 256)
INK2V_PAIR_ROWS_KERNEL(q6_K, 256)
INK2V_PAIR_ROWS_KERNEL(iq2_xs, 256)
INK2V_PAIR_ROWS_KERNEL(iq2_s, 256)
INK2V_PAIR_ROWS_KERNEL(iq3_xxs, 256)
INK2V_PAIR_ROWS_KERNEL(iq3_s, 256)
INK2V_PAIR_ROWS_KERNEL(iq4_nl, 32)
INK2V_PAIR_ROWS_KERNEL(nvfp4, 64)
INK2V_SCALAR_PAIR_ROWS_KERNEL(f32, float)
INK2V_SCALAR_PAIR_ROWS_KERNEL(f16, half)
INK2V_SCALAR_PAIR_ROWS_KERNEL(bf16, ushort)

struct Ink2VExpertsShape {
    // Pair i reads activation row i/xDiv and writes y + i*rows.
    uint npairs, xStride, xDiv;
};

#define INK2V_GROUP_MAX 1024

// The routed pairs of a verify chunk grouped by expert into tab: tab[0] is the
// group count, then (expert, first slot, pairs) per group in order of each
// expert's first pair, then per slot the activation row and the pair, the
// pairs of a group in ascending order. One threadgroup.
kernel void ink2v_group_pairs(device const uint *ids [[buffer(0)]],
                              device uint *tab [[buffer(1)]],
                              constant Ink2VExpertsShape &s [[buffer(2)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint threads [[threads_per_threadgroup]]) {
    threadgroup uint cnt[INK2V_GROUP_MAX];
    const uint n = s.npairs;
    device uint *groups = tab + 1, *xrow = groups + 3 * n, *pairOf = xrow + n;
    for (uint i = tid; i < n; i += threads) {
        const uint e = ids[i];
        uint c = 0;
        bool first = true;
        for (uint j = 0; j < n; j++)
            if (ids[j] == e) {
                first = first && j >= i;
                c++;
            }
        cnt[i] = first ? c : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += threads) {
        const uint e = ids[i];
        uint lead = i, before = 0;
        for (uint j = 0; j < i; j++)
            if (ids[j] == e) {
                lead = min(lead, j);
                before++;
            }
        uint start = 0, rank = 0;
        for (uint j = 0; j < lead; j++) {
            start += cnt[j];
            rank += cnt[j] != 0;
        }
        xrow[start + before] = i / s.xDiv;
        pairOf[start + before] = i;
        if (lead == i) {
            groups[3 * rank] = e;
            groups[3 * rank + 1] = start;
            groups[3 * rank + 2] = cnt[i];
        }
    }
    if (tid == 0) {
        uint g = 0;
        for (uint j = 0; j < n; j++) g += cnt[j] != 0;
        tab[0] = g;
    }
}

// Threadgroup (x, g) runs weight rows 8x..8x+7 of group g's expert over all
// its pairs, two rows a SIMD group as ink2v_pair_rows_, so each routed expert
// row is read once for the chunk. A pair keeps routed_<type>'s lane map and
// fma chain in its own accumulator: its bits are the decode expert GEMV's,
// however the chunk routes.
#define INK2V_EXPERTS_KERNEL(TYPE, BLOCK) \
kernel void ink2v_experts_##TYPE(device const uchar *raw [[buffer(0)]], \
                                 device const float *x [[buffer(1)]], \
                                 device float *y [[buffer(2)]], \
                                 constant PackedShape &shape [[buffer(3)]], \
                                 device const uint *tab [[buffer(4)]], \
                                 constant Ink2VExpertsShape &batch [[buffer(5)]], \
                                 uint2 group [[threadgroup_position_in_grid]], \
                                 uint simd [[simdgroup_index_in_threadgroup]], \
                                 uint lane [[thread_index_in_simdgroup]]) { \
    const uint row0 = (group.x * 4 + simd) * INK2V_PAIR_R, n = batch.npairs; \
    if (row0 >= shape.rows || group.y >= tab[0]) return; \
    device const uint *gr = tab + 1 + 3 * group.y, *xrow = tab + 1 + 3 * n, *pairOf = xrow + n; \
    const uint e = gr[0], first = gr[1], pairs = gr[2]; \
    const uint nr = min(uint(INK2V_PAIR_R), shape.rows - row0), rowBlocks = shape.cols / BLOCK; \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + \
            (size_t(e) * shape.rows + row0) * rowBlocks; \
    for (uint c0 = 0; c0 < pairs; c0 += INK2V_PAIR_P) { \
        const uint s0 = first + c0, count = min(uint(INK2V_PAIR_P), pairs - c0); \
        device const float *xp[INK2V_PAIR_P]; \
        float4 acc[INK2V_PAIR_R][INK2V_PAIR_P]; \
        _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) { \
            xp[p] = x + size_t(xrow[s0 + min(p, count - 1)]) * batch.xStride; \
            _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) acc[r][p] = float4(0.0f); \
        } \
        INK2V_PAIR_SWEEP(TYPE, BLOCK, INK2V_PAIR_PAIR_ROW) \
        _Pragma("unroll") for (uint r = 0; r < INK2V_PAIR_R; r++) \
            _Pragma("unroll") for (uint p = 0; p < INK2V_PAIR_P; p++) \
                if (r < nr && p < count) { \
                    float total = simd_sum(acc[r][p].x + acc[r][p].y + acc[r][p].z + acc[r][p].w); \
                    if (lane == 0) y[size_t(pairOf[s0 + p]) * shape.rows + row0 + r] = total; \
                } \
    } \
}
#define INK2V_PAIR_PAIR_ROW(p) xp[p]

INK2V_EXPERTS_KERNEL(q8_0, 32)
INK2V_EXPERTS_KERNEL(q3_K, 256)
INK2V_EXPERTS_KERNEL(q4_K, 256)
INK2V_EXPERTS_KERNEL(q5_K, 256)
INK2V_EXPERTS_KERNEL(q6_K, 256)
INK2V_EXPERTS_KERNEL(iq2_xs, 256)
INK2V_EXPERTS_KERNEL(iq2_s, 256)
INK2V_EXPERTS_KERNEL(iq3_xxs, 256)
INK2V_EXPERTS_KERNEL(iq3_s, 256)
INK2V_EXPERTS_KERNEL(iq4_nl, 32)
INK2V_EXPERTS_KERNEL(nvfp4, 64)
