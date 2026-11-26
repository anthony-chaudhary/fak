// The weight-reading kernels of the batched inkling2 prefill. Not a standalone
// source: it follows packed_gguf_codebooks.metal, packed_gguf_dequant.metal and
// packed_gguf_gemv.metal, whose block structs, dequantize_<type>, decode_scalar
// and PackedShape it uses.
//
// Every output keeps the lane map and the fma chain of row_<type>, then one
// simd_sum, so it equals the decode mv_<type> / routed_<type> of that row bit
// for bit. A lane holds a register tile of weight rows by activation rows: a
// decoded weight block serves every activation row of the tile, and a loaded
// activation quad every weight row.

// INK2P_ROWS_R (weight rows per SIMD group), INK2P_ROWS_TILE (activation rows
// per tile) and INK2P_GROUP_TILE (routed pairs per tile) are defined ahead of
// this text by the Go side, which sizes the grids from the same numbers.

struct PackedRows {
    // Activation row t is x + t*xStride; its output is y + t*yStride.
    uint tokens, xStride, yStride;
};

#define ROWS_ARGS \
    device const uchar *raw [[buffer(0)]], \
    device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], \
    constant PackedShape &shape [[buffer(3)]], \
    constant PackedRows &batch [[buffer(4)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

// Y[t][row] = W[row,:] . X[t,:]. A SIMD group owns INK2P_ROWS_R weight rows by
// INK2P_ROWS_TILE activation rows: group.x counts four of those row blocks,
// group.y the activation tiles. An activation quad is loaded once for all the
// weight rows of the block, which is what bounds the decode GEMV. Rows past
// either end repeat the last one unwritten, so that every loop has a fixed
// trip count and the tile stays in registers.
#define PACKED_ROWS_KERNEL(TYPE, BLOCK) \
kernel void rows_##TYPE(ROWS_ARGS) { \
    const uint row0 = (group.x * 4 + simd) * INK2P_ROWS_R; \
    if (row0 >= shape.rows) return; \
    const uint t0 = group.y * INK2P_ROWS_TILE; \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset); \
    size_t wr[INK2P_ROWS_R], xo[INK2P_ROWS_TILE]; \
    float4 acc[INK2P_ROWS_R][INK2P_ROWS_TILE]; \
    for (uint r = 0; r < INK2P_ROWS_R; r++) { \
        wr[r] = size_t(min(row0 + r, shape.rows - 1)) * (shape.cols / BLOCK); \
        for (uint p = 0; p < INK2P_ROWS_TILE; p++) acc[r][p] = float4(0.0f); \
    } \
    for (uint p = 0; p < INK2P_ROWS_TILE; p++) xo[p] = size_t(min(t0 + p, batch.tokens - 1)) * batch.xStride; \
    for (uint g = lane; g < shape.cols / 16; g += 32) { \
        float4x4 values[INK2P_ROWS_R]; \
        for (uint r = 0; r < INK2P_ROWS_R; r++) \
            dequantize_##TYPE(w + wr[r] + g / (BLOCK/16), g % (BLOCK/16), values[r]); \
        for (uint p = 0; p < INK2P_ROWS_TILE; p++) { \
            device const float4 *v = (device const float4 *)(x + xo[p] + 16*g); \
            for (int i = 0; i < 4; i++) { \
                const float4 xv = v[i]; \
                for (uint r = 0; r < INK2P_ROWS_R; r++) acc[r][p] = fma(values[r][i], xv, acc[r][p]); \
            } \
        } \
    } \
    for (uint r = 0; r < INK2P_ROWS_R; r++) \
        for (uint p = 0; p < INK2P_ROWS_TILE; p++) { \
            float total = simd_sum(acc[r][p].x + acc[r][p].y + acc[r][p].z + acc[r][p].w); \
            if (lane == 0 && row0 + r < shape.rows && t0 + p < batch.tokens) \
                y[size_t(t0 + p) * batch.yStride + row0 + r] = total; \
        } \
}

#define SCALAR_ROWS_KERNEL(NAME, CT) \
kernel void rows_##NAME(ROWS_ARGS) { \
    const uint row0 = (group.x * 4 + simd) * INK2P_ROWS_R; \
    const uint t0 = group.y * INK2P_ROWS_TILE; \
    for (uint row = row0; row < min(row0 + INK2P_ROWS_R, shape.rows); row++) { \
        device const CT *w = (device const CT *)(raw + shape.byte_offset) + size_t(row) * shape.cols; \
        float acc[INK2P_ROWS_TILE]; \
        size_t xo[INK2P_ROWS_TILE]; \
        for (uint p = 0; p < INK2P_ROWS_TILE; p++) { \
            acc[p] = 0.0f; \
            xo[p] = size_t(min(t0 + p, batch.tokens - 1)) * batch.xStride; \
        } \
        for (uint col = lane; col < shape.cols; col += 32) { \
            float value = decode_scalar(w[col]); \
            for (uint p = 0; p < INK2P_ROWS_TILE; p++) acc[p] = fma(value, x[xo[p] + col], acc[p]); \
        } \
        for (uint p = 0; p < INK2P_ROWS_TILE; p++) { \
            float total = simd_sum(acc[p]); \
            if (lane == 0 && t0 + p < batch.tokens) y[size_t(t0 + p) * batch.yStride + row] = total; \
        } \
    } \
}

PACKED_ROWS_KERNEL(q8_0, 32)
PACKED_ROWS_KERNEL(q3_K, 256)
PACKED_ROWS_KERNEL(q4_K, 256)
PACKED_ROWS_KERNEL(q5_K, 256)
PACKED_ROWS_KERNEL(q6_K, 256)
PACKED_ROWS_KERNEL(iq2_xs, 256)
PACKED_ROWS_KERNEL(iq2_s, 256)
PACKED_ROWS_KERNEL(iq3_xxs, 256)
PACKED_ROWS_KERNEL(iq3_s, 256)
PACKED_ROWS_KERNEL(iq4_nl, 32)
PACKED_ROWS_KERNEL(nvfp4, 64)
SCALAR_ROWS_KERNEL(f32, float)
SCALAR_ROWS_KERNEL(f16, half)
SCALAR_ROWS_KERNEL(bf16, ushort)

struct PackedGroup {
    // Pair i reads activation row (i / xDiv) * xStride and writes y row i.
    uint npairs, xDiv, xStride;
    // Nonzero: multiply each output by escale[expert].
    uint scaled;
};

#define GROUPED_ARGS \
    device const uchar *raw [[buffer(0)]], \
    device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], \
    constant PackedShape &shape [[buffer(3)]], \
    device const int *ids [[buffer(4)]], \
    device const int *order [[buffer(5)]], \
    device const float *escale [[buffer(6)]], \
    constant PackedGroup &grp [[buffer(7)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

// Expert-grouped GEMM over the routed (token, slot) pairs: order lists the
// pairs sorted by expert, group.y is a tile of consecutive entries of it and
// group.x four blocks of INK2P_ROWS_R rows of the expert's matrix. Inside a
// tile each run of pairs that share an expert decodes that expert's blocks
// once; walking the runs here keeps the grid a host constant, so nothing is
// read back to size it. shape.rows is the rows of one expert. The tile is
// walked with fixed trip counts, as in rows_<type>: a pair outside the run
// skips its sums, and entries past the end repeat the last pair unwritten.
#define PACKED_GROUPED_KERNEL(TYPE, BLOCK) \
kernel void grouped_##TYPE(GROUPED_ARGS) { \
    const uint row0 = (group.x * 4 + simd) * INK2P_ROWS_R; \
    if (row0 >= shape.rows) return; \
    const uint first = group.y * INK2P_GROUP_TILE; \
    const uint count = min(uint(INK2P_GROUP_TILE), grp.npairs - first); \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset); \
    uint pair[INK2P_GROUP_TILE], expert[INK2P_GROUP_TILE]; \
    size_t xo[INK2P_GROUP_TILE]; \
    float4 acc[INK2P_ROWS_R][INK2P_GROUP_TILE]; \
    for (uint p = 0; p < INK2P_GROUP_TILE; p++) { \
        pair[p] = uint(order[first + min(p, count - 1)]); \
        expert[p] = uint(ids[pair[p]]); \
        xo[p] = size_t(pair[p] / grp.xDiv) * grp.xStride; \
        for (uint r = 0; r < INK2P_ROWS_R; r++) acc[r][p] = float4(0.0f); \
    } \
    uint start = 0; \
    while (start < count) { \
        uint e = 0, end = start + 1; \
        for (uint p = 0; p < INK2P_GROUP_TILE; p++) { \
            if (p == start) e = expert[p]; \
            if (p == end && p < count && expert[p] == e) end++; \
        } \
        size_t wr[INK2P_ROWS_R]; \
        for (uint r = 0; r < INK2P_ROWS_R; r++) \
            wr[r] = (size_t(e) * shape.rows + min(row0 + r, shape.rows - 1)) * (shape.cols / BLOCK); \
        for (uint g = lane; g < shape.cols / 16; g += 32) { \
            float4x4 values[INK2P_ROWS_R]; \
            for (uint r = 0; r < INK2P_ROWS_R; r++) \
                dequantize_##TYPE(w + wr[r] + g / (BLOCK/16), g % (BLOCK/16), values[r]); \
            for (uint p = 0; p < INK2P_GROUP_TILE; p++) { \
                if (p < start || p >= end) continue; \
                device const float4 *v = (device const float4 *)(x + xo[p] + 16*g); \
                for (int i = 0; i < 4; i++) { \
                    const float4 xv = v[i]; \
                    for (uint r = 0; r < INK2P_ROWS_R; r++) acc[r][p] = fma(values[r][i], xv, acc[r][p]); \
                } \
            } \
        } \
        start = end; \
    } \
    for (uint r = 0; r < INK2P_ROWS_R; r++) \
        for (uint p = 0; p < INK2P_GROUP_TILE; p++) { \
            float total = simd_sum(acc[r][p].x + acc[r][p].y + acc[r][p].z + acc[r][p].w); \
            if (grp.scaled != 0) total *= escale[expert[p]]; \
            if (lane == 0 && row0 + r < shape.rows && p < count) y[size_t(pair[p]) * shape.rows + row0 + r] = total; \
        } \
}

#define SCALAR_GROUPED_KERNEL(NAME, CT) \
kernel void grouped_##NAME(GROUPED_ARGS) { \
    const uint row0 = (group.x * 4 + simd) * INK2P_ROWS_R; \
    const uint first = group.y * INK2P_GROUP_TILE; \
    const uint count = min(uint(INK2P_GROUP_TILE), grp.npairs - first); \
    for (uint row = row0; row < min(row0 + INK2P_ROWS_R, shape.rows); row++) { \
        uint start = 0; \
        while (start < count) { \
            const uint e = uint(ids[order[first + start]]); \
            uint end = start + 1; \
            while (end < count && uint(ids[order[first + end]]) == e) end++; \
            device const CT *w = (device const CT *)(raw + shape.byte_offset) + (size_t(e) * shape.rows + row) * shape.cols; \
            float acc[INK2P_GROUP_TILE]; \
            uint pair[INK2P_GROUP_TILE]; \
            for (uint p = 0; p < INK2P_GROUP_TILE; p++) { \
                acc[p] = 0.0f; \
                pair[p] = p >= start && p < end ? uint(order[first + p]) : 0u; \
            } \
            for (uint col = lane; col < shape.cols; col += 32) { \
                float value = decode_scalar(w[col]); \
                for (uint p = start; p < end; p++) \
                    acc[p] = fma(value, x[size_t(pair[p] / grp.xDiv) * grp.xStride + col], acc[p]); \
            } \
            for (uint p = start; p < end; p++) { \
                float total = simd_sum(acc[p]); \
                if (grp.scaled != 0) total *= escale[e]; \
                if (lane == 0) y[size_t(pair[p]) * shape.rows + row] = total; \
            } \
            start = end; \
        } \
    } \
}

PACKED_GROUPED_KERNEL(q8_0, 32)
PACKED_GROUPED_KERNEL(q3_K, 256)
PACKED_GROUPED_KERNEL(q4_K, 256)
PACKED_GROUPED_KERNEL(q5_K, 256)
PACKED_GROUPED_KERNEL(q6_K, 256)
PACKED_GROUPED_KERNEL(iq2_xs, 256)
PACKED_GROUPED_KERNEL(iq2_s, 256)
PACKED_GROUPED_KERNEL(iq3_xxs, 256)
PACKED_GROUPED_KERNEL(iq3_s, 256)
PACKED_GROUPED_KERNEL(iq4_nl, 32)
PACKED_GROUPED_KERNEL(nvfp4, 64)
SCALAR_GROUPED_KERNEL(f32, float)
SCALAR_GROUPED_KERNEL(f16, half)
SCALAR_GROUPED_KERNEL(bf16, ushort)
