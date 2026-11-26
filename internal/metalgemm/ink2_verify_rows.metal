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
