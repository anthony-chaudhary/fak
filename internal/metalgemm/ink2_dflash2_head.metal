// The DFlash2 draft's LM head over its proposal rows: Y[t] = W.X[t] for up
// to eight activation rows of a packed weight. Not a standalone source: it
// follows packed_gguf_codebooks.metal and packed_gguf_dequant.metal, whose
// block decoders it uses.
//
// A threadgroup dequantizes a 32-row by 64-column weight tile into
// threadgroup memory once and multiplies it with the activation tile by 8x8
// SIMD-group matrices, so each activation load serves eight weight rows
// rather than one; the verify rows kernel rereads the activations per weight
// row, which bounds it at the cache, not the weight, bandwidth.

struct Ink2DF2HeadShape {
    uint rows, cols, byteOffset, tokens, xStride, yStride;
};

#define INK2DF2_HEAD_ROWS 32
#define INK2DF2_HEAD_K 64

#define INK2DF2_HEAD_KERNEL(TYPE, BLOCK) \
kernel void ink2_df2_head_##TYPE(device const uchar *raw [[buffer(0)]], \
                                 device const float *x [[buffer(1)]], \
                                 device float *y [[buffer(2)]], \
                                 constant Ink2DF2HeadShape &s [[buffer(3)]], \
                                 uint group [[threadgroup_position_in_grid]], \
                                 uint tid [[thread_index_in_threadgroup]], \
                                 uint sg [[simdgroup_index_in_threadgroup]], \
                                 uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float wt[INK2DF2_HEAD_ROWS * INK2DF2_HEAD_K]; \
    threadgroup float xt[8 * INK2DF2_HEAD_K]; \
    threadgroup float ct[4 * 64]; \
    const uint row0 = group * INK2DF2_HEAD_ROWS; \
    /* thread tid dequantizes 16 columns of one tile row */ \
    const uint wr = tid / 4, wg = tid % 4; \
    const uint wrow = min(row0 + wr, s.rows - 1); \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + s.byteOffset) + ulong(wrow) * (s.cols / BLOCK); \
    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f); \
    for (uint k0 = 0; k0 < s.cols; k0 += INK2DF2_HEAD_K) { \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        { \
            const uint col = k0 + 16 * wg; \
            float4x4 v; \
            dequantize_##TYPE(w + col / BLOCK, short((col % BLOCK) / 16), v); \
            threadgroup float4 *dst = (threadgroup float4 *)(wt + wr * INK2DF2_HEAD_K + 16 * wg); \
            for (int i = 0; i < 4; i++) dst[i] = v[i]; \
        } \
        for (uint i = tid; i < 8 * INK2DF2_HEAD_K; i += 128) { \
            const uint t = i / INK2DF2_HEAD_K, c = i % INK2DF2_HEAD_K; \
            xt[i] = t < s.tokens ? x[ulong(t) * s.xStride + k0 + c] : 0.0f; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint kk = 0; kk < INK2DF2_HEAD_K; kk += 8) { \
            simdgroup_float8x8 a, b; \
            simdgroup_load(a, wt + sg * 8 * INK2DF2_HEAD_K + kk, INK2DF2_HEAD_K); \
            simdgroup_load(b, xt + kk, INK2DF2_HEAD_K, ulong2(0, 0), true); \
            simdgroup_multiply_accumulate(acc, a, b, acc); \
        } \
    } \
    simdgroup_store(acc, ct + sg * 64, 8); \
    simdgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint i = lane; i < 64; i += 32) { \
        const uint r = i / 8, t = i % 8, row = row0 + sg * 8 + r; \
        if (t < s.tokens && row < s.rows) y[ulong(t) * s.yStride + row] = ct[sg * 64 + i]; \
    } \
}

INK2DF2_HEAD_KERNEL(q8_0, 32)
INK2DF2_HEAD_KERNEL(q4_K, 256)
INK2DF2_HEAD_KERNEL(q5_K, 256)
INK2DF2_HEAD_KERNEL(q6_K, 256)
