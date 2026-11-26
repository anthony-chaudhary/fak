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

// The q5_K head without threadgroup staging: each SIMD group owns 8 weight
// rows and feeds the 8x8 matrices from registers, so no barrier stalls the
// weight stream. Lane (fm, fn) of a fragment holds A[fm][fn..fn+1] and
// B[fm][fn..fn+1]; within a 256-column block, fragment column kk of step j is
// block column j + 32*kk, so a lane walks sub-blocks fn and fn+1 of its row
// (one qs byte holds both nibbles) while B row fm walks sub-block fm of the
// activations. Needs 4-byte aligned blocks and 16-byte aligned activation rows.
kernel void ink2_df2_head_q5_K_reg(device const uchar *raw [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant Ink2DF2HeadShape &s [[buffer(3)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint sgs [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint row0 = (group * sgs + sg) * 8;
    if (row0 >= s.rows) return;
    const uint nb = s.cols / 256, row = row0 + fm;
    device const block_q5_K *w = (device const block_q5_K *)(raw + s.byteOffset) + ulong(min(row, s.rows - 1)) * nb;
    const bool v0 = fn < s.tokens, v1 = fn + 1 < s.tokens;
    device const float *x0 = x + ulong(v0 ? fn : 0) * s.xStride + 32 * fm;
    device const float *x1 = x + ulong(v1 ? fn + 1 : 0) * s.xStride + 32 * fm;
    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    for (uint blk = 0; blk < nb; blk++) {
        device const block_q5_K *b = w + blk;
        const uchar2 s0 = get_scale_min_k4_just2(fn, 0, b->scales), s1 = get_scale_min_k4_just2(fn, 1, b->scales);
        const float d = b->d, dm = b->dmin;
        const float dl0 = d * s0[0], ml0 = dm * s0[1], dl1 = d * s1[0], ml1 = dm * s1[1];
        device const uint *qs = (device const uint *)(b->qs + 16 * fn), *qh = (device const uint *)b->qh;
        device const float4 *xa = (device const float4 *)(x0 + 256 * blk), *xb = (device const float4 *)(x1 + 256 * blk);
        for (uint j = 0; j < 8; j++) {
            const float4 a4 = v0 ? xa[j] : float4(0.0f), b4 = v1 ? xb[j] : float4(0.0f);
            const uint q4 = qs[j], h4 = qh[j] >> fn;
            for (uint i = 0; i < 4; i++) {
                const uint q = (q4 >> (8 * i)) & 0xffu, h = h4 >> (8 * i);
                simdgroup_float8x8 am, bm;
                am.thread_elements()[0] = dl0 * float((q & 0xfu) + ((h & 1u) << 4)) - ml0;
                am.thread_elements()[1] = dl1 * float((q >> 4) + ((h & 2u) << 3)) - ml1;
                bm.thread_elements()[0] = a4[i];
                bm.thread_elements()[1] = b4[i];
                simdgroup_multiply_accumulate(acc, am, bm, acc);
            }
        }
    }
    if (row >= s.rows) return;
    if (v0) y[ulong(fn) * s.yStride + row] = acc.thread_elements()[0];
    if (v1) y[ulong(fn + 1) * s.yStride + row] = acc.thread_elements()[1];
}
