// Y[t] = W.X[t] for the rows of a verify chunk, width-invariant, on fp32 8x8
// simdgroup_matrix products (ink2vm_) or, for one to four rows, the same
// arithmetic on scalar FMAs (ink2vs_). Not a standalone source: it follows packed_gguf_codebooks.metal,
// packed_gguf_dequant.metal and packed_gguf_gemv.metal, whose block decoders,
// decode_scalar and PackedShape it uses.
//
// The fixed arithmetic of an output element: K splits into 64-column groups,
// group g feeding chain g % 8. A chain is one fma chain from +0 over its
// groups in order, a group in the column order phys(k, j) = 16*(k/2) + 2j +
// k%2 for j = 0..7, k = 0..7. S_s = chain s + chain s+4, and the output is
// ((S_0 + S_1) + S_2) + S_3. An fp32 8x8x8 simdgroup MMA is the fma chain
// over k = 0..7 from its accumulator (TestInk2VMMAProbeOrder), so MMA j of a
// group runs columns phys(0..7, j) and both kernel forms give the same bits.
// Neither depends on the row count: a row's bits are the same at every width
// and in either form. They are not the decode GEMV's, so a session that
// verifies here must take its plain steps here too.
//
// The MMA form keeps every operand in registers: the register MMA pattern of
// incoai/splash@2afc7337 runtime/metal/kernels/common/sgmatrix.h (Apache-2.0).
// Lane l holds elements (fm, fn) and (fm, fn+1) of each fragment, so it
// decodes the 16 columns of weight row fm at 16*(fn/2) in the group, and those
// weights feed the group's 8 MMAs in pairs.
#pragma clang fp reassociate(off)

#define INK2VM_SG 4
#define INK2VM_CHAINS 8
// Activation rows one MMA pass covers: two 8-row blocks.
#define INK2VM_ROWS 16
// The most rows a scalar pass carries.
#define INK2VM_SCALAR_MAX 4

struct Ink2VMRowsShape {
    // Activation row t is x + t*xStride; its output is y + t*yStride.
    uint tokens, xStride, yStride, scalarMax;
};

struct Ink2VMExpertsShape {
    // Pair i reads activation row i/xDiv and writes y + i*rows. A group of at
    // most scalarMax pairs runs in the scalar form, any other in the MMA form.
    uint npairs, xStride, xDiv, scalarMax;
};

// The 16 weights of row `row` at column k.
#define INK2VM_DECODE_PACKED(TYPE, BLOCK) \
inline float4x4 ink2vm_w_##TYPE(device const uchar *raw, constant PackedShape &shape, size_t row, uint k) { \
    float4x4 v; \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + row * (shape.cols / BLOCK); \
    dequantize_##TYPE(w + k / BLOCK, short((k % BLOCK) / 16), v); \
    return v; \
}

#define INK2VM_DECODE_SCALAR(NAME, CT) \
inline float4x4 ink2vm_w_##NAME(device const uchar *raw, constant PackedShape &shape, size_t row, uint k) { \
    float4x4 v; \
    device const CT *w = (device const CT *)(raw + shape.byte_offset) + row * shape.cols + k; \
    for (int i = 0; i < 16; i++) v[i / 4][i % 4] = decode_scalar(w[i]); \
    return v; \
}

// c + a x b, the matrices living only inside the call so the operands and the
// accumulator stay register pairs.
inline float2 ink2vm_mma(float2 a, float2 b, float2 c) {
    simdgroup_float8x8 A, B, C;
    A.thread_elements()[0] = a.x;
    A.thread_elements()[1] = a.y;
    B.thread_elements()[0] = b.x;
    B.thread_elements()[1] = b.y;
    C.thread_elements()[0] = c.x;
    C.thread_elements()[1] = c.y;
    simdgroup_multiply_accumulate(C, A, B, C);
    return float2(C.thread_elements()[0], C.thread_elements()[1]);
}

#define INK2VM_ARGS \
    uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

// One MMA pass of up to INK2VM_ROWS activation rows over weight rows
// row0..row0+7: C[weight row][activation row] = W . X^T. SIMD group s runs
// chains s and s+4, so groups s, s+4, s+8, ...; XROW(r) is activation row r
// of the pass, r < count. Leaves S_s in part[s] ([activation row][8]).
#define INK2VM_MMA_PASS(TYPE, XROW) \
    { \
        const uint qid = lane / 4, fm = (qid & 4) | ((lane / 2) & 3), fn = ((qid & 2) << 1) | ((lane & 1) << 1); \
        const uint blocks = (count + 7) / 8; \
        device const float *xr[2][2]; \
        bool live[2][2]; \
        for (uint b = 0; b < 2; b++) \
            for (uint h = 0; h < 2; h++) { \
                const uint r = 8 * b + fn + h; \
                live[b][h] = r < count; \
                xr[b][h] = XROW(min(r, count - 1)); \
            } \
        float2 acc[2][2] = {{float2(0.0f), float2(0.0f)}, {float2(0.0f), float2(0.0f)}}; \
        const size_t wrow = wbase + min(row0 + fm, shape.rows - 1); \
        const uint wq = 16 * (fn / 2), xq = 16 * (fm / 2); \
        const bool odd = fm % 2; \
        for (uint g = simd; g < ng; g += INK2VM_SG) { \
            const uint c = (g % INK2VM_CHAINS) / INK2VM_SG, k0 = 64 * g; \
            const float4x4 v = ink2vm_w_##TYPE(raw, shape, wrow, k0 + wq); \
            _Pragma("unroll") for (uint b = 0; b < 2; b++) \
                if (b < blocks) { \
                    /* columns k0 + xq + 2j + fm%2 of the lane's two rows */ \
                    float4 xs[2][4]; \
                    _Pragma("unroll") for (uint h = 0; h < 2; h++) { \
                        device const float4 *x4 = (device const float4 *)(xr[b][h] + k0 + xq); \
                        _Pragma("unroll") for (uint i = 0; i < 4; i++) xs[h][i] = live[b][h] ? x4[i] : float4(0.0f); \
                    } \
                    float2 cc = c == 0 ? acc[b][0] : acc[b][1]; \
                    _Pragma("unroll") for (uint j = 0; j < 8; j++) { \
                        const float2 a = float2(v[j / 2][2 * (j % 2)], v[j / 2][2 * (j % 2) + 1]); \
                        const float2 p0 = (j % 2) ? xs[0][j / 2].zw : xs[0][j / 2].xy, p1 = (j % 2) ? xs[1][j / 2].zw : xs[1][j / 2].xy; \
                        cc = ink2vm_mma(a, odd ? float2(p0.y, p1.y) : float2(p0.x, p1.x), cc); \
                    } \
                    if (c == 0) acc[b][0] = cc; \
                    else acc[b][1] = cc; \
                } \
        } \
        _Pragma("unroll") for (uint b = 0; b < 2; b++) \
            if (b < blocks) { \
                const float2 v = acc[b][0] + acc[b][1]; \
                part[simd][(8 * b + fn) * 8 + fm] = v.x; \
                part[simd][(8 * b + fn + 1) * 8 + fm] = v.y; \
            } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    }

// The output element e ([activation row][8]) of an MMA pass.
inline float ink2vm_sum(threadgroup const float (&part)[INK2VM_SG][INK2VM_ROWS * 8], uint e) {
    return ((part[0][e] + part[1][e]) + part[2][e]) + part[3][e];
}

// group.x is a tile of 8 weight rows, group.y a pass of INK2VM_ROWS activation
// rows.
#define INK2VM_ROWS_KERNEL(TYPE) \
kernel void ink2vm_rows_##TYPE(device const uchar *raw [[buffer(0)]], \
                               device const float *x [[buffer(1)]], \
                               device float *y [[buffer(2)]], \
                               constant PackedShape &shape [[buffer(3)]], \
                               constant Ink2VMRowsShape &batch [[buffer(4)]], \
                               INK2VM_ARGS) { \
    threadgroup float part[INK2VM_SG][INK2VM_ROWS * 8]; \
    const uint ng = shape.cols / 64, row0 = group.x * 8, t0 = group.y * INK2VM_ROWS; \
    const uint count = min(uint(INK2VM_ROWS), batch.tokens - t0); \
    const size_t wbase = 0; \
    INK2VM_MMA_PASS(TYPE, INK2VM_TOKEN_ROW) \
    for (uint e = tid; e < count * 8; e += 32 * INK2VM_SG) { \
        const uint t = e / 8, n = row0 + e % 8; \
        if (n < shape.rows) y[size_t(t0 + t) * batch.yStride + n] = ink2vm_sum(part, e); \
    } \
}
#define INK2VM_TOKEN_ROW(r) (x + size_t(t0 + (r)) * batch.xStride)

// Threadgroup (x, g) runs weight rows 8x..8x+7 of group g's expert (tab as
// ink2v_group_pairs lays it out) over all its pairs when they are more than
// scalarMax, so each routed expert row is read once a pass. A pair's bits do
// not depend on which pairs share its expert.
#define INK2VM_EXPERTS_KERNEL(TYPE) \
kernel void ink2vm_experts_##TYPE(device const uchar *raw [[buffer(0)]], \
                                  device const float *x [[buffer(1)]], \
                                  device float *y [[buffer(2)]], \
                                  constant PackedShape &shape [[buffer(3)]], \
                                  device const uint *tab [[buffer(4)]], \
                                  constant Ink2VMExpertsShape &batch [[buffer(5)]], \
                                  INK2VM_ARGS) { \
    threadgroup float part[INK2VM_SG][INK2VM_ROWS * 8]; \
    if (group.y >= tab[0]) return; \
    const uint n = batch.npairs, ng = shape.cols / 64; \
    device const uint *gr = tab + 1 + 3 * group.y, *xrows = tab + 1 + 3 * n, *pairOf = xrows + n; \
    const uint e = gr[0], first = gr[1], pairs = gr[2], row0 = group.x * 8; \
    if (pairs <= batch.scalarMax) return; \
    const size_t wbase = size_t(e) * shape.rows; \
    for (uint c0 = 0; c0 < pairs; c0 += INK2VM_ROWS) { \
        const uint s0 = first + c0, count = min(uint(INK2VM_ROWS), pairs - c0); \
        INK2VM_MMA_PASS(TYPE, INK2VM_PAIR_ROW) \
        for (uint i = tid; i < count * 8; i += 32 * INK2VM_SG) { \
            const uint p = i / 8, r = row0 + i % 8; \
            if (r < shape.rows) y[size_t(pairOf[s0 + p]) * shape.rows + r] = ink2vm_sum(part, i); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
}
#define INK2VM_PAIR_ROW(r) (x + size_t(xrows[s0 + (r)]) * batch.xStride)

// The scalar form over count <= INK2VM_SCALAR_MAX activation rows (XROW(p),
// p < count): SIMD group s runs weight rows row0+4s..row0+4s+3, lane 8r+c
// chain c of row r. Lane 8r of each row leaves the outputs in out[p].
#define INK2VS_PASS(TYPE, XROW) \
    float out[INK2VM_SCALAR_MAX]; \
    const uint rr = lane / 8, chain = lane % 8, rowN = row0 + 4 * simd + rr; \
    { \
        float acc[INK2VM_SCALAR_MAX]; \
        device const float *xp[INK2VM_SCALAR_MAX]; \
        _Pragma("unroll") for (uint p = 0; p < INK2VM_SCALAR_MAX; p++) { \
            acc[p] = 0.0f; \
            xp[p] = XROW(min(p, count - 1)); \
        } \
        const size_t wrow = wbase + min(rowN, shape.rows - 1); \
        for (uint g = chain; g < ng; g += INK2VM_CHAINS) { \
            float4x4 v[4]; \
            _Pragma("unroll") for (uint q = 0; q < 4; q++) v[q] = ink2vm_w_##TYPE(raw, shape, wrow, 64 * g + 16 * q); \
            _Pragma("unroll") for (uint p = 0; p < INK2VM_SCALAR_MAX; p++) \
                if (p < count) { \
                    device const float2 *x2 = (device const float2 *)(xp[p] + 64 * g); \
                    _Pragma("unroll") for (uint j = 0; j < 8; j++) \
                        _Pragma("unroll") for (uint q = 0; q < 4; q++) { \
                            const float2 a = x2[8 * q + j]; \
                            acc[p] = fma(v[q][j / 2][2 * (j % 2)], a.x, acc[p]); \
                            acc[p] = fma(v[q][j / 2][2 * (j % 2) + 1], a.y, acc[p]); \
                        } \
                } \
        } \
        const ushort base = ushort(lane & ~7u); \
        _Pragma("unroll") for (uint p = 0; p < INK2VM_SCALAR_MAX; p++) { \
            float S[INK2VM_SG]; \
            for (ushort s = 0; s < INK2VM_SG; s++) \
                S[s] = simd_shuffle(acc[p], ushort(base + s)) + simd_shuffle(acc[p], ushort(base + s + INK2VM_SG)); \
            out[p] = ((S[0] + S[1]) + S[2]) + S[3]; \
        } \
    }

// group.x is a tile of 16 weight rows.
#define INK2VS_ROWS_KERNEL(TYPE) \
kernel void ink2vs_rows_##TYPE(device const uchar *raw [[buffer(0)]], \
                               device const float *x [[buffer(1)]], \
                               device float *y [[buffer(2)]], \
                               constant PackedShape &shape [[buffer(3)]], \
                               constant Ink2VMRowsShape &batch [[buffer(4)]], \
                               INK2VM_ARGS) { \
    const uint ng = shape.cols / 64, row0 = group.x * 16, count = batch.tokens; \
    const size_t wbase = 0; \
    INK2VS_PASS(TYPE, INK2VS_TOKEN_ROW) \
    if (chain == 0 && rowN < shape.rows) \
        for (uint p = 0; p < count; p++) y[size_t(p) * batch.yStride + rowN] = out[p]; \
}
#define INK2VS_TOKEN_ROW(r) (x + size_t(r) * batch.xStride)

// Threadgroup (x, g) runs weight rows 16x..16x+15 of group g's expert when it
// has at most scalarMax pairs.
#define INK2VS_EXPERTS_KERNEL(TYPE) \
kernel void ink2vs_experts_##TYPE(device const uchar *raw [[buffer(0)]], \
                                  device const float *x [[buffer(1)]], \
                                  device float *y [[buffer(2)]], \
                                  constant PackedShape &shape [[buffer(3)]], \
                                  device const uint *tab [[buffer(4)]], \
                                  constant Ink2VMExpertsShape &batch [[buffer(5)]], \
                                  INK2VM_ARGS) { \
    if (group.y >= tab[0]) return; \
    const uint n = batch.npairs, ng = shape.cols / 64; \
    device const uint *gr = tab + 1 + 3 * group.y, *xrows = tab + 1 + 3 * n, *pairOf = xrows + n; \
    const uint e = gr[0], s0 = gr[1], count = gr[2], row0 = group.x * 16; \
    if (count > batch.scalarMax) return; \
    const size_t wbase = size_t(e) * shape.rows; \
    INK2VS_PASS(TYPE, INK2VM_PAIR_ROW) \
    if (chain == 0 && rowN < shape.rows) \
        for (uint p = 0; p < count; p++) y[size_t(pairOf[s0 + p]) * shape.rows + rowN] = out[p]; \
}

#define INK2VM_PACKED(TYPE, BLOCK) INK2VM_DECODE_PACKED(TYPE, BLOCK) INK2VM_ROWS_KERNEL(TYPE) INK2VM_EXPERTS_KERNEL(TYPE) \
    INK2VS_ROWS_KERNEL(TYPE) INK2VS_EXPERTS_KERNEL(TYPE)
#define INK2VM_SCALAR(NAME, CT) INK2VM_DECODE_SCALAR(NAME, CT) INK2VM_ROWS_KERNEL(NAME) INK2VS_ROWS_KERNEL(NAME)

INK2VM_PACKED(q8_0, 32)
INK2VM_PACKED(q3_K, 256)
INK2VM_PACKED(q4_K, 256)
INK2VM_PACKED(q5_K, 256)
INK2VM_PACKED(q6_K, 256)
INK2VM_PACKED(iq2_xs, 256)
INK2VM_PACKED(iq2_s, 256)
INK2VM_PACKED(iq3_xxs, 256)
INK2VM_PACKED(iq3_s, 256)
INK2VM_PACKED(iq4_nl, 32)
INK2VM_PACKED(nvfp4, 64)
INK2VM_SCALAR(f32, float)
INK2VM_SCALAR(f16, half)
INK2VM_SCALAR(bf16, ushort)
