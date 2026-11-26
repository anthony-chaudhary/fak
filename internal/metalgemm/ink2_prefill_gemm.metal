// The tiled GEMM forms of the batched inkling2 prefill weight kernels. Not a
// standalone source: it follows ink2_prefill_packed.metal, whose PackedRows,
// PackedGroup and the packed decoders it uses.
//
// A threadgroup of four SIMD groups owns INK2G_BM activation rows (or routed
// pairs of one expert) by INK2G_BN weight rows. Each K step of INK2G_BK decodes the
// weight slab into threadgroup memory once for all INK2G_BM rows and feeds
// fp32 8x8 simdgroup_matrix products. The sum runs in another order than the
// decode GEMV's, so these kernels are not bit-equal to it: the Go side gives
// them only chunks wider than any speculative verify chunk.

// Row pitch of the threadgroup tiles: 4 floats of padding staggers the banks
// of the 8 rows a simdgroup_load reads.
#define INK2G_PITCH (INK2G_BK + 4)

// The operand slabs, and after the K loop the output tile. fp32 operands: half
// ones measured no more than 8% faster on an M3 Pro.
union Ink2GTiles {
    struct {
        float x[INK2G_BM * INK2G_PITCH];
        float w[INK2G_BN * INK2G_PITCH];
    } op;
    float c[INK2G_BM * INK2G_BN];
};

// The four SIMD groups split the tile 2 x 2: each owns INK2G_BM/2 activation
// rows by INK2G_BN/2 weight rows, as 8x8 accumulators.
#define INK2G_SM 2
#define INK2G_SN (INK2G_BN / 16)

// rows is the tile's live activation rows: 8-row blocks past them skip their
// products, which is most of a short expert run's tile.
inline void ink2g_mma(threadgroup const Ink2GTiles &tl, uint simd, uint rows, thread simdgroup_float8x8 (&acc)[INK2G_SM][INK2G_SN]) {
    const uint mr = (simd / 2) * 16, nr = (simd % 2) * (INK2G_BN / 2);
    static_assert(INK2G_SM == 2, "a SIMD group holds two 8-row blocks");
    if (mr >= rows) return;
    const bool both = mr + 8 < rows;
    for (uint k = 0; k < INK2G_BK; k += 8) {
        simdgroup_float8x8 a[INK2G_SM], b[INK2G_SN];
        for (uint i = 0; i < INK2G_SM; i++) simdgroup_load(a[i], tl.op.x + (mr + 8 * i) * INK2G_PITCH + k, INK2G_PITCH);
        // B[k][n] = W[n][k]: the weight tile read transposed.
        for (uint j = 0; j < INK2G_SN; j++) simdgroup_load(b[j], tl.op.w + (nr + 8 * j) * INK2G_PITCH + k, INK2G_PITCH, ulong2(0), true);
        for (uint j = 0; j < INK2G_SN; j++) simdgroup_multiply_accumulate(acc[0][j], a[0], b[j], acc[0][j]);
        if (both)
            for (uint j = 0; j < INK2G_SN; j++) simdgroup_multiply_accumulate(acc[1][j], a[1], b[j], acc[1][j]);
    }
}

// Leaves the accumulators as the [INK2G_BM][INK2G_BN] tile tl.c for the caller
// to write out.
inline void ink2g_spill(threadgroup Ink2GTiles &tl, uint simd, thread simdgroup_float8x8 (&acc)[INK2G_SM][INK2G_SN]) {
    const uint mr = (simd / 2) * 16, nr = (simd % 2) * (INK2G_BN / 2);
    threadgroup float *c = tl.c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < INK2G_SM; i++)
        for (uint j = 0; j < INK2G_SN; j++) simdgroup_store(acc[i][j], c + (mr + 8 * i) * INK2G_BN + nr + 8 * j, INK2G_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A SIMD group per 8 activation rows: 4 threads load each row of the
// activation slab, and the first 2*INK2G_BN decode the weight slab.
#define INK2G_THREADS (4 * INK2G_BM)
static_assert(INK2G_BN * 2 <= INK2G_THREADS, "a thread decodes half a weight row");
static_assert(INK2G_BK % 32 == 0, "a thread decodes 16 weights of each 32 columns");

// Thread tid decodes 16 weights of each 32 columns of slab row tid/2;
// row is the absolute weight row (expert stacks add the expert's base).
#define INK2G_DECODE_PACKED(TYPE, BLOCK) \
inline void ink2g_w_##TYPE(device const uchar *raw, constant PackedShape &shape, size_t row, uint k0, uint tid, threadgroup Ink2GTiles &tl) { \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + row * (shape.cols / BLOCK); \
    for (uint h = 0; h < INK2G_BK / 32; h++) { \
        const uint k = k0 + 32 * h + 16 * (tid % 2); \
        float4x4 v; \
        dequantize_##TYPE(w + k / BLOCK, short((k % BLOCK) / 16), v); \
        threadgroup float4 *dst = (threadgroup float4 *)(tl.op.w + (tid / 2) * INK2G_PITCH + 32 * h + 16 * (tid % 2)); \
        for (int i = 0; i < 4; i++) dst[i] = v[i]; \
    } \
}

#define INK2G_DECODE_SCALAR(NAME, CT) \
inline void ink2g_w_##NAME(device const uchar *raw, constant PackedShape &shape, size_t row, uint k0, uint tid, threadgroup Ink2GTiles &tl) { \
    for (uint h = 0; h < INK2G_BK / 32; h++) { \
        device const CT *w = (device const CT *)(raw + shape.byte_offset) + row * shape.cols + k0 + 32 * h + 16 * (tid % 2); \
        threadgroup float *dst = tl.op.w + (tid / 2) * INK2G_PITCH + 32 * h + 16 * (tid % 2); \
        for (int i = 0; i < 16; i++) dst[i] = decode_scalar(w[i]); \
    } \
}

// Thread tid loads 8 floats of each 32 columns of each 32-row band of the
// activation slab.
inline void ink2g_x(device const float *x, size_t xo, uint k0, uint tid, threadgroup Ink2GTiles &tl) {
    for (uint h = 0; h < INK2G_BK / 32; h++) {
            device const float4 *src = (device const float4 *)(x + xo + k0 + 32 * h + 8 * (tid % 4));
            threadgroup float4 *dst = (threadgroup float4 *)(tl.op.x + (tid / 4) * INK2G_PITCH + 32 * h + 8 * (tid % 4));
            dst[0] = src[0];
            dst[1] = src[1];
        }
}

#define GEMM_ROWS_ARGS \
    device const uchar *raw [[buffer(0)]], \
    device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], \
    constant PackedShape &shape [[buffer(3)]], \
    constant PackedRows &batch [[buffer(4)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]

// Y[t][n] = W[n,:] . X[t,:]: group.x a tile of INK2G_BN weight rows, group.y of
// INK2G_BM activation rows. Rows past either end repeat the last one unwritten.
#define GEMM_ROWS_KERNEL(TYPE) \
kernel void gemm_rows_##TYPE(GEMM_ROWS_ARGS) { \
    threadgroup Ink2GTiles tl; \
    const uint n0 = group.x * INK2G_BN, t0 = group.y * INK2G_BM; \
    const size_t wrow = min(n0 + tid / 2, shape.rows - 1); \
    const size_t xo = size_t(min(t0 + tid / 4, batch.tokens - 1)) * batch.xStride; \
    simdgroup_float8x8 acc[INK2G_SM][INK2G_SN]; \
    for (uint i = 0; i < INK2G_SM; i++) \
        for (uint j = 0; j < INK2G_SN; j++) acc[i][j] = simdgroup_float8x8(0.0f); \
    for (uint k0 = 0; k0 < shape.cols; k0 += INK2G_BK) { \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid < 2 * INK2G_BN) ink2g_w_##TYPE(raw, shape, wrow, k0, tid, tl); \
        ink2g_x(x, xo, k0, tid, tl); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        ink2g_mma(tl, simd, batch.tokens - t0, acc); \
    } \
    ink2g_spill(tl, simd, acc); \
    for (uint e = tid; e < INK2G_BM * INK2G_BN; e += INK2G_THREADS) { \
        const uint t = t0 + e / INK2G_BN, n = n0 + e % INK2G_BN; \
        if (t < batch.tokens && n < shape.rows) y[size_t(t) * batch.yStride + n] = tl.c[e]; \
    } \
}

#define GEMM_GROUPED_ARGS \
    device const uchar *raw [[buffer(0)]], \
    device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], \
    constant PackedShape &shape [[buffer(3)]], \
    device const int *ids [[buffer(4)]], \
    device const int *order [[buffer(5)]], \
    device const float *escale [[buffer(6)]], \
    constant PackedGroup &grp [[buffer(7)]], \
    device const int *offsets [[buffer(8)]], \
    constant uint &nexp [[buffer(9)]], \
    uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]

// The expert-grouped GEMM over the routed pairs PBucket sorted: group.y is
// one INK2G_BM-pair tile of one expert's run in order (offsets[e]..[e+1]),
// numbered across experts in expert order; the grid's bound of
// npairs/INK2G_BM + nexp tiles leaves the surplus groups idle. shape.rows is
// the rows of one expert.
#define GEMM_GROUPED_KERNEL(TYPE) \
kernel void gemm_grouped_##TYPE(GEMM_GROUPED_ARGS) { \
    threadgroup Ink2GTiles tl; \
    uint e = 0, first = 0, tiles = 0; \
    for (; e < nexp; e++) { \
        const uint lo = uint(offsets[e]), hi = uint(offsets[e + 1]); \
        const uint n = (hi - lo + INK2G_BM - 1) / INK2G_BM; \
        if (group.y < tiles + n) { first = lo + (group.y - tiles) * INK2G_BM; break; } \
        tiles += n; \
    } \
    if (e == nexp) return; \
    const uint count = min(uint(INK2G_BM), uint(offsets[e + 1]) - first); \
    const uint n0 = group.x * INK2G_BN; \
    const size_t wrow = size_t(e) * shape.rows + min(n0 + tid / 2, shape.rows - 1); \
    const size_t xo = size_t(uint(order[first + min(tid / 4, count - 1)]) / grp.xDiv) * grp.xStride; \
    simdgroup_float8x8 acc[INK2G_SM][INK2G_SN]; \
    for (uint i = 0; i < INK2G_SM; i++) \
        for (uint j = 0; j < INK2G_SN; j++) acc[i][j] = simdgroup_float8x8(0.0f); \
    for (uint k0 = 0; k0 < shape.cols; k0 += INK2G_BK) { \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid < 2 * INK2G_BN) ink2g_w_##TYPE(raw, shape, wrow, k0, tid, tl); \
        ink2g_x(x, xo, k0, tid, tl); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        ink2g_mma(tl, simd, count, acc); \
    } \
    ink2g_spill(tl, simd, acc); \
    const float s = grp.scaled != 0 ? escale[e] : 1.0f; \
    for (uint i = tid; i < INK2G_BM * INK2G_BN; i += INK2G_THREADS) { \
        const uint p = i / INK2G_BN, n = n0 + i % INK2G_BN; \
        if (p < count && n < shape.rows) { \
            const float v = tl.c[i]; \
            y[size_t(order[first + p]) * shape.rows + n] = grp.scaled != 0 ? v * s : v; \
        } \
    } \
}

#define INK2G_PACKED(TYPE, BLOCK) INK2G_DECODE_PACKED(TYPE, BLOCK) GEMM_ROWS_KERNEL(TYPE) GEMM_GROUPED_KERNEL(TYPE)
#define INK2G_SCALAR(NAME, CT) INK2G_DECODE_SCALAR(NAME, CT) GEMM_ROWS_KERNEL(NAME) GEMM_GROUPED_KERNEL(NAME)

INK2G_PACKED(q8_0, 32)
INK2G_PACKED(q3_K, 256)
INK2G_PACKED(q4_K, 256)
INK2G_PACKED(q5_K, 256)
INK2G_PACKED(q6_K, 256)
INK2G_PACKED(iq2_xs, 256)
INK2G_PACKED(iq2_s, 256)
INK2G_PACKED(iq3_xxs, 256)
INK2G_PACKED(iq3_s, 256)
INK2G_PACKED(iq4_nl, 32)
INK2G_PACKED(nvfp4, 64)
INK2G_SCALAR(f32, float)
INK2G_SCALAR(f16, half)
INK2G_SCALAR(bf16, ushort)
