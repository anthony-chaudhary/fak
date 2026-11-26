// The tiled GEMM forms of the batched inkling2 prefill weight kernels. Not a
// standalone source: it follows ink2_prefill_packed.metal, whose PackedRows,
// PackedGroup and the packed decoders it uses.
//
// A threadgroup of four SIMD groups owns BM activation rows (or routed pairs
// of one expert) by BN weight rows, the SIMD groups split GM x GN, each a
// block of SM x SN 8x8 fp32 accumulators. Each K step of INK2G_BK decodes the
// weight slab into threadgroup memory once for all BM rows and feeds 8x8
// simdgroup_matrix products, which then go straight from the accumulators to
// the output. The sum runs in another order than the decode
// GEMV's, so these kernels are not bit-equal to it: the Go side gives them
// only chunks wider than any speculative verify chunk.

#define INK2G_THREADS 128
static_assert(INK2G_BK % 16 == 0, "a thread decodes 16 weights at a time");

// Activations are staged as float, weights as TW: half for the formats whose
// every decoded weight a half holds exactly (nvfp4, f16), which halves the
// slab and so lifts occupancy with the products unchanged. 16 bytes of row
// padding stagger the banks of the 8 rows a simdgroup_load reads.
#define INK2G_PITCH(T) (INK2G_BK + 16 / sizeof(T))

// The threadgroup floats of a tile's operand slabs.
#define INK2G_TG_FLOATS(BM, BN, TW) (((BN) * INK2G_PITCH(TW) * sizeof(TW) + (BM) * INK2G_PITCH(float) * 4) / 4)

// live is the tile's live activation rows: 8-row blocks past them skip their
// products, which is most of a short expert run's tile.
template <typename TW, uint GN, uint SM, uint SN>
inline void ink2g_mma(threadgroup const float *xs, threadgroup const TW *ws, uint simd, uint live,
                      thread simdgroup_float8x8 (&acc)[SM][SN]) {
    const uint mr = (simd / GN) * SM * 8, nr = (simd % GN) * SN * 8;
    if (mr >= live) return;
    const uint left = live - mr;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < INK2G_BK; k += 8) {
        simdgroup_float8x8 a[SM];
        simdgroup_matrix<TW, 8, 8> b[SN];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < SM; i++)
            if (8 * i < left) simdgroup_load(a[i], xs + (mr + 8 * i) * INK2G_PITCH(float) + k, INK2G_PITCH(float));
        // B[k][n] = W[n][k]: the weight tile read transposed.
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) simdgroup_load(b[j], ws + (nr + 8 * j) * INK2G_PITCH(TW) + k, INK2G_PITCH(TW), ulong2(0), true);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < SM; i++)
            if (8 * i < left)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < SN; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
}

// Each decoder writes the 16 weights of row from column k (a multiple of 16).
#define INK2G_DECODE_PACKED(TYPE, BLOCK) \
struct Ink2GDec_##TYPE { \
    template <typename TW> \
    static inline void chunk(device const uchar *raw, constant PackedShape &shape, size_t row, uint k, threadgroup TW *dst) { \
        device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + row * (shape.cols / BLOCK); \
        float4x4 v; \
        dequantize_##TYPE(w + k / BLOCK, short((k % BLOCK) / 16), v); \
        threadgroup vec<TW, 4> *d4 = (threadgroup vec<TW, 4> *)dst; \
        for (int i = 0; i < 4; i++) d4[i] = vec<TW, 4>(v[i]); \
    } \
};

#define INK2G_DECODE_SCALAR(NAME, CT) \
struct Ink2GDec_##NAME { \
    template <typename TW> \
    static inline void chunk(device const uchar *raw, constant PackedShape &shape, size_t row, uint k, threadgroup TW *dst) { \
        device const CT *w = (device const CT *)(raw + shape.byte_offset) + row * shape.cols + k; \
        for (int i = 0; i < 16; i++) dst[i] = TW(decode_scalar(w[i])); \
    } \
};

// The K loop shared by both kernels, handing each element (m, n) of the
// [BM][BN] product tile to store(m, n, v). Thread tid loads the activation
// float4s tid + 128r of each slab, from the rows whose offsets xo holds, and
// decodes the 16-weight chunks tid + 128r of each weight slab, of rows
// wbase + min(n0 + slab row, shape.rows - 1).
template <typename D, typename TW, uint GM, uint GN, uint SM, uint SN, uint XR, typename S>
inline void ink2g_k_loop(device const uchar *raw, constant PackedShape &shape, device const float *x, thread const size_t (&xo)[XR],
                         size_t wbase, uint n0, uint live, threadgroup float *tg, uint tid, uint simd, S store) {
    constexpr uint BM = GM * SM * 8, BN = GN * SN * 8;
    constexpr uint WR = BN * INK2G_BK / 16 / INK2G_THREADS, PW = INK2G_PITCH(TW), PX = INK2G_PITCH(float);
    static_assert(XR * 4 * INK2G_THREADS == BM * INK2G_BK, "the activation slab splits evenly");
    static_assert(WR * 16 * INK2G_THREADS == BN * INK2G_BK, "the weight slab splits evenly");
    threadgroup TW *ws = (threadgroup TW *)tg;
    threadgroup float *xs = (threadgroup float *)(ws + BN * PW);
    size_t wrow[WR];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < WR; r++) wrow[r] = wbase + min(n0 + (tid + INK2G_THREADS * r) / (INK2G_BK / 16), shape.rows - 1);
    simdgroup_float8x8 acc[SM][SN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < shape.cols; k0 += INK2G_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < WR; r++) {
            const uint c = tid + INK2G_THREADS * r, row = c / (INK2G_BK / 16), col = 16 * (c % (INK2G_BK / 16));
            D::template chunk<TW>(raw, shape, wrow[r], k0 + col, ws + row * PW + col);
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < XR; r++) {
            const uint c = tid + INK2G_THREADS * r, row = c / (INK2G_BK / 4), col = 4 * (c % (INK2G_BK / 4));
            *(threadgroup float4 *)(xs + row * PX + col) = *(device const float4 *)(x + xo[r] + k0 + col);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ink2g_mma<TW, GN, SM, SN>(xs, ws, simd, live, acc);
    }
    // Lane l of a SIMD group holds elements (fm, fn) and (fm, fn + 1) of each
    // 8x8 product.
    const uint mr = (simd / GN) * SM * 8, nr = (simd % GN) * SN * 8;
    const uint lane = tid % 32, q = lane / 4;
    const uint fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) {
            const thread vec<float, 2> &v = reinterpret_cast<const thread vec<float, 2> &>(acc[i][j].thread_elements());
            store(mr + 8 * i + fm, nr + 8 * j + fn, v[0]);
            store(mr + 8 * i + fm, nr + 8 * j + fn + 1, v[1]);
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

#define INK2G_R_BM (INK2G_R_GM * INK2G_R_SM * 8)
#define INK2G_R_BN (INK2G_R_GN * INK2G_R_SN * 8)
#define INK2G_G_BM (INK2G_G_GM * INK2G_G_SM * 8)
#define INK2G_G_BN (INK2G_G_GN * INK2G_G_SN * 8)
static_assert(INK2G_R_GM * INK2G_R_GN * 32 == INK2G_THREADS, "four SIMD groups");
static_assert(INK2G_G_GM * INK2G_G_GN * 32 == INK2G_THREADS, "four SIMD groups");

// Y[t][n] = W[n,:] . X[t,:]: group.x a tile of INK2G_R_BN weight rows,
// group.y of INK2G_R_BM activation rows. Rows past either end repeat the last
// one unwritten.
#define GEMM_ROWS_KERNEL(TYPE, TW) \
kernel void gemm_rows_##TYPE(GEMM_ROWS_ARGS) { \
    constexpr uint BM = INK2G_R_BM, BN = INK2G_R_BN, XR = BM * INK2G_BK / 4 / INK2G_THREADS; \
    threadgroup float tg[INK2G_TG_FLOATS(INK2G_R_BM, INK2G_R_BN, TW)]; \
    const uint n0 = group.x * BN, t0 = group.y * BM; \
    size_t xo[XR]; \
    for (uint r = 0; r < XR; r++) \
        xo[r] = size_t(min(t0 + (tid + INK2G_THREADS * r) / (INK2G_BK / 4), batch.tokens - 1)) * batch.xStride; \
    ink2g_k_loop<Ink2GDec_##TYPE, TW, INK2G_R_GM, INK2G_R_GN, INK2G_R_SM, INK2G_R_SN>(raw, shape, x, xo, 0, n0, batch.tokens - t0, tg, tid, simd, \
        [&](uint m, uint c, float v) { \
            const uint t = t0 + m, n = n0 + c; \
            if (t < batch.tokens && n < shape.rows) y[size_t(t) * batch.yStride + n] = v; \
        }); \
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

// Finds the tile group.y names, BM pairs from first in expert e's run of
// count; false for a surplus group.
template <uint BM>
inline bool ink2g_grouped_find(device const int *offsets, uint nexp, uint gy, thread uint &e, thread uint &first, thread uint &count) {
    uint tiles = 0;
    for (e = 0; e < nexp; e++) {
        const uint lo = uint(offsets[e]), hi = uint(offsets[e + 1]);
        const uint n = (hi - lo + BM - 1) / BM;
        if (gy < tiles + n) {
            first = lo + (gy - tiles) * BM;
            count = min(BM, hi - first);
            return true;
        }
        tiles += n;
    }
    return false;
}

// One tile of the expert-grouped GEMM: count (at most BM) pairs of expert e
// from order[first], by BN weight rows from n0.
template <typename D, typename TW, uint GM, uint GN, uint SM, uint SN>
inline void ink2g_grouped_tile(device const uchar *raw, constant PackedShape &shape, device const float *x, device float *y,
                               device const int *order, device const float *escale, constant PackedGroup &grp,
                               uint e, uint first, uint count, uint n0, threadgroup float *tg, uint tid, uint simd) {
    constexpr uint XR = GM * SM * 8 * INK2G_BK / 4 / INK2G_THREADS;
    size_t xo[XR];
    for (uint r = 0; r < XR; r++)
        xo[r] = size_t(uint(order[first + min((tid + INK2G_THREADS * r) / (INK2G_BK / 4), count - 1)]) / grp.xDiv) * grp.xStride;
    ink2g_k_loop<D, TW, GM, GN, SM, SN>(raw, shape, x, xo, size_t(e) * shape.rows, n0, count, tg, tid, simd,
        [&](uint p, uint c, float v) {
            const uint n = n0 + c;
            if (p < count && n < shape.rows) y[size_t(order[first + p]) * shape.rows + n] = grp.scaled != 0 ? v * escale[e] : v;
        });
}

// The expert-grouped GEMM over the routed pairs PBucket sorted: group.y is
// one INK2G_G_BM-pair tile of one expert's run in order (offsets[e]..[e+1]),
// numbered across experts in expert order; the grid's bound of
// npairs/INK2G_G_BM + nexp tiles leaves the surplus groups idle. shape.rows is
// the rows of one expert.
#define GEMM_GROUPED_KERNEL(TYPE, TW) \
kernel void gemm_grouped_##TYPE(GEMM_GROUPED_ARGS) { \
    threadgroup float tg[INK2G_TG_FLOATS(INK2G_G_BM, INK2G_G_BN, TW)]; \
    uint e, first, count; \
    if (!ink2g_grouped_find<INK2G_G_BM>(offsets, nexp, group.y, e, first, count)) return; \
    ink2g_grouped_tile<Ink2GDec_##TYPE, TW, INK2G_G_GM, INK2G_G_GN, INK2G_G_SM, INK2G_G_SN>(raw, shape, x, y, order, escale, grp, \
        e, first, count, group.x * INK2G_G_BN, tg, tid, simd); \
}

#define INK2G_PACKED(TYPE, BLOCK, TW) INK2G_DECODE_PACKED(TYPE, BLOCK) GEMM_ROWS_KERNEL(TYPE, TW) GEMM_GROUPED_KERNEL(TYPE, TW)
#define INK2G_SCALAR(NAME, CT, TW) INK2G_DECODE_SCALAR(NAME, CT) GEMM_ROWS_KERNEL(NAME, TW) GEMM_GROUPED_KERNEL(NAME, TW)

INK2G_PACKED(q8_0, 32, float)
INK2G_PACKED(q3_K, 256, float)
INK2G_PACKED(q4_K, 256, float)
INK2G_PACKED(q5_K, 256, float)
INK2G_PACKED(q6_K, 256, float)
INK2G_PACKED(iq2_xs, 256, float)
INK2G_PACKED(iq2_s, 256, float)
INK2G_PACKED(iq3_xxs, 256, float)
INK2G_PACKED(iq3_s, 256, float)
INK2G_PACKED(iq4_nl, 32, float)
// A 2-bit-mantissa value times a UE4M3 scale fits a half's 11 bits and range.
INK2G_PACKED(nvfp4, 64, half)
INK2G_SCALAR(f32, float, float)
INK2G_SCALAR(f16, half, half)
INK2G_SCALAR(bf16, ushort, float)

// The nvfp4 decoder on word loads and half products, which are exact: a
// 2-bit-mantissa value times a UE4M3 scale needs 6 bits and stays normal.
constexpr constant static half ink2g_kv_h[16] = {
    0, .5h, 1.h, 1.5h, 2.h, 3.h, 4.h, 6.h, -0.h, -.5h, -1.h, -1.5h, -2.h, -3.h, -4.h, -6.h
};
struct Ink2GDec_nvfp4h {
    template <typename TW>
    static inline void chunk(device const uchar *raw, constant PackedShape &shape, size_t row, uint k, threadgroup TW *dst) {
        device const uchar *b = raw + shape.byte_offset + (row * (shape.cols / 64) + k / 64) * 36;
        const uint g = (k % 64) / 16, scale = b[g] & 127, exponent = scale >> 3, mantissa = scale & 7;
        const half d = half(scale == 127 ? 0 : exponent == 0 ? float(mantissa) / 512.f : ldexp(float(8 + mantissa), int(exponent) - 10));
        device const uint *q = (device const uint *)(b + 4 + 8 * g);
        const uint u[2] = {q[0], q[1]};
        threadgroup vec<TW, 4> *d4 = (threadgroup vec<TW, 4> *)dst;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; i++) {
            const uint w = u[i];
            d4[i] = vec<TW, 4>(half4(ink2g_kv_h[w & 15], ink2g_kv_h[(w >> 8) & 15], ink2g_kv_h[(w >> 16) & 15], ink2g_kv_h[(w >> 24) & 15]) * d);
            d4[2 + i] = vec<TW, 4>(half4(ink2g_kv_h[(w >> 4) & 15], ink2g_kv_h[(w >> 12) & 15], ink2g_kv_h[(w >> 20) & 15], ink2g_kv_h[w >> 28]) * d);
        }
    }
};

// The nvfp4 grouped GEMM in 32-pair by 128-row tiles whose SIMD groups split
// 2 x 2 over a tile of at most 16 live pairs, so a short expert run keeps all
// four busy, and 4 x 1 over a fuller one. Both sum each element in the order
// gemm_grouped_nvfp4 does.
kernel void gemm_grouped_adapt_nvfp4(GEMM_GROUPED_ARGS) {
    threadgroup float tg[INK2G_TG_FLOATS(32, 128, half)];
    uint e, first, count;
    if (!ink2g_grouped_find<32>(offsets, nexp, group.y, e, first, count)) return;
    if (count <= 16)
        ink2g_grouped_tile<Ink2GDec_nvfp4h, half, 2, 2, 1, 8>(raw, shape, x, y, order, escale, grp, e, first, count, group.x * 128, tg, tid, simd);
    else
        ink2g_grouped_tile<Ink2GDec_nvfp4h, half, 4, 1, 1, 16>(raw, shape, x, y, order, escale, grp, e, first, count, group.x * 128, tg, tid, simd);
}
