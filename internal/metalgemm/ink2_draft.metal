#include <metal_stdlib>
using namespace metal;

// Kernels of the DFlash2 draft block: BF16 linear (the fc tap fuse and every
// projection), the grouped dynamic short conv, the non-causal sliding-window
// attention and the codebook candidate selector. Activations are f32; weights
// stay BF16 as the checkpoint ships them, or an opt-in per-row int8 copy. The
// key-split attention and int8 matrix kernels use SIMD-group matrices, which
// round their sums in their own order. The selector's SIMD-group top-k
// extraction is adapted from incoai/splash@2afc7337 (Apache-2.0),
// draft_select_edges.
//
// The conv, the attention's dot products and value sums and the selector's
// edge scores keep the host reference's evaluation order, so a multiply must
// not fuse into the add that follows it.
#pragma STDC FP_CONTRACT OFF

static inline float ink2_df_bf16(ushort v) { return as_type<float>(uint(v) << 16); }

// ---- linear ---------------------------------------------------------------

constant constexpr uint kInk2DFTokens = 8;

struct Ink2DFLinearShape {
    uint outRows, cols;
    // The input is segs consecutive column segments of seg columns, each its
    // own [tokens][seg] matrix, so concatenated taps need no copy.
    uint seg, segs, tokens;
    // Elements past the 4-byte aligned binding where row 0 starts (0 on the
    // vector kernel's 16-byte aligned path).
    uint shift;
};

// One SIMD group per output row: lanes stride the columns and reduce at the
// end, every token row of the launch sharing each weight load.
kernel void ink2_df_linear(device const ushort *w [[buffer(0)]],
                           device const float *x0 [[buffer(1)]],
                           device const float *x1 [[buffer(2)]],
                           device const float *x2 [[buffer(3)]],
                           device const float *x3 [[buffer(4)]],
                           device float *y [[buffer(5)]],
                           constant Ink2DFLinearShape &s [[buffer(6)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint h = group * simds + sg;
    if (h >= s.outRows) return;
    device const ushort *row = w + s.shift + ulong(h) * s.cols;
    float acc[kInk2DFTokens];
    for (uint t = 0; t < kInk2DFTokens; t++) acc[t] = 0.0f;
    for (uint part = 0; part < s.segs; part++) {
        device const float *x = part == 0 ? x0 : part == 1 ? x1 : part == 2 ? x2 : x3;
        device const ushort *ws = row + ulong(part) * s.seg;
        for (uint k = lane; k < s.seg; k += 32) {
            const float wv = ink2_df_bf16(ws[k]);
            for (uint t = 0; t < kInk2DFTokens; t++)
                if (t < s.tokens) acc[t] = fma(wv, x[ulong(t) * s.seg + k], acc[t]);
        }
    }
    for (uint t = 0; t < kInk2DFTokens; t++) {
        if (t >= s.tokens) break;
        const float v = simd_sum(acc[t]);
        if (lane == 0) y[ulong(t) * s.outRows + h] = v;
    }
}

// The most SIMD groups a vector linear threadgroup splits its columns among.
constant constexpr uint kInk2DFLinearSimds = 16;

// The vector path: R output rows per threadgroup, so each token row's float4
// loads feed R weight rows, and eight BF16 columns per lane from one 16-byte
// load. The SIMD groups split the columns and reduce through threadgroup
// memory, which keeps enough groups in flight for the 2304-row fc fuse. T is
// the token bucket; token rows past tokens reread the last one. NW weights
// over the same input share its loads; each output keeps the arithmetic of a
// one-weight launch.
template <uint T, uint R, uint NW>
static inline void ink2_df_linear_tile(device const ushort *const thread *w, const thread uint *outRows, uint h0,
                                       uint cols, uint seg, uint segs, uint tokens,
                                       device const float *x0, device const float *x1,
                                       device const float *x2, device const float *x3,
                                       uint simds, uint sg, uint lane, threadgroup float *red,
                                       thread float (&out)[NW][R * T], uint tid) {
    device const uint4 *rows[NW][R];
    for (uint i = 0; i < NW; i++)
        for (uint r = 0; r < R; r++)
            rows[i][r] = (device const uint4 *)(w[i] + ulong(min(h0 + r, outRows[i] - 1)) * cols);
    uint tok[T];
    for (uint t = 0; t < T; t++) tok[t] = min(t, tokens - 1) * (seg / 4);
    float acc[NW][R][T];
    for (uint i = 0; i < NW; i++)
        for (uint r = 0; r < R; r++)
            for (uint t = 0; t < T; t++) acc[i][r][t] = 0.0f;
    const uint seg8 = seg / 8;
    for (uint part = 0; part < segs; part++) {
        device const float4 *x = (device const float4 *)(part == 0 ? x0 : part == 1 ? x1 : part == 2 ? x2 : x3);
        const uint wbase = part * seg8;
        for (uint k = lane + 32 * sg; k < seg8; k += 32 * simds) {
            float4 lo[NW][R], hi[NW][R];
            for (uint i = 0; i < NW; i++)
                for (uint r = 0; r < R; r++) {
                    const uint4 u = rows[i][r][wbase + k];
                    lo[i][r] = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                                      as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
                    hi[i][r] = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                                      as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
                }
            for (uint t = 0; t < T; t++) {
                const float4 a = x[tok[t] + 2 * k], b = x[tok[t] + 2 * k + 1];
                for (uint i = 0; i < NW; i++)
                    for (uint r = 0; r < R; r++)
                        acc[i][r][t] = fma(hi[i][r].w, b.w, fma(hi[i][r].z, b.z, fma(hi[i][r].y, b.y, fma(hi[i][r].x, b.x,
                                       fma(lo[i][r].w, a.w, fma(lo[i][r].z, a.z, fma(lo[i][r].y, a.y, fma(lo[i][r].x, a.x, acc[i][r][t]))))))));
            }
        }
    }
    for (uint i = 0; i < NW; i++)
        for (uint r = 0; r < R; r++)
            for (uint t = 0; t < T; t++) {
                const float v = simd_sum(acc[i][r][t]);
                if (lane == 0) red[(sg * NW + i) * R * T + r * T + t] = v;
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R * T)
        for (uint i = 0; i < NW; i++) {
            float v = red[i * R * T + tid];
            for (uint g = 1; g < simds; g++) v += red[(g * NW + i) * R * T + tid];
            out[i][tid] = v;
        }
}

template <uint T, uint R>
kernel void ink2_df_linear_vec(device const ushort *w [[buffer(0)]],
                               device const float *x0 [[buffer(1)]],
                               device const float *x1 [[buffer(2)]],
                               device const float *x2 [[buffer(3)]],
                               device const float *x3 [[buffer(4)]],
                               device float *y [[buffer(5)]],
                               constant Ink2DFLinearShape &s [[buffer(6)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[kInk2DFLinearSimds * R * T];
    const uint h0 = group * R;
    if (h0 >= s.outRows) return;
    device const ushort *ws[1] = {w};
    const uint rows[1] = {s.outRows};
    float out[1][R * T];
    ink2_df_linear_tile<T, R, 1>(ws, rows, h0, s.cols, s.seg, s.segs, s.tokens, x0, x1, x2, x3, simds, sg, lane, red, out, tid);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        if (t < s.tokens && h0 + r < s.outRows) y[ulong(t) * s.outRows + h0 + r] = out[0][tid];
    }
}

typedef decltype(ink2_df_linear_vec<8, 2>) ink2_df_linear_vec_t;
template [[host_name("ink2_df_linear_t1")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<1, 2>;
template [[host_name("ink2_df_linear_t2")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<2, 2>;
template [[host_name("ink2_df_linear_t4")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<4, 2>;
template [[host_name("ink2_df_linear_t8")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<8, 2>;

struct Ink2DFMultiShape {
    uint rows0, rows1, rows2;
    // Threadgroups [0, end0) cover weight 0, [end0, end1) weight 1, the rest 2.
    uint end0, end1;
    uint cols, seg, segs, tokens;
};

// Up to three weights over one input in one launch, the q/k/v or k/v
// projections: each threadgroup covers R rows of one of them.
template <uint T, uint R>
kernel void ink2_df_linear_multi(device const ushort *w0 [[buffer(0)]],
                                 device const ushort *w1 [[buffer(1)]],
                                 device const ushort *w2 [[buffer(2)]],
                                 device const float *x0 [[buffer(3)]],
                                 device const float *x1 [[buffer(4)]],
                                 device const float *x2 [[buffer(5)]],
                                 device const float *x3 [[buffer(6)]],
                                 device float *y0 [[buffer(7)]],
                                 device float *y1 [[buffer(8)]],
                                 device float *y2 [[buffer(9)]],
                                 constant Ink2DFMultiShape &s [[buffer(10)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint simds [[simdgroups_per_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[kInk2DFLinearSimds * R * T];
    const uint which = group < s.end0 ? 0 : group < s.end1 ? 1 : 2;
    const uint h0 = (group - (which == 0 ? 0 : which == 1 ? s.end0 : s.end1)) * R;
    const uint outRows = which == 0 ? s.rows0 : which == 1 ? s.rows1 : s.rows2;
    device float *y = which == 0 ? y0 : which == 1 ? y1 : y2;
    if (h0 >= outRows) return;
    device const ushort *ws[1] = {which == 0 ? w0 : which == 1 ? w1 : w2};
    const uint rows[1] = {outRows};
    float out[1][R * T];
    ink2_df_linear_tile<T, R, 1>(ws, rows, h0, s.cols, s.seg, s.segs, s.tokens, x0, x1, x2, x3, simds, sg, lane, red, out, tid);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        if (t < s.tokens && h0 + r < outRows) y[ulong(t) * outRows + h0 + r] = out[0][tid];
    }
}

typedef decltype(ink2_df_linear_multi<8, 2>) ink2_df_linear_multi_t;
template [[host_name("ink2_df_linear_multi_t1")]] kernel ink2_df_linear_multi_t ink2_df_linear_multi<1, 2>;
template [[host_name("ink2_df_linear_multi_t2")]] kernel ink2_df_linear_multi_t ink2_df_linear_multi<2, 2>;
template [[host_name("ink2_df_linear_multi_t4")]] kernel ink2_df_linear_multi_t ink2_df_linear_multi<4, 2>;
template [[host_name("ink2_df_linear_multi_t8")]] kernel ink2_df_linear_multi_t ink2_df_linear_multi<8, 2>;

// silu(x Wg^T) * (x Wu^T): the gate and up rows of a threadgroup share the
// input loads, and the product is the one DraftSiluMul forms.
template <uint T, uint R>
kernel void ink2_df_linear_silu(device const ushort *wg [[buffer(0)]],
                                device const ushort *wu [[buffer(1)]],
                                device const float *x0 [[buffer(2)]],
                                device float *y [[buffer(3)]],
                                constant Ink2DFLinearShape &s [[buffer(4)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[kInk2DFLinearSimds * 2 * R * T];
    const uint h0 = group * R;
    if (h0 >= s.outRows) return;
    device const ushort *ws[2] = {wg, wu};
    const uint rows[2] = {s.outRows, s.outRows};
    float out[2][R * T];
    ink2_df_linear_tile<T, R, 2>(ws, rows, h0, s.cols, s.seg, 1, s.tokens, x0, x0, x0, x0, simds, sg, lane, red, out, tid);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        const float v = out[0][tid];
        if (t < s.tokens && h0 + r < s.outRows)
            y[ulong(t) * s.outRows + h0 + r] = precise::divide(v, 1.0f + precise::exp(-v)) * out[1][tid];
    }
}

typedef decltype(ink2_df_linear_silu<8, 2>) ink2_df_linear_silu_t;
template [[host_name("ink2_df_linear_silu_t1")]] kernel ink2_df_linear_silu_t ink2_df_linear_silu<1, 2>;
template [[host_name("ink2_df_linear_silu_t2")]] kernel ink2_df_linear_silu_t ink2_df_linear_silu<2, 2>;
template [[host_name("ink2_df_linear_silu_t4")]] kernel ink2_df_linear_silu_t ink2_df_linear_silu<4, 2>;
template [[host_name("ink2_df_linear_silu_t8")]] kernel ink2_df_linear_silu_t ink2_df_linear_silu<8, 2>;

// ---- dynamic short conv ----------------------------------------------------

struct Ink2DFConvShape {
    uint rows, hidden, group, side, kn;
    // Floats per row of the kernel projection: [side][tap][conv group].
    uint dynStride;
    uint baseShift, residual;
};

// One thread per (row, channel). The causal taps reset at the block's first
// row; base is the BF16 [side][tap][channel] kernel.
kernel void ink2_df_conv(device const float *x [[buffer(0)]],
                         device const float *dyn [[buffer(1)]],
                         device const ushort *base [[buffer(2)]],
                         device const float *resid [[buffer(3)]],
                         device float *out [[buffer(4)]],
                         constant Ink2DFConvShape &s [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= s.rows * s.hidden) return;
    const uint row = i / s.hidden, c = i % s.hidden, groups = s.hidden / s.group;
    float y = 0.0f;
    for (uint tap = 0; tap < s.kn && tap <= row; tap++) {
        const float v = x[ulong(row - tap) * s.hidden + c];
        const uint k = s.side * s.kn + tap;
        y += ink2_df_bf16(base[s.baseShift + ulong(k) * s.hidden + c]) * v;
        y += dyn[ulong(row) * s.dynStride + k * groups + c / s.group] * v;
    }
    if (s.residual != 0) y = resid[i] + y;
    out[i] = y;
}

// ---- non-causal sliding-window attention -----------------------------------

constant constexpr uint kInk2DFMaxDim = 128;
// The widest key span: a 2048-position window's 2047 context keys plus a block.
constant constexpr uint kInk2DFMaxKeys = 2047 + kInk2DFTokens;

struct Ink2DFAttendShape {
    uint rows, heads, kvHeads, dim;
    // A query at position pos+row sees context positions within window-1 of it
    // (no limit at 0) that the ring of cap rows still holds.
    uint window, cap;
    float scale;
};

// One threadgroup per (query head, block row); every block key is visible to
// every row. Each thread scores whole keys, so a dot product, like the softmax
// sum and each output's value sum after it, runs serially in the host's key
// order; loads are batched ahead of the in-order adds to hide their latency.
kernel void ink2_df_attend(device const float *q [[buffer(0)]],
                           device const float *kb [[buffer(1)]],
                           device const float *vb [[buffer(2)]],
                           device const float *kc [[buffer(3)]],
                           device const float *vc [[buffer(4)]],
                           device float *out [[buffer(5)]],
                           device const int *meta [[buffer(6)]],
                           constant Ink2DFAttendShape &s [[buffer(7)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint2 tpg [[threads_per_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.x, row = group.y, dim = s.dim, threads = tpg.x;
    const uint kvw = s.kvHeads * dim, off = (h / (s.heads / s.kvHeads)) * dim;
    const int pos = meta[0];
    int lo = max(0, pos - int(s.cap));
    if (s.window > 0) lo = max(lo, pos + int(row) - int(s.window) + 1);
    const uint nctx = uint(max(0, pos - lo)), total = nctx + s.rows, start = uint(lo) % s.cap;

    threadgroup float qs[kInk2DFMaxDim];
    threadgroup float p[kInk2DFMaxKeys];
    threadgroup float red[32];
    threadgroup float den;
    for (uint i = tid; i < dim; i += threads) qs[i] = q[(ulong(row) * s.heads + h) * dim + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local = -INFINITY;
    for (uint k = tid; k < total; k += threads) {
        uint slot = start + k;
        if (slot >= s.cap) slot -= s.cap;
        device const float *key = (k < nctx ? kc + ulong(slot) * kvw : kb + ulong(k - nctx) * kvw) + off;
        float sum = 0.0f;
        for (uint d = 0; d < dim; d += 4) {
            const float k0 = key[d], k1 = key[d + 1], k2 = key[d + 2], k3 = key[d + 3];
            sum += qs[d] * k0;
            sum += qs[d + 1] * k1;
            sum += qs[d + 2] * k2;
            sum += qs[d + 3] * k3;
        }
        p[k] = sum * s.scale;
        local = max(local, p[k]);
    }
    local = simd_max(local);
    if (simd_is_first()) red[sg] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = red[0];
    for (uint g = 1; g < simds; g++) m = max(m, red[g]);
    for (uint k = tid; k < total; k += threads) p[k] = precise::exp(p[k] - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < total; k++) sum += p[k];
        den = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < total; k += threads) p[k] = p[k] / den;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < dim; d += threads) {
        float acc = 0.0f;
        uint slot = start, k = 0;
        for (; k + 8 <= nctx; k += 8) {
            float v[8];
            for (uint i = 0; i < 8; i++) {
                v[i] = vc[ulong(slot) * kvw + off + d];
                slot = slot + 1 == s.cap ? 0 : slot + 1;
            }
            for (uint i = 0; i < 8; i++) acc += p[k + i] * v[i];
        }
        for (; k < nctx; k++) {
            acc += p[k] * vc[ulong(slot) * kvw + off + d];
            slot = slot + 1 == s.cap ? 0 : slot + 1;
        }
        for (uint b = 0; b < s.rows; b++) acc += p[nctx + b] * vb[ulong(b) * kvw + off + d];
        out[(ulong(row) * s.heads + h) * dim + d] = acc;
    }
}

// ---- codebook candidate selector -------------------------------------------

constant constexpr uint kInk2DFTopK = 16;
constant constexpr uint kInk2DFMaxRank = 256;

// The selection order: the larger value, then the lower token id.
static inline bool ink2_df_better(float v, uint id, float bv, uint bid) {
    return v > bv || (v == bv && id < bid);
}

// Inserts (v, id) into a thread's sorted list, keeping its best kInk2DFTopK.
static inline void ink2_df_insert(thread float *vals, thread uint *ids, float v, uint id) {
    if (!ink2_df_better(v, id, vals[kInk2DFTopK - 1], ids[kInk2DFTopK - 1])) return;
    for (uint i = 0; i < kInk2DFTopK; i++) {
        if (ink2_df_better(v, id, vals[i], ids[i])) {
            const float tv = vals[i];
            const uint ti = ids[i];
            vals[i] = v;
            ids[i] = id;
            v = tv;
            id = ti;
        }
    }
}

// The SIMD group's best k of its lanes' sorted lists, best first, into
// gv/gi; lists are consumed.
static inline void ink2_df_simd_top(thread float *vals, thread uint *ids, uint k, uint lane,
                                    threadgroup float *gv, threadgroup uint *gi) {
    for (uint r = 0; r < k; r++) {
        const float best = simd_max(vals[0]);
        const uint bid = simd_min(vals[0] == best ? ids[0] : 0xffffffffu);
        if (vals[0] == best && ids[0] == bid) {
            for (uint i = 0; i + 1 < kInk2DFTopK; i++) {
                vals[i] = vals[i + 1];
                ids[i] = ids[i + 1];
            }
            vals[kInk2DFTopK - 1] = -INFINITY;
            ids[kInk2DFTopK - 1] = 0xffffffffu;
        }
        if (lane == 0) {
            gv[r] = best;
            gi[r] = bid;
        }
    }
}

// The first SIMD group merges the lists sorted lists of k entries each in
// gv/gi (one per lane, lists <= 32) into the best k, best first, at outV/outI.
static inline void ink2_df_merge(threadgroup const float *gv, threadgroup const uint *gi, uint lists, uint k, uint lane,
                                 threadgroup float *outV, threadgroup uint *outI) {
    float vals[kInk2DFTopK];
    uint ids[kInk2DFTopK];
    for (uint i = 0; i < kInk2DFTopK; i++) {
        const bool held = lane < lists && i < k;
        vals[i] = held ? gv[lane * k + i] : -INFINITY;
        ids[i] = held ? gi[lane * k + i] : 0xffffffffu;
    }
    ink2_df_simd_top(vals, ids, k, lane, outV, outI);
}

struct Ink2DFTopKShape {
    uint vocab, rowStride, k, shards, chunk;
};

// Grid (shards, rows): the best k of one vocabulary chunk of one row, into
// partial[row][shard][k] values and ids. A chunk holding a NaN or infinity
// writes a NaN first value, which the selector rejects.
kernel void ink2_df_topk_partial(device const float *logits [[buffer(0)]],
                                 device float *pv [[buffer(1)]],
                                 device uint *pi [[buffer(2)]],
                                 constant Ink2DFTopKShape &s [[buffer(3)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint2 tpg [[threads_per_threadgroup]],
                                 uint simds [[simdgroups_per_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint shard = group.x, row = group.y, threads = tpg.x;
    threadgroup float gv[32 * kInk2DFTopK];
    threadgroup uint gi[32 * kInk2DFTopK];
    float vals[kInk2DFTopK];
    uint ids[kInk2DFTopK];
    for (uint i = 0; i < kInk2DFTopK; i++) {
        vals[i] = -INFINITY;
        ids[i] = 0xffffffffu;
    }
    threadgroup atomic_uint bad;
    if (tid == 0) atomic_store_explicit(&bad, 0u, memory_order_relaxed);
    device const float *lr = logits + ulong(row) * s.rowStride;
    const uint first = shard * s.chunk, last = min(s.vocab, first + s.chunk);
    bool finite = true;
    for (uint t = first + tid; t < last; t += threads) {
        const float v = lr[t];
        finite = finite && isfinite(v);
        ink2_df_insert(vals, ids, v, t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!finite) atomic_store_explicit(&bad, 1u, memory_order_relaxed);
    ink2_df_simd_top(vals, ids, s.k, lane, gv + sg * s.k, gi + sg * s.k);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float mv[kInk2DFTopK];
    threadgroup uint mi[kInk2DFTopK];
    if (sg == 0) ink2_df_merge(gv, gi, simds, s.k, lane, mv, mi);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < s.k) {
        const ulong at = (ulong(row) * s.shards + shard) * s.k;
        const bool nonfinite = tid == 0 && atomic_load_explicit(&bad, memory_order_relaxed) != 0;
        pv[at + tid] = nonfinite ? NAN : mv[tid];
        pi[at + tid] = mi[tid];
    }
}

struct Ink2DFSelectShape {
    uint rows, k, shards, rank, vocab;
    uint predShift, succShift, candidates;
    float softcap;
};

// One threadgroup walks the greedy chain: per row it merges the shard
// partials into the row's top k, scores each candidate as its logit plus the
// predecessor-hidden-successor codebook edge, softcapped after that
// correction, and takes the best (the earlier candidate on a tie) as the next
// row's predecessor. The first predecessor is the anchor at anchor[0]. A row
// with a non-finite logit or edge score ends the chain: it and every later
// row get path (and candidates) -1.
kernel void ink2_df_select(device const float *pv [[buffer(0)]],
                           device const uint *pi [[buffer(1)]],
                           device const float *proj [[buffer(2)]],
                           device const ushort *pred [[buffer(3)]],
                           device const ushort *succ [[buffer(4)]],
                           device const int *anchor [[buffer(5)]],
                           device int *path [[buffer(6)]],
                           device int *cand [[buffer(7)]],
                           constant Ink2DFSelectShape &s [[buffer(8)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float gv[32 * kInk2DFTopK];
    threadgroup uint gi[32 * kInk2DFTopK];
    threadgroup float cv[kInk2DFTopK], capped[kInk2DFTopK];
    threadgroup uint ci[kInk2DFTopK];
    threadgroup uint predecessor;
    threadgroup atomic_uint dead;
    // The predecessor code times the row's projected hidden, the left product
    // of each edge term.
    threadgroup float edge[kInk2DFMaxRank];
    if (tid == 0) {
        predecessor = min(uint(anchor[0]), s.vocab - 1u);
        atomic_store_explicit(&dead, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < s.rows; row++) {
        if (atomic_load_explicit(&dead, memory_order_relaxed) != 0) {
            if (tid == 0) path[row] = -1;
            if (s.candidates != 0 && tid < s.k) cand[ulong(row) * s.k + tid] = -1;
            continue;
        }
        float vals[kInk2DFTopK];
        uint ids[kInk2DFTopK];
        for (uint i = 0; i < kInk2DFTopK; i++) {
            vals[i] = -INFINITY;
            ids[i] = 0xffffffffu;
        }
        const ulong base = ulong(row) * s.shards * s.k;
        bool finite = true;
        for (uint e = tid; e < s.shards * s.k; e += threads) {
            const float v = pv[base + e];
            finite = finite && !isnan(v);
            ink2_df_insert(vals, ids, v, pi[base + e]);
        }
        if (!finite) atomic_store_explicit(&dead, 1u, memory_order_relaxed);
        ink2_df_simd_top(vals, ids, s.k, lane, gv + sg * s.k, gi + sg * s.k);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) ink2_df_merge(gv, gi, simds, s.k, lane, cv, ci);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            device const ushort *pc = pred + s.predShift + ulong(predecessor) * s.rank;
            device const float *h = proj + ulong(row) * s.rank;
            for (uint k = tid; k < s.rank; k += threads) edge[k] = ink2_df_bf16(pc[k]) * h[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < s.k) {
            // The clamp keeps a short shard's sentinel id inside the codebooks.
            device const ushort *qc = succ + s.succShift + ulong(min(ci[tid], s.vocab - 1u)) * s.rank;
            float score = cv[tid];
            uint k = 0;
            for (; k + 8 <= s.rank; k += 8) {
                float c[8];
                for (uint i = 0; i < 8; i++) c[i] = ink2_df_bf16(qc[k + i]);
                for (uint i = 0; i < 8; i++) score += edge[k + i] * c[i];
            }
            for (; k < s.rank; k++) score += edge[k] * ink2_df_bf16(qc[k]);
            if (!isfinite(score)) atomic_store_explicit(&dead, 1u, memory_order_relaxed);
            capped[tid] = s.softcap > 0.0f ? s.softcap * precise::tanh(score / s.softcap) : score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool rejected = atomic_load_explicit(&dead, memory_order_relaxed) != 0;
        if (s.candidates != 0 && tid < s.k) cand[ulong(row) * s.k + tid] = rejected ? -1 : int(ci[tid]);
        if (rejected) {
            if (tid == 0) path[row] = -1;
            continue;
        }
        if (tid == 0) {
            uint best = 0;
            for (uint r = 1; r < s.k; r++)
                if (capped[r] > capped[best]) best = r;
            path[row] = int(ci[best]);
            predecessor = min(ci[best], s.vocab - 1u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- key-split attention ---------------------------------------------------

// The key-split kernel's head width, keys staged at once, keys per
// threadgroup and most splits per query.
constant constexpr uint kInk2DFSplitDim = 128;
constant constexpr uint kInk2DFSplitSub = 16;
constant constexpr uint kInk2DFSplitKeys = 64;
constant constexpr uint kInk2DFSplitMax = 64;

struct Ink2DFAttendSplitShape {
    uint rows, heads, kvHeads, dim, window, cap;
    float scale;
    uint splits;
};

// The keys of row 0's context-then-block order: every later row sees a
// suffix of its context, so the splits are shared and a row masks the rest.
struct Ink2DFSplitKeys {
    int pos, lo;
    uint nctx, total, start;
};

static inline Ink2DFSplitKeys ink2_df_split_keys(int pos, constant Ink2DFAttendSplitShape &s) {
    Ink2DFSplitKeys k;
    k.pos = pos;
    k.lo = max(0, pos - int(s.cap));
    if (s.window > 0) k.lo = max(k.lo, pos - int(s.window) + 1);
    k.nctx = uint(max(0, pos - k.lo));
    k.total = k.nctx + s.rows;
    k.start = uint(k.lo) % s.cap;
    return k;
}

// Stages keys [first, first+16) of K and V, zero past the keys; each thread
// issues all its loads before its stores so their latencies overlap.
static inline void ink2_df_split_stage(threadgroup float *ks, threadgroup float *vs, device const float *kc,
                                       device const float *vc, device const float *kb, device const float *vb,
                                       uint first, Ink2DFSplitKeys k, uint kvw, uint off, uint cap, uint tid, uint threads) {
    constexpr uint n = kInk2DFSplitSub * kInk2DFSplitDim;
    for (uint base = 0; base < n; base += 8 * threads) {
        float kv[8], vv[8];
        for (uint j = 0; j < 8; j++) {
            const uint i = base + j * threads + tid;
            const uint kk = first + i / kInk2DFSplitDim, e = i % kInk2DFSplitDim;
            kv[j] = vv[j] = 0.0f;
            if (i < n && kk < k.nctx) {
                uint slot = k.start + kk;
                if (slot >= cap) slot -= cap;
                kv[j] = kc[ulong(slot) * kvw + off + e];
                vv[j] = vc[ulong(slot) * kvw + off + e];
            } else if (i < n && kk < k.total) {
                kv[j] = kb[ulong(kk - k.nctx) * kvw + off + e];
                vv[j] = vb[ulong(kk - k.nctx) * kvw + off + e];
            }
        }
        for (uint j = 0; j < 8; j++) {
            const uint i = base + j * threads + tid;
            if (i < n) {
                ks[i] = kv[j];
                vs[i] = vv[j];
            }
        }
    }
}

// Grid (KV head, split), one SIMD group per query head of the KV head, each
// covering the 8 block rows: flash decoding over keys [split*64, split*64
// +64) staged 16 at a time and shared by every query, scores and value sums
// on SIMD-group matrices with an online softmax across the stages. Each query
// writes its max, exp sum and unnormalized value sums to
// part[(row*heads+head)*splits+split][dim+2]; a split past the keys writes
// nothing.
kernel void ink2_df_attend_split(device const float *q [[buffer(0)]],
                                 device const float *kb [[buffer(1)]],
                                 device const float *vb [[buffer(2)]],
                                 device const float *kc [[buffer(3)]],
                                 device const float *vc [[buffer(4)]],
                                 device float *part [[buffer(5)]],
                                 device const int *meta [[buffer(6)]],
                                 constant Ink2DFAttendSplitShape &s [[buffer(7)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint2 tpg [[threads_per_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint threads = tpg.x;
    const uint kvh = group.x, split = group.y, h = kvh * (s.heads / s.kvHeads) + sg;
    const uint kvw = s.kvHeads * kInk2DFSplitDim, off = kvh * kInk2DFSplitDim;
    const Ink2DFSplitKeys k = ink2_df_split_keys(meta[0], s);
    const uint k0 = split * kInk2DFSplitKeys;
    if (k0 >= k.total) return;

    threadgroup float ks[kInk2DFSplitSub * kInk2DFSplitDim], vs[kInk2DFSplitSub * kInk2DFSplitDim];
    threadgroup float pall[8 * 8 * kInk2DFSplitSub];
    threadgroup float dall[8 * 64];
    threadgroup float *ps = pall + sg * 8 * kInk2DFSplitSub, *dg = dall + sg * 64;

    simdgroup_float8x8 o[kInk2DFSplitDim / 8];
    for (uint i = 0; i < kInk2DFSplitDim / 8; i++) o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float mr[8], lr[8];
    for (uint r = 0; r < 8; r++) {
        mr[r] = -INFINITY;
        lr[r] = 0.0f;
    }
    simdgroup_float8x8 qm[kInk2DFSplitDim / 8];
    for (uint d = 0; d < kInk2DFSplitDim / 8; d++) simdgroup_load(qm[d], q + h * kInk2DFSplitDim + 8 * d, s.heads * kInk2DFSplitDim);

    for (uint sub = 0; sub < kInk2DFSplitKeys / kInk2DFSplitSub; sub++) {
        const uint first = k0 + sub * kInk2DFSplitSub;
        if (first >= k.total) break;
        ink2_df_split_stage(ks, vs, kc, vc, kb, vb, first, k, kvw, off, s.cap, tid, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sc[kInk2DFSplitSub / 8];
        for (uint t = 0; t < kInk2DFSplitSub / 8; t++) sc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < kInk2DFSplitDim / 8; d++)
            for (uint t = 0; t < kInk2DFSplitSub / 8; t++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, ks + t * 8 * kInk2DFSplitDim + 8 * d, kInk2DFSplitDim, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sc[t], qm[d], b, sc[t]);
            }
        for (uint t = 0; t < kInk2DFSplitSub / 8; t++) simdgroup_store(sc[t], ps + t * 8, kInk2DFSplitSub);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint kk = first + lane;
        for (uint r = 0; r < 8; r++) {
            // Row r's window starts r positions later than row 0's.
            const bool masked = lane >= kInk2DFSplitSub || kk >= k.total ||
                                (s.window > 0 && kk < k.nctx && int(kk) + k.lo < k.pos + int(r) - int(s.window) + 1);
            const float v = masked ? -INFINITY : ps[r * kInk2DFSplitSub + min(lane, kInk2DFSplitSub - 1)] * s.scale;
            const float mn = max(mr[r], simd_max(v));
            const float alpha = mn == -INFINITY ? 1.0f : precise::exp(mr[r] - mn);
            const float e = masked ? 0.0f : precise::exp(v - mn);
            if (lane < kInk2DFSplitSub) ps[r * kInk2DFSplitSub + lane] = e;
            lr[r] = lr[r] * alpha + simd_sum(e);
            mr[r] = mn;
            if (lane < 8) dg[r * 8 + lane] = lane == r ? alpha : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dm, pm[kInk2DFSplitSub / 8];
        simdgroup_load(dm, dg, 8);
        for (uint t = 0; t < kInk2DFSplitSub / 8; t++) simdgroup_load(pm[t], ps + t * 8, kInk2DFSplitSub);
        for (uint dt = 0; dt < kInk2DFSplitDim / 8; dt++) {
            simdgroup_float8x8 acc;
            simdgroup_multiply(acc, dm, o[dt]);
            for (uint t = 0; t < kInk2DFSplitSub / 8; t++) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, vs + t * 8 * kInk2DFSplitDim + dt * 8, kInk2DFSplitDim);
                simdgroup_multiply_accumulate(acc, pm[t], vt, acc);
            }
            o[dt] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint stride = kInk2DFSplitDim + 2;
    device float *dst = part + (ulong(h) * s.splits + split) * stride;
    const ulong rowStride = ulong(s.heads) * s.splits * stride;
    for (uint r = 0; r < 8; r++)
        if (lane == r) {
            dst[r * rowStride] = mr[r];
            dst[r * rowStride + 1] = lr[r];
        }
    for (uint dt = 0; dt < kInk2DFSplitDim / 8; dt++) simdgroup_store(o[dt], dst + 2 + dt * 8, rowStride);
}

// Grid (head, row): merges the written split partials in split order,
// rescaling each to the overall max, into the normalized attention output.
kernel void ink2_df_attend_merge(device const float *part [[buffer(0)]],
                                 device float *out [[buffer(1)]],
                                 device const int *meta [[buffer(2)]],
                                 constant Ink2DFAttendSplitShape &s [[buffer(3)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint2 tpg [[threads_per_threadgroup]]) {
    const uint h = group.x, row = group.y, dim = s.dim, stride = dim + 2;
    const Ink2DFSplitKeys k = ink2_df_split_keys(meta[0], s);
    const uint used = min(s.splits, (k.total + kInk2DFSplitKeys - 1) / kInk2DFSplitKeys);
    device const float *src = part + (ulong(row) * s.heads + h) * s.splits * stride;
    threadgroup float w[kInk2DFSplitMax];
    float m = -INFINITY;
    for (uint i = 0; i < used; i++) m = max(m, src[i * stride]);
    for (uint i = tid; i < used; i += tpg.x) w[i] = precise::exp(src[i * stride] - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float l = 0.0f;
    for (uint i = 0; i < used; i++) l += src[i * stride + 1] * w[i];
    for (uint d = tid; d < dim; d += tpg.x) {
        float acc = 0.0f;
        for (uint i = 0; i < used; i++) acc += src[i * stride + 2 + d] * w[i];
        out[(ulong(row) * s.heads + h) * dim + d] = acc / l;
    }
}

// ---- int8 draft linears ----------------------------------------------------

// Grid rows: one BF16 weight row to symmetric int8 with an f32 scale, q =
// rint(w / scale) for scale = max|w| / 127 (a zero row keeps scale 0).
kernel void ink2_df_q8_quantize(device const ushort *w [[buffer(0)]],
                                device char *qw [[buffer(1)]],
                                device float *scales [[buffer(2)]],
                                constant Ink2DFLinearShape &s [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    device const ushort *src = w + s.shift + ulong(row) * s.cols;
    threadgroup float red[32];
    float a = 0.0f;
    for (uint c = tid; c < s.cols; c += threads) a = max(a, fabs(ink2_df_bf16(src[c])));
    a = simd_max(a);
    if (simd_is_first()) red[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = red[0];
    for (uint g = 1; g < simds; g++) amax = max(amax, red[g]);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (uint c = tid; c < s.cols; c += threads)
        qw[ulong(row) * s.cols + c] = char(clamp(rint(ink2_df_bf16(src[c]) * inv), -127.0f, 127.0f));
    if (tid == 0) scales[row] = amax / 127.0f;
}

// Weight rows per int8 threadgroup.
constant constexpr uint kInk2DFQ8Rows = 2;

// ink2_df_linear_tile over int8 rows, sixteen columns per 16-byte lane load
// so a lane keeps as many weight bytes in flight as the BF16 kernel; out
// holds the unscaled sums, which the caller multiplies by the row scale.
template <uint T, uint R, uint NW>
static inline void ink2_df_linear_q8_tile(device const char *const thread *w, const thread uint *outRows, uint h0,
                                          uint cols, uint seg, uint segs, uint tokens,
                                          device const float *x0, device const float *x1,
                                          device const float *x2, device const float *x3,
                                          uint simds, uint sg, uint lane, threadgroup float *red,
                                          thread float (&out)[NW][R * T], uint tid) {
    device const uint4 *rows[NW][R];
    for (uint i = 0; i < NW; i++)
        for (uint r = 0; r < R; r++)
            rows[i][r] = (device const uint4 *)(w[i] + ulong(min(h0 + r, outRows[i] - 1)) * cols);
    uint tok[T];
    for (uint t = 0; t < T; t++) tok[t] = min(t, tokens - 1) * (seg / 4);
    float acc[NW][R][T];
    for (uint i = 0; i < NW; i++)
        for (uint r = 0; r < R; r++)
            for (uint t = 0; t < T; t++) acc[i][r][t] = 0.0f;
    const uint seg16 = seg / 16;
    for (uint part = 0; part < segs; part++) {
        device const float4 *x = (device const float4 *)(part == 0 ? x0 : part == 1 ? x1 : part == 2 ? x2 : x3);
        const uint wbase = part * seg16;
        for (uint k = lane + 32 * sg; k < seg16; k += 32 * simds) {
            uint4 u[NW][R];
            for (uint i = 0; i < NW; i++)
                for (uint r = 0; r < R; r++) u[i][r] = rows[i][r][wbase + k];
            for (uint t = 0; t < T; t++) {
                const float4 a = x[tok[t] + 4 * k], b = x[tok[t] + 4 * k + 1], c = x[tok[t] + 4 * k + 2], e = x[tok[t] + 4 * k + 3];
                for (uint i = 0; i < NW; i++)
                    for (uint r = 0; r < R; r++) {
                        const float4 w0 = float4(as_type<char4>(u[i][r].x)), w1 = float4(as_type<char4>(u[i][r].y));
                        const float4 w2 = float4(as_type<char4>(u[i][r].z)), w3 = float4(as_type<char4>(u[i][r].w));
                        float v = acc[i][r][t];
                        v = fma(w0.w, a.w, fma(w0.z, a.z, fma(w0.y, a.y, fma(w0.x, a.x, v))));
                        v = fma(w1.w, b.w, fma(w1.z, b.z, fma(w1.y, b.y, fma(w1.x, b.x, v))));
                        v = fma(w2.w, c.w, fma(w2.z, c.z, fma(w2.y, c.y, fma(w2.x, c.x, v))));
                        acc[i][r][t] = fma(w3.w, e.w, fma(w3.z, e.z, fma(w3.y, e.y, fma(w3.x, e.x, v))));
                    }
            }
        }
    }
    for (uint i = 0; i < NW; i++)
        for (uint r = 0; r < R; r++)
            for (uint t = 0; t < T; t++) {
                const float v = simd_sum(acc[i][r][t]);
                if (lane == 0) red[(sg * NW + i) * R * T + r * T + t] = v;
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R * T)
        for (uint i = 0; i < NW; i++) {
            float v = red[i * R * T + tid];
            for (uint g = 1; g < simds; g++) v += red[(g * NW + i) * R * T + tid];
            out[i][tid] = v;
        }
}

template <uint T, uint R>
kernel void ink2_df_linear_q8(device const char *w [[buffer(0)]],
                              device const float *x0 [[buffer(1)]],
                              device const float *x1 [[buffer(2)]],
                              device const float *x2 [[buffer(3)]],
                              device const float *x3 [[buffer(4)]],
                              device float *y [[buffer(5)]],
                              constant Ink2DFLinearShape &s [[buffer(6)]],
                              device const float *scale [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simds [[simdgroups_per_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[kInk2DFLinearSimds * R * T];
    const uint h0 = group * R;
    if (h0 >= s.outRows) return;
    device const char *ws[1] = {w};
    const uint rows[1] = {s.outRows};
    float out[1][R * T];
    ink2_df_linear_q8_tile<T, R, 1>(ws, rows, h0, s.cols, s.seg, s.segs, s.tokens, x0, x1, x2, x3, simds, sg, lane, red, out, tid);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        if (t < s.tokens && h0 + r < s.outRows) y[ulong(t) * s.outRows + h0 + r] = out[0][tid] * scale[h0 + r];
    }
}

typedef decltype(ink2_df_linear_q8<8, kInk2DFQ8Rows>) ink2_df_linear_q8_t;
template [[host_name("ink2_df_linear_q8_t1")]] kernel ink2_df_linear_q8_t ink2_df_linear_q8<1, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_q8_t2")]] kernel ink2_df_linear_q8_t ink2_df_linear_q8<2, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_q8_t4")]] kernel ink2_df_linear_q8_t ink2_df_linear_q8<4, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_q8_t8")]] kernel ink2_df_linear_q8_t ink2_df_linear_q8<8, kInk2DFQ8Rows>;

template <uint T, uint R>
kernel void ink2_df_linear_multi_q8(device const char *w0 [[buffer(0)]],
                                    device const char *w1 [[buffer(1)]],
                                    device const char *w2 [[buffer(2)]],
                                    device const float *x0 [[buffer(3)]],
                                    device const float *x1 [[buffer(4)]],
                                    device const float *x2 [[buffer(5)]],
                                    device const float *x3 [[buffer(6)]],
                                    device float *y0 [[buffer(7)]],
                                    device float *y1 [[buffer(8)]],
                                    device float *y2 [[buffer(9)]],
                                    constant Ink2DFMultiShape &s [[buffer(10)]],
                                    device const float *sc0 [[buffer(11)]],
                                    device const float *sc1 [[buffer(12)]],
                                    device const float *sc2 [[buffer(13)]],
                                    uint group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint simds [[simdgroups_per_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[kInk2DFLinearSimds * R * T];
    const uint which = group < s.end0 ? 0 : group < s.end1 ? 1 : 2;
    const uint h0 = (group - (which == 0 ? 0 : which == 1 ? s.end0 : s.end1)) * R;
    const uint outRows = which == 0 ? s.rows0 : which == 1 ? s.rows1 : s.rows2;
    device float *y = which == 0 ? y0 : which == 1 ? y1 : y2;
    device const float *scale = which == 0 ? sc0 : which == 1 ? sc1 : sc2;
    if (h0 >= outRows) return;
    device const char *ws[1] = {which == 0 ? w0 : which == 1 ? w1 : w2};
    const uint rows[1] = {outRows};
    float out[1][R * T];
    ink2_df_linear_q8_tile<T, R, 1>(ws, rows, h0, s.cols, s.seg, s.segs, s.tokens, x0, x1, x2, x3, simds, sg, lane, red, out, tid);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        if (t < s.tokens && h0 + r < outRows) y[ulong(t) * outRows + h0 + r] = out[0][tid] * scale[h0 + r];
    }
}

typedef decltype(ink2_df_linear_multi_q8<8, kInk2DFQ8Rows>) ink2_df_linear_multi_q8_t;
template [[host_name("ink2_df_linear_multi_q8_t1")]] kernel ink2_df_linear_multi_q8_t ink2_df_linear_multi_q8<1, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_multi_q8_t2")]] kernel ink2_df_linear_multi_q8_t ink2_df_linear_multi_q8<2, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_multi_q8_t4")]] kernel ink2_df_linear_multi_q8_t ink2_df_linear_multi_q8<4, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_multi_q8_t8")]] kernel ink2_df_linear_multi_q8_t ink2_df_linear_multi_q8<8, kInk2DFQ8Rows>;

template <uint T, uint R>
kernel void ink2_df_linear_silu_q8(device const char *wg [[buffer(0)]],
                                   device const char *wu [[buffer(1)]],
                                   device const float *x0 [[buffer(2)]],
                                   device float *y [[buffer(3)]],
                                   constant Ink2DFLinearShape &s [[buffer(4)]],
                                   device const float *sg8 [[buffer(5)]],
                                   device const float *su8 [[buffer(6)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint simds [[simdgroups_per_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[kInk2DFLinearSimds * 2 * R * T];
    const uint h0 = group * R;
    if (h0 >= s.outRows) return;
    device const char *ws[2] = {wg, wu};
    const uint rows[2] = {s.outRows, s.outRows};
    float out[2][R * T];
    ink2_df_linear_q8_tile<T, R, 2>(ws, rows, h0, s.cols, s.seg, 1, s.tokens, x0, x0, x0, x0, simds, sg, lane, red, out, tid);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        if (t < s.tokens && h0 + r < s.outRows) {
            const float v = out[0][tid] * sg8[h0 + r];
            y[ulong(t) * s.outRows + h0 + r] = precise::divide(v, 1.0f + precise::exp(-v)) * (out[1][tid] * su8[h0 + r]);
        }
    }
}

typedef decltype(ink2_df_linear_silu_q8<8, kInk2DFQ8Rows>) ink2_df_linear_silu_q8_t;
template [[host_name("ink2_df_linear_silu_q8_t1")]] kernel ink2_df_linear_silu_q8_t ink2_df_linear_silu_q8<1, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_silu_q8_t2")]] kernel ink2_df_linear_silu_q8_t ink2_df_linear_silu_q8<2, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_silu_q8_t4")]] kernel ink2_df_linear_silu_q8_t ink2_df_linear_silu_q8<4, kInk2DFQ8Rows>;
template [[host_name("ink2_df_linear_silu_q8_t8")]] kernel ink2_df_linear_silu_q8_t ink2_df_linear_silu_q8<8, kInk2DFQ8Rows>;

// The 8-row int8 linears on SIMD-group matrices: a SIMD group covers 8
// weight rows, staging 64 dequantized columns of them into threadgroup memory
// (one 16-byte load per lane, the next stage's issued before this one's
// products) and multiplying the 8 token rows by them; c[i] is [token][row].
template <uint NW>
static inline void ink2_df_q8_mma(device const char *const thread *w, uint outRows, uint r0, uint cols, uint seg,
                                  device const float *x0, device const float *x1, device const float *x2, device const float *x3,
                                  threadgroup float *wt, uint lane, thread simdgroup_float8x8 (&c)[NW]) {
    for (uint i = 0; i < NW; i++) c[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint lr = lane / 4, lc = (lane % 4) * 16, stages = cols / 64;
    const ulong row = ulong(min(r0 + lr, outRows - 1)) * cols;
    uint4 u[NW];
    for (uint i = 0; i < NW; i++) u[i] = *(device const uint4 *)(w[i] + row + lc);
    for (uint st = 0; st < stages; st++) {
        for (uint i = 0; i < NW; i++) {
            threadgroup float4 *dst = (threadgroup float4 *)(wt + i * 512 + lr * 64 + lc);
            dst[0] = float4(as_type<char4>(u[i].x));
            dst[1] = float4(as_type<char4>(u[i].y));
            dst[2] = float4(as_type<char4>(u[i].z));
            dst[3] = float4(as_type<char4>(u[i].w));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (st + 1 < stages)
            for (uint i = 0; i < NW; i++) u[i] = *(device const uint4 *)(w[i] + row + (st + 1) * 64 + lc);
        const uint col = st * 64, part = col / seg, c0 = col % seg;
        device const float *x = (part == 0 ? x0 : part == 1 ? x1 : part == 2 ? x2 : x3) + c0;
        for (uint b = 0; b < 8; b++) {
            simdgroup_float8x8 a;
            simdgroup_load(a, x + b * 8, seg);
            for (uint i = 0; i < NW; i++) {
                simdgroup_float8x8 bt;
                simdgroup_load(bt, wt + i * 512 + b * 8, 64, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(c[i], a, bt, c[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < NW; i++) simdgroup_store(c[i], wt + i * 512, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// 32 weight rows per threadgroup of 4 SIMD groups.
kernel void ink2_df_linear_q8_mma(device const char *w [[buffer(0)]],
                                  device const float *x0 [[buffer(1)]],
                                  device const float *x1 [[buffer(2)]],
                                  device const float *x2 [[buffer(3)]],
                                  device const float *x3 [[buffer(4)]],
                                  device float *y [[buffer(5)]],
                                  constant Ink2DFLinearShape &s [[buffer(6)]],
                                  device const float *scale [[buffer(7)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float wtall[4 * 512];
    const uint r0 = group * 32 + sg * 8;
    if (r0 >= s.outRows) return;
    threadgroup float *wt = wtall + sg * 512;
    device const char *ws[1] = {w};
    simdgroup_float8x8 c[1];
    ink2_df_q8_mma<1>(ws, s.outRows, r0, s.cols, s.seg, x0, x1, x2, x3, wt, lane, c);
    for (uint e = lane; e < 64; e += 32) {
        const uint t = e / 8, r = r0 + e % 8;
        if (r < s.outRows) y[ulong(t) * s.outRows + r] = wt[e] * scale[r];
    }
}

kernel void ink2_df_linear_multi_q8_mma(device const char *w0 [[buffer(0)]],
                                        device const char *w1 [[buffer(1)]],
                                        device const char *w2 [[buffer(2)]],
                                        device const float *x0 [[buffer(3)]],
                                        device const float *x1 [[buffer(4)]],
                                        device const float *x2 [[buffer(5)]],
                                        device const float *x3 [[buffer(6)]],
                                        device float *y0 [[buffer(7)]],
                                        device float *y1 [[buffer(8)]],
                                        device float *y2 [[buffer(9)]],
                                        constant Ink2DFMultiShape &s [[buffer(10)]],
                                        device const float *sc0 [[buffer(11)]],
                                        device const float *sc1 [[buffer(12)]],
                                        device const float *sc2 [[buffer(13)]],
                                        uint group [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float wtall[4 * 512];
    const uint which = group < s.end0 ? 0 : group < s.end1 ? 1 : 2;
    const uint r0 = (group - (which == 0 ? 0 : which == 1 ? s.end0 : s.end1)) * 32 + sg * 8;
    const uint outRows = which == 0 ? s.rows0 : which == 1 ? s.rows1 : s.rows2;
    device float *y = which == 0 ? y0 : which == 1 ? y1 : y2;
    device const float *scale = which == 0 ? sc0 : which == 1 ? sc1 : sc2;
    if (r0 >= outRows) return;
    threadgroup float *wt = wtall + sg * 512;
    device const char *ws[1] = {which == 0 ? w0 : which == 1 ? w1 : w2};
    simdgroup_float8x8 c[1];
    ink2_df_q8_mma<1>(ws, outRows, r0, s.cols, s.seg, x0, x1, x2, x3, wt, lane, c);
    for (uint e = lane; e < 64; e += 32) {
        const uint t = e / 8, r = r0 + e % 8;
        if (r < outRows) y[ulong(t) * outRows + r] = wt[e] * scale[r];
    }
}

kernel void ink2_df_linear_silu_q8_mma(device const char *wg [[buffer(0)]],
                                       device const char *wu [[buffer(1)]],
                                       device const float *x0 [[buffer(2)]],
                                       device float *y [[buffer(3)]],
                                       constant Ink2DFLinearShape &s [[buffer(4)]],
                                       device const float *sg8 [[buffer(5)]],
                                       device const float *su8 [[buffer(6)]],
                                       uint group [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float wtall[4 * 2 * 512];
    const uint r0 = group * 32 + sg * 8;
    if (r0 >= s.outRows) return;
    threadgroup float *wt = wtall + sg * 2 * 512;
    device const char *ws[2] = {wg, wu};
    simdgroup_float8x8 c[2];
    ink2_df_q8_mma<2>(ws, s.outRows, r0, s.cols, s.seg, x0, x0, x0, x0, wt, lane, c);
    for (uint e = lane; e < 64; e += 32) {
        const uint t = e / 8, r = r0 + e % 8;
        if (r < s.outRows) {
            const float v = wt[e] * sg8[r];
            y[ulong(t) * s.outRows + r] = precise::divide(v, 1.0f + precise::exp(-v)) * (wt[512 + e] * su8[r]);
        }
    }
}
