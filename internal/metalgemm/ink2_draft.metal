#include <metal_stdlib>
using namespace metal;

// Kernels of the DFlash2 draft block: BF16 linear (the fc tap fuse and every
// projection), the grouped dynamic short conv, the non-causal sliding-window
// attention and the codebook candidate selector. Activations are f32; weights
// stay BF16 as the checkpoint ships them. The selector's SIMD-group top-k
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
// the token bucket; token rows past s.tokens reread the last one.
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
    threadgroup float red[kInk2DFLinearSimds][R * T];
    const uint h0 = group * R;
    if (h0 >= s.outRows) return;
    device const uint4 *rows[R];
    for (uint r = 0; r < R; r++)
        rows[r] = (device const uint4 *)(w + ulong(min(h0 + r, s.outRows - 1)) * s.cols);
    uint tok[T];
    for (uint t = 0; t < T; t++) tok[t] = min(t, s.tokens - 1) * (s.seg / 4);
    float acc[R][T];
    for (uint r = 0; r < R; r++)
        for (uint t = 0; t < T; t++) acc[r][t] = 0.0f;
    const uint seg8 = s.seg / 8;
    for (uint part = 0; part < s.segs; part++) {
        device const float4 *x = (device const float4 *)(part == 0 ? x0 : part == 1 ? x1 : part == 2 ? x2 : x3);
        const uint wbase = part * seg8;
        for (uint k = lane + 32 * sg; k < seg8; k += 32 * simds) {
            float4 lo[R], hi[R];
            for (uint r = 0; r < R; r++) {
                const uint4 u = rows[r][wbase + k];
                lo[r] = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                               as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
                hi[r] = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                               as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
            }
            for (uint t = 0; t < T; t++) {
                const float4 a = x[tok[t] + 2 * k], b = x[tok[t] + 2 * k + 1];
                for (uint r = 0; r < R; r++)
                    acc[r][t] = fma(hi[r].w, b.w, fma(hi[r].z, b.z, fma(hi[r].y, b.y, fma(hi[r].x, b.x,
                                fma(lo[r].w, a.w, fma(lo[r].z, a.z, fma(lo[r].y, a.y, fma(lo[r].x, a.x, acc[r][t]))))))));
            }
        }
    }
    for (uint r = 0; r < R; r++)
        for (uint t = 0; t < T; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0) red[sg][r * T + t] = v;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R * T) {
        const uint r = tid / T, t = tid % T;
        float v = red[0][tid];
        for (uint g = 1; g < simds; g++) v += red[g][tid];
        if (t < s.tokens && h0 + r < s.outRows) y[ulong(t) * s.outRows + h0 + r] = v;
    }
}

typedef decltype(ink2_df_linear_vec<8, 2>) ink2_df_linear_vec_t;
template [[host_name("ink2_df_linear_t1")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<1, 2>;
template [[host_name("ink2_df_linear_t2")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<2, 2>;
template [[host_name("ink2_df_linear_t4")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<4, 2>;
template [[host_name("ink2_df_linear_t8")]] kernel ink2_df_linear_vec_t ink2_df_linear_vec<8, 2>;

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
