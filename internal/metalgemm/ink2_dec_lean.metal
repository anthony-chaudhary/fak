#include <metal_stdlib>
using namespace metal;

// Lean forms of reducing kernels of ink2_ops.metal. Each keeps that file's
// summation order (per thread in column order, across the SIMD group, then over
// the SIMD groups in order), so its results are the same bits. What changes is
// what a launch waits on: a thread's loads are issued together instead of one
// per loop iteration, and work one thread can do is not repeated by all.
#pragma STDC FP_CONTRACT OFF

#define INK2L_TG 256
#define INK2L_SG (INK2L_TG / 32)
#define INK2L_STATES 16
#define INK2L_ROUTE_THREADS 32
#define INK2L_MAX_TOPK 16

static inline float ink2l_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float ink2l_inv_rms(float sumSquares, uint n, float eps) {
    return 1.0f / sqrt(sumSquares / float(n) + eps);
}

template <typename T> static inline T ink2l_pick4(uint i, T a, T b, T c, T d) {
    return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;
}

static inline float ink2l_fold(threadgroup float *part) {
    float total = 0.0f;
    for (uint g = 0; g < INK2L_SG; g++) total += part[g];
    return total;
}

// A thread's rows are elements tid + INK2L_TG * k of its operand. The rows are
// named, so every load of a phase is issued before the first use: a loop would
// issue one load per iteration, each waiting on the last. The host compiles
// this source per row count, INK2L_EXACT when the width is exactly that many
// rows; otherwise a row past n reads the last element and stays out of the math.
#ifndef INK2L_ROWS
#define INK2L_ROWS 12
#endif
#if INK2L_ROWS == 8
#define INK2L_EACH_ROW(X, a) X(a, 0) X(a, 1) X(a, 2) X(a, 3) X(a, 4) X(a, 5) X(a, 6) X(a, 7)
#elif INK2L_ROWS == 9
#define INK2L_EACH_ROW(X, a) X(a, 0) X(a, 1) X(a, 2) X(a, 3) X(a, 4) X(a, 5) X(a, 6) X(a, 7) X(a, 8)
#elif INK2L_ROWS == 12
#define INK2L_EACH_ROW(X, a) \
    X(a, 0) X(a, 1) X(a, 2) X(a, 3) X(a, 4) X(a, 5) X(a, 6) X(a, 7) X(a, 8) X(a, 9) X(a, 10) X(a, 11)
#else
#define INK2L_EACH_ROW(X, a)                                                                           \
    X(a, 0) X(a, 1) X(a, 2) X(a, 3) X(a, 4) X(a, 5) X(a, 6) X(a, 7) X(a, 8) X(a, 9) X(a, 10) X(a, 11) \
    X(a, 12) X(a, 13) X(a, 14) X(a, 15)
#endif
#ifdef INK2L_EXACT
#define INK2L_INDEX(a, k)                      \
    const uint i##k = tid + INK2L_TG * k;      \
    const bool on##k = true;                   \
    const uint c##k = i##k;
#else
#define INK2L_INDEX(a, k)                      \
    const uint i##k = tid + INK2L_TG * k;      \
    const bool on##k = i##k < n;               \
    const uint c##k = min(i##k, n - 1);
#endif

struct Ink2LRMSShape {
    uint n;
    float eps;
    uint weighted, perRow;
};

kernel void ink2_rms_lean(device const float *x [[buffer(0)]],
                          device float *y [[buffer(1)]],
                          device const float *w0 [[buffer(2)]],
                          device const float *w1 [[buffer(3)]],
                          device const float *w2 [[buffer(4)]],
                          device const float *w3 [[buffer(5)]],
                          constant Ink2LRMSShape &p [[buffer(6)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[INK2L_SG];
    const uint row = group.x, slot = p.perRow ? row : 0, n = p.n;
    device const float *xr = x + ulong(row) * n;
    device float *yr = y + ulong(row) * n;
    device const float *w = ink2l_pick4(slot, w0, w1, w2, w3);
    const bool weighted = (p.weighted >> slot) & 1;
    INK2L_EACH_ROW(INK2L_INDEX, _)
#define INK2L_LOAD(a, k) const float x##k = xr[c##k], g##k = weighted ? w[c##k] : 1.0f;
    INK2L_EACH_ROW(INK2L_LOAD, _)
#undef INK2L_LOAD
    float ss = 0.0f;
#define INK2L_SQUARE(a, k) \
    if (on##k) ss = fma(x##k, x##k, ss);
    INK2L_EACH_ROW(INK2L_SQUARE, _)
#undef INK2L_SQUARE
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every thread folds the partial sums: cheaper than a second barrier
    // behind one thread's fold.
    const float s = ink2l_inv_rms(ink2l_fold(part), n, p.eps);
#define INK2L_STORE(a, k)          \
    if (on##k) {                   \
        float v = x##k * s;        \
        if (weighted) v *= g##k;   \
        yr[i##k] = v;              \
    }
    INK2L_EACH_ROW(INK2L_STORE, _)
#undef INK2L_STORE
}

struct Ink2LMixShape {
    uint n, states;
    uint norm, weighted;
    float eps, invSqrtN;
};

#define INK2L_EACH_STATE(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

#ifndef INK2L_MIX_HELD
#define INK2L_MIX_HELD 9
#endif

struct Ink2LStates {
    device const float *at[INK2L_STATES];
};

struct Ink2LMixShared {
    // Per state: its sum of squares, then its raw query dot, per SIMD group.
    float part[2 * INK2L_STATES * INK2L_SG];
    float scale[INK2L_STATES], share[INK2L_STATES];
    float outPart[INK2L_SG];
};

// The depth mix of ink2_mix. One SIMD group takes the softmax, a state per
// lane, and the rest read its result: the scores are a few dozen operations
// that every thread would otherwise repeat. S, when nonzero, is the state
// count at compile time, so no load waits on a state-count branch.
template <uint S>
static inline void ink2l_mix(thread const Ink2LStates &st, device const float *queries,
                             device const float *norm0, device const float *norm1, device const float *norm2,
                             device const float *norm3, device float *outs, constant Ink2LMixShape &p, uint q, uint tid,
                             uint lane, uint sg, threadgroup Ink2LMixShared &sh) {
    const uint n = p.n, states = S ? S : p.states;
    device const float *query = queries + ulong(q) * n;
    device float *out = outs + ulong(q) * n;
    const bool weighted = p.norm != 0 && ((p.weighted >> q) & 1);
    device const float *w = ink2l_pick4(q, norm0, norm1, norm2, norm3);
    INK2L_EACH_ROW(INK2L_INDEX, _)
#define INK2L_QUERY(a, k) const float q##k = query[c##k], g##k = weighted ? w[c##k] : 1.0f;
    INK2L_EACH_ROW(INK2L_QUERY, _)
#undef INK2L_QUERY

    // With a few states fixed at compile time every state's elements load at
    // once, held for both the scores and the blend; more would spill.
    const bool held = S != 0 && S <= INK2L_MIX_HELD;
#define INK2L_HOLD(a, k) const float x##a##_##k = held && S > a ? st.at[a][c##k] : 0.0f;
#define INK2L_HOLD_STATE(a) INK2L_EACH_ROW(INK2L_HOLD, a)
    INK2L_EACH_STATE(INK2L_HOLD_STATE)
#undef INK2L_HOLD_STATE
#undef INK2L_HOLD
#define INK2L_LOAD(a, k) const float x##k = held ? x##a##_##k : st.at[a][c##k];
#define INK2L_CHAIN(a, k)              \
    if (on##k) {                       \
        ss = fma(x##k, x##k, ss);      \
        dot = fma(x##k, q##k, dot);    \
    }
#define INK2L_GATHER(a)                                  \
    if (states > a) {                                    \
        float ss = 0.0f, dot = 0.0f;                     \
        INK2L_EACH_ROW(INK2L_LOAD, a)                    \
        INK2L_EACH_ROW(INK2L_CHAIN, a)                   \
        ss = simd_sum(ss);                               \
        dot = simd_sum(dot);                             \
        if (lane == 0) {                                 \
            sh.part[(2 * a) * INK2L_SG + sg] = ss;       \
            sh.part[(2 * a + 1) * INK2L_SG + sg] = dot;  \
        }                                                \
    }
    INK2L_EACH_STATE(INK2L_GATHER)
#undef INK2L_GATHER
#undef INK2L_CHAIN
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float rms = 0.0f, score = -INFINITY;
        if (lane < states) {
            rms = ink2l_inv_rms(ink2l_fold(sh.part + (2 * lane) * INK2L_SG), n, 1e-6f);
            score = (rms * ink2l_fold(sh.part + (2 * lane + 1) * INK2L_SG)) * p.invSqrtN;
        }
        const float e = exp(score - simd_max(score));
        // The shares sum in state order, as the scores do on one thread.
        float sum = 0.0f;
        for (ushort s = 0; s < states; s++) sum += simd_broadcast(e, s);
        if (lane < states) {
            sh.scale[lane] = rms;
            sh.share[lane] = e / sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#define INK2L_ZERO(a, k) float m##k = 0.0f;
    INK2L_EACH_ROW(INK2L_ZERO, _)
#undef INK2L_ZERO
#define INK2L_TERM(a, k) m##k += sw * (x##k * sc);
#define INK2L_BLEND(a)                                    \
    if (states > a) {                                     \
        const float sw = sh.share[a], sc = sh.scale[a];   \
        INK2L_EACH_ROW(INK2L_LOAD, a)                     \
        INK2L_EACH_ROW(INK2L_TERM, a)                     \
    }
    INK2L_EACH_STATE(INK2L_BLEND)
#undef INK2L_BLEND
#undef INK2L_TERM
#undef INK2L_LOAD
    if (p.norm == 0) {
#define INK2L_PUT(a, k) \
    if (on##k) out[i##k] = m##k;
        INK2L_EACH_ROW(INK2L_PUT, _)
#undef INK2L_PUT
        return;
    }
    float squares = 0.0f;
#define INK2L_SQUARE(a, k) \
    if (on##k) squares = fma(m##k, m##k, squares);
    INK2L_EACH_ROW(INK2L_SQUARE, _)
#undef INK2L_SQUARE
    squares = simd_sum(squares);
    if (lane == 0) sh.outPart[sg] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = ink2l_inv_rms(ink2l_fold(sh.outPart), n, p.eps);
#define INK2L_STORE(a, k)          \
    if (on##k) {                   \
        float v = m##k * s;        \
        if (weighted) v *= g##k;   \
        out[i##k] = v;             \
    }
    INK2L_EACH_ROW(INK2L_STORE, _)
#undef INK2L_STORE
}

#define INK2L_MIX_PARAMS                                       \
    INK2L_EACH_STATE(INK2L_PARAM)                              \
    device const float *queries [[buffer(16)]],                \
    device const float *norm0 [[buffer(17)]],                  \
    device const float *norm1 [[buffer(18)]],                  \
    device const float *norm2 [[buffer(19)]],                  \
    device const float *norm3 [[buffer(20)]],                  \
    device float *outs [[buffer(21)]],                         \
    constant Ink2LMixShape &p [[buffer(22)]],
#define INK2L_PARAM(k) device const float *st##k [[buffer(k)]],
#define INK2L_STATE_ITEM(k) st##k,

// ink2_mix, bound alike.
template <uint S>
kernel void ink2_mix_lean(INK2L_MIX_PARAMS
                          uint3 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Ink2LMixShared sh;
    const Ink2LStates st = {{INK2L_EACH_STATE(INK2L_STATE_ITEM)}};
    ink2l_mix<S>(st, queries, norm0, norm1, norm2, norm3, outs, p, group.x, tid, lane, sg, sh);
}

#define INK2L_MIX_KERNEL(S, NAME)                                                                       \
    template [[host_name(NAME)]] kernel void ink2_mix_lean<S>(INK2L_MIX_PARAMS uint3, uint, uint, uint);
#undef INK2L_PARAM
#define INK2L_PARAM(k) device const float *,
#undef INK2L_MIX_PARAMS
#define INK2L_MIX_PARAMS                                                                                \
    INK2L_EACH_STATE(INK2L_PARAM) device const float *, device const float *, device const float *,     \
        device const float *, device const float *, device float *, constant Ink2LMixShape &,
INK2L_MIX_KERNEL(0, "ink2_mix_lean")
#define INK2L_MIX_FIXED(S) INK2L_MIX_KERNEL(S, "ink2_mix_lean_" #S)
INK2L_MIX_FIXED(1)
INK2L_MIX_FIXED(2)
INK2L_MIX_FIXED(3)
INK2L_MIX_FIXED(4)
INK2L_MIX_FIXED(5)
INK2L_MIX_FIXED(6)
INK2L_MIX_FIXED(7)
INK2L_MIX_FIXED(8)
INK2L_MIX_FIXED(9)
INK2L_MIX_FIXED(10)
INK2L_MIX_FIXED(11)
INK2L_MIX_FIXED(12)
INK2L_MIX_FIXED(13)
INK2L_MIX_FIXED(14)
INK2L_MIX_FIXED(15)
INK2L_MIX_FIXED(16)
#undef INK2L_MIX_FIXED
#undef INK2L_MIX_KERNEL

#undef INK2L_STATE_ITEM
#undef INK2L_PARAM
#undef INK2L_MIX_PARAMS

struct Ink2LRouteShape {
    uint experts, slots, biased;
    float weightScale;
};

// The routing of ink2_route for up to 4 * INK2L_ROUTE_THREADS experts: a lane's
// four experts are named, so a round is register selects and SIMD reductions.
kernel void ink2_route_lean(device const float *logits [[buffer(0)]],
                            device const float *bias [[buffer(1)]],
                            device int *ids [[buffer(2)]],
                            device float *wts [[buffer(3)]],
                            constant Ink2LRouteShape &p [[buffer(4)]],
                            uint lane [[thread_index_in_threadgroup]],
                            uint tok [[threadgroup_position_in_grid]]) {
    logits += ulong(tok) * p.experts;
    ids += ulong(tok) * p.slots;
    wts += ulong(tok) * p.slots;
    const uint none = 0xffffffffu;
#define INK2L_EACH_EXPERT(X) X(0) X(1) X(2) X(3)
#define INK2L_LOAD(c)                                              \
    const uint e##c = lane + c * INK2L_ROUTE_THREADS;              \
    bool open##c = e##c < p.experts;                               \
    float prob##c = 0.0f, sel##c = 0.0f;                           \
    if (open##c) {                                                 \
        prob##c = ink2l_sigmoid(logits[e##c]);                     \
        sel##c = p.biased ? prob##c + bias[e##c] : prob##c;        \
    }
    INK2L_EACH_EXPERT(INK2L_LOAD)
#undef INK2L_LOAD
    float picked[INK2L_MAX_TOPK];
    float sum = 0.0f;
    for (uint t = 0; t < p.slots; t++) {
        float best = -INFINITY;
        uint at = none;
#define INK2L_OFFER(c)                                         \
    if (open##c && (at == none || sel##c > best)) {            \
        best = sel##c;                                         \
        at = e##c;                                             \
    }
        INK2L_EACH_EXPERT(INK2L_OFFER)
#undef INK2L_OFFER
        const float top = simd_max(best);
        const uint win = simd_min(at != none && best == top ? at : none);
        float mine = 0.0f;
#define INK2L_TAKE(c)          \
    if (win == e##c) {         \
        open##c = false;       \
        mine = prob##c;        \
    }
        INK2L_EACH_EXPERT(INK2L_TAKE)
#undef INK2L_TAKE
#undef INK2L_EACH_EXPERT
        // Every other lane offers zero, below any probability.
        const float pr = simd_max(mine);
        sum += pr;
        if (lane == 0) {
            picked[t] = pr;
            ids[t] = int(win);
        }
    }
    if (lane != 0) return;
    for (uint t = 0; t < p.slots; t++) {
        float v = picked[t] / (sum + 1e-20f);
        if (p.weightScale != 0.0f) v *= p.weightScale;
        wts[t] = v;
    }
}

struct Ink2LCtrlShape {
    uint n, rank;
};

// A controller weight is float32, or bfloat16 when every value is exactly one:
// its float32 is the bfloat16 bits followed by sixteen zero bits, so widening
// it is the same value at half the bytes read.
static inline float ink2l_wt(device const float *w, uint i) { return w[i]; }
static inline float ink2l_wt(device const ushort *w, uint i) { return as_type<float>(uint(w[i]) << 16); }

// ink2_ctrl_down with each thread's columns named, bound alike.
template <typename W>
kernel void ink2_ctrl_down_lean(device const float *cur0 [[buffer(0)]],
                                device const float *cur1 [[buffer(1)]],
                                device const float *cur2 [[buffer(2)]],
                                device const float *cur3 [[buffer(3)]],
                                device const float *msg0 [[buffer(4)]],
                                device const float *msg1 [[buffer(5)]],
                                device const float *msg2 [[buffer(6)]],
                                device const float *msg3 [[buffer(7)]],
                                device const W *down0 [[buffer(8)]],
                                device const W *down1 [[buffer(9)]],
                                device const W *down2 [[buffer(10)]],
                                device const W *down3 [[buffer(11)]],
                                device float *g [[buffer(12)]],
                                constant Ink2LCtrlShape &p [[buffer(13)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4 * INK2L_SG];
    const uint r = group.x, j = group.y, n = p.n;
    device const float *cur = ink2l_pick4(j, cur0, cur1, cur2, cur3);
    device const float *msg = ink2l_pick4(j, msg0, msg1, msg2, msg3);
    device const W *wc = ink2l_pick4(j, down0, down1, down2, down3) + ulong(r) * 2 * n;
    device const W *wm = wc + n;
    INK2L_EACH_ROW(INK2L_INDEX, _)
#define INK2L_LOAD(a, k) \
    const float x##k = cur[c##k], y##k = msg[c##k], u##k = ink2l_wt(wc, c##k), v##k = ink2l_wt(wm, c##k);
    INK2L_EACH_ROW(INK2L_LOAD, _)
#undef INK2L_LOAD
    float ssc = 0.0f, ssm = 0.0f, dc = 0.0f, dm = 0.0f;
#define INK2L_CHAIN(a, k)              \
    if (on##k) {                       \
        ssc = fma(x##k, x##k, ssc);    \
        ssm = fma(y##k, y##k, ssm);    \
        dc = fma(u##k, x##k, dc);      \
        dm = fma(v##k, y##k, dm);      \
    }
    INK2L_EACH_ROW(INK2L_CHAIN, _)
#undef INK2L_CHAIN
    ssc = simd_sum(ssc);
    ssm = simd_sum(ssm);
    dc = simd_sum(dc);
    dm = simd_sum(dm);
    if (lane == 0) {
        part[sg] = ssc;
        part[INK2L_SG + sg] = ssm;
        part[2 * INK2L_SG + sg] = dc;
        part[3 * INK2L_SG + sg] = dm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float sc = ink2l_inv_rms(ink2l_fold(part), n, 1e-6f), sm = ink2l_inv_rms(ink2l_fold(part + INK2L_SG), n, 1e-6f);
        g[ulong(j) * p.rank + r] = fma(sc, ink2l_fold(part + 2 * INK2L_SG), sm * ink2l_fold(part + 3 * INK2L_SG));
    }
}

#define INK2L_CTRL_DOWN(W, NAME)                                                                                   \
    template [[host_name(NAME)]] kernel void ink2_ctrl_down_lean<W>(                                               \
        device const float *, device const float *, device const float *, device const float *, device const float *, \
        device const float *, device const float *, device const float *, device const W *, device const W *,       \
        device const W *, device const W *, device float *, constant Ink2LCtrlShape &, uint3, uint, uint, uint);
INK2L_CTRL_DOWN(float, "ink2_ctrl_down_lean")
INK2L_CTRL_DOWN(ushort, "ink2_ctrl_down_lean_bf16")
#undef INK2L_CTRL_DOWN

// ink2_ctrl_apply for a rank fixed at compile time, so a thread's loads of
// both rank-long rows issue together instead of four per loop trip.
template <uint R, typename W>
kernel void ink2_ctrl_apply_lean(device const float *cur0 [[buffer(0)]],
                                 device const float *cur1 [[buffer(1)]],
                                 device const float *cur2 [[buffer(2)]],
                                 device const float *cur3 [[buffer(3)]],
                                 device const float *msg0 [[buffer(4)]],
                                 device const float *msg1 [[buffer(5)]],
                                 device const float *msg2 [[buffer(6)]],
                                 device const float *msg3 [[buffer(7)]],
                                 device const W *gain0 [[buffer(8)]],
                                 device const W *gain1 [[buffer(9)]],
                                 device const W *gain2 [[buffer(10)]],
                                 device const W *gain3 [[buffer(11)]],
                                 device const W *forget0 [[buffer(12)]],
                                 device const W *forget1 [[buffer(13)]],
                                 device const W *forget2 [[buffer(14)]],
                                 device const W *forget3 [[buffer(15)]],
                                 device const float *g [[buffer(16)]],
                                 device float *out [[buffer(17)]],
                                 constant Ink2LCtrlShape &p [[buffer(18)]],
                                 uint2 at [[thread_position_in_grid]]) {
    const uint i = at.x, j = at.y;
    device const float *gj = g + ulong(j) * R;
    device const W *gr = ink2l_pick4(j, gain0, gain1, gain2, gain3) + ulong(i) * R;
    device const W *fr = ink2l_pick4(j, forget0, forget1, forget2, forget3) + ulong(i) * R;
    float4 gain4 = float4(0.0f), forget4 = float4(0.0f);
    uint c = 0;
#pragma unroll
    for (; c + 4 <= R; c += 4) {
        const float4 gv = float4(gj[c], gj[c + 1], gj[c + 2], gj[c + 3]);
        gain4 = fma(float4(ink2l_wt(gr, c), ink2l_wt(gr, c + 1), ink2l_wt(gr, c + 2), ink2l_wt(gr, c + 3)), gv, gain4);
        forget4 = fma(float4(ink2l_wt(fr, c), ink2l_wt(fr, c + 1), ink2l_wt(fr, c + 2), ink2l_wt(fr, c + 3)), gv, forget4);
    }
    float gain = (gain4.x + gain4.y) + (gain4.z + gain4.w), forget = (forget4.x + forget4.y) + (forget4.z + forget4.w);
#pragma unroll
    for (; c < R; c++) {
        gain = fma(ink2l_wt(gr, c), gj[c], gain);
        forget = fma(ink2l_wt(fr, c), gj[c], forget);
    }
    const float x = ink2l_pick4(j, cur0, cur1, cur2, cur3)[i], m = ink2l_pick4(j, msg0, msg1, msg2, msg3)[i];
    const float f = tanh(forget), t = tanh(gain);
    out[ulong(j) * p.n + i] = (x - f * x) + t * m;
}

#define INK2L_CTRL_APPLY_AS(R, W, NAME)                                                                              \
    template [[host_name(NAME)]] kernel void ink2_ctrl_apply_lean<R, W>(                                              \
        device const float *, device const float *, device const float *, device const float *, device const float *, \
        device const float *, device const float *, device const float *, device const W *, device const W *,       \
        device const W *, device const W *, device const W *, device const W *, device const W *, device const W *, \
        device const float *, device float *, constant Ink2LCtrlShape &, uint2);
#define INK2L_CTRL_APPLY(R)                                        \
    INK2L_CTRL_APPLY_AS(R, float, "ink2_ctrl_apply_lean_" #R)      \
    INK2L_CTRL_APPLY_AS(R, ushort, "ink2_ctrl_apply_lean_bf16_" #R)
INK2L_CTRL_APPLY(16)
INK2L_CTRL_APPLY(32)
INK2L_CTRL_APPLY(48)
INK2L_CTRL_APPLY(64)
INK2L_CTRL_APPLY(96)
INK2L_CTRL_APPLY(128)
