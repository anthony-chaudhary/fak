#include <metal_stdlib>
using namespace metal;

// The sums below keep the host forward's evaluation order, so a multiply must
// not fuse into the add that follows it; fma() marks the fusions that are meant.
#pragma STDC FP_CONTRACT OFF

// Threads of a reducing threadgroup, in 32-wide SIMD groups.
#define INK2_TG 256
#define INK2_SG (INK2_TG / 32)
#define INK2_MAX_STATES 16
#define INK2_MAX_EXPERTS 512
#define INK2_MAX_TOPK 16
#define INK2_ROUTE_THREADS 32
#define INK2_ROUTE_STRIDE (INK2_MAX_EXPERTS / INK2_ROUTE_THREADS)

// MSL has no float64, so every sum the host forward takes in float64 is a
// float32 one here: per thread in column order, then across the SIMD group,
// then over the SIMD groups in order.

static inline float ink2_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float ink2_inv_rms(float sumSquares, uint n, float eps) {
    return 1.0f / sqrt(sumSquares / float(n) + eps);
}

template <typename T> static inline T ink2_pick4(uint i, T a, T b, T c, T d) {
    return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;
}

struct Ink2RMSShared {
    float part[INK2_SG];
    float scale;
};

// y = x * s (* w) with s = 1/sqrt(mean(x^2) + eps), one threadgroup per row.
// x may be y: a thread reads an element before it writes it, and every sum is
// taken before the first write.
static void ink2_rms_row(device const float *x, device const float *w, bool weighted, device float *y, uint n,
                         float eps, uint tid, uint lane, uint sg, threadgroup Ink2RMSShared &sh) {
    float ss = 0.0f;
    for (uint i = tid; i < n; i += INK2_TG) ss = fma(x[i], x[i], ss);
    ss = simd_sum(ss);
    if (lane == 0) sh.part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint g = 0; g < INK2_SG; g++) total += sh.part[g];
        sh.scale = ink2_inv_rms(total, n, eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = sh.scale;
    for (uint i = tid; i < n; i += INK2_TG) {
        float v = x[i] * s;
        if (weighted) v *= w[i];
        y[i] = v;
    }
}

struct Ink2RMSShape {
    uint n;
    float eps;
    // Bit i: weight slot i is bound. Row r reads slot r when perRow, else slot 0.
    uint weighted, perRow;
};

kernel void ink2_rms(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     device const float *w0 [[buffer(2)]],
                     device const float *w1 [[buffer(3)]],
                     device const float *w2 [[buffer(4)]],
                     device const float *w3 [[buffer(5)]],
                     constant Ink2RMSShape &p [[buffer(6)]],
                     uint3 group [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Ink2RMSShared sh;
    const uint row = group.x, slot = p.perRow ? row : 0;
    ink2_rms_row(x + ulong(row) * p.n, ink2_pick4(slot, w0, w1, w2, w3), (p.weighted >> slot) & 1,
                 y + ulong(row) * p.n, p.n, p.eps, tid, lane, sg, sh);
}

// The halves of a controller pair row: rms(cur) then rms(msg), unweighted.
kernel void ink2_ctrl_pair(device const float *cur [[buffer(0)]],
                           device const float *msg [[buffer(1)]],
                           device float *pair [[buffer(2)]],
                           constant uint &n [[buffer(3)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Ink2RMSShared sh;
    device const float *x = group.x == 0 ? cur : msg;
    ink2_rms_row(x, x, false, pair + ulong(group.x) * n, n, 1e-6f, tid, lane, sg, sh);
}

struct Ink2MixShape {
    uint n, states;
    // norm: RMS-normalize each mixed row by eps; weighted bit q: and scale
    // query q's row by its bound weight.
    uint norm, weighted;
    float eps, invSqrtN;
};

struct Ink2MixShared {
    // Per SIMD group: each state's sum of squares, then each state's query dot.
    float part[2 * INK2_MAX_STATES * INK2_SG];
    float scale[INK2_MAX_STATES];
    // The raw query dots, then the softmax weights.
    float weight[INK2_MAX_STATES];
    float outPart[INK2_SG];
    float outScale;
};

#define INK2_STATE_ARGS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

// Depth mix of one query (a threadgroup per query, which also takes the
// states' norms): every state RMS-normalized (eps 1e-6), scored against the
// query, softmaxed, and summed in state order. One pass gathers every sum of
// squares and raw dot, so a score is scale * raw dot. Phases hand over through
// threadgroup memory alone and no thread keeps an array across a barrier,
// which is why each phase lists the states again.
// Technique adapted from incoai/splash runtime/metal/kernels/shared/block_residual_mix.metal@2afc7337 (Apache-2.0).
kernel void ink2_mix(
#define INK2_STATE_PARAM(i) device const float *s##i [[buffer(i)]],
    INK2_STATE_ARGS(INK2_STATE_PARAM)
#undef INK2_STATE_PARAM
    device const float *queries [[buffer(16)]],
    device const float *norm0 [[buffer(17)]],
    device const float *norm1 [[buffer(18)]],
    device const float *norm2 [[buffer(19)]],
    device const float *norm3 [[buffer(20)]],
    device float *outs [[buffer(21)]],
    constant Ink2MixShape &p [[buffer(22)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Ink2MixShared sh;
#define INK2_STATE_ITEM(i) s##i,
#define INK2_STATES device const float *st[INK2_MAX_STATES] = {INK2_STATE_ARGS(INK2_STATE_ITEM)}
    const uint n = p.n, states = p.states;
    device const float *query = queries + ulong(group.x) * n;
    device float *out = outs + ulong(group.x) * n;
    {
        INK2_STATES;
        float acc[2 * INK2_MAX_STATES];
        for (uint v = 0; v < 2 * INK2_MAX_STATES; v++) acc[v] = 0.0f;
        for (uint i = tid; i < n; i += INK2_TG) {
            const float qv = query[i];
            for (uint s = 0; s < INK2_MAX_STATES; s++)
                if (s < states) {
                    const float x = st[s][i];
                    acc[s] = fma(x, x, acc[s]);
                    acc[INK2_MAX_STATES + s] = fma(x, qv, acc[INK2_MAX_STATES + s]);
                }
        }
        for (uint v = 0; v < 2 * INK2_MAX_STATES; v++) {
            const float sum = simd_sum(acc[v]);
            if (lane == 0) sh.part[v * INK2_SG + sg] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2 * INK2_MAX_STATES) {
        float total = 0.0f;
        for (uint g = 0; g < INK2_SG; g++) total += sh.part[tid * INK2_SG + g];
        if (tid < INK2_MAX_STATES) sh.scale[tid] = ink2_inv_rms(total, n, 1e-6f);
        else sh.weight[tid - INK2_MAX_STATES] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sc[INK2_MAX_STATES];
        float top = -INFINITY;
        for (uint s = 0; s < states; s++) {
            sc[s] = (sh.scale[s] * sh.weight[s]) * p.invSqrtN;
            top = max(top, sc[s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < states; s++) {
            sc[s] = exp(sc[s] - top);
            sum += sc[s];
        }
        for (uint s = 0; s < states; s++) sh.weight[s] = sc[s] / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float squares = 0.0f;
    {
        INK2_STATES;
        for (uint i = tid; i < n; i += INK2_TG) {
            float a = 0.0f;
            for (uint s = 0; s < INK2_MAX_STATES; s++)
                if (s < states) a += sh.weight[s] * (st[s][i] * sh.scale[s]);
            out[i] = a;
            squares = fma(a, a, squares);
        }
    }
#undef INK2_STATES
#undef INK2_STATE_ITEM
    if (p.norm == 0) return;
    squares = simd_sum(squares);
    if (lane == 0) sh.outPart[sg] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint g = 0; g < INK2_SG; g++) total += sh.outPart[g];
        sh.outScale = ink2_inv_rms(total, n, p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = sh.outScale;
    const bool weighted = (p.weighted >> group.x) & 1;
    device const float *w = ink2_pick4(group.x, norm0, norm1, norm2, norm3);
    for (uint i = tid; i < n; i += INK2_TG) {
        float v = out[i] * s;
        if (weighted) v *= w[i];
        out[i] = v;
    }
}

struct Ink2CtrlShape {
    uint n, rank;
};

// g[j][r] = Down_j[r,:] . [rms(cur_j), rms(msg_j)] with both RMS scales
// (eps 1e-6) folded out of the dot: a threadgroup per (rank row, controller).
// Technique adapted from incoai/splash runtime/metal/kernels/shared/message_gate.metal@2afc7337 (Apache-2.0).
kernel void ink2_ctrl_down(device const float *cur0 [[buffer(0)]],
                           device const float *cur1 [[buffer(1)]],
                           device const float *cur2 [[buffer(2)]],
                           device const float *cur3 [[buffer(3)]],
                           device const float *msg0 [[buffer(4)]],
                           device const float *msg1 [[buffer(5)]],
                           device const float *msg2 [[buffer(6)]],
                           device const float *msg3 [[buffer(7)]],
                           device const float *down0 [[buffer(8)]],
                           device const float *down1 [[buffer(9)]],
                           device const float *down2 [[buffer(10)]],
                           device const float *down3 [[buffer(11)]],
                           device float *g [[buffer(12)]],
                           constant Ink2CtrlShape &p [[buffer(13)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4 * INK2_SG];
    const uint r = group.x, j = group.y, n = p.n;
    device const float *cur = ink2_pick4(j, cur0, cur1, cur2, cur3);
    device const float *msg = ink2_pick4(j, msg0, msg1, msg2, msg3);
    device const float *wc = ink2_pick4(j, down0, down1, down2, down3) + ulong(r) * 2 * n;
    device const float *wm = wc + n;
    float ssc = 0.0f, ssm = 0.0f, dc = 0.0f, dm = 0.0f;
    for (uint i = tid; i < n; i += INK2_TG) {
        const float c = cur[i], m = msg[i];
        ssc = fma(c, c, ssc);
        ssm = fma(m, m, ssm);
        dc = fma(wc[i], c, dc);
        dm = fma(wm[i], m, dm);
    }
    ssc = simd_sum(ssc);
    ssm = simd_sum(ssm);
    dc = simd_sum(dc);
    dm = simd_sum(dm);
    if (lane == 0) {
        part[sg] = ssc;
        part[INK2_SG + sg] = ssm;
        part[2 * INK2_SG + sg] = dc;
        part[3 * INK2_SG + sg] = dm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint v = 0; v < 4; v++)
            for (uint s = 0; s < INK2_SG; s++) t[v] += part[v * INK2_SG + s];
        const float sc = ink2_inv_rms(t[0], n, 1e-6f), sm = ink2_inv_rms(t[1], n, 1e-6f);
        g[ulong(j) * p.rank + r] = fma(sc, t[2], sm * t[3]);
    }
}

// out[j][i] = cur - tanh(F_j[i,:].g_j)*cur + tanh(G_j[i,:].g_j)*msg, one
// thread per output of each controller.
kernel void ink2_ctrl_apply(device const float *cur0 [[buffer(0)]],
                            device const float *cur1 [[buffer(1)]],
                            device const float *cur2 [[buffer(2)]],
                            device const float *cur3 [[buffer(3)]],
                            device const float *msg0 [[buffer(4)]],
                            device const float *msg1 [[buffer(5)]],
                            device const float *msg2 [[buffer(6)]],
                            device const float *msg3 [[buffer(7)]],
                            device const float *gain0 [[buffer(8)]],
                            device const float *gain1 [[buffer(9)]],
                            device const float *gain2 [[buffer(10)]],
                            device const float *gain3 [[buffer(11)]],
                            device const float *forget0 [[buffer(12)]],
                            device const float *forget1 [[buffer(13)]],
                            device const float *forget2 [[buffer(14)]],
                            device const float *forget3 [[buffer(15)]],
                            device const float *g [[buffer(16)]],
                            device float *out [[buffer(17)]],
                            constant Ink2CtrlShape &p [[buffer(18)]],
                            uint2 at [[thread_position_in_grid]]) {
    const uint i = at.x, j = at.y, rank = p.rank;
    device const float *gj = g + ulong(j) * rank;
    device const float *gr = ink2_pick4(j, gain0, gain1, gain2, gain3) + ulong(i) * rank;
    device const float *fr = ink2_pick4(j, forget0, forget1, forget2, forget3) + ulong(i) * rank;
    // Four running sums keep the rank-long dot off one dependency chain.
    float4 gain4 = float4(0.0f), forget4 = float4(0.0f);
    uint c = 0;
    for (; c + 4 <= rank; c += 4) {
        const float4 gv = float4(gj[c], gj[c + 1], gj[c + 2], gj[c + 3]);
        gain4 = fma(float4(gr[c], gr[c + 1], gr[c + 2], gr[c + 3]), gv, gain4);
        forget4 = fma(float4(fr[c], fr[c + 1], fr[c + 2], fr[c + 3]), gv, forget4);
    }
    float gain = (gain4.x + gain4.y) + (gain4.z + gain4.w), forget = (forget4.x + forget4.y) + (forget4.z + forget4.w);
    for (; c < rank; c++) {
        gain = fma(gr[c], gj[c], gain);
        forget = fma(fr[c], gj[c], forget);
    }
    const float x = ink2_pick4(j, cur0, cur1, cur2, cur3)[i], m = ink2_pick4(j, msg0, msg1, msg2, msg3)[i];
    const float f = tanh(forget), t = tanh(gain);
    out[ulong(j) * p.n + i] = (x - f * x) + t * m;
}

kernel void ink2_sigmoid_mul(device float *x [[buffer(0)]],
                             device const float *gate [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    x[i] = x[i] * ink2_sigmoid(gate[i]);
}

struct Ink2RouteShape {
    uint experts, slots, biased;
    float weightScale;
};

// MoE routing: sigmoid probabilities, top-k of probs + bias with ties to the
// lower index, weights renormalized over the selected probabilities and
// scaled. One SIMD group: lane l keeps experts l, l+32, ... in registers and a
// round is three SIMD reductions, so no round waits on a barrier.
// Adapted from packed_route, which ranks the same way over a fused
// [bias, scale, scale] table and overwrites its logits.
kernel void ink2_route(device const float *logits [[buffer(0)]],
                       device const float *bias [[buffer(1)]],
                       device int *ids [[buffer(2)]],
                       device float *wts [[buffer(3)]],
                       constant Ink2RouteShape &p [[buffer(4)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint tok [[threadgroup_position_in_grid]]) {
    // A batched route runs token tok's logits row into its slots.
    logits += ulong(tok) * p.experts;
    ids += ulong(tok) * p.slots;
    wts += ulong(tok) * p.slots;
    const uint none = 0xffffffffu;
    const uint strides = (p.experts + INK2_ROUTE_THREADS - 1) / INK2_ROUTE_THREADS;
    float probs[INK2_ROUTE_STRIDE], sel[INK2_ROUTE_STRIDE];
    for (uint s = 0; s < INK2_ROUTE_STRIDE; s++) {
        const uint e = lane + s * INK2_ROUTE_THREADS;
        probs[s] = 0.0f;
        sel[s] = 0.0f;
        if (e < p.experts) {
            probs[s] = ink2_sigmoid(logits[e]);
            sel[s] = p.biased ? probs[s] + bias[e] : probs[s];
        }
    }
    float picked[INK2_MAX_TOPK];
    for (uint t = 0; t < INK2_MAX_TOPK; t++) picked[t] = 0.0f;
    // Bit s: this lane's expert s is already routed.
    uint taken = 0;
    float sum = 0.0f;
    for (uint t = 0; t < p.slots; t++) {
        float best = -INFINITY;
        uint at = none;
        for (uint s = 0; s < strides; s++) {
            const uint e = lane + s * INK2_ROUTE_THREADS;
            if (e < p.experts && ((taken >> s) & 1) == 0 && (at == none || sel[s] > best)) {
                best = sel[s];
                at = e;
            }
        }
        const float top = simd_max(best);
        const uint win = simd_min(at != none && best == top ? at : none);
        const bool mine = win != none && win % INK2_ROUTE_THREADS == lane;
        const uint slot = mine ? win / INK2_ROUTE_THREADS : 0;
        if (mine) taken |= 1u << slot;
        // Every other lane offers zero, below any probability.
        const float pr = simd_max(mine ? probs[slot] : 0.0f);
        picked[t] = pr;
        sum += pr;
        if (lane == 0) ids[t] = int(win);
    }
    if (lane != 0) return;
    for (uint t = 0; t < p.slots; t++) {
        float v = picked[t] / (sum + 1e-20f);
        if (p.weightScale != 0.0f) v *= p.weightScale;
        wts[t] = v;
    }
}

struct Ink2SituShape {
    uint width, actStride;
    float beta, linearBeta;
};

// SITU activation of the routed slots: act = beta*tanh(g/beta)*sigmoid(g) *
// lb*tanh(u/lb) over each slot's [gate | up] row.
// Adapted from packed_situ, which also folds a per-slot scale into the row.
kernel void ink2_situ(device const float *gu [[buffer(0)]],
                      device float *act [[buffer(1)]],
                      constant Ink2SituShape &p [[buffer(2)]],
                      uint2 at [[thread_position_in_grid]]) {
    device const float *row = gu + ulong(at.y) * 2 * p.width;
    const float g = row[at.x];
    float u = row[p.width + at.x];
    const float a = p.beta * tanh(g / p.beta) * ink2_sigmoid(g);
    if (p.linearBeta > 0.0f) u = p.linearBeta * tanh(u / p.linearBeta);
    act[ulong(at.y) * p.actStride + at.x] = a * u;
}

struct Ink2CombineShape {
    uint slots, n;
};

// out[j] = sum over routed slots, in routing order, of weight * down[slot][j].
// Adapted from packed_expert_sum, which reads its weights from a fused
// per-slot scale table.
kernel void ink2_combine(device const float *down [[buffer(0)]],
                         device const float *wts [[buffer(1)]],
                         device float *out [[buffer(2)]],
                         constant Ink2CombineShape &p [[buffer(3)]],
                         uint2 at [[thread_position_in_grid]]) {
    // A batched combine runs token at.y's slots into its row.
    const uint j = at.x;
    down += ulong(at.y) * p.slots * p.n;
    wts += ulong(at.y) * p.slots;
    out += ulong(at.y) * p.n;
    float acc = 0.0f;
    for (uint t = 0; t < p.slots; t++) acc += wts[t] * down[ulong(t) * p.n + j];
    out[j] = acc;
}

kernel void ink2_add(device const float *a [[buffer(0)]],
                     device const float *b [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

// The argmax runs as three launches over slot = the caller's four index words:
// slot[0] collects the lowest index holding the largest key, slot[2] the key.

// Order-preserving map of a float onto unsigned integers.
static inline uint ink2_argkey(float v) {
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

kernel void ink2_argmax_begin(device uint *slot [[buffer(0)]]) {
    slot[0] = 0xffffffffu;
    slot[2] = 0u;
}

struct Ink2ArgmaxShape {
    uint n;
    float softcap;
};

// Softcaps the logits in place (softcap != 0) and folds the largest key.
kernel void ink2_softcap_max(device float *logits [[buffer(0)]],
                             device atomic_uint *slot [[buffer(1)]],
                             constant Ink2ArgmaxShape &p [[buffer(2)]],
                             uint i [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint key = 0u;
    if (i < p.n) {
        float v = logits[i];
        if (p.softcap != 0.0f) {
            v = p.softcap * tanh(v / p.softcap);
            logits[i] = v;
        }
        key = ink2_argkey(v);
    }
    key = simd_max(key);
    if (lane == 0) atomic_fetch_max_explicit(&slot[2], key, memory_order_relaxed);
}

kernel void ink2_argmax_index(device const float *logits [[buffer(0)]],
                              device atomic_uint *slot [[buffer(1)]],
                              constant Ink2ArgmaxShape &p [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < p.n && ink2_argkey(logits[i]) == atomic_load_explicit(&slot[2], memory_order_relaxed))
        atomic_fetch_min_explicit(&slot[0], i, memory_order_relaxed);
}
