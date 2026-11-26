// Token-axis batches of the resident inkling2_loop decode kernels. Token t of
// a T-token chunk sits at position pos0 + t and every activation is
// [T][width] rows. Each kernel is its decode kernel of ink2_ops.metal or
// ink2_conv.metal with a token axis on the grid, so a row keeps that kernel's
// arithmetic and order and equals it bit for bit.

#include <metal_stdlib>
using namespace metal;

// The sums below keep the host forward's evaluation order, so a multiply must
// not fuse into the add that follows it; fma() marks the fusions that are meant.
#pragma STDC FP_CONTRACT OFF

// The limits and helpers of ink2_ops.metal, repeated: a source compiles alone.
#define INK2_TG 256
#define INK2_SG (INK2_TG / 32)
#define INK2_MAX_STATES 16
#define INK2_MAX_EXPERTS 512
#define INK2_MAX_TOPK 16
#define INK2_ROUTE_THREADS 32
#define INK2_ROUTE_STRIDE (INK2_MAX_EXPERTS / INK2_ROUTE_THREADS)
#define INK2P_MAX_KN 8

static inline float ink2_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float ink2_inv_rms(float sumSquares, uint n, float eps) {
    return 1.0f / sqrt(sumSquares / float(n) + eps);
}

template <typename T> static inline T ink2_pick4(uint i, T a, T b, T c, T d) {
    return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;
}

struct Ink2PMixShape {
    uint n, states, tokens;
    float invSqrtN;
};

struct Ink2PMixShared {
    float part[2 * INK2_MAX_STATES * INK2_SG];
    float scale[INK2_MAX_STATES];
    float weight[INK2_MAX_STATES];
};

#define INK2_STATE_ARGS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

// Depth mix, a threadgroup per (query, token): token t's states are row t of
// each [T][n] state and the output is query-major [nq][T][n].
// Technique adapted from incoai/splash runtime/metal/kernels/shared/block_residual_mix.metal@2afc7337 (Apache-2.0).
kernel void ink2p_mix(
#define INK2_STATE_PARAM(i) device const float *s##i [[buffer(i)]],
    INK2_STATE_ARGS(INK2_STATE_PARAM)
#undef INK2_STATE_PARAM
    device const float *queries [[buffer(16)]],
    device float *outs [[buffer(17)]],
    constant Ink2PMixShape &p [[buffer(18)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Ink2PMixShared sh;
#define INK2_STATE_ITEM(i) s##i,
#define INK2_STATES device const float *st[INK2_MAX_STATES] = {INK2_STATE_ARGS(INK2_STATE_ITEM)}
    const uint n = p.n, states = p.states;
    const ulong toff = ulong(group.y) * n;
    device const float *query = queries + ulong(group.x) * n;
    device float *out = outs + (ulong(group.x) * p.tokens + group.y) * n;
    {
        INK2_STATES;
        float acc[2 * INK2_MAX_STATES];
        for (uint v = 0; v < 2 * INK2_MAX_STATES; v++) acc[v] = 0.0f;
        for (uint i = tid; i < n; i += INK2_TG) {
            const float qv = query[i];
            for (uint s = 0; s < INK2_MAX_STATES; s++)
                if (s < states) {
                    const float x = st[s][toff + i];
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
    {
        INK2_STATES;
        for (uint i = tid; i < n; i += INK2_TG) {
            float a = 0.0f;
            for (uint s = 0; s < INK2_MAX_STATES; s++)
                if (s < states) a += sh.weight[s] * (st[s][toff + i] * sh.scale[s]);
            out[i] = a;
        }
    }
#undef INK2_STATES
#undef INK2_STATE_ITEM
}

struct Ink2PCtrlShape {
    uint n, rank, tokens;
};

// g_j[t][r] = Down_j[r,:] . [rms(cur_j[t]), rms(msg_j[t])]: a threadgroup per
// (rank row, token, controller).
// Technique adapted from incoai/splash runtime/metal/kernels/shared/message_gate.metal@2afc7337 (Apache-2.0).
kernel void ink2p_ctrl_down(device const float *cur0 [[buffer(0)]],
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
                            device float *g0 [[buffer(12)]],
                            device float *g1 [[buffer(13)]],
                            device float *g2 [[buffer(14)]],
                            device float *g3 [[buffer(15)]],
                            constant Ink2PCtrlShape &p [[buffer(16)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4 * INK2_SG];
    const uint r = group.x, tok = group.y, j = group.z, n = p.n;
    const ulong toff = ulong(tok) * n;
    device const float *cur = ink2_pick4(j, cur0, cur1, cur2, cur3) + toff;
    device const float *msg = ink2_pick4(j, msg0, msg1, msg2, msg3) + toff;
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
        ink2_pick4(j, g0, g1, g2, g3)[ulong(tok) * p.rank + r] = fma(sc, t[2], sm * t[3]);
    }
}

// out_j[t][i] = cur - tanh(F_j[i,:].g_j[t])*cur + tanh(G_j[i,:].g_j[t])*msg,
// one thread per (output, token, controller).
kernel void ink2p_ctrl_apply(device const float *cur0 [[buffer(0)]],
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
                             device const float *g0 [[buffer(16)]],
                             device const float *g1 [[buffer(17)]],
                             device const float *g2 [[buffer(18)]],
                             device const float *g3 [[buffer(19)]],
                             device float *out0 [[buffer(20)]],
                             device float *out1 [[buffer(21)]],
                             device float *out2 [[buffer(22)]],
                             device float *out3 [[buffer(23)]],
                             constant Ink2PCtrlShape &p [[buffer(24)]],
                             uint3 at [[thread_position_in_grid]]) {
    const uint i = at.x, tok = at.y, j = at.z, rank = p.rank;
    if (i >= p.n || tok >= p.tokens) return;
    device const float *gj = ink2_pick4(j, g0, g1, g2, g3) + ulong(tok) * rank;
    device const float *gr = ink2_pick4(j, gain0, gain1, gain2, gain3) + ulong(i) * rank;
    device const float *fr = ink2_pick4(j, forget0, forget1, forget2, forget3) + ulong(i) * rank;
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
    const ulong o = ulong(tok) * p.n + i;
    const float x = ink2_pick4(j, cur0, cur1, cur2, cur3)[o], m = ink2_pick4(j, msg0, msg1, msg2, msg3)[o];
    const float f = tanh(forget), t = tanh(gain);
    ink2_pick4(j, out0, out1, out2, out3)[o] = (x - f * x) + t * m;
}

struct Ink2PConvShape {
    uint n, perHead, kn, tokens;
    uint dynamic, residual;
    float norm;
};

// Causal short conv along the token axis, one thread per channel: it walks
// the chunk in token order with its history in registers and writes the final
// history once, so token t sees exactly the inputs of tokens < t. dyn is
// [T][heads][kn]. out must not alias x or resid: a thread reads token t+1's
// input after writing token t's output.
kernel void ink2p_conv(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *dyn [[buffer(2)]],
                       device float *hist [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       device const float *resid [[buffer(5)]],
                       constant Ink2PConvShape &s [[buffer(6)]],
                       uint c [[thread_position_in_grid]]) {
    if (c >= s.n) return;
    const uint n = s.n, kn = s.kn, past = kn - 1;
    device float *hc = hist + ulong(c) * past;
    device const float *wc = w + ulong(c) * kn;
    float h[INK2P_MAX_KN], wv[INK2P_MAX_KN];
    for (uint t = 0; t < INK2P_MAX_KN; t++) {
        h[t] = 0.0f;
        wv[t] = 0.0f;
    }
    for (uint t = 0; t < past; t++) h[t] = hc[t];
    for (uint t = 0; t < kn; t++) wv[t] = wc[t];
    const uint dstride = (n / s.perHead) * kn;
    const uint head = c / s.perHead;
    for (uint tok = 0; tok < s.tokens; tok++) {
        const float xc = x[ulong(tok) * n + c];
        float acc = 0.0f;
        for (uint t = 0; t < past; t++) acc += h[t] * wv[t];
        acc += xc * wv[past];
        if (s.dynamic != 0) {
            device const float *dr = dyn + ulong(tok) * dstride + head * kn;
            for (uint lag = 0; lag < kn; lag++) acc += dr[lag] * (lag > 0 ? h[past - lag] : xc);
        }
        float o = (xc + acc) * s.norm;
        for (uint t = 0; t + 1 < past; t++) h[t] = h[t + 1];
        h[past - 1] = xc;
        if (s.residual != 0) o = resid[ulong(tok) * n + c] + o;
        out[ulong(tok) * n + c] = o;
    }
    for (uint t = 0; t < past; t++) hc[t] = h[t];
}

struct Ink2PQKVShape {
    uint heads, kvHeads, headDim, relLen;
    uint cap, lanes;
    uint global, pos0;
    float qkClip, relClip, eps, alpha, floorTok;
};

// ink2_conv.metal's ink2_clip, repeated.
static inline void ink2_clip(device float *x, device const float *scale, bool scaled, float thr, uint n, uint lane,
                             uint lanes, float gain, bool gained) {
    float ss = 0.0f;
    for (uint i = lane; i < n; i += lanes) {
        float v = x[i];
        if (scaled) v *= scale[i];
        x[i] = v;
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    float f = thr / (sqrt(ss) + 1e-6f);
    // Not clamp(): a NaN factor has to reach the output as it does on the host.
    if (f > 1.0f) f = 1.0f;
    if (f < 0.0f) f = 0.0f;
    for (uint i = lane; i < n; i += lanes) {
        float v = x[i] * f;
        if (gained) v *= gain;
        x[i] = v;
    }
}

// Post-projection attention prep, a threadgroup (one SIMD group of lanes
// threads) per (head block, token): blocks [0, heads) clip q and r in place,
// then kvHeads clip k into the cache row of position pos0+t, then kvHeads
// RMS-normalize v into it. Tokens write distinct cache rows only while
// T <= cap.
kernel void ink2p_qkv_post(device float *q [[buffer(0)]],
                           device float *r [[buffer(1)]],
                           device const float *k [[buffer(2)]],
                           device const float *v [[buffer(3)]],
                           device const float *qclip [[buffer(4)]],
                           device const float *rclip [[buffer(5)]],
                           device float *kcache [[buffer(6)]],
                           device float *vcache [[buffer(7)]],
                           constant Ink2PQKVShape &s [[buffer(8)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    const uint b = group.x, tok = group.y;
    const uint pos = s.pos0 + tok;
    const uint kvDim = s.kvHeads * s.headDim;
    if (b < s.heads) {
        float tau = 1.0f;
        if (s.global != 0) tau = 1.0f + s.alpha * log(max(1.0f, float(pos + 1) / s.floorTok));
        ink2_clip(q + (ulong(tok) * s.heads + b) * s.headDim, qclip + ulong(b) * s.headDim, true, s.qkClip, s.headDim,
                  lane, s.lanes, tau, s.global != 0);
        ink2_clip(r + (ulong(tok) * s.heads + b) * s.relLen, rclip + ulong(b) * s.relLen, true, s.relClip, s.relLen,
                  lane, s.lanes, tau, s.global != 0);
        return;
    }
    const ulong row = ulong(pos % s.cap) * kvDim;
    if (b < s.heads + s.kvHeads) {
        const uint first = (b - s.heads) * s.headDim;
        device float *dst = kcache + row + first;
        device const float *src = k + ulong(tok) * kvDim + first;
        for (uint i = lane; i < s.headDim; i += s.lanes) dst[i] = src[i];
        ink2_clip(dst, kcache, false, s.qkClip, s.headDim, lane, s.lanes, 1.0f, false);
        return;
    }
    const uint first = (b - s.heads - s.kvHeads) * s.headDim;
    device float *dst = vcache + row + first;
    device const float *src = v + ulong(tok) * kvDim + first;
    // The unnormalized value is staged in the cache row: a lane holds more
    // than one element across the reduction.
    float ss = 0.0f;
    for (uint i = lane; i < s.headDim; i += s.lanes) {
        const float x = src[i];
        dst[i] = x;
        ss = fma(x, x, ss);
    }
    ss = simd_sum(ss);
    const float scale = 1.0f / sqrt(ss / float(s.headDim) + s.eps);
    for (uint i = lane; i < s.headDim; i += s.lanes) dst[i] *= scale;
}

struct Ink2PRouteShape {
    uint experts, slots, biased;
    float weightScale;
};

// MoE routing, a threadgroup (one SIMD group) per token: logits row t routes
// into ids and wts [t][k].
kernel void ink2p_route(device const float *logits [[buffer(0)]],
                        device const float *bias [[buffer(1)]],
                        device int *ids [[buffer(2)]],
                        device float *wts [[buffer(3)]],
                        constant Ink2PRouteShape &p [[buffer(4)]],
                        uint tok [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
    const uint none = 0xffffffffu;
    const uint strides = (p.experts + INK2_ROUTE_THREADS - 1) / INK2_ROUTE_THREADS;
    device const float *row = logits + ulong(tok) * p.experts;
    device int *idr = ids + ulong(tok) * p.slots;
    device float *wtr = wts + ulong(tok) * p.slots;
    float probs[INK2_ROUTE_STRIDE], sel[INK2_ROUTE_STRIDE];
    for (uint s = 0; s < INK2_ROUTE_STRIDE; s++) {
        const uint e = lane + s * INK2_ROUTE_THREADS;
        probs[s] = 0.0f;
        sel[s] = 0.0f;
        if (e < p.experts) {
            probs[s] = ink2_sigmoid(row[e]);
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
        if (lane == 0) idr[t] = int(win);
    }
    if (lane != 0) return;
    for (uint t = 0; t < p.slots; t++) {
        float v = picked[t] / (sum + 1e-20f);
        if (p.weightScale != 0.0f) v *= p.weightScale;
        wtr[t] = v;
    }
}

struct Ink2PCombineShape {
    uint slots, n, tokens;
};

// out[t][j] = sum over token t's routed slots, in routing order, of
// wts[t][s] * down[t*k + s][j].
kernel void ink2p_combine(device const float *down [[buffer(0)]],
                          device const float *wts [[buffer(1)]],
                          device float *out [[buffer(2)]],
                          constant Ink2PCombineShape &p [[buffer(3)]],
                          uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.n || at.y >= p.tokens) return;
    const ulong first = ulong(at.y) * p.slots;
    float acc = 0.0f;
    for (uint s = 0; s < p.slots; s++) acc += wts[first + s] * down[(first + s) * p.n + at.x];
    out[ulong(at.y) * p.n + at.x] = acc;
}

struct Ink2PBucketShape {
    uint npairs, nexp;
    // Threads of the one scan threadgroup.
    uint threads;
};

// counts[e] and the exclusive scan offsets[0..nexp] of the routed ids, in one
// threadgroup: thread e counts expert e (and e + threads, ...), then thread 0
// scans. Counting per expert needs neither atomics nor a cleared buffer.
kernel void ink2p_bucket_scan(device const int *ids [[buffer(0)]],
                              device int *counts [[buffer(1)]],
                              device int *offsets [[buffer(2)]],
                              constant Ink2PBucketShape &p [[buffer(3)]],
                              uint tid [[thread_index_in_threadgroup]]) {
    for (uint e = tid; e < p.nexp; e += p.threads) {
        int count = 0;
        for (uint i = 0; i < p.npairs; i++) count += ids[i] == int(e) ? 1 : 0;
        counts[e] = count;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid != 0) return;
    int acc = 0;
    for (uint e = 0; e < p.nexp; e++) {
        offsets[e] = acc;
        acc += counts[e];
    }
    offsets[p.nexp] = acc;
}

// order[offsets[e] + rank] = i, rank being the number of earlier pairs routed
// to the same expert: a stable sort of the pairs by expert, one thread per
// pair. A thread reads up to npairs ids, so this is quadratic in the chunk.
kernel void ink2p_bucket_fill(device const int *ids [[buffer(0)]],
                              device const int *offsets [[buffer(1)]],
                              device int *order [[buffer(2)]],
                              constant Ink2PBucketShape &p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.npairs) return;
    const int e = ids[i];
    int rank = 0;
    for (uint j = 0; j < i; j++) rank += ids[j] == e ? 1 : 0;
    order[offsets[e] + rank] = int(i);
}

// The controller of a chunk wider than any verify chunk, as two GEMMs around
// these kernels; unlike the kernels above it does not keep the decode
// controller's order, so it is not bit-equal to it.
struct Ink2PCtrlRows { uint n, tokens; };

// xs[t] = [rms(cur[t]), rms(msg[t])], the GEMM input of the down projection:
// a threadgroup per token.
kernel void ink2p_ctrl_norm(device const float *cur [[buffer(0)]],
                            device const float *msg [[buffer(1)]],
                            device float *xs [[buffer(2)]],
                            constant Ink2PCtrlRows &p [[buffer(3)]],
                            uint tok [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2 * INK2_SG];
    const uint n = p.n;
    device const float *c = cur + ulong(tok) * n, *m = msg + ulong(tok) * n;
    float ssc = 0.0f, ssm = 0.0f;
    for (uint i = tid; i < n; i += INK2_TG) {
        ssc = fma(c[i], c[i], ssc);
        ssm = fma(m[i], m[i], ssm);
    }
    ssc = simd_sum(ssc);
    ssm = simd_sum(ssm);
    if (lane == 0) {
        part[sg] = ssc;
        part[INK2_SG + sg] = ssm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tc = 0.0f, tm = 0.0f;
    for (uint s = 0; s < INK2_SG; s++) {
        tc += part[s];
        tm += part[INK2_SG + s];
    }
    const float sc = ink2_inv_rms(tc, n, 1e-6f), sm = ink2_inv_rms(tm, n, 1e-6f);
    device float *x = xs + ulong(tok) * 2 * n;
    for (uint i = tid; i < n; i += INK2_TG) {
        x[i] = c[i] * sc;
        x[n + i] = m[i] * sm;
    }
}

// out = cur - tanh(forget)*cur + tanh(gain)*msg from the GEMMs' gain and forget.
kernel void ink2p_ctrl_gate(device const float *cur [[buffer(0)]],
                            device const float *msg [[buffer(1)]],
                            device const float *gain [[buffer(2)]],
                            device const float *forget [[buffer(3)]],
                            device float *out [[buffer(4)]],
                            constant Ink2PCtrlRows &p [[buffer(5)]],
                            uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.n || at.y >= p.tokens) return;
    const ulong o = ulong(at.y) * p.n + at.x;
    const float x = cur[o], f = tanh(forget[o]), t = tanh(gain[o]);
    out[o] = (x - f * x) + t * msg[o];
}
