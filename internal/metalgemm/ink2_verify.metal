// Row-batched forms of the resident inkling2_loop decode kernels for a
// speculative verify chunk: token t of a T-token chunk sits at position
// pos + t and every activation is [T][width] rows. A row is one grid
// coordinate, so its arithmetic is the decode kernel's whatever T is: verifying
// a token alone or beside eight others gives the same bits.
//
// Kernel shapes adapted from internal/compute/cuda/inkling2_prefill.cuh and
// inkling2_verify_rows.cuh; each body is the decode kernel of ink2_ops.metal or
// ink2_conv.metal with a token offset.

#include <metal_stdlib>
using namespace metal;

// The sums below keep the host forward's evaluation order, so a multiply must
// not fuse into the add that follows it; fma() marks the fusions that are meant.
#pragma STDC FP_CONTRACT OFF

#define INK2_TG 256
#define INK2_SG (INK2_TG / 32)
#define INK2_MAX_STATES 16
#define INK2V_MAX_KN 8

static inline float ink2_inv_rms(float sumSquares, uint n, float eps) {
    return 1.0f / sqrt(sumSquares / float(n) + eps);
}

template <typename T> static inline T ink2_pick4(uint i, T a, T b, T c, T d) {
    return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;
}

struct Ink2VMixShape {
    uint n, states, tokens;
    // norm: RMS-normalize each mixed row by eps; weighted bit q: and scale
    // query q's row by its bound weight.
    uint norm, weighted;
    float eps, invSqrtN;
};

struct Ink2VMixShared {
    float part[2 * INK2_MAX_STATES * INK2_SG];
    float scale[INK2_MAX_STATES];
    float weight[INK2_MAX_STATES];
    float outPart[INK2_SG];
    float outScale;
};

#define INK2_STATE_ARGS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)

// ink2_mix with a threadgroup per (query, token): token t's states are row t
// of each [T][n] state and the output is query-major [nq][T][n].
// Technique adapted from incoai/splash runtime/metal/kernels/shared/block_residual_mix.metal@2afc7337 (Apache-2.0).
kernel void ink2v_mix(
#define INK2_STATE_PARAM(i) device const float *s##i [[buffer(i)]],
    INK2_STATE_ARGS(INK2_STATE_PARAM)
#undef INK2_STATE_PARAM
    device const float *queries [[buffer(16)]],
    device const float *norm0 [[buffer(17)]],
    device const float *norm1 [[buffer(18)]],
    device const float *norm2 [[buffer(19)]],
    device const float *norm3 [[buffer(20)]],
    device float *outs [[buffer(21)]],
    constant Ink2VMixShape &p [[buffer(22)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Ink2VMixShared sh;
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
    float squares = 0.0f;
    {
        INK2_STATES;
        for (uint i = tid; i < n; i += INK2_TG) {
            float a = 0.0f;
            for (uint s = 0; s < INK2_MAX_STATES; s++)
                if (s < states) a += sh.weight[s] * (st[s][toff + i] * sh.scale[s]);
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

struct Ink2VCtrlShape {
    uint n, rank, tokens;
};

// g_j[t][r] = Down_j[r,:] . [rms(cur_j[t]), rms(msg_j[t])]: ink2_ctrl_down
// with a threadgroup per (rank row, token, controller).
// Technique adapted from incoai/splash runtime/metal/kernels/shared/message_gate.metal@2afc7337 (Apache-2.0).
kernel void ink2v_ctrl_down(device const float *cur0 [[buffer(0)]],
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
                            constant Ink2VCtrlShape &p [[buffer(16)]],
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

// out_j[t][i] = cur - tanh(F_j[i,:].g_j[t])*cur + tanh(G_j[i,:].g_j[t])*msg:
// ink2_ctrl_apply with one thread per (output, token, controller).
kernel void ink2v_ctrl_apply(device const float *cur0 [[buffer(0)]],
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
                             constant Ink2VCtrlShape &p [[buffer(24)]],
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

// One channel walks the chunk in token order with its history in registers and
// writes the final history once, so token t sees exactly the inputs of tokens
// < t: ink2_conv_chan run T times, whatever T is. dyn is [T][heads][kn]. With
// logged the channel's log row is [past + T]: the entry history, then every
// input, which is what a commit rewinds from.
static inline void ink2v_conv_walk(device const float *x, device const float *w, device const float *dyn, bool dynamic,
                                   device float *hist, device float *out, device const float *resid, bool residual,
                                   device float *log, bool logged, uint c, uint n, uint perHead, uint kn, uint tokens,
                                   float norm) {
    const uint past = kn - 1;
    device float *hc = hist + ulong(c) * past;
    device const float *wc = w + ulong(c) * kn;
    float h[INK2V_MAX_KN], wv[INK2V_MAX_KN];
    for (uint t = 0; t < INK2V_MAX_KN; t++) {
        h[t] = 0.0f;
        wv[t] = 0.0f;
    }
    for (uint t = 0; t < past; t++) h[t] = hc[t];
    for (uint t = 0; t < kn; t++) wv[t] = wc[t];
    device float *lc = log + ulong(c) * (past + tokens);
    if (logged)
        for (uint t = 0; t < past; t++) lc[t] = h[t];
    const uint dstride = (n / perHead) * kn;
    const uint head = c / perHead;
    for (uint tok = 0; tok < tokens; tok++) {
        const float xc = x[ulong(tok) * n + c];
        if (logged) lc[past + tok] = xc;
        float acc = 0.0f;
        for (uint t = 0; t < past; t++) acc += h[t] * wv[t];
        acc += xc * wv[past];
        if (dynamic) {
            device const float *dr = dyn + ulong(tok) * dstride + head * kn;
            for (uint lag = 0; lag < kn; lag++) acc += dr[lag] * (lag > 0 ? h[past - lag] : xc);
        }
        float o = (xc + acc) * norm;
        for (uint t = 0; t + 1 < past; t++) h[t] = h[t + 1];
        if (past > 0) h[past - 1] = xc;
        if (residual) o = resid[ulong(tok) * n + c] + o;
        out[ulong(tok) * n + c] = o;
    }
    for (uint t = 0; t < past; t++) hc[t] = h[t];
}

struct Ink2VConvShape {
    uint n, perHead, kn, tokens;
    uint dynamic, residual, logged;
    float norm;
};

// One thread per channel. out must not alias x or resid: a thread reads token
// t+1's input after writing token t's output.
kernel void ink2v_conv(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *dyn [[buffer(2)]],
                       device float *hist [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       device const float *resid [[buffer(5)]],
                       device float *log [[buffer(6)]],
                       constant Ink2VConvShape &s [[buffer(7)]],
                       uint c [[thread_position_in_grid]]) {
    if (c >= s.n) return;
    ink2v_conv_walk(x, w, dyn, s.dynamic != 0, hist, out, resid, s.residual != 0, log, s.logged != 0, c, s.n, s.perHead,
                    s.kn, s.tokens, s.norm);
}

struct Ink2VConvPairShape {
    uint n, perHead, kn, tokens;
    // Bit j: conv j has dynamic taps.
    uint dynamic;
    float norm0, norm1;
};

// The logged k and v short convs of one verify visit in one launch: at.y
// selects the conv, each of them ink2v_conv without a residual.
kernel void ink2v_conv_pair(device const float *x0 [[buffer(0)]],
                            device const float *x1 [[buffer(1)]],
                            device const float *w0 [[buffer(2)]],
                            device const float *w1 [[buffer(3)]],
                            device const float *dyn0 [[buffer(4)]],
                            device const float *dyn1 [[buffer(5)]],
                            device float *hist0 [[buffer(6)]],
                            device float *hist1 [[buffer(7)]],
                            device float *out0 [[buffer(8)]],
                            device float *out1 [[buffer(9)]],
                            device float *log0 [[buffer(10)]],
                            device float *log1 [[buffer(11)]],
                            constant Ink2VConvPairShape &s [[buffer(12)]],
                            uint2 at [[thread_position_in_grid]]) {
    if (at.x >= s.n) return;
    const bool second = at.y != 0;
    ink2v_conv_walk(second ? x1 : x0, second ? w1 : w0, second ? dyn1 : dyn0, ((s.dynamic >> at.y) & 1) != 0,
                    second ? hist1 : hist0, second ? out1 : out0, x0, false, second ? log1 : log0, true, at.x, s.n,
                    s.perHead, s.kn, s.tokens, second ? s.norm1 : s.norm0);
}

struct Ink2VCommitShape {
    uint channels, past, rowLen;
    uint keep, keepFromDevice;
};

// The history after the first keep tokens of a logged chunk is
// log[c][keep .. keep+past). keepDev carries the int32 a host write left in a
// float word.
kernel void ink2v_conv_commit(device float *hist [[buffer(0)]],
                              device const float *log [[buffer(1)]],
                              device const int *keepDev [[buffer(2)]],
                              constant Ink2VCommitShape &s [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= s.channels * s.past) return;
    const uint keep = s.keepFromDevice != 0 ? uint(keepDev[0]) : s.keep;
    const uint c = i / s.past, j = i % s.past;
    hist[i] = log[ulong(c) * s.rowLen + keep + j];
}

struct Ink2VQKVShape {
    uint heads, kvHeads, headDim, relLen;
    uint cap, lanes;
    uint global, posFromDevice;
    int pos0;
    float qkClip, relClip, eps, alpha, floorTok;
};

// ink2_conv.metal's ink2_clip.
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

// ink2_qkv_post without its folded convs, a threadgroup (one SIMD group of
// lanes threads) per (head block, token): blocks [0, heads) clip q and r in
// place, then kvHeads clip k into the cache row of position pos+t, then
// kvHeads RMS-normalize v into it. posDev holds the chunk's first position
// when posFromDevice. Tokens write distinct cache rows only while T <= cap.
kernel void ink2v_qkv_post(device float *q [[buffer(0)]],
                           device float *r [[buffer(1)]],
                           device const float *k [[buffer(2)]],
                           device const float *v [[buffer(3)]],
                           device const float *qclip [[buffer(4)]],
                           device const float *rclip [[buffer(5)]],
                           device float *kcache [[buffer(6)]],
                           device float *vcache [[buffer(7)]],
                           device const int *posDev [[buffer(8)]],
                           constant Ink2VQKVShape &s [[buffer(9)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    const uint b = group.x, tok = group.y;
    const uint pos = uint((s.posFromDevice != 0 ? posDev[0] : s.pos0) + int(tok));
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

struct Ink2VPairShape {
    uint npairs, width, div, stride;
};

// out[t*k + j] = x[t]: each (token, slot) pair gets its token's input row.
// Shape: npairs = rows*k, width = n, div = k, stride = n.
kernel void ink2v_repeat_rows(device const float *x [[buffer(0)]],
                              device float *out [[buffer(1)]],
                              constant Ink2VPairShape &p [[buffer(2)]],
                              uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.width || at.y >= p.npairs) return;
    out[ulong(at.y) * p.width + at.x] = x[ulong(at.y / p.div) * p.stride + at.x];
}

