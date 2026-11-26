#include <metal_stdlib>
using namespace metal;

// The conv keeps the host forward's evaluation order, so a multiply must not
// fuse into the add that follows it; the sums of squares fuse by name.
#pragma STDC FP_CONTRACT OFF

// One channel of the causal short conv: returns (x + taps) * norm and shifts x
// into the channel's history, which holds its kn-1 past inputs oldest first.
// Dynamic taps are laid out [head][tap], head = channel / perHead.
static inline float ink2_conv_chan(float xc, uint c, device const float *w, device const float *dyn, bool dynamic,
                                   device float *hist, uint perHead, uint kn, float norm) {
    const uint past = kn - 1;
    device float *hc = hist + ulong(c) * past;
    device const float *wc = w + ulong(c) * kn;
    float acc = 0.0f;
    for (uint t = 0; t < past; t++) acc += hc[t] * wc[t];
    acc += xc * wc[past];
    if (dynamic) {
        device const float *dh = dyn + ulong(c / perHead) * kn;
        for (uint lag = 0; lag < kn; lag++) acc += dh[lag] * (lag > 0 ? hc[past - lag] : xc);
    }
    const float o = (xc + acc) * norm;
    for (uint t = 0; t + 1 < past; t++) hc[t] = hc[t + 1];
    hc[past - 1] = xc;
    return o;
}

struct Ink2ConvShape {
    uint n, perHead, kn;
    // Nonzero makes the input the combine of that many routed rows of x by wts.
    uint slots;
    uint dynamic, residual;
    float norm;
};

// One thread per channel. With slots the combine sums in routing order, as the
// standalone combine does, so folding it here changes no bits.
kernel void ink2_conv(device const float *x [[buffer(0)]],
                      device const float *wts [[buffer(1)]],
                      device const float *w [[buffer(2)]],
                      device const float *dyn [[buffer(3)]],
                      device float *hist [[buffer(4)]],
                      device float *out [[buffer(5)]],
                      device const float *resid [[buffer(6)]],
                      constant Ink2ConvShape &s [[buffer(7)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= s.n) return;
    float xc;
    if (s.slots != 0) {
        xc = 0.0f;
        for (uint t = 0; t < s.slots; t++) xc += wts[t] * x[ulong(t) * s.n + c];
    } else {
        xc = x[c];
    }
    float o = ink2_conv_chan(xc, c, w, dyn, s.dynamic != 0, hist, s.perHead, s.kn, s.norm);
    if (s.residual != 0) o = resid[c] + o;
    out[c] = o;
}

struct Ink2QKVShape {
    uint heads, kvHeads, headDim, relLen;
    uint cap, lanes;
    uint global, conv, kn, kDynamic, vDynamic;
    float qkClip, relClip, eps, alpha, floorTok, kNorm, vNorm;
};

// Shrinks x (scaled per channel first when scaled) to an L2 norm of at most
// thr, then multiplies by gain. Each lane owns every lanes-th element.
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

// Post-projection attention prep of one layer visit: per-head q and r clips
// with the global log-scale tau, the k clip, the v RMS, and the append of k and
// v into the slot cache row of the current position (a ring for local slots),
// each after its short conv when conv is set. Threadgroups [0, heads) are the
// q and r heads, then kvHeads k heads, then kvHeads v heads; a threadgroup is
// one SIMD group of lanes threads.
kernel void ink2_qkv_post(device float *q [[buffer(0)]],
                          device float *r [[buffer(1)]],
                          device const float *k [[buffer(2)]],
                          device const float *v [[buffer(3)]],
                          device const float *qclip [[buffer(4)]],
                          device const float *rclip [[buffer(5)]],
                          device float *kcache [[buffer(6)]],
                          device float *vcache [[buffer(7)]],
                          device const int *meta [[buffer(8)]],
                          device const float *kw [[buffer(9)]],
                          device const float *vw [[buffer(10)]],
                          device const float *kdyn [[buffer(11)]],
                          device const float *vdyn [[buffer(12)]],
                          device float *khist [[buffer(13)]],
                          device float *vhist [[buffer(14)]],
                          constant Ink2QKVShape &s [[buffer(15)]],
                          uint b [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    const uint pos = uint(meta[0]);
    const uint kvDim = s.kvHeads * s.headDim;
    if (b < s.heads) {
        float tau = 1.0f;
        if (s.global != 0) tau = 1.0f + s.alpha * log(max(1.0f, float(pos + 1) / s.floorTok));
        ink2_clip(q + ulong(b) * s.headDim, qclip + ulong(b) * s.headDim, true, s.qkClip, s.headDim, lane, s.lanes,
                  tau, s.global != 0);
        ink2_clip(r + ulong(b) * s.relLen, rclip + ulong(b) * s.relLen, true, s.relClip, s.relLen, lane, s.lanes,
                  tau, s.global != 0);
        return;
    }
    const ulong row = ulong(pos % s.cap) * kvDim;
    if (b < s.heads + s.kvHeads) {
        const uint first = (b - s.heads) * s.headDim;
        device float *dst = kcache + row + first;
        for (uint i = lane; i < s.headDim; i += s.lanes) {
            const uint c = first + i;
            dst[i] = s.conv != 0 ? ink2_conv_chan(k[c], c, kw, kdyn, s.kDynamic != 0, khist, s.headDim, s.kn, s.kNorm)
                                 : k[c];
        }
        ink2_clip(dst, kcache, false, s.qkClip, s.headDim, lane, s.lanes, 1.0f, false);
        return;
    }
    const uint first = (b - s.heads - s.kvHeads) * s.headDim;
    device float *dst = vcache + row + first;
    // The unnormalized value is staged in the cache row: a lane holds more
    // than one element across the reduction.
    float ss = 0.0f;
    for (uint i = lane; i < s.headDim; i += s.lanes) {
        const uint c = first + i;
        const float x = s.conv != 0
                            ? ink2_conv_chan(v[c], c, vw, vdyn, s.vDynamic != 0, vhist, s.headDim, s.kn, s.vNorm)
                            : v[c];
        dst[i] = x;
        ss = fma(x, x, ss);
    }
    ss = simd_sum(ss);
    const float scale = 1.0f / sqrt(ss / float(s.headDim) + s.eps);
    for (uint i = lane; i < s.headDim; i += s.lanes) dst[i] *= scale;
}
