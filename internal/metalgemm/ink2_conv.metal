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

#ifdef INK2_QKV_HALF
typedef half ink2_qkv_kv_t;
#else
typedef float ink2_qkv_kv_t;
#endif

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

// The host builds this source again with INK2_QKV_NAMED for heads of 128
// dims, 16 relative states, 32 lanes and convs of 4 taps. That build's
// ink2_qkv_post (the general one is described below) keeps every operation and
// its order but loads a lane's elements, and a conv's taps and history, before
// its first store.
#ifdef INK2_QKV_NAMED
#define INK2_QKV_PER 4u
#define INK2_QKV_LANES 32u
#define INK2_QKV_HD (INK2_QKV_PER * INK2_QKV_LANES)
#define INK2_QKV_REL 16u

// ink2_clip of this lane's values v, already loaded, into x.
static inline void ink2_clip_regs(thread float *v, uint per, device float *x, device const float *scale, bool scaled,
                                  float thr, uint lane, float gain, bool gained) {
    float ss = 0.0f;
    for (uint t = 0; t < per; t++) {
        if (scaled) v[t] *= scale[lane + t * INK2_QKV_LANES];
        ss = fma(v[t], v[t], ss);
    }
    ss = simd_sum(ss);
    float f = thr / (sqrt(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (f < 0.0f) f = 0.0f;
    for (uint t = 0; t < per; t++) {
        float u = v[t] * f;
        if (gained) u *= gain;
        x[lane + t * INK2_QKV_LANES] = u;
    }
}

static inline void ink2_clip_regs_kv(thread float *v, uint per, device ink2_qkv_kv_t *x, float thr, uint lane) {
    float ss = 0.0f;
    for (uint t = 0; t < per; t++) ss = fma(v[t], v[t], ss);
    ss = simd_sum(ss);
    float f = thr / (sqrt(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (f < 0.0f) f = 0.0f;
    for (uint t = 0; t < per; t++) x[lane + t * INK2_QKV_LANES] = ink2_qkv_kv_t(v[t] * f);
}

// ink2_conv_chan of four taps for INK2_QKV_PER channels c0 + t * INK2_QKV_LANES.
static inline void ink2_conv4_regs(thread float *xs, uint c0, device const float *w, device const float *dyn,
                                   bool dynamic, device float *hist, float norm) {
    float h[INK2_QKV_PER][3], wt[INK2_QKV_PER][4], dt[INK2_QKV_PER][4];
    for (uint t = 0; t < INK2_QKV_PER; t++) {
        const uint c = c0 + t * INK2_QKV_LANES;
        device const float *hc = hist + ulong(c) * 3;
        device const float *wc = w + ulong(c) * 4;
        device const float *dh = dyn + ulong(c / INK2_QKV_HD) * 4;
        for (uint i = 0; i < 3; i++) h[t][i] = hc[i];
        for (uint i = 0; i < 4; i++) wt[t][i] = wc[i];
        for (uint i = 0; i < 4; i++) dt[t][i] = 0.0f;
        if (dynamic)
            for (uint i = 0; i < 4; i++) dt[t][i] = dh[i];
    }
    for (uint t = 0; t < INK2_QKV_PER; t++) {
        const float xc = xs[t];
        float acc = 0.0f;
        for (uint i = 0; i < 3; i++) acc += h[t][i] * wt[t][i];
        acc += xc * wt[t][3];
        if (dynamic) {
            acc += dt[t][0] * xc;
            for (uint lag = 1; lag < 4; lag++) acc += dt[t][lag] * h[t][3 - lag];
        }
        xs[t] = (xc + acc) * norm;
        device float *hc = hist + ulong(c0 + t * INK2_QKV_LANES) * 3;
        hc[0] = h[t][1];
        hc[1] = h[t][2];
        hc[2] = xc;
    }
}

kernel void ink2_qkv_post(device float *q [[buffer(0)]],
                          device float *r [[buffer(1)]],
                          device const float *k [[buffer(2)]],
                          device const float *v [[buffer(3)]],
                          device const float *qclip [[buffer(4)]],
                          device const float *rclip [[buffer(5)]],
                          device ink2_qkv_kv_t *kcache [[buffer(6)]],
                          device ink2_qkv_kv_t *vcache [[buffer(7)]],
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
    const uint kvDim = s.kvHeads * INK2_QKV_HD;
    float xs[INK2_QKV_PER];
    if (b < s.heads) {
        float tau = 1.0f;
        if (s.global != 0) tau = 1.0f + s.alpha * log(max(1.0f, float(pos + 1) / s.floorTok));
        device float *qh = q + ulong(b) * INK2_QKV_HD;
        device float *rh = r + ulong(b) * INK2_QKV_REL;
        for (uint t = 0; t < INK2_QKV_PER; t++) xs[t] = qh[lane + t * INK2_QKV_LANES];
        float rv = lane < INK2_QKV_REL ? rh[lane] : 0.0f;
        ink2_clip_regs(xs, INK2_QKV_PER, qh, qclip + ulong(b) * INK2_QKV_HD, true, s.qkClip, lane, tau, s.global != 0);
        // A lane past the relative states holds a zero and stores nothing.
        float rs = 0.0f;
        if (lane < INK2_QKV_REL) {
            rv *= rclip[ulong(b) * INK2_QKV_REL + lane];
            rs = fma(rv, rv, rs);
        }
        rs = simd_sum(rs);
        float f = s.relClip / (sqrt(rs) + 1e-6f);
        if (f > 1.0f) f = 1.0f;
        if (f < 0.0f) f = 0.0f;
        if (lane < INK2_QKV_REL) {
            float u = rv * f;
            if (s.global != 0) u *= tau;
            rh[lane] = u;
        }
        return;
    }
    const ulong row = ulong(pos % s.cap) * kvDim;
    const bool isK = b < s.heads + s.kvHeads;
    const uint first = (isK ? b - s.heads : b - s.heads - s.kvHeads) * INK2_QKV_HD;
    device const float *src = isK ? k : v;
    for (uint t = 0; t < INK2_QKV_PER; t++) xs[t] = src[first + lane + t * INK2_QKV_LANES];
    if (s.conv != 0) {
        if (isK) ink2_conv4_regs(xs, first + lane, kw, kdyn, s.kDynamic != 0, khist, s.kNorm);
        else ink2_conv4_regs(xs, first + lane, vw, vdyn, s.vDynamic != 0, vhist, s.vNorm);
    }
    if (isK) {
        ink2_clip_regs_kv(xs, INK2_QKV_PER, kcache + row + first, s.qkClip, lane);
        return;
    }
    float ss = 0.0f;
    for (uint t = 0; t < INK2_QKV_PER; t++) ss = fma(xs[t], xs[t], ss);
    ss = simd_sum(ss);
    const float scale = 1.0f / sqrt(ss / float(INK2_QKV_HD) + s.eps);
    device ink2_qkv_kv_t *dst = vcache + row + first;
    for (uint t = 0; t < INK2_QKV_PER; t++) dst[lane + t * INK2_QKV_LANES] = ink2_qkv_kv_t(xs[t] * scale);
}
#else
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
                          device ink2_qkv_kv_t *kcache [[buffer(6)]],
                          device ink2_qkv_kv_t *vcache [[buffer(7)]],
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
#ifdef INK2_QKV_HALF
    threadgroup float tmp[512];
    if (b < s.heads + s.kvHeads) {
        const uint first = (b - s.heads) * s.headDim;
        device half *dst = kcache + row + first;
        float ss = 0.0f;
        for (uint i = lane; i < s.headDim; i += s.lanes) {
            const uint c = first + i;
            const float x = s.conv != 0
                                ? ink2_conv_chan(k[c], c, kw, kdyn, s.kDynamic != 0, khist, s.headDim, s.kn, s.kNorm)
                                : k[c];
            tmp[i] = x;
            ss = fma(x, x, ss);
        }
        ss = simd_sum(ss);
        float f = s.qkClip / (sqrt(ss) + 1e-6f);
        if (f > 1.0f) f = 1.0f;
        if (f < 0.0f) f = 0.0f;
        for (uint i = lane; i < s.headDim; i += s.lanes) dst[i] = half(tmp[i] * f);
        return;
    }
    const uint first = (b - s.heads - s.kvHeads) * s.headDim;
    device half *dst = vcache + row + first;
    float ss = 0.0f;
    for (uint i = lane; i < s.headDim; i += s.lanes) {
        const uint c = first + i;
        const float x = s.conv != 0
                            ? ink2_conv_chan(v[c], c, vw, vdyn, s.vDynamic != 0, vhist, s.headDim, s.kn, s.vNorm)
                            : v[c];
        tmp[i] = x;
        ss = fma(x, x, ss);
    }
    ss = simd_sum(ss);
    const float scale = 1.0f / sqrt(ss / float(s.headDim) + s.eps);
    for (uint i = lane; i < s.headDim; i += s.lanes) dst[i] = half(tmp[i] * scale);
#else
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
#endif
}
#endif
