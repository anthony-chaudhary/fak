// Tree forms of the verify chunk kernels of ink2_verify.metal. Row t of a
// T-row token tree has parent[t] < t (-1 for the root, row 0) and depth its
// count of ancestors: it is the token at position pos + depth along its root
// path, while its cache row is still pos + t, so siblings never share one. A
// row computes exactly what the chain kernel computes for the last token of
// its root path, so a chain tree (parent[t] = t-1) gives the chain's bits.
// Tree layout and path commit follow incoai/splash@2afc7337, Apache-2.0
// (VerifyTree.h, kv_path_commit.metal).

#include <metal_stdlib>
using namespace metal;

// ink2_verify.metal's contract: the chain kernels compile with it, so these must.
#pragma STDC FP_CONTRACT OFF

#define INK2T_MAX_NODES 32
#define INK2T_MAX_KN 8

// Ancestors of row t, bounded so that a malformed parent table cannot loop.
static inline uint ink2t_depth(device const int *parent, uint t) {
    uint d = 0;
    int at = int(t);
    while (d < INK2T_MAX_NODES && parent[at] >= 0 && parent[at] < at) {
        at = parent[at];
        d++;
    }
    return d;
}

struct Ink2TQKVShape {
    uint heads, kvHeads, headDim, relLen;
    uint cap, lanes;
    uint global, posFromDevice;
    int pos0;
    float qkClip, relClip, eps, alpha, floorTok;
};

// ink2_verify.metal's ink2_clip.
static inline void ink2t_clip(device float *x, device const float *scale, bool scaled, float thr, uint n, uint lane,
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
    if (f > 1.0f) f = 1.0f;
    if (f < 0.0f) f = 0.0f;
    for (uint i = lane; i < n; i += lanes) {
        float v = x[i] * f;
        if (gained) v *= gain;
        x[i] = v;
    }
}

// ink2v_qkv_post for a tree: the global log-scale temperature takes the
// row's tree position pos + depth, the cache row stays pos + t.
kernel void ink2t_qkv_post(device float *q [[buffer(0)]],
                           device float *r [[buffer(1)]],
                           device const float *k [[buffer(2)]],
                           device const float *v [[buffer(3)]],
                           device const float *qclip [[buffer(4)]],
                           device const float *rclip [[buffer(5)]],
                           device float *kcache [[buffer(6)]],
                           device float *vcache [[buffer(7)]],
                           device const int *posDev [[buffer(8)]],
                           device const int *parent [[buffer(9)]],
                           constant Ink2TQKVShape &s [[buffer(10)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    const uint b = group.x, tok = group.y;
    const uint pos0 = uint(s.posFromDevice != 0 ? posDev[0] : s.pos0);
    const uint kvDim = s.kvHeads * s.headDim;
    if (b < s.heads) {
        const uint pos = pos0 + ink2t_depth(parent, tok);
        float tau = 1.0f;
        if (s.global != 0) tau = 1.0f + s.alpha * log(max(1.0f, float(pos + 1) / s.floorTok));
        ink2t_clip(q + (ulong(tok) * s.heads + b) * s.headDim, qclip + ulong(b) * s.headDim, true, s.qkClip, s.headDim,
                   lane, s.lanes, tau, s.global != 0);
        ink2t_clip(r + (ulong(tok) * s.heads + b) * s.relLen, rclip + ulong(b) * s.relLen, true, s.relClip, s.relLen,
                   lane, s.lanes, tau, s.global != 0);
        return;
    }
    const ulong row = ulong((pos0 + tok) % s.cap) * kvDim;
    if (b < s.heads + s.kvHeads) {
        const uint first = (b - s.heads) * s.headDim;
        device float *dst = kcache + row + first;
        device const float *src = k + ulong(tok) * kvDim + first;
        for (uint i = lane; i < s.headDim; i += s.lanes) dst[i] = src[i];
        ink2t_clip(dst, kcache, false, s.qkClip, s.headDim, lane, s.lanes, 1.0f, false);
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

struct Ink2TConvShape {
    uint n, perHead, kn, tokens;
    uint dynamic, residual, logged;
    float norm;
};

// ink2v_conv over a tree: row t's kn-1 history inputs are its nearest
// ancestors' inputs, then the entry history. The entry history is left as it
// is, since the kept path is only known after the verify; the log ([past + T]
// per channel, entry history then every row's input) is what
// ink2t_conv_commit rebuilds it from. out must not alias x or resid.
kernel void ink2t_conv(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *dyn [[buffer(2)]],
                       device const float *hist [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       device const float *resid [[buffer(5)]],
                       device float *log [[buffer(6)]],
                       device const int *parent [[buffer(7)]],
                       constant Ink2TConvShape &s [[buffer(8)]],
                       uint c [[thread_position_in_grid]]) {
    if (c >= s.n) return;
    const uint n = s.n, kn = s.kn, past = kn - 1;
    device const float *hc = hist + ulong(c) * past;
    device const float *wc = w + ulong(c) * kn;
    float wv[INK2T_MAX_KN];
    for (uint t = 0; t < INK2T_MAX_KN; t++) wv[t] = 0.0f;
    for (uint t = 0; t < kn; t++) wv[t] = wc[t];
    device float *lc = log + ulong(c) * (past + s.tokens);
    if (s.logged != 0)
        for (uint t = 0; t < past; t++) lc[t] = hc[t];
    const uint dstride = (n / s.perHead) * kn;
    const uint head = c / s.perHead;
    for (uint tok = 0; tok < s.tokens; tok++) {
        const float xc = x[ulong(tok) * n + c];
        if (s.logged != 0) lc[past + tok] = xc;
        float h[INK2T_MAX_KN];
        for (uint t = 0; t < INK2T_MAX_KN; t++) h[t] = 0.0f;
        uint used = 0;
        for (int at = parent[tok]; used < past && at >= 0 && uint(at) < tok; at = parent[at]) {
            used++;
            h[past - used] = x[ulong(at) * n + c];
        }
        for (uint t = 0; t + used < past; t++) h[t] = hc[t + used];
        float acc = 0.0f;
        for (uint t = 0; t < past; t++) acc += h[t] * wv[t];
        acc += xc * wv[past];
        if (s.dynamic != 0) {
            device const float *dr = dyn + ulong(tok) * dstride + head * kn;
            for (uint lag = 0; lag < kn; lag++) acc += dr[lag] * (lag > 0 ? h[past - lag] : xc);
        }
        float o = (xc + acc) * s.norm;
        if (s.residual != 0) o = resid[ulong(tok) * n + c] + o;
        out[ulong(tok) * n + c] = o;
    }
}

struct Ink2TCommitShape {
    uint channels, past, rowLen, fromDevice;
    uint cols[INK2T_MAX_KN];
};

// hist[c][j] = log[c][cols[j]]: the history after a kept root path, cols
// naming its last past entries of the entry history followed by the path's
// logged inputs. colsDev, when bound, holds the cols as int32.
kernel void ink2t_conv_commit(device float *hist [[buffer(0)]],
                              device const float *log [[buffer(1)]],
                              device const int *colsDev [[buffer(2)]],
                              constant Ink2TCommitShape &s [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= s.channels * s.past) return;
    const uint c = i / s.past, j = i % s.past;
    const uint col = s.fromDevice != 0 ? uint(colsDev[j]) : s.cols[j];
    hist[i] = log[ulong(c) * s.rowLen + col];
}

struct Ink2TPathShape {
    // Floats of a row, the ring's rows and the first row's ring index.
    uint width, cap, pos0, posFromDevice;
    // Rows the kept path holds, its row indices, and whether pathDev ([count,
    // rows...]) supplies them instead.
    uint count, fromDevice, buffers, pad;
    uint rows[INK2T_MAX_NODES];
};

// Moves the kept path's cache rows pos + path[i] to pos + i in each bound ring.
// path is increasing with path[i] >= i, so walking i upward reads every source
// before any write reaches it; a thread owns one column of every row.
kernel void ink2t_kv_compact(device float *a [[buffer(0)]],
                             device float *b [[buffer(1)]],
                             device const int *pathDev [[buffer(2)]],
                             device const int *posDev [[buffer(3)]],
                             constant Ink2TPathShape &s [[buffer(4)]],
                             uint2 at [[thread_position_in_grid]]) {
    if (at.x >= s.width || at.y >= s.buffers) return;
    device float *ring = at.y == 0 ? a : b;
    const uint pos0 = s.posFromDevice != 0 ? uint(posDev[0]) : s.pos0;
    const uint count = min(s.fromDevice != 0 ? uint(pathDev[0]) : s.count, uint(INK2T_MAX_NODES));
    for (uint i = 0; i < count; i++) {
        const uint src = s.fromDevice != 0 ? uint(pathDev[1 + i]) : s.rows[i];
        if (src == i) continue;
        const ulong from = ulong((pos0 + src) % s.cap) * s.width, to = ulong((pos0 + i) % s.cap) * s.width;
        ring[to + at.x] = ring[from + at.x];
    }
}

struct Ink2TGatherShape {
    uint width, inStride, count, fromDevice;
    uint rows[INK2T_MAX_NODES];
};

// out[i] = in[path[i]] for rows of width floats, in rows inStride apart.
kernel void ink2t_gather_rows(device const float *in [[buffer(0)]],
                              device float *out [[buffer(1)]],
                              device const int *pathDev [[buffer(2)]],
                              constant Ink2TGatherShape &s [[buffer(3)]],
                              uint2 at [[thread_position_in_grid]]) {
    const uint count = min(s.fromDevice != 0 ? uint(pathDev[0]) : s.count, uint(INK2T_MAX_NODES));
    if (at.x >= s.width || at.y >= count) return;
    const uint src = s.fromDevice != 0 ? uint(pathDev[1 + at.y]) : s.rows[at.y];
    out[ulong(at.y) * s.width + at.x] = in[ulong(src) * s.inStride + at.x];
}
