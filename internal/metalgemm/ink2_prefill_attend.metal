// The batched attention of the resident inkling2_loop prefill: ink2_attend.metal
// with a token axis on the grid. It is a source of its own so that it compiles
// under that kernel's floating-point contraction, not ink2_prefill.metal's: a
// row then keeps the decode kernel's value sums bit for bit.

#include <metal_stdlib>
using namespace metal;

#define INK2_ATT_MAX_THREADS 512
#define INK2_ATT_MAX_SIMD 64
#define INK2_ATT_Q_TERMS 4

struct Ink2PAttendShape {
    uint heads, per_kv, head_dim, rel_len;
    uint kv_dim, extent, window, cap;
    uint threads, width;
    // Keys a (token, head) scores row holds.
    uint span;
    // First token of this launch: a chunk is attended in token ranges so that
    // one bounded scores scratch serves all of it.
    uint tok0;
    uint pos0;
    float scale;
};

static inline float4 ink2_att_load(const device float *p, uint have) {
    if (have >= 4) return float4(*(const device packed_float4 *)p);
    float4 v = float4(0.0f);
    for (uint i = 0; i < have; ++i) v[i] = p[i];
    return v;
}

// Attention of every query of a chunk, a threadgroup per (query head, token):
// ink2_attend with the query at pos0+t and a scores row per (token, head) of
// this launch. Every query sees the keys of its own and earlier positions,
// which ink2p_qkv_post appended for the whole chunk beforehand.
kernel void ink2p_attend(device const float *q [[buffer(0)]],
                         device const float *r [[buffer(1)]],
                         device const float *sinks [[buffer(2)]],
                         device const float *rel_bias [[buffer(3)]],
                         device const float *kcache [[buffer(4)]],
                         device const float *vcache [[buffer(5)]],
                         device float *scores [[buffer(6)]],
                         device float *out [[buffer(7)]],
                         constant Ink2PAttendShape &s [[buffer(8)]],
                         uint3 grid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tmax[INK2_ATT_MAX_SIMD];
    threadgroup float tsum[INK2_ATT_MAX_SIMD];
    threadgroup float4 gacc[INK2_ATT_MAX_THREADS];

    const uint h = grid.x, tok = grid.y + s.tok0;
    const uint hd = s.head_dim;
    const uint quads = (hd + 3u) / 4u;
    const uint kvh = h / s.per_kv;
    const uint pos = s.pos0 + tok;
    uint n = pos + 1u;
    if (s.window > 0 && n > s.window) n = s.window;
    n = min(n, s.cap);
    // Key j is position pos + 1 - n + j; n <= cap, so its ring row wraps at most once.
    const uint base = (pos + 1u - n) % s.cap;
    const uint simds = s.threads / s.width;
    const ulong th = ulong(tok) * s.heads + h;
    device float *sc = scores + (ulong(grid.y) * s.heads + h) * s.span;

    const device float *qh = q + th * hd;
    const device float *rh = r + th * s.rel_len;
    float4 qreg[INK2_ATT_Q_TERMS];
    for (uint t = 0; t < INK2_ATT_Q_TERMS; ++t) {
        const uint c = lane + t * s.width;
        qreg[t] = c < quads ? ink2_att_load(qh + 4u * c, hd - 4u * c) : float4(0.0f);
    }
    const float sink = -sinks[th];
    // The zero sink logit bounds the softmax shift from below.
    float mx = 0.0f;
    for (uint j = simd; j < n; j += simds) {
        uint row = base + j;
        row -= row >= s.cap ? s.cap : 0u;
        const device float *kr = kcache + row * s.kv_dim + kvh * hd;
        float d = 0.0f;
        for (uint t = 0, c = lane; c < quads; ++t, c += s.width)
            d += dot(qreg[t], ink2_att_load(kr + 4u * c, hd - 4u * c));
        d *= s.scale;
        const uint dist = n - 1u - j;
        if (dist < s.extent) {
            const device float *rb = rel_bias + dist * s.rel_len;
            for (uint i = lane; i < s.rel_len; i += s.width) d = fma(rb[i], rh[i], d);
        }
        const float score = simd_sum(d) + sink;
        if (lane == 0) sc[j] = score;
        mx = max(mx, score);
    }
    if (lane == 0) tmax[simd] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);

    float m = 0.0f;
    for (uint w = 0; w < simds; ++w) m = max(m, tmax[w]);
    float e = 0.0f;
    for (uint j = tid; j < n; j += s.threads) {
        const float x = exp(sc[j] - m);
        sc[j] = x;
        e += x;
    }
    e = simd_sum(e);
    if (lane == 0) tsum[simd] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);

    float denom = exp(-m);
    for (uint w = 0; w < simds; ++w) denom += tsum[w];

    const uint groups = s.threads / quads;
    const uint c = tid % quads;
    const uint have = hd - 4u * c;
    if (tid < groups * quads) {
        const device float *vb = vcache + kvh * hd + 4u * c;
        float4 acc = float4(0.0f);
        for (uint j = tid / quads; j < n; j += groups) {
            uint row = base + j;
            row -= row >= s.cap ? s.cap : 0u;
            acc += sc[j] * ink2_att_load(vb + row * s.kv_dim, have);
        }
        gacc[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < quads) {
        float4 acc = float4(0.0f);
        for (uint g = 0; g < groups; ++g) acc += gacc[g * quads + tid];
        acc /= denom;
        for (uint i = 0; i < min(have, 4u); ++i) out[th * hd + 4u * tid + i] = acc[i];
    }
}
