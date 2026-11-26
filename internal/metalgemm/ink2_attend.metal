// Causal attention of the newest token over one KV slot: relative-position
// bias within extent, a per-head sink logit, a zero sink in the softmax
// denominator and grouped-query heads. One threadgroup per query head, in three
// phases joined by barriers: a SIMD group per key scores it, a thread per key
// exponentiates, a thread per (key group, four dims) sums the values.
//
// The cache is [cap][kv_dim] with position p at row p % cap; meta[0] is the
// newest position as an int. The scores scratch holds one float per (head,
// key), keys oldest first.

#include <metal_stdlib>
using namespace metal;

#define INK2_ATT_MAX_THREADS 512
#define INK2_ATT_MAX_SIMD 64
#define INK2_ATT_Q_TERMS 4

struct Ink2AttendShape {
    uint heads, per_kv, head_dim, rel_len;
    uint kv_dim, extent, window, cap;
    uint threads, width;
    float scale;
};

// Four dims of a row, of which the row holds only `have`. Unaligned: an
// operand sits at any float offset of its buffer.
static inline float4 ink2_att_load(const device float *p, uint have) {
    if (have >= 4) return float4(*(const device packed_float4 *)p);
    float4 v = float4(0.0f);
    for (uint i = 0; i < have; ++i) v[i] = p[i];
    return v;
}

#ifdef INK2_ATT_NAMED
// Key j's score before the SIMD sum: this lane's share of q.k, scaled, plus
// its share of the relative bias within extent.
static inline float ink2_att_part(thread const float4 *qreg, const device float *kcache, const device float *rel_bias,
                                  const device float *rh, constant Ink2AttendShape &s, uint base, uint n, uint j,
                                  uint kvh, uint hd, uint quads, uint lane) {
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
    return d;
}
#endif

// The host builds this source again with INK2_ATT_NAMED: the score and value
// loops then take several keys per step, their loads issued together, with
// every sum and max in the same order.
kernel void ink2_attend(device const float *q [[buffer(0)]],
                        device const float *r [[buffer(1)]],
                        device const float *sinks [[buffer(2)]],
                        device const float *rel_bias [[buffer(3)]],
                        device const float *kcache [[buffer(4)]],
                        device const float *vcache [[buffer(5)]],
                        device float *scores [[buffer(6)]],
                        device float *out [[buffer(7)]],
                        device const int *meta [[buffer(8)]],
                        constant Ink2AttendShape &s [[buffer(9)]],
                        uint3 grid [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tmax[INK2_ATT_MAX_SIMD];
    threadgroup float tsum[INK2_ATT_MAX_SIMD];
    threadgroup float4 gacc[INK2_ATT_MAX_THREADS];

    const uint h = grid.x;
    const uint hd = s.head_dim;
    const uint quads = (hd + 3u) / 4u;
    const uint kvh = h / s.per_kv;
    const int pos = meta[0];
    uint n = pos < 0 ? 0u : uint(pos) + 1u;
    if (s.window > 0 && n > s.window) n = s.window;
    n = min(n, s.cap);
    // Key j is position pos + 1 - n + j; n <= cap, so its ring row wraps at most once.
    const uint base = n == 0 ? 0u : (uint(pos) + 1u - n) % s.cap;
    const uint simds = s.threads / s.width;
    device float *sc = scores + h * s.cap;

    const device float *qh = q + h * hd;
    const device float *rh = r + h * s.rel_len;
    float4 qreg[INK2_ATT_Q_TERMS];
    for (uint t = 0; t < INK2_ATT_Q_TERMS; ++t) {
        const uint c = lane + t * s.width;
        qreg[t] = c < quads ? ink2_att_load(qh + 4u * c, hd - 4u * c) : float4(0.0f);
    }
    const float sink = -sinks[h];
    // The zero sink logit bounds the softmax shift from below.
    float mx = 0.0f;
#ifdef INK2_ATT_NAMED
    uint j = simd;
    for (; j + 3u * simds < n; j += 4u * simds) {
        const float d0 = ink2_att_part(qreg, kcache, rel_bias, rh, s, base, n, j, kvh, hd, quads, lane);
        const float d1 = ink2_att_part(qreg, kcache, rel_bias, rh, s, base, n, j + simds, kvh, hd, quads, lane);
        const float d2 = ink2_att_part(qreg, kcache, rel_bias, rh, s, base, n, j + 2u * simds, kvh, hd, quads, lane);
        const float d3 = ink2_att_part(qreg, kcache, rel_bias, rh, s, base, n, j + 3u * simds, kvh, hd, quads, lane);
        const float s0 = simd_sum(d0) + sink, s1 = simd_sum(d1) + sink;
        const float s2 = simd_sum(d2) + sink, s3 = simd_sum(d3) + sink;
        if (lane == 0) {
            sc[j] = s0;
            sc[j + simds] = s1;
            sc[j + 2u * simds] = s2;
            sc[j + 3u * simds] = s3;
        }
        mx = max(mx, s0);
        mx = max(mx, s1);
        mx = max(mx, s2);
        mx = max(mx, s3);
    }
    for (; j < n; j += simds) {
        const float score = simd_sum(ink2_att_part(qreg, kcache, rel_bias, rh, s, base, n, j, kvh, hd, quads, lane)) + sink;
        if (lane == 0) sc[j] = score;
        mx = max(mx, score);
    }
#else
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
#endif
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
#ifdef INK2_ATT_NAMED
        uint j = tid / quads;
        for (; j + 3u * groups < n; j += 4u * groups) {
            uint r0 = base + j, r1 = r0 + groups, r2 = r1 + groups, r3 = r2 + groups;
            r0 -= r0 >= s.cap ? s.cap : 0u;
            r1 -= r1 >= s.cap ? s.cap : 0u;
            r2 -= r2 >= s.cap ? s.cap : 0u;
            r3 -= r3 >= s.cap ? s.cap : 0u;
            const float w0 = sc[j], w1 = sc[j + groups], w2 = sc[j + 2u * groups], w3 = sc[j + 3u * groups];
            const float4 v0 = ink2_att_load(vb + r0 * s.kv_dim, have), v1 = ink2_att_load(vb + r1 * s.kv_dim, have);
            const float4 v2 = ink2_att_load(vb + r2 * s.kv_dim, have), v3 = ink2_att_load(vb + r3 * s.kv_dim, have);
            acc += w0 * v0;
            acc += w1 * v1;
            acc += w2 * v2;
            acc += w3 * v3;
        }
        for (; j < n; j += groups) {
            uint row = base + j;
            row -= row >= s.cap ? s.cap : 0u;
            acc += sc[j] * ink2_att_load(vb + row * s.kv_dim, have);
        }
#else
        for (uint j = tid / quads; j < n; j += groups) {
            uint row = base + j;
            row -= row >= s.cap ? s.cap : 0u;
            acc += sc[j] * ink2_att_load(vb + row * s.kv_dim, have);
        }
#endif
        gacc[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < quads) {
        float4 acc = float4(0.0f);
        for (uint g = 0; g < groups; ++g) acc += gacc[g * quads + tid];
        acc /= denom;
        for (uint i = 0; i < min(have, 4u); ++i) out[h * hd + 4u * tid + i] = acc[i];
    }
}
