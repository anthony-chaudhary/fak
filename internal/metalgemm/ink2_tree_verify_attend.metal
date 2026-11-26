// ink2v_attend over a token tree (parent[t] < t, -1 for the root at row 0).
// Row t is the query at position pos + depth(t) and attends the contiguous
// position range ending there, exactly as ink2v_attend's row of that position
// does, except that a position pos + d >= pos reads the cache row pos + a_d of
// its depth-d ancestor a_d: the committed prefix plus its own root path, never
// a sibling branch. A chain tree reads ink2v_attend's rows, so it gives its
// bits. Like ink2_verify_attend.metal this source carries no FP_CONTRACT
// pragma: the two must compile alike.

#include <metal_stdlib>
using namespace metal;

#define INK2_ATT_MAX_THREADS 512
#define INK2_ATT_MAX_SIMD 64
#define INK2_ATT_Q_TERMS 4
#define INK2T_MAX_NODES 32

struct Ink2TAttendShape {
    uint heads, per_kv, head_dim, rel_len;
    uint kv_dim, extent, window, cap;
    uint threads, width;
    uint span;
    uint tok0;
    uint pos_from_device;
    int pos0;
    float scale;
};

static inline float4 ink2t_att_load(const device float *p, uint have) {
    if (have >= 4) return float4(*(const device packed_float4 *)p);
    float4 v = float4(0.0f);
    for (uint i = 0; i < have; ++i) v[i] = p[i];
    return v;
}

// The ring row of key j of the row's range.
#define INK2T_ATT_ROW(j) (((first + (j)) >= uint(pos0) ? uint(pos0) + path[first + (j) - uint(pos0)] : first + (j)) % s.cap)

#define INK2T_ATT_SCORE(j, score)                                                                           \
    float score;                                                                                            \
    {                                                                                                       \
        const uint row = INK2T_ATT_ROW(j);                                                                  \
        const device float *kr = kcache + row * s.kv_dim + kvh * hd;                                        \
        float d = 0.0f;                                                                                     \
        for (uint t = 0, c = lane; c < quads; ++t, c += s.width)                                            \
            d += dot(qreg[t], ink2t_att_load(kr + 4u * c, hd - 4u * c));                                    \
        d *= s.scale;                                                                                       \
        const uint dist = n - 1u - (j);                                                                     \
        if (dist < s.extent) {                                                                              \
            const device float *rb = rel_bias + dist * s.rel_len;                                           \
            for (uint i = lane; i < s.rel_len; i += s.width) d = fma(rb[i], rh[i], d);                      \
        }                                                                                                   \
        score = simd_sum(d) + sink;                                                                         \
    }

kernel void ink2t_attend(device const float *q [[buffer(0)]],
                         device const float *r [[buffer(1)]],
                         device const float *sinks [[buffer(2)]],
                         device const float *rel_bias [[buffer(3)]],
                         device const float *kcache [[buffer(4)]],
                         device const float *vcache [[buffer(5)]],
                         device float *scores [[buffer(6)]],
                         device float *out [[buffer(7)]],
                         device const int *posDev [[buffer(8)]],
                         device const int *parent [[buffer(9)]],
                         constant Ink2TAttendShape &s [[buffer(10)]],
                         uint3 grid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tmax[INK2_ATT_MAX_SIMD];
    threadgroup float tsum[INK2_ATT_MAX_SIMD];
    threadgroup float4 gacc[INK2_ATT_MAX_THREADS];
    // path[d]: the row's depth-d ancestor (itself at its own depth).
    threadgroup uint path[INK2T_MAX_NODES];
    threadgroup uint depthOf;

    const uint h = grid.x, tok = grid.y + s.tok0;
    if (tid == 0) {
        uint up[INK2T_MAX_NODES];
        uint d = 0;
        int at = int(tok);
        up[d++] = uint(at);
        while (d < INK2T_MAX_NODES && parent[at] >= 0 && parent[at] < at) {
            at = parent[at];
            up[d++] = uint(at);
        }
        for (uint i = 0; i < d; ++i) path[i] = up[d - 1u - i];
        depthOf = d - 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint hd = s.head_dim;
    const uint quads = (hd + 3u) / 4u;
    const uint kvh = h / s.per_kv;
    const int pos0 = s.pos_from_device != 0 ? posDev[0] : s.pos0;
    const int pos = pos0 + int(depthOf);
    uint n = pos < 0 ? 0u : uint(pos) + 1u;
    if (s.window > 0 && n > s.window) n = s.window;
    n = min(n, s.cap);
    const uint first = n == 0 ? 0u : uint(pos) + 1u - n;
    const uint simds = s.threads / s.width;
    const ulong th = ulong(tok) * s.heads + h;
    device float *sc = scores + (ulong(grid.y) * s.heads + h) * s.span;

    const device float *qh = q + th * hd;
    const device float *rh = r + th * s.rel_len;
    float4 qreg[INK2_ATT_Q_TERMS];
    for (uint t = 0; t < INK2_ATT_Q_TERMS; ++t) {
        const uint c = lane + t * s.width;
        qreg[t] = c < quads ? ink2t_att_load(qh + 4u * c, hd - 4u * c) : float4(0.0f);
    }
    const float sink = -sinks[th];
    float mx = 0.0f;
    for (uint j = simd; j < n; j += simds) {
        INK2T_ATT_SCORE(j, score)
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
            const uint row = INK2T_ATT_ROW(j);
            acc += sc[j] * ink2t_att_load(vb + row * s.kv_dim, have);
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
