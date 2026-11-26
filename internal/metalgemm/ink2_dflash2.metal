#include <metal_stdlib>
using namespace metal;

// Elementwise kernels the DFlash2 draft block needs beside ink2_draft.metal:
// per-head RMS norm with neox RoPE (into row order or a context ring), a ring
// store, and the SwiGLU product.

struct Ink2DFRopeShape {
    uint rows, heads, dim, cap;
    // Nonzero writes row r to ring slot (pos+r)%cap instead of row r.
    uint ring;
    // Added to the int at meta: row r sits at position meta[0]+base+r.
    uint base;
    float eps, log2Theta;
};

// One threadgroup per (head, row). Every thread reads only the two elements it
// writes after the reduction barrier, so x may alias dst in row order.
kernel void ink2_df_qk_rope(device const float *x [[buffer(0)]],
                            device const float *w [[buffer(1)]],
                            device float *dst [[buffer(2)]],
                            device const int *meta [[buffer(3)]],
                            constant Ink2DFRopeShape &s [[buffer(4)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 tid2 [[thread_position_in_threadgroup]],
                            uint2 tpg [[threads_per_threadgroup]]) {
    const uint h = group.x, row = group.y, dim = s.dim, hd = dim / 2, tid = tid2.x, threads = tpg.x;
    device const float *src = x + (ulong(row) * s.heads + h) * dim;
    threadgroup float red[256];
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += threads) ss += src[i] * src[i];
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = precise::rsqrt(red[0] / float(dim) + s.eps);
    const int pos = meta[0] + int(s.base) + int(row);
    const ulong out = s.ring != 0 ? ulong(uint(pos) % s.cap) : ulong(row);
    device float *o = dst + (out * s.heads + h) * dim;
    for (uint i = tid; i < hd; i += threads) {
        const float a = src[i] * scale * w[i], b = src[i + hd] * scale * w[i + hd];
        const float freq = precise::exp2(-float(2 * i) / float(dim) * s.log2Theta);
        float c;
        const float sn = precise::sincos(float(pos) * freq, c);
        o[i] = a * c - b * sn;
        o[i + hd] = b * c + a * sn;
    }
}

struct Ink2DFRingShape {
    uint rows, width, cap, base;
};

kernel void ink2_df_ring_store(device const float *src [[buffer(0)]],
                               device float *ring [[buffer(1)]],
                               device const int *meta [[buffer(2)]],
                               constant Ink2DFRingShape &s [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= s.rows * s.width) return;
    const uint row = i / s.width, col = i % s.width;
    const uint slot = uint(meta[0] + int(s.base) + int(row)) % s.cap;
    ring[ulong(slot) * s.width + col] = src[i];
}

kernel void ink2_df_silu_mul(device const float *g [[buffer(0)]],
                             device const float *u [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant uint &n [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = g[i];
    out[i] = v / (1.0f + precise::exp(-v)) * u[i];
}
