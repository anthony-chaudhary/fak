#include <metal_stdlib>
using namespace metal;

struct Ink2EmbedShape {
    uint rows, cols, byteOffset, pos;
};

// One thread per value of the row idx[0] of a Q5_K table, plus one that writes
// pos after it. Each value is the host's d1*q - m1 with the single rounding of
// the fused multiply-subtract the host compiles it to.
kernel void ink2_dec_embed_q5_K(device const uchar *w [[buffer(0)]],
                                device const uint *idx [[buffer(1)]],
                                device float *out [[buffer(2)]],
                                constant Ink2EmbedShape &s [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= s.cols) {
        if (i == s.cols) out[i] = as_type<float>(s.pos);
        return;
    }
    const ulong row = ulong(min(idx[0], s.rows - 1u));
    device const uchar *b = w + s.byteOffset + row * ulong(s.cols / 256u) * 176ul + ulong(i / 256u) * 176ul;
    const uint e = i % 256u, j = e / 64u, hi = (e % 64u) / 32u, l = e % 32u, is = 2u * j + hi;
    const float d = float(as_type<half>(ushort(b[0] | (ushort(b[1]) << 8))));
    const float dmin = float(as_type<half>(ushort(b[2] | (ushort(b[3]) << 8))));
    device const uchar *sc = b + 4;
    uint scale, m;
    if (is < 4u) {
        scale = sc[is] & 63u;
        m = sc[is + 4u] & 63u;
    } else {
        scale = (sc[is + 4u] & 0x0fu) | ((sc[is - 4u] >> 6) << 4);
        m = (sc[is + 4u] >> 4) | ((sc[is] >> 6) << 4);
    }
    const float d1 = d * float(scale), m1 = dmin * float(m);
    device const uchar *qh = b + 16, *ql = b + 48;
    uint q = ql[32u * j + l];
    q = hi != 0u ? q >> 4 : q & 0x0fu;
    if ((qh[l] >> is) & 1u) q += 16u;
    out[i] = fma(d1, float(q), -m1);
}

struct Ink2CopyShape {
    uint n[8];
};

// Copies up to eight spans, span k from src k to dst k, as raw words.
kernel void ink2_dec_copy8(device const uint *s0 [[buffer(0)]], device const uint *s1 [[buffer(1)]],
                           device const uint *s2 [[buffer(2)]], device const uint *s3 [[buffer(3)]],
                           device const uint *s4 [[buffer(4)]], device const uint *s5 [[buffer(5)]],
                           device const uint *s6 [[buffer(6)]], device const uint *s7 [[buffer(7)]],
                           device uint *d0 [[buffer(8)]], device uint *d1 [[buffer(9)]],
                           device uint *d2 [[buffer(10)]], device uint *d3 [[buffer(11)]],
                           device uint *d4 [[buffer(12)]], device uint *d5 [[buffer(13)]],
                           device uint *d6 [[buffer(14)]], device uint *d7 [[buffer(15)]],
                           constant Ink2CopyShape &s [[buffer(16)]],
                           uint2 at [[thread_position_in_grid]]) {
    const uint i = at.x, k = at.y;
    if (i >= s.n[k]) return;
    switch (k) {
    case 0: d0[i] = s0[i]; break;
    case 1: d1[i] = s1[i]; break;
    case 2: d2[i] = s2[i]; break;
    case 3: d3[i] = s3[i]; break;
    case 4: d4[i] = s4[i]; break;
    case 5: d5[i] = s5[i]; break;
    case 6: d6[i] = s6[i]; break;
    default: d7[i] = s7[i]; break;
    }
}
