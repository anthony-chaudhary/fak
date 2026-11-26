// The tiled attention of a batched inkling2 prefill chunk wider than any
// verify chunk: ink2p_attend's scores and softmax, with an online softmax over
// blocks of keys and fp32 8x8 simdgroup_matrix products for Q K^T and P V. It
// sums in another order than ink2p_attend, so a row is not bit-equal to it.

#include <metal_stdlib>
using namespace metal;

#define FL_BQ 16    // queries of one head per threadgroup
#define FL_BK 32    // keys per block: 8 per SIMD group
#define FL_HD 128   // head dim
#define FL_RL 32    // most relative-bias terms
#define FL_PITCH (FL_BK + 4)

struct Ink2PFlashShape {
    uint heads, perKV, relLen, kvDim;
    uint extent, window, cap, tokens;
    uint pos0;
    float scale;
};

// A threadgroup per (head, FL_BQ queries); grid.y counts the query tiles.
kernel void ink2p_flash(device const float *q [[buffer(0)]],
                        device const float *r [[buffer(1)]],
                        device const float *sinks [[buffer(2)]],
                        device const float *relBias [[buffer(3)]],
                        device const float *kcache [[buffer(4)]],
                        device const float *vcache [[buffer(5)]],
                        device float *out [[buffer(6)]],
                        constant Ink2PFlashShape &s [[buffer(7)]],
                        uint2 grid [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[FL_BQ * FL_HD];
    // Scores, then probabilities, of the block; after the last block, O.
    threadgroup float ps[FL_BQ * FL_HD];
    threadgroup float rq[FL_BQ * FL_RL];
    threadgroup float diag[2 * 64];
    threadgroup float rowM[FL_BQ], rowL[FL_BQ], rowA[FL_BQ], rowSink[FL_BQ];
    threadgroup uint rowPos[FL_BQ], rowFirst[FL_BQ];

    const uint h = grid.x, t0 = grid.y * FL_BQ;
    const uint tLast = min(t0 + FL_BQ, s.tokens) - 1;
    const uint kvh = h / s.perKV;
    const uint headOff = kvh * FL_HD;

    for (uint e = tid; e < FL_BQ * FL_HD / 4; e += 128) {
        const uint row = e / (FL_HD / 4), c = 4 * (e % (FL_HD / 4));
        const uint t = min(t0 + row, tLast);
        *(threadgroup float4 *)(qs + row * FL_HD + c) = *(device const float4 *)(q + (ulong(t) * s.heads + h) * FL_HD + c);
    }
    for (uint e = tid; e < FL_BQ * s.relLen; e += 128) {
        const uint row = e / s.relLen, i = e % s.relLen;
        rq[row * FL_RL + i] = r[(ulong(min(t0 + row, tLast)) * s.heads + h) * s.relLen + i];
    }
    if (tid < FL_BQ) {
        const uint t = min(t0 + tid, tLast), pos = s.pos0 + t;
        uint n = pos + 1u;
        if (s.window > 0 && n > s.window) n = s.window;
        n = min(n, s.cap);
        rowPos[tid] = pos;
        rowFirst[tid] = pos + 1u - n;
        rowSink[tid] = -sinks[ulong(t) * s.heads + h];
        // The zero sink logit: the shift starts at 0 with its exp(0) in the sum.
        rowM[tid] = 0.0f;
        rowL[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD group g owns O columns [32g, 32g+32) of both 8-query blocks.
    simdgroup_float8x8 o[2][4];
    for (uint i = 0; i < 2; i++)
        for (uint c = 0; c < 4; c++) o[i][c] = simdgroup_float8x8(0.0f);

    const uint kFirst = rowFirst[0], kLast = rowPos[tLast - t0];
    // Blocks start 8-aligned, so with cap a multiple of 8 no 8-key run wraps the ring.
    for (uint kb = kFirst & ~7u; kb <= kLast; kb += FL_BK) {
        {
            const uint kp = kb + 8 * simd;
            device const float *kr = kcache + ulong(kp % s.cap) * s.kvDim + headOff;
            simdgroup_float8x8 a0, a1, b, s0 = simdgroup_float8x8(0.0f), s1 = simdgroup_float8x8(0.0f);
            for (uint d = 0; d < FL_HD; d += 8) {
                simdgroup_load(a0, qs + d, FL_HD);
                simdgroup_load(a1, qs + 8 * FL_HD + d, FL_HD);
                simdgroup_load(b, kr + d, s.kvDim, ulong2(0), true);
                simdgroup_multiply_accumulate(s0, a0, b, s0);
                simdgroup_multiply_accumulate(s1, a1, b, s1);
            }
            simdgroup_store(s0, ps + 8 * simd, FL_PITCH);
            simdgroup_store(s1, ps + 8 * FL_PITCH + 8 * simd, FL_PITCH);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Eight consecutive lanes share a query row, four keys each.
            const uint row = tid / 8, c0 = 4 * (tid % 8);
            const uint pos = rowPos[row], first = rowFirst[row];
            float sc[4], mx = -INFINITY;
            for (uint i = 0; i < 4; i++) {
                const uint kp = kb + c0 + i;
                float v = -INFINITY;
                if (kp >= first && kp <= pos) {
                    v = ps[row * FL_PITCH + c0 + i] * s.scale;
                    const uint dist = pos - kp;
                    if (dist < s.extent) {
                        device const float *rb = relBias + ulong(dist) * s.relLen;
                        for (uint j = 0; j < s.relLen; j++) v = fma(rb[j], rq[row * FL_RL + j], v);
                    }
                    v += rowSink[row];
                }
                sc[i] = v;
                mx = max(mx, v);
            }
            mx = max(mx, simd_shuffle_xor(mx, 1));
            mx = max(mx, simd_shuffle_xor(mx, 2));
            mx = max(mx, simd_shuffle_xor(mx, 4));
            const float mOld = rowM[row], mNew = max(mOld, mx);
            float sum = 0.0f;
            for (uint i = 0; i < 4; i++) {
                const float p = sc[i] == -INFINITY ? 0.0f : exp(sc[i] - mNew);
                ps[row * FL_PITCH + c0 + i] = p;
                sum += p;
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 2);
            sum += simd_shuffle_xor(sum, 4);
            if (tid % 8 == 0) {
                const float alpha = exp(mOld - mNew);
                rowA[row] = alpha;
                rowM[row] = mNew;
                rowL[row] = rowL[row] * alpha + sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint b = tid / 64, i = (tid / 8) % 8, j = tid % 8;
            diag[tid] = i == j ? rowA[8 * b + i] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 dm[2], pa[2], vb;
            simdgroup_load(dm[0], diag, 8);
            simdgroup_load(dm[1], diag + 64, 8);
            for (uint i = 0; i < 2; i++)
                for (uint c = 0; c < 4; c++) simdgroup_multiply(o[i][c], dm[i], o[i][c]);
            for (uint kk = 0; kk < FL_BK; kk += 8) {
                simdgroup_load(pa[0], ps + kk, FL_PITCH);
                simdgroup_load(pa[1], ps + 8 * FL_PITCH + kk, FL_PITCH);
                device const float *vr = vcache + ulong((kb + kk) % s.cap) * s.kvDim + headOff + 32 * simd;
                for (uint c = 0; c < 4; c++) {
                    simdgroup_load(vb, vr + 8 * c, s.kvDim);
                    simdgroup_multiply_accumulate(o[0][c], pa[0], vb, o[0][c]);
                    simdgroup_multiply_accumulate(o[1][c], pa[1], vb, o[1][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++)
        for (uint c = 0; c < 4; c++) simdgroup_store(o[i][c], ps + 8 * i * FL_HD + 32 * simd + 8 * c, FL_HD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < FL_BQ * FL_HD; e += 128) {
        const uint row = e / FL_HD, d = e % FL_HD;
        if (t0 + row <= tLast) out[(ulong(t0 + row) * s.heads + h) * FL_HD + d] = ps[e] / rowL[row];
    }
}
