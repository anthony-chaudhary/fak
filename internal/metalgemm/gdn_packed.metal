// gdn_packed.metal — 8-row B-tree SIMDgroup packing for Qwen3.5/3.8 GatedDeltaNet linear attention
//
// Key architectural mechanisms:
//  1. 8-row SIMDgroup packing: 32 threads in each SIMDgroup are organized as 8 rows (row_id = lane_id / 4)
//     with 4 lanes per row (sub_lane = lane_id % 4).
//  2. Register footprint: each lane holds 32 state elements for its value row (4 * 32 = 128 = D_k),
//     held in registers as float4 st[8].
//  3. B-Tree local reduction & 4-lane intra-row butterfly shuffle: 8-part binary addition tree followed by
//     simd_shuffle_xor(1) and simd_shuffle_xor(2) reduces 128 elements within each row with ZERO cross-row leakage.
//  4. Fused recurrent scan: in-place delta update, readout, RMSNorm, and gated SiLU writeback.

#include <metal_stdlib>
using namespace metal;

constant int DK = 128;
constant int ROWS_PER_SIMD = 8;
constant int LANES_PER_ROW = 4;

#ifndef GDN_SILU_DEFINED
#define GDN_SILU_DEFINED
inline float gdn_silu(float x) { return x / (1.0f + exp(-x)); }
inline float gdn_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }
#endif

// One threadgroup per value head. Threads are partitioned into SIMDgroups of 32 lanes.
// Each SIMDgroup owns 8 rows of value dimension; 4 lanes cooperatively process 128 elements of D_k per row.
kernel void gdn_recurrent_packed_8row(
    device const float *convOut [[buffer(0)]],
    device const float *qNorm [[buffer(1)]],
    device const float *kNorm [[buffer(2)]],
    device const float *z [[buffer(3)]],
    device const float *b [[buffer(4)]],
    device const float *a [[buffer(5)]],
    device const float *aLog [[buffer(6)]],
    device const float *dtBias [[buffer(7)]],
    device const float *norm [[buffer(8)]],
    device float *state [[buffer(9)]],
    device float *core [[buffer(10)]],
    constant int& tokens [[buffer(11)]],
    constant int& convDim [[buffer(12)]],
    constant int& nK [[buffer(13)]],
    constant int& nV [[buffer(14)]],
    constant int& kHd [[buffer(15)]],
    constant int& vHd [[buffer(16)]],
    constant float& eps [[buffer(17)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    if (head >= (uint)nV) return;

    int repeat = nV / nK;
    int keyHead = (int)head / repeat;
    int keyDim = nK * kHd;

    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    uint row_id = lane_id / 4;
    uint sub_lane = lane_id % 4;
    uint v_idx = simd_id * 8 + row_id;
    uint k_start = sub_lane * 32;

    float4 st[8];
    if (v_idx < (uint)vHd) {
        for (int j = 0; j < 8; ++j) {
            float4 s;
            for (int c = 0; c < 4; ++c) {
                s[c] = state[((long)head * kHd + (sub_lane * 32 + j * 4 + c)) * vHd + v_idx];
            }
            st[j] = s;
        }
    } else {
        for (int j = 0; j < 8; ++j) {
            st[j] = float4(0.0f);
        }
    }

    threadgroup float tg_sq[32];

    for (int token = 0; token < tokens; ++token) {
        device const float *qRow = qNorm + ((long)token * nK + keyHead) * kHd;
        device const float *kRow = kNorm + ((long)token * nK + keyHead) * kHd;

        float4 k_vec[8];
        float4 q_vec[8];
        device const float4 *k_ptr = (device const float4 *)(kRow + k_start);
        device const float4 *q_ptr = (device const float4 *)(qRow + k_start);
        for (int j = 0; j < 8; ++j) {
            k_vec[j] = k_ptr[j];
            q_vec[j] = q_ptr[j];
        }

        float beta = 1.0f / (1.0f + exp(-b[(long)token * nV + head]));
        float decay = exp(-exp(aLog[head]) * gdn_softplus(a[(long)token * nV + head] + dtBias[head]));

        float v_row = 0.0f;
        if (v_idx < (uint)vHd) {
            long valueIndex = (long)token * convDim + 2L * keyDim + (long)head * vHd + v_idx;
            v_row = convOut[valueIndex];
        }

        float part[8];
        for (int j = 0; j < 8; ++j) {
            st[j] *= decay;
            part[j] = dot(st[j], k_vec[j]);
        }
        float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) +
                       ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);

        float delta = (v_row - kv_mem) * beta;

        float r_part[8];
        for (int j = 0; j < 8; ++j) {
            st[j] += delta * k_vec[j];
            r_part[j] = dot(st[j], q_vec[j]);
        }
        float readout = ((r_part[0] + r_part[1]) + (r_part[2] + r_part[3])) +
                        ((r_part[4] + r_part[5]) + (r_part[6] + r_part[7]));
        readout += simd_shuffle_xor(readout, 1);
        readout += simd_shuffle_xor(readout, 2);

        float sq = (sub_lane == 0 && v_idx < (uint)vHd) ? (readout * readout) : 0.0f;
        float simd_sq = simd_sum(sq);
        if (lane_id == 0) {
            tg_sq[simd_id] = simd_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = 0.0f;
        uint num_simds = lanes / 32;
        for (uint s = 0; s < num_simds; ++s) {
            total_sq += tg_sq[s];
        }
        float inv = rsqrt(total_sq / (float)vHd + eps);

        if (sub_lane == 0 && v_idx < (uint)vHd) {
            long vd = (long)head * vHd + v_idx;
            core[(long)token * nV * vHd + vd] = norm[v_idx] * readout * inv * gdn_silu(z[(long)token * nV * vHd + vd]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (v_idx < (uint)vHd) {
        for (int j = 0; j < 8; ++j) {
            for (int c = 0; c < 4; ++c) {
                state[((long)head * kHd + (sub_lane * 32 + j * 4 + c)) * vHd + v_idx] = st[j][c];
            }
        }
    }
}
