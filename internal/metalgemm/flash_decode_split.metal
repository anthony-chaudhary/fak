// flash_decode_split.metal — Metal Shading Language kernels for Split-KV FlashDecoding.
// Prior-art: vllm-project/vllm-metal:paged_ops.cpp:occupancy-gate
//
// Stage 1 (split_kv_decode_stage1):
//   Partitions the sequence dimension (kv_tokens) along grid.z into chunks of chunk_size (512 tokens).
//   Each threadgroup (32 threads / 1 SIMDgroup) owns one (batch, head, split).
//   Loads Q once into registers, streams K and V, computes Q·K via SIMD shuffle reduction,
//   maintains local running max (m_s), local sum (l_s), and partial accumulator (acc_s).
//   Writes partial_out, partial_max, partial_sum to global buffers.
//
// Stage 2 (split_kv_decode_stage2):
//   Reduction kernel combining partition logits across S splits for each (batch, head) into final output.
//   Computes global max M = max(m_s) and normalizer L = sum(l_s * exp(m_s - M)).
//   Normalizes and writes the final output vector Out = (1/L) * sum(acc_s * exp(m_s - M)).

#include <metal_stdlib>
using namespace metal;

struct SplitKVParams {
    uint batch;
    uint q_tokens;
    uint kv_tokens;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint num_splits;
    uint chunk_size;
    float scale;
    int causal;
    int sliding_window;
};

// split_kv_decode_stage1: Pass 1 of Split-KV FlashDecoding
kernel void split_kv_decode_stage1(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* partial_out [[buffer(3)]],
    device float* partial_max [[buffer(4)]],
    device float* partial_sum [[buffer(5)]],
    constant SplitKVParams& p [[buffer(6)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint h = tg_pos.x;
    uint b_qi = tg_pos.y;
    uint s = tg_pos.z;

    uint total_bq = p.batch * p.q_tokens;
    if (h >= p.num_heads || b_qi >= total_bq || s >= p.num_splits) return;

    uint b = b_qi / p.q_tokens;
    uint qi = b_qi % p.q_tokens;

    uint grp = p.num_heads / p.num_kv_heads;
    uint kvh = h / grp;

    uint k_start = s * p.chunk_size;
    uint k_end = min((s + 1) * p.chunk_size, p.kv_tokens);

    // Q offset for this head & batch
    uint q_offset = (b * p.q_tokens + qi) * (p.num_heads * p.head_dim) + h * p.head_dim;

    // Load Q into registers for this lane (up to 256 dimensions: 8 dims per lane)
    float q_reg[8];
    float acc_reg[8];
    for (uint i = 0; i < 8; i++) {
        acc_reg[i] = 0.0f;
        uint d = simd_lane + i * 32;
        if (d < p.head_dim) {
            q_reg[i] = Q[q_offset + d];
        } else {
            q_reg[i] = 0.0f;
        }
    }

    float m_s = -1e30f;
    float l_s = 0.0f;

    uint global_q_pos = qi;
    if (p.kv_tokens >= p.q_tokens) {
        global_q_pos = (p.kv_tokens - p.q_tokens) + qi;
    }

    for (uint kj = k_start; kj < k_end; kj++) {
        if (p.causal && kj > global_q_pos) {
            continue;
        }
        if (p.sliding_window > 0 && ((int)kj + p.sliding_window <= (int)global_q_pos)) {
            continue;
        }

        uint k_offset = (b * p.kv_tokens + kj) * (p.num_kv_heads * p.head_dim) + kvh * p.head_dim;
        float dot_part = 0.0f;
        for (uint i = 0; i < 8; i++) {
            uint d = simd_lane + i * 32;
            if (d < p.head_dim) {
                dot_part += q_reg[i] * K[k_offset + d];
            }
        }

        float dot = simd_sum(dot_part);
        float score = dot * p.scale;

        float new_m = max(m_s, score);
        float alpha = (m_s > -1e30f) ? exp(m_s - new_m) : 0.0f;
        float p_val = exp(score - new_m);
        l_s = l_s * alpha + p_val;
        m_s = new_m;

        uint v_offset = (b * p.kv_tokens + kj) * (p.num_kv_heads * p.head_dim) + kvh * p.head_dim;
        for (uint i = 0; i < 8; i++) {
            uint d = simd_lane + i * 32;
            if (d < p.head_dim) {
                acc_reg[i] = acc_reg[i] * alpha + p_val * V[v_offset + d];
            }
        }
    }

    // Write partial outputs for this split
    uint split_out_offset = ((b_qi * p.num_heads + h) * p.num_splits + s) * p.head_dim;
    for (uint i = 0; i < 8; i++) {
        uint d = simd_lane + i * 32;
        if (d < p.head_dim) {
            partial_out[split_out_offset + d] = acc_reg[i];
        }
    }

    if (simd_lane == 0) {
        uint meta_idx = (b_qi * p.num_heads + h) * p.num_splits + s;
        partial_max[meta_idx] = m_s;
        partial_sum[meta_idx] = l_s;
    }
}

// split_kv_decode_stage2: Pass 2 of Split-KV FlashDecoding
kernel void split_kv_decode_stage2(
    device const float* partial_out [[buffer(0)]],
    device const float* partial_max [[buffer(1)]],
    device const float* partial_sum [[buffer(2)]],
    device float* Out [[buffer(3)]],
    constant SplitKVParams& p [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint h = tg_pos.x;
    uint b_qi = tg_pos.y;

    uint total_bq = p.batch * p.q_tokens;
    if (h >= p.num_heads || b_qi >= total_bq) return;

    uint meta_base = (b_qi * p.num_heads + h) * p.num_splits;

    // Step 1: Find global max M across all S splits
    float thread_max = -1e30f;
    for (uint s = simd_lane; s < p.num_splits; s += 32) {
        float ms = partial_max[meta_base + s];
        if (ms > thread_max) {
            thread_max = ms;
        }
    }
    float M = simd_max(thread_max);

    // Step 2: Compute normalizer L = sum(l_s * exp(m_s - M))
    float thread_l = 0.0f;
    for (uint s = simd_lane; s < p.num_splits; s += 32) {
        float ms = partial_max[meta_base + s];
        float ls = partial_sum[meta_base + s];
        if (ms > -1e30f && ls > 0.0f) {
            thread_l += ls * exp(ms - M);
        }
    }
    float L = simd_sum(thread_l);

    // Step 3: Accumulate partition outputs
    float acc[8];
    for (uint i = 0; i < 8; i++) {
        acc[i] = 0.0f;
    }

    if (M > -1e30f && L > 0.0f) {
        float inv_L = 1.0f / L;
        for (uint s = 0; s < p.num_splits; s++) {
            float ms = partial_max[meta_base + s];
            if (ms > -1e30f) {
                float factor = exp(ms - M) * inv_L;
                if (factor > 0.0f) {
                    uint split_out_offset = (meta_base + s) * p.head_dim;
                    for (uint i = 0; i < 8; i++) {
                        uint d = simd_lane + i * 32;
                        if (d < p.head_dim) {
                            acc[i] += partial_out[split_out_offset + d] * factor;
                        }
                    }
                }
            }
        }
    }

    // Write final output vector
    uint out_offset = (b_qi * p.num_heads + h) * p.head_dim;
    for (uint i = 0; i < 8; i++) {
        uint d = simd_lane + i * 32;
        if (d < p.head_dim) {
            Out[out_offset + d] = acc[i];
        }
    }
}
