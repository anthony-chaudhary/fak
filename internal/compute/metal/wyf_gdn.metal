// wyf_gdn.metal — WYF Chunkwise Parallel Recurrence for GatedDeltaNet Linear Attention
//
// Mathematical Basis:
//   GatedDeltaNet linear recurrence:
//     S'_t = g_t * S_{t-1}
//     kvmem_t = S'_t * k_t
//     delta_t = beta_t * (v_t - kvmem_t)
//     S_t = S'_t + delta_t * k_t^T
//     out_t = S_t * q_t
//
//   WYF Chunkwise Parallel Formulation (chunk size C = 32, head dim D = 128):
//     1. Parallel Prefix Scan of Decays:
//        gamma_t = prod_{j=0}^t g_j
//        Gamma(s, t) = gamma_t / gamma_s  (for s <= t)
//     2. Inter-Chunk Projections from Incoming State S_0:
//        b_t = gamma_t * (S_0 * k_t)
//        O_inter_t = gamma_t * (S_0 * q_t)
//     3. Intra-Chunk Triangular Cross-Token Kernel:
//        c_{t, s} = Gamma(s, t) * (k_t^T * k_s)  (for s < t)
//        M_{t, s} = beta_t * c_{t, s},  with M_{t, t} = 1
//        R_t = beta_t * (v_t - b_t)
//     4. Triangular Forward Substitution:
//        delta_t = R_t - sum_{s < t} M_{t, s} * delta_s
//     5. Intra-Chunk Attention Readout:
//        P_{t, s} = Gamma(s, t) * (q_t^T * k_s)  (for s <= t)
//        O_intra_t = sum_{s <= t} P_{t, s} * delta_s
//        out_t = O_inter_t + O_intra_t
//     6. Chunk-End Recurrent State Update (Rank-1 Outer Product):
//        tilde_k_s = Gamma(s, C-1) * k_s
//        S_end = gamma_{C-1} * S_0 + sum_{s=0}^{C-1} delta_s * tilde_k_s^T
//
// Designed for Apple Silicon (M1/M2/M3/M4) unified memory and 32-lane SIMD groups.

#include <metal_stdlib>
using namespace metal;

constant int WYF_CHUNK_SIZE = 32;
constant int WYF_HEAD_DIM = 128;
constant int WYF_SIMD_WIDTH = 32;

// wyf_prefix_decay_scan: Parallel prefix product of per-token decays g_t in (0, 1]
// across a 32-thread SIMD group using warp shuffle up in 5 cycles.
kernel void wyf_prefix_decay_scan(
    device const float *g_in        [[buffer(0)]],
    device float *gamma_out        [[buffer(1)]],
    device float *log_gamma_out    [[buffer(2)]],
    constant int &numTokens        [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint3 groupPos                 [[threadgroup_position_in_grid]])
{
    int chunk_idx = groupPos.x;
    int head_idx = groupPos.y;
    int batch_idx = groupPos.z;
    int t0 = chunk_idx * WYF_CHUNK_SIZE;
    int lane = tid % WYF_SIMD_WIDTH;

    int cur_tokens = min(WYF_CHUNK_SIZE, numTokens - t0);
    if (lane >= cur_tokens) return;

    long offset = ((long)batch_idx * numTokens + (t0 + lane)) * groupPos.y + head_idx;
    float g_val = g_in[offset];
    if (g_val <= 0.0f) {
        g_val = 1e-7f;
    }

    // SIMD prefix scan via hardware shuffle
    float scan_prod = g_val;
    for (ushort delta = 1; delta < 32; delta <<= 1) {
        float up = simd_shuffle_up(scan_prod, delta);
        if (lane >= delta) {
            scan_prod *= up;
        }
    }

    long out_offset = ((long)batch_idx * numTokens + (t0 + lane)) * groupPos.y + head_idx;
    gamma_out[out_offset] = scan_prod;
    if (log_gamma_out != nullptr) {
        log_gamma_out[out_offset] = log(scan_prod);
    }
}

// wyf_cross_token_kernel: Computes 32x32 strictly lower-triangular cross-token
// kernel M_{t, s} and causal attention matrix P_{t, s} for one chunk.
kernel void wyf_cross_token_kernel(
    device const float *k_dev      [[buffer(0)]],
    device const float *q_dev      [[buffer(1)]],
    device const float *gamma_dev  [[buffer(2)]],
    device const float *beta_dev   [[buffer(3)]],
    device float *m_out            [[buffer(4)]],
    device float *p_out            [[buffer(5)]],
    constant int &numTokens        [[buffer(6)]],
    constant int &numHeads         [[buffer(7)]],
    uint3 groupPos                 [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]])
{
    int chunk_idx = groupPos.x;
    int head_idx = groupPos.y;
    int batch_idx = groupPos.z;
    int t0 = chunk_idx * WYF_CHUNK_SIZE;
    int cur_tokens = min(WYF_CHUNK_SIZE, numTokens - t0);

    // Cooperative loading of K and Q into threadgroup memory
    threadgroup float sh_k[WYF_CHUNK_SIZE * WYF_HEAD_DIM];
    threadgroup float sh_q[WYF_CHUNK_SIZE * WYF_HEAD_DIM];
    threadgroup float sh_gamma[WYF_CHUNK_SIZE];
    threadgroup float sh_beta[WYF_CHUNK_SIZE];

    // Load gamma and beta
    if (tid < (uint)cur_tokens) {
        long gb_off = ((long)batch_idx * numTokens + (t0 + tid)) * numHeads + head_idx;
        sh_gamma[tid] = gamma_dev[gb_off];
        sh_beta[tid] = beta_dev[gb_off];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load K and Q (32 x 128 = 4096 floats = 1024 float4s)
    int total_vec4 = cur_tokens * 32;
    for (int idx = tid; idx < total_vec4; idx += 256) {
        int tok = idx / 32;
        int d4 = idx % 32;
        long kq_off = (((long)batch_idx * numTokens + (t0 + tok)) * numHeads + head_idx) * WYF_HEAD_DIM + d4 * 4;
        const device float4 *k_ptr = (const device float4 *)(k_dev + kq_off);
        const device float4 *q_ptr = (const device float4 *)(q_dev + kq_off);
        threadgroup float4 *sh_k4 = (threadgroup float4 *)sh_k;
        threadgroup float4 *sh_q4 = (threadgroup float4 *)sh_q;
        sh_k4[idx] = *k_ptr;
        sh_q4[idx] = *q_ptr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute lower-triangular M and P: each thread computes one (t, s) entry
    // t in [0, cur_tokens), s in [0, t]
    for (int entry = tid; entry < cur_tokens * cur_tokens; entry += 256) {
        int t = entry / cur_tokens;
        int s = entry % cur_tokens;

        long mat_off = (((long)batch_idx * (numTokens / WYF_CHUNK_SIZE + 1) + chunk_idx) * numHeads + head_idx) * (WYF_CHUNK_SIZE * WYF_CHUNK_SIZE) + t * WYF_CHUNK_SIZE + s;

        if (s > t) {
            m_out[mat_off] = 0.0f;
            p_out[mat_off] = 0.0f;
            continue;
        }

        // Dot products dot(k_t, k_s) and dot(q_t, k_s)
        threadgroup const float4 *k_t_ptr = (threadgroup const float4 *)(sh_k + t * WYF_HEAD_DIM);
        threadgroup const float4 *k_s_ptr = (threadgroup const float4 *)(sh_k + s * WYF_HEAD_DIM);
        threadgroup const float4 *q_t_ptr = (threadgroup const float4 *)(sh_q + t * WYF_HEAD_DIM);

        float dot_kk = 0.0f;
        float dot_qk = 0.0f;
        for (int i = 0; i < 32; ++i) {
            dot_kk += dot(k_t_ptr[i], k_s_ptr[i]);
            dot_qk += dot(q_t_ptr[i], k_s_ptr[i]);
        }

        float decay_st = (s == t) ? 1.0f : (sh_gamma[t] / max(sh_gamma[s], 1e-12f));

        if (s == t) {
            m_out[mat_off] = 1.0f;
        } else {
            m_out[mat_off] = sh_beta[t] * decay_st * dot_kk;
        }
        p_out[mat_off] = decay_st * dot_qk;
    }
}

// wyf_triangular_solve: Solves M * Delta = R via triangular forward substitution
// across value head dimensions Dv in parallel.
kernel void wyf_triangular_solve(
    device const float *m_in       [[buffer(0)]],
    device const float *r_in       [[buffer(1)]],
    device float *delta_out        [[buffer(2)]],
    constant int &curTokens        [[buffer(3)]],
    constant int &headDim          [[buffer(4)]],
    uint3 groupPos                 [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]])
{
    // Each thread solves for one dv dimension across curTokens tokens
    int dv = groupPos.x * 32 + (tid % 32);
    if (dv >= headDim) return;

    long chunk_head_idx = groupPos.y;
    long m_base = chunk_head_idx * (WYF_CHUNK_SIZE * WYF_CHUNK_SIZE);
    long rd_base = chunk_head_idx * (WYF_CHUNK_SIZE * headDim);

    // Forward substitution across 32 tokens
    float local_delta[WYF_CHUNK_SIZE];
    for (int t = 0; t < curTokens; ++t) {
        float val = r_in[rd_base + t * headDim + dv];
        for (int s = 0; s < t; ++s) {
            float m_ts = m_in[m_base + t * WYF_CHUNK_SIZE + s];
            val -= m_ts * local_delta[s];
        }
        local_delta[t] = val;
        delta_out[rd_base + t * headDim + dv] = val;
    }
}

// wyf_recurrent_state_update: Updates chunk-end recurrent state with rank-1 outer products:
//   S_end = gamma_{C-1} * S_0 + sum_{s=0}^{C-1} delta_s * tilde_k_s^T
kernel void wyf_recurrent_state_update(
    device const float *delta_in   [[buffer(0)]],
    device const float *k_in       [[buffer(1)]],
    device const float *gamma_in   [[buffer(2)]],
    device float *state_io         [[buffer(3)]],
    constant int &curTokens        [[buffer(4)]],
    constant int &headDim          [[buffer(5)]],
    uint3 groupPos                 [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]])
{
    int dv = groupPos.x;
    int head_idx = groupPos.y;
    int batch_idx = groupPos.z;
    if (dv >= headDim) return;

    long state_row_offset = ((long)batch_idx * groupPos.y + head_idx) * (headDim * headDim) + (long)dv * headDim;
    float gamma_last = gamma_in[curTokens - 1];

    // Decay the existing row state
    for (int dk = tid; dk < headDim; dk += 256) {
        state_io[state_row_offset + dk] *= gamma_last;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Accumulate outer product sum_{s=0}^{C-1} delta_s[dv] * tilde_k_s[dk]
    for (int s = 0; s < curTokens; ++s) {
        float gamma_s = gamma_in[s];
        float decay_k = gamma_last / max(gamma_s, 1e-12f);
        float delta_val = delta_in[s * headDim + dv];

        for (int dk = tid; dk < headDim; dk += 256) {
            float k_val = k_in[s * headDim + dk] * decay_k;
            state_io[state_row_offset + dk] += delta_val * k_val;
        }
    }
}

// wyf_chunkwise_readout: Computes final chunk tokens output O = O_inter + P * Delta
kernel void wyf_chunkwise_readout(
    device const float *o_inter_in [[buffer(0)]],
    device const float *p_in       [[buffer(1)]],
    device const float *delta_in   [[buffer(2)]],
    device float *out_dev          [[buffer(3)]],
    constant int &curTokens        [[buffer(4)]],
    constant int &headDim          [[buffer(5)]],
    uint3 groupPos                 [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]])
{
    int dv = groupPos.x * 32 + (tid % 32);
    if (dv >= headDim) return;

    long chunk_head_idx = groupPos.y;
    long p_base = chunk_head_idx * (WYF_CHUNK_SIZE * WYF_CHUNK_SIZE);
    long d_base = chunk_head_idx * (WYF_CHUNK_SIZE * headDim);

    for (int t = 0; t < curTokens; ++t) {
        float acc = o_inter_in[d_base + t * headDim + dv];
        for (int s = 0; s <= t; ++s) {
            float p_ts = p_in[p_base + t * WYF_CHUNK_SIZE + s];
            acc += p_ts * delta_in[d_base + s * headDim + dv];
        }
        out_dev[d_base + t * headDim + dv] = acc;
    }
}
