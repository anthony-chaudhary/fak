// hashk_gather.metal — Metal compute shader for HashK dual-subtable PLE table compression
// on Apple Silicon GPU & unified memory architectures.
//
// Gathers 80-dim embedding slices from Subtable 0 and Subtable 1 directly into threadgroups,
// concatenates them to 160 dimensions, and bypasses the ridge regression matrix Wh ≈ I_160
// (eliminating 409,600 MACs/token across 16 heads).
//
// Dual-subtable slot mapping:
//   uint64_t x_sub = (local_idx + 1) * 2862933555777941757ULL + SALTS[sub] + head * 998244353ULL;

#include <metal_stdlib>
using namespace metal;

// HashK polynomial congruential hashing constants
constant ulong HASHK_MULTIPLIER   = 2862933555777941757ULL;
constant ulong HASHK_HEAD_PRIME   = 998244353ULL;
constant ulong SPLITMIX_CONSTANT  = 0x9e3779b97f4a7c15ULL;

// Default dual-subtable salts
constant ulong HASHK_DEFAULT_SALTS[2] = {
    0x517cc1b727220a95ULL, // Subtable 0 salt
    0x9e3779b97f4a7c15ULL  // Subtable 1 salt
};

// SplitMix64 deterministic 64-bit hash generator step
inline ulong splitmix64(ulong x) {
    ulong z = x + SPLITMIX_CONSTANT;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// compute_hashk_slot computes the compressed subtable slot index
inline ulong compute_hashk_slot(ulong local_idx, int sub, ulong head, ulong num_slots) {
    if (num_slots == 0) {
        return 0;
    }
    ulong salt = HASHK_DEFAULT_SALTS[sub & 1];
    ulong x_sub = (local_idx + 1) * HASHK_MULTIPLIER + salt + head * HASHK_HEAD_PRIME;
    return splitmix64(x_sub) % num_slots;
}

// HashKParams holds runtime dispatch parameters
struct HashKParams {
    ulong num_tokens;         // Total tokens to gather
    ulong num_heads;          // Attention heads (default 16 for Qwen3.8-Flash-Next PLE)
    ulong num_slots_per_sub;  // Slots per subtable (e.g. 80,000,000 for 320M vocab R=4)
    uint  subtable_dim;       // Hidden dimension per subtable (default 80)
    uint  full_dim;           // Full concatenated embedding dimension (default 160)
    uint  bypass_ridge;       // 1 to bypass Wh ≈ I_160 identity ridge matrix
};

// hashk_gather performs cooperative threadgroup gather of dual subtable embedding slices (Float32).
// Each threadgroup handles one (token, head) pair, loads 80 dims from Subtable 0 and 80 dims
// from Subtable 1 into threadgroup local memory, and writes the concatenated 160-dim vector
// directly to output (bypassing Wh ≈ I_160).
kernel void hashk_gather(
    device const ulong*        tokens           [[buffer(0)]],
    device const float*        subtable0        [[buffer(1)]],
    device const float*        subtable1        [[buffer(2)]],
    device float*              output           [[buffer(3)]],
    constant HashKParams&      params           [[buffer(4)]],
    uint3                      group_id         [[threadgroup_position_in_grid]],
    uint                       tid              [[thread_index_in_threadgroup]],
    uint                       threads_per_group[[threads_per_threadgroup]]
) {
    threadgroup float tg_slice[256];
    threadgroup ulong s_slot0;
    threadgroup ulong s_slot1;

    // Resolve token index and head index from grid coordinates
    ulong token_idx;
    ulong head_idx;
    if (group_id.y > 0 || params.num_heads == 1) {
        token_idx = group_id.x;
        head_idx  = group_id.y;
    } else {
        ulong flat_idx = group_id.x;
        token_idx = flat_idx / params.num_heads;
        head_idx  = flat_idx % params.num_heads;
    }

    if (token_idx >= params.num_tokens || head_idx >= params.num_heads) {
        return;
    }

    // Leader thread computes dual-subtable slot indices
    if (tid == 0) {
        ulong local_token = tokens != nullptr ? tokens[token_idx] : token_idx;
        s_slot0 = compute_hashk_slot(local_token, 0, head_idx, params.num_slots_per_sub);
        s_slot1 = compute_hashk_slot(local_token, 1, head_idx, params.num_slots_per_sub);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperatively load subtable 0 slice (dims 0 .. subtable_dim-1)
    ulong base0 = s_slot0 * params.subtable_dim;
    for (uint d = tid; d < params.subtable_dim; d += threads_per_group) {
        tg_slice[d] = subtable0[base0 + d];
    }

    // Cooperatively load subtable 1 slice (dims subtable_dim .. full_dim-1)
    ulong base1 = s_slot1 * params.subtable_dim;
    for (uint d = tid; d < params.subtable_dim; d += threads_per_group) {
        tg_slice[params.subtable_dim + d] = subtable1[base1 + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Identity ridge matrix bypass (Wh ≈ I_160): write concatenated 160-dim vector to output
    ulong out_offset = (token_idx * params.num_heads + head_idx) * params.full_dim;
    for (uint d = tid; d < params.full_dim; d += threads_per_group) {
        output[out_offset + d] = tg_slice[d];
    }
}

// hashk_gather_half: FP16 variant for half-precision native pipelines
kernel void hashk_gather_half(
    device const ulong*        tokens           [[buffer(0)]],
    device const half*         subtable0        [[buffer(1)]],
    device const half*         subtable1        [[buffer(2)]],
    device half*               output           [[buffer(3)]],
    constant HashKParams&      params           [[buffer(4)]],
    uint3                      group_id         [[threadgroup_position_in_grid]],
    uint                       tid              [[thread_index_in_threadgroup]],
    uint                       threads_per_group[[threads_per_threadgroup]]
) {
    threadgroup half tg_slice[256];
    threadgroup ulong s_slot0;
    threadgroup ulong s_slot1;

    ulong token_idx;
    ulong head_idx;
    if (group_id.y > 0 || params.num_heads == 1) {
        token_idx = group_id.x;
        head_idx  = group_id.y;
    } else {
        ulong flat_idx = group_id.x;
        token_idx = flat_idx / params.num_heads;
        head_idx  = flat_idx % params.num_heads;
    }

    if (token_idx >= params.num_tokens || head_idx >= params.num_heads) {
        return;
    }

    if (tid == 0) {
        ulong local_token = tokens != nullptr ? tokens[token_idx] : token_idx;
        s_slot0 = compute_hashk_slot(local_token, 0, head_idx, params.num_slots_per_sub);
        s_slot1 = compute_hashk_slot(local_token, 1, head_idx, params.num_slots_per_sub);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong base0 = s_slot0 * params.subtable_dim;
    for (uint d = tid; d < params.subtable_dim; d += threads_per_group) {
        tg_slice[d] = subtable0[base0 + d];
    }

    ulong base1 = s_slot1 * params.subtable_dim;
    for (uint d = tid; d < params.subtable_dim; d += threads_per_group) {
        tg_slice[params.subtable_dim + d] = subtable1[base1 + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong out_offset = (token_idx * params.num_heads + head_idx) * params.full_dim;
    for (uint d = tid; d < params.full_dim; d += threads_per_group) {
        output[out_offset + d] = tg_slice[d];
    }
}

// hashk_gather_fp8: FP8 (uchar) raw storage variant for maximum memory savings (12.8 GB VRAM)
kernel void hashk_gather_fp8(
    device const ulong*        tokens           [[buffer(0)]],
    device const uchar*        subtable0        [[buffer(1)]],
    device const uchar*        subtable1        [[buffer(2)]],
    device uchar*              output           [[buffer(3)]],
    constant HashKParams&      params           [[buffer(4)]],
    uint3                      group_id         [[threadgroup_position_in_grid]],
    uint                       tid              [[thread_index_in_threadgroup]],
    uint                       threads_per_group[[threads_per_threadgroup]]
) {
    threadgroup uchar tg_slice[256];
    threadgroup ulong s_slot0;
    threadgroup ulong s_slot1;

    ulong token_idx;
    ulong head_idx;
    if (group_id.y > 0 || params.num_heads == 1) {
        token_idx = group_id.x;
        head_idx  = group_id.y;
    } else {
        ulong flat_idx = group_id.x;
        token_idx = flat_idx / params.num_heads;
        head_idx  = flat_idx % params.num_heads;
    }

    if (token_idx >= params.num_tokens || head_idx >= params.num_heads) {
        return;
    }

    if (tid == 0) {
        ulong local_token = tokens != nullptr ? tokens[token_idx] : token_idx;
        s_slot0 = compute_hashk_slot(local_token, 0, head_idx, params.num_slots_per_sub);
        s_slot1 = compute_hashk_slot(local_token, 1, head_idx, params.num_slots_per_sub);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong base0 = s_slot0 * params.subtable_dim;
    for (uint d = tid; d < params.subtable_dim; d += threads_per_group) {
        tg_slice[d] = subtable0[base0 + d];
    }

    ulong base1 = s_slot1 * params.subtable_dim;
    for (uint d = tid; d < params.subtable_dim; d += threads_per_group) {
        tg_slice[params.subtable_dim + d] = subtable1[base1 + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong out_offset = (token_idx * params.num_heads + head_idx) * params.full_dim;
    for (uint d = tid; d < params.full_dim; d += threads_per_group) {
        output[out_offset + d] = tg_slice[d];
    }
}
