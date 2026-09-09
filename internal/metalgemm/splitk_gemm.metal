// splitk_gemm.metal — Split-K workgroup saturation for low-batch quantized GEMM on Apple Silicon.
//
// Study Provenance: EricLBuehler/mistral.rs:mistralrs-quant:afq_qmm_splitk
//
// Stage 1: Partitioned K-dimension quantized GEMV/GEMM kernels (Q8_0 and Q4_0 dequantization).
//          Tile sizing: output tile BM=32, BN=32 (or 1D tile for decode M=1: BN=32).
//          Each threadgroup along grid.z processes slice tid.z of size k_partition_size.
//          Accumulates in float32 into scratch buffer [split_k, M, N].
//
// Stage 2: Reduction kernel splitk_reduce summing along the split_k axis across slices.

#include <metal_stdlib>
using namespace metal;

// Quantization block structures matching standard GGML / candle / mistral.rs formats.
// Q8_0: 32 elements per block. 16-bit half scale, 32 signed int8 codes. Total 34 bytes.
struct block_q8_0 {
    half d;
    int8_t qs[32];
};

// Q4_0: 32 elements per block. 16-bit half scale, 16 packed 4-bit nibbles. Total 18 bytes.
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

// splitk_gemv_q8_0: 1D decode GEMV (M=1) for Q8_0 weights.
// Each threadgroup computes BN=32 output columns along grid.x for one K-slice along grid.z.
kernel void splitk_gemv_q8_0(
    device const float*       X                [[buffer(0)]], // [1, K]
    device const block_q8_0*  W                [[buffer(1)]], // [N, K/32]
    device float*             scratch          [[buffer(2)]], // [split_k, M, N]
    constant int&             M                [[buffer(3)]],
    constant int&             N                [[buffer(4)]],
    constant int&             K                [[buffer(5)]],
    constant int&             k_partition_size [[buffer(6)]],
    constant int&             split_k          [[buffer(7)]],
    uint3                     tg_pos           [[threadgroup_position_in_grid]],
    uint3                     tid              [[thread_position_in_threadgroup]]
) {
    uint col = tg_pos.x * 32 + tid.x;
    if (col >= (uint)N) return;
    uint slice = tg_pos.z;
    if (slice >= (uint)split_k) return;

    uint k_start = slice * (uint)k_partition_size;
    uint b_start = k_start / 32;
    uint num_blocks = (uint)k_partition_size / 32;
    uint num_blocks_per_row = (uint)K / 32;

    device const block_q8_0* w_row = W + (ulong)col * num_blocks_per_row;
    device const float* x_slice = X + k_start;

    float acc = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        block_q8_0 blk = w_row[b_start + b];
        device const float* x_blk = x_slice + b * 32;
        float bsum = 0.0f;
        for (int i = 0; i < 32; i++) {
            bsum += (float)blk.qs[i] * x_blk[i];
        }
        acc += float(blk.d) * bsum;
    }

    ulong scratch_idx = (ulong)slice * ((ulong)M * (ulong)N) + (ulong)col;
    scratch[scratch_idx] = acc;
}

// splitk_gemv_q4_0: 1D decode GEMV (M=1) for Q4_0 weights.
// Each threadgroup computes BN=32 output columns along grid.x for one K-slice along grid.z.
kernel void splitk_gemv_q4_0(
    device const float*       X                [[buffer(0)]], // [1, K]
    device const block_q4_0*  W                [[buffer(1)]], // [N, K/32]
    device float*             scratch          [[buffer(2)]], // [split_k, M, N]
    constant int&             M                [[buffer(3)]],
    constant int&             N                [[buffer(4)]],
    constant int&             K                [[buffer(5)]],
    constant int&             k_partition_size [[buffer(6)]],
    constant int&             split_k          [[buffer(7)]],
    uint3                     tg_pos           [[threadgroup_position_in_grid]],
    uint3                     tid              [[thread_position_in_threadgroup]]
) {
    uint col = tg_pos.x * 32 + tid.x;
    if (col >= (uint)N) return;
    uint slice = tg_pos.z;
    if (slice >= (uint)split_k) return;

    uint k_start = slice * (uint)k_partition_size;
    uint b_start = k_start / 32;
    uint num_blocks = (uint)k_partition_size / 32;
    uint num_blocks_per_row = (uint)K / 32;

    device const block_q4_0* w_row = W + (ulong)col * num_blocks_per_row;
    device const float* x_slice = X + k_start;

    float acc = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        block_q4_0 blk = w_row[b_start + b];
        device const float* x_blk = x_slice + b * 32;
        float bsum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t q = blk.qs[j];
            int q0 = int(q & 0x0f) - 8;
            int q1 = int(q >> 4) - 8;
            bsum += (float)q0 * x_blk[2 * j] + (float)q1 * x_blk[2 * j + 1];
        }
        acc += float(blk.d) * bsum;
    }

    ulong scratch_idx = (ulong)slice * ((ulong)M * (ulong)N) + (ulong)col;
    scratch[scratch_idx] = acc;
}

// splitk_gemm_q8_0: 2D tiled GEMM for Q8_0 weights (BM=32, BN=32).
kernel void splitk_gemm_q8_0(
    device const float*       X                [[buffer(0)]], // [M, K]
    device const block_q8_0*  W                [[buffer(1)]], // [N, K/32]
    device float*             scratch          [[buffer(2)]], // [split_k, M, N]
    constant int&             M                [[buffer(3)]],
    constant int&             N                [[buffer(4)]],
    constant int&             K                [[buffer(5)]],
    constant int&             k_partition_size [[buffer(6)]],
    constant int&             split_k          [[buffer(7)]],
    uint3                     tg_pos           [[threadgroup_position_in_grid]],
    uint3                     tid              [[thread_position_in_threadgroup]],
    uint3                     tg_size          [[threads_per_threadgroup]]
) {
    uint col = tg_pos.x * tg_size.x + tid.x;
    uint row = tg_pos.y * tg_size.y + tid.y;
    uint slice = tg_pos.z;

    if (row >= (uint)M || col >= (uint)N || slice >= (uint)split_k) return;

    uint k_start = slice * (uint)k_partition_size;
    uint b_start = k_start / 32;
    uint num_blocks = (uint)k_partition_size / 32;
    uint num_blocks_per_row = (uint)K / 32;

    device const block_q8_0* w_row = W + (ulong)col * num_blocks_per_row;
    device const float* x_row = X + (ulong)row * (ulong)K + (ulong)k_start;

    float acc = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        block_q8_0 blk = w_row[b_start + b];
        device const float* x_blk = x_row + b * 32;
        float bsum = 0.0f;
        for (int i = 0; i < 32; i++) {
            bsum += (float)blk.qs[i] * x_blk[i];
        }
        acc += float(blk.d) * bsum;
    }

    ulong scratch_idx = (ulong)slice * ((ulong)M * (ulong)N) + (ulong)row * (ulong)N + (ulong)col;
    scratch[scratch_idx] = acc;
}

// splitk_gemm_q4_0: 2D tiled GEMM for Q4_0 weights (BM=32, BN=32).
kernel void splitk_gemm_q4_0(
    device const float*       X                [[buffer(0)]], // [M, K]
    device const block_q4_0*  W                [[buffer(1)]], // [N, K/32]
    device float*             scratch          [[buffer(2)]], // [split_k, M, N]
    constant int&             M                [[buffer(3)]],
    constant int&             N                [[buffer(4)]],
    constant int&             K                [[buffer(5)]],
    constant int&             k_partition_size [[buffer(6)]],
    constant int&             split_k          [[buffer(7)]],
    uint3                     tg_pos           [[threadgroup_position_in_grid]],
    uint3                     tid              [[thread_position_in_threadgroup]],
    uint3                     tg_size          [[threads_per_threadgroup]]
) {
    uint col = tg_pos.x * tg_size.x + tid.x;
    uint row = tg_pos.y * tg_size.y + tid.y;
    uint slice = tg_pos.z;

    if (row >= (uint)M || col >= (uint)N || slice >= (uint)split_k) return;

    uint k_start = slice * (uint)k_partition_size;
    uint b_start = k_start / 32;
    uint num_blocks = (uint)k_partition_size / 32;
    uint num_blocks_per_row = (uint)K / 32;

    device const block_q4_0* w_row = W + (ulong)col * num_blocks_per_row;
    device const float* x_row = X + (ulong)row * (ulong)K + (ulong)k_start;

    float acc = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        block_q4_0 blk = w_row[b_start + b];
        device const float* x_blk = x_row + b * 32;
        float bsum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t q = blk.qs[j];
            int q0 = int(q & 0x0f) - 8;
            int q1 = int(q >> 4) - 8;
            bsum += (float)q0 * x_blk[2 * j] + (float)q1 * x_blk[2 * j + 1];
        }
        acc += float(blk.d) * bsum;
    }

    ulong scratch_idx = (ulong)slice * ((ulong)M * (ulong)N) + (ulong)row * (ulong)N + (ulong)col;
    scratch[scratch_idx] = acc;
}

// Stage 2: Reduction kernel splitk_reduce summing along the split_k axis across slices.
kernel void splitk_reduce(
    device const float* scratch   [[buffer(0)]], // [split_k, M, N]
    device float*       final_out [[buffer(1)]], // [M, N]
    constant int&       M         [[buffer(2)]],
    constant int&       N         [[buffer(3)]],
    constant int&       split_k   [[buffer(4)]],
    uint2               pos       [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    if (row >= (uint)M || col >= (uint)N) {
        return;
    }
    float sum = 0.0f;
    uint mn = (uint)(M * N);
    uint out_idx = row * (uint)N + col;
    for (int s = 0; s < split_k; s++) {
        sum += scratch[(uint)s * mn + out_idx];
    }
    final_out[out_idx] = sum;
}
