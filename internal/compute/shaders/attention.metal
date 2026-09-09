// attention.metal — threadgroup-tiled FlashAttention for Apple Silicon Metal backend.
//
// Solves #12521: threadgroup-tiled FlashAttention with online softmax and SIMD shuffle
// reductions for decode attention on Apple Silicon unified memory (UMA).
//
// Replaces 1-thread-per-head scalar dispatch with 128/256-thread threadgroup tiles:
// - Threadgroup-tiled grid: one threadgroup per query head (grid size = nH threadgroups).
// - Query vector caching: q[h, :] cached into threadgroup memory (qs[256]).
// - SIMDgroup shuffle reduction: partial dot-products reduced via simd_sum across 32 lanes.
// - Threadgroup tree reduction: inter-SIMD reduction across threadgroup shared sums (tg_sums).
// - Register-resident online softmax: running max (m) and running sum (l) computed in registers.
// - Slice accumulator: each thread accumulates only its strided slice of headDim (acc[4]),
//   eliminating the 256-float scalar accumulator register spill.
// - Zero global scratch DRAM allocation: FlashAttention streaming online softmax.

#include <metal_stdlib>
using namespace metal;

kernel void attention_f32(device const float* q [[buffer(0)]],
                          device const float* K [[buffer(1)]],
                          device const float* V [[buffer(2)]],
                          device float* outp [[buffer(3)]],
                          constant int& nPos [[buffer(4)]],
                          constant int& nH [[buffer(5)]],
                          constant int& nKV [[buffer(6)]],
                          constant int& hd [[buffer(7)]],
                          constant float& scale [[buffer(8)]],
                          uint h [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    if ((int)h >= nH) return;

    int grp = nH / nKV;
    int kvh = (int)h / grp;
    int w = nKV * hd;
    uint qbase = h * (uint)hd;
    uint obase = h * (uint)hd;

    // Cache query row in threadgroup shared memory
    threadgroup float qs[256];
    for (uint d = tid; d < (uint)hd; d += tg_size) {
        qs[d] = q[qbase + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction shared memory across SIMD groups (up to 256 threads / 8 SIMD groups)
    threadgroup float tg_sums[8];
    uint simd_id = tid / 32;
    uint lane_id = tid % 32;
    uint num_simd = (tg_size + 31) / 32;

    // Register-resident online softmax state (avoids register spilling: acc[8] covers hd<=256 for tg_size>=32)
    float m = -3.402823466e38f;
    float l = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Stream through key/value sequence positions
    for (int j = 0; j < nPos; j++) {
        uint kvbase = (uint)(j * w + kvh * hd);

        // 1. Thread-local partial dot product over strided head dimensions
        float partial_dot = 0.0f;
        for (uint d = tid; d < (uint)hd; d += tg_size) {
            partial_dot += qs[d] * K[kvbase + d];
        }

        // 2. SIMDgroup reduction across 32 lanes
        float simd_s = simd_sum(partial_dot);
        if (lane_id == 0) {
            tg_sums[simd_id] = simd_s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Threadgroup reduction across SIMD groups
        float score = 0.0f;
        for (uint s = 0; s < num_simd; s++) {
            score += tg_sums[s];
        }
        score *= scale;

        // 4. Online softmax recurrence
        float mnew = max(m, score);
        float corr = (l > 0.0f) ? exp(m - mnew) : 0.0f;
        float p = exp(score - mnew);
        l = l * corr + p;

        // 5. Accumulate weighted values into thread's register slice
        for (uint d = tid; d < (uint)hd; d += tg_size) {
            uint idx = d / tg_size;
            acc[idx] = acc[idx] * corr + p * V[kvbase + d];
        }
        m = mnew;

        // Ensure all threads read tg_sums before next iteration overwrites it
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 6. Final normalization and output write
    float invl = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint d = tid; d < (uint)hd; d += tg_size) {
        uint idx = d / tg_size;
        outp[obase + d] = acc[idx] * invl;
    }
}

// flash_attention_tiled_f32 — alias entrypoint for explicit threadgroup-tiled dispatch
kernel void flash_attention_tiled_f32(device const float* q [[buffer(0)]],
                                      device const float* K [[buffer(1)]],
                                      device const float* V [[buffer(2)]],
                                      device float* outp [[buffer(3)]],
                                      constant int& nPos [[buffer(4)]],
                                      constant int& nH [[buffer(5)]],
                                      constant int& nKV [[buffer(6)]],
                                      constant int& hd [[buffer(7)]],
                                      constant float& scale [[buffer(8)]],
                                      uint h [[threadgroup_position_in_grid]],
                                      uint tid [[thread_position_in_threadgroup]],
                                      uint tg_size [[threads_per_threadgroup]]) {
    attention_f32(q, K, V, outp, nPos, nH, nKV, hd, scale, h, tid, tg_size);
}
