// Private decode candidate; reuses the licensed q4k_mmvq.cuh kernels unchanged.
// Included immediately after q4k_mmvq.cuh in the sole CUDA translation unit.
#include <stdint.h>
#include <limits.h>

namespace fak_q4k_shared_input {
struct span { uintptr_t begin, end; };
static bool make_span(const void *ptr, size_t bytes, span *out) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    if (!ptr || bytes == 0 || bytes > UINTPTR_MAX - begin) return false;
    *out = {begin, begin + bytes};
    return true;
}
static bool overlaps(span a, span b) { return a.begin < b.end && b.begin < a.end; }

// Read-only operands may alias each other. Writes must not alias any read or
// another write, including partial overlap; this protects the quantized input
// throughout all three projections, not merely at the quantization launch.
static bool valid_ranges(const uint8_t *const weights[3], const float *x,
                         float *const outputs[3], const int widths[3], int in,
                         void *workspace, size_t workspaceBytes) {
    using namespace fak_q4k_mmvq;
    if (in <= 0 || in > INT_MAX - 255 || in % QK_K != 0 ||
        (size_t)in > SIZE_MAX / sizeof(float)) return false;
    const size_t blocks = (size_t)in / QK_K;
    const size_t scratchBytes = (size_t)(in / QK8_1) * sizeof(block_q8_1);
    if (workspaceBytes < scratchBytes || reinterpret_cast<uintptr_t>(workspace) % alignof(block_q8_1) != 0 ||
        reinterpret_cast<uintptr_t>(x) % alignof(float) != 0) return false;
    span reads[4], writes[4];
    if (!make_span(x, (size_t)in * sizeof(float), &reads[3]) ||
        !make_span(workspace, scratchBytes, &writes[3])) return false;
    for (int i = 0; i < 3; ++i) {
        // The existing dot kernel computes its row/block offset in int.
        if (widths[i] <= 0 || widths[i] > INT_MAX / (int)blocks ||
            reinterpret_cast<uintptr_t>(weights[i]) % alignof(block_q4_K) != 0 ||
            reinterpret_cast<uintptr_t>(outputs[i]) % alignof(float) != 0) return false;
        const size_t count = (size_t)widths[i] * blocks;
        if (count > SIZE_MAX / sizeof(block_q4_K) || (size_t)widths[i] > SIZE_MAX / sizeof(float) ||
            !make_span(weights[i], count * sizeof(block_q4_K), &reads[i]) ||
            !make_span(outputs[i], (size_t)widths[i] * sizeof(float), &writes[i])) return false;
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) if (overlaps(writes[i], reads[j])) return false;
        for (int j = 0; j < i; ++j) if (overlaps(writes[i], writes[j])) return false;
    }
    return true;
}
} // namespace fak_q4k_shared_input

extern "C" int fcuda_q4k_qkv_shared_input_f32(
    const uint8_t *wQ, const uint8_t *wK, const uint8_t *wV,
    const float *x, float *yQ, float *yK, float *yV,
    int outQ, int outK, int outV, int in, void *workspace, size_t workspaceBytes) {
    using namespace fak_q4k_mmvq;
#ifdef FAK_CUDA_V41_CHECKED_STREAM
    // An uncertain stream is a fault, not an admission refusal/fallback license.
    if (v41_candidates_stream_uncertain()) return 70000 + (int)cudaErrorUnknown;
#endif
    const uint8_t *weights[3] = {wQ, wK, wV};
    float *outputs[3] = {yQ, yK, yV};
    const int widths[3] = {outQ, outK, outV};
    if (g_capture_open || !fak_q4k_shared_input::valid_ranges(weights, x, outputs, widths, in, workspace, workspaceBytes))
        return FCUDA_Q4K_MMVQ_REFUSED;
    // Do not clear and hide an earlier asynchronous fault before publishing work.
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return 70000 + (int)status;
    block_q8_1 *quantized = static_cast<block_q8_1 *>(workspace);
    dim3 qgrid((in + 255) / 256, 1);
    FCUDA_TS("quantize_q8_1", qgrid, quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(x, quantized, in));
    status = cudaGetLastError();
    if (status != cudaSuccess) return 70000 + (int)status;
    for (int i = 0; i < 3; ++i) {
        dim3 grid(widths[i]), block(warp_size, calc_nwarps(1));
        FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid, block, 0, g_stream>>>(weights[i], quantized, outputs[i], in, widths[i]));
        status = cudaGetLastError();
        if (status != cudaSuccess) return 70000 + (int)status;
        ++g_q4k_mmvq_launches;
    }
    return 0;
}
