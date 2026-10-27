// Original FAK SIMT implementation of the native V4.1 activation oracle.
// Scale ABI follows DeepGEMM 1fc9c66975ee322d628b3288534ca0e9ce12a64d,
// csrc/jit_kernels/impls/smxx_layout.hpp (MIT); no donor kernel copied.
// DeepInfer 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01 linear.py requests
// recipe32 / packed UE8M0 / TMA-aligned column-major scales. This is NOT its
// tile_kernels casting implementation and is not a DeepGEMM GEMM port.
#include "../cuda_mxfp8_cast.h"
#if CUDART_VERSION >= 12090
#include <cuda_fp8.h>

// One CTA owns one row and four 32-value blocks; no byte-store races in scales.
static __global__ void k_mxfp8_cast_f32(const float *__restrict__ input,
                                      uint8_t *__restrict__ codes,
                                      uint32_t *__restrict__ scales,
                                      uint32_t *status, int rows, int cols,
                                      int scale_stride) {
    const int chunks = cols / 128;
    const int row = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
    const int lane = threadIdx.x & 31, group = threadIdx.x >> 5;
    const int col = chunk * 128 + threadIdx.x;
    __shared__ uint32_t sf[4];
    float x = row < rows ? input[(size_t)row * cols + col] : 0.0f;
    const unsigned bad = __ballot_sync(0xffffffffu, !isfinite(x));
    if (bad) x = 0.0f; // Entire rejected block is harmless; never accept its output.
    float amax = fmaxf(fabsf(x), 1.0e-4f);
    for (int delta = 16; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    const float ratio = __fmul_rn(amax, 1.0f / 448.0f);
    const unsigned bits = __float_as_uint(ratio);
    const unsigned exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
    const float scale = __uint_as_float(exponent << 23);
    if (row < rows) {
        const float normalized = __fdiv_rn(x, scale);
        codes[(size_t)row * cols + col] = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
    }
    if (lane == 0) {
        sf[group] = row < rows ? exponent : 0;
        if (bad && row < rows) atomicOr(status, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        scales[(size_t)chunk * scale_stride + row] = sf[0] | (sf[1] << 8) | (sf[2] << 16) | (sf[3] << 24);
}
#endif

extern "C" int fcuda_mxfp8_cast_supported(void) {
#if CUDART_VERSION >= 12090
    int runtime = 0, device = 0;
    cudaError_t e = cudaRuntimeGetVersion(&runtime);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    if (runtime < 12090) return FAK_MXFP8_CAST_UNSUPPORTED;
    e = cudaGetDevice(&device);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    cudaDeviceProp p;
    e = cudaGetDeviceProperties(&p, device);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    return p.major == 10 && (p.minor == 0 || p.minor == 3) ? FAK_MXFP8_CAST_OK : FAK_MXFP8_CAST_UNSUPPORTED;
#else
    return FAK_MXFP8_CAST_UNSUPPORTED;
#endif
}

static bool mxfp8_ranges_overlap(uintptr_t a, size_t an, uintptr_t b, size_t bn) {
    return a < b + bn && b < a + an;
}

extern "C" int fcuda_mxfp8_cast_f32(const float *input, size_t input_bytes,
                                    uint8_t *codes, size_t code_bytes,
                                    uint32_t *scales, size_t scale_bytes,
                                    uint32_t *device_status, int rows, int cols) {
    fak_mxfp8_cast_layout p;
    int rc = fak_mxfp8_cast_plan(rows, cols, &p);
    if (rc) return rc;
    rc = fcuda_mxfp8_cast_supported();
    if (rc) return rc;
#if CUDART_VERSION >= 12090
    const uintptr_t ptr[4] = {(uintptr_t)input, (uintptr_t)codes, (uintptr_t)scales, (uintptr_t)device_status};
    const size_t sizes[4] = {p.input_bytes, p.code_bytes, p.scale_bytes, sizeof(uint32_t)};
    if (input_bytes < sizes[0] || code_bytes < sizes[1] || scale_bytes < sizes[2])
        return FAK_MXFP8_CAST_BUFFER;
    int device = 0;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    for (int i = 0; i < 4; ++i) {
        if (!ptr[i] || ptr[i] % (i == 3 ? 4 : 16) || sizes[i] > UINTPTR_MAX - ptr[i])
            return FAK_MXFP8_CAST_BUFFER;
        cudaPointerAttributes attr;
        e = cudaPointerGetAttributes(&attr, (const void *)ptr[i]);
        if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
        if (attr.type != cudaMemoryTypeDevice || attr.device != device) return FAK_MXFP8_CAST_BUFFER;
        for (int j = 0; j < i; ++j)
            if (mxfp8_ranges_overlap(ptr[i], sizes[i], ptr[j], sizes[j])) return FAK_MXFP8_CAST_BUFFER;
    }
    cudaStreamCaptureStatus capture;
    e = cudaStreamIsCapturing(g_stream, &capture);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    if (g_capture_open || capture != cudaStreamCaptureStatusNone) return FAK_MXFP8_CAST_CAPTURE;
    e = cudaMemsetAsync(device_status, 0, sizeof(uint32_t), g_stream);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    const int blocks = p.scale_stride_words * (cols / 128);
    k_mxfp8_cast_f32<<<blocks, 128, 0, g_stream>>>(input, codes, scales, device_status, rows, cols, p.scale_stride_words);
    e = cudaGetLastError();
    // Drain even after a launch error: memset may already be in flight.
    const cudaError_t drain = cudaStreamSynchronize(g_stream);
    if (drain != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)drain;
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    uint32_t status = 0;
    e = cudaMemcpy(&status, device_status, sizeof(status), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return FAK_MXFP8_CAST_CUDA_ERROR_BASE + (int)e;
    return status ? FAK_MXFP8_CAST_NONFINITE : FAK_MXFP8_CAST_OK;
#else
    return FAK_MXFP8_CAST_UNSUPPORTED;
#endif
}
