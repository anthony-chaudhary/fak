// Original build-lane witness for #13981, not a serving GEMM.
// PTX contract: NVIDIA PTX ISA 9.0, tcgen05.alloc/dealloc and
// tcgen05.relinquish_alloc_permit (CUDA 13.0).
// https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf
#include <cuda_runtime.h>
#include "cuda_sm10x_probe.h"

static __global__ void fak_sm10x_tmem_probe(uint32_t *result) {
    __shared__ uint32_t address;
    // Exactly one converged warp allocates and releases 32 TMEM columns.
    const uint32_t shared = (uint32_t)__cvta_generic_to_shared(&address);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"(shared) : "memory");
    __syncwarp();
    const uint32_t tmem = address;
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                 :: "r"(tmem) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    if (threadIdx.x == 0) *result = 1;
}

// Caller provides a device uint32 and synchronizes the stream before reading.
// The API reports enqueue status only. No backend init launches this witness.
extern "C" int fcuda_sm10x_probe(uint32_t *result, cudaStream_t stream) {
    if (!result) return (int)cudaErrorInvalidValue;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return (int)err;
    cudaDeviceProp properties;
    err = cudaGetDeviceProperties(&properties, device);
    if (err != cudaSuccess) return (int)err;
    // Do not treat an arch-specific 10.0 object as covering every 10.x device.
    if (!(properties.major == 10 && (properties.minor == 0 || properties.minor == 3)))
        return (int)cudaErrorNotSupported;
    cudaStreamCaptureStatus capture;
    err = cudaStreamIsCapturing(stream, &capture);
    if (err != cudaSuccess) return (int)err;
    if (capture != cudaStreamCaptureStatusNone) return (int)cudaErrorNotSupported;
    fak_sm10x_tmem_probe<<<1, 32, 0, stream>>>(result);
    return (int)cudaPeekAtLastError();
}
