#ifndef FAK_CUDA_SM10X_PROBE_H
#define FAK_CUDA_SM10X_PROBE_H
#include <cuda_runtime_api.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Explicit hardware witness only; never called at backend initialization.
// result is one writable uint32 on the current CUDA device; caller retains it
// until stream synchronization. Success only means enqueue, not completion.
int fcuda_sm10x_probe(uint32_t *result, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
