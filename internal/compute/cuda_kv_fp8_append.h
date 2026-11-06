#ifndef FAK_CUDA_KV_FP8_APPEND_H
#define FAK_CUDA_KV_FP8_APPEND_H
#include "cuda_kv_fp8_lifecycle.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct fcuda_kv_fp8_append_input {
  const float *raw;
  const float *key;
  const float *value;
  size_t bytes;
} fcuda_kv_fp8_append_input;
typedef struct fcuda_kv_fp8_append_destination {
  void *plane[5];       // exact owner order: raw, key, value, key scale, value scale
  size_t bytes[5];
} fcuda_kv_fp8_append_destination;
enum {
  FCUDA_KV_FP8_PHASE_APPEND_VALIDATE = 12,
  FCUDA_KV_FP8_PHASE_APPEND_INGEST = 13
};
// Caller holds cudaMu, roots every input owner and excludes Free/Recycle.
// Inputs are exact-base ordinary device-local activation allocations. Borrowed,
// managed, packed, weight, pooled and graph buffers are not an alternate route.
// Returned zero is submission only. The checked lifecycle finish must precede
// ingest, attention, publication and input retirement/reuse.
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_append_validate_owned(
    const fcuda_kv_fp8_append_input *input, int elements, int *status);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_append_ingest_owned(
    const fcuda_kv_fp8_append_destination *destination,
    const fcuda_kv_fp8_append_input *input,
    int prefix, int tokens, int heads, int head_dim, int *status);
#ifdef __cplusplus
}
#endif
#endif
