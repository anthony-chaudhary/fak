#ifndef FAK_CUDA_KV_FP8_H
#define FAK_CUDA_KV_FP8_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum {
  FCUDA_KV_FP8_NONFINITE = 1,
  FCUDA_KV_FP8_BAD_SCALE = 2,
  FCUDA_KV_FP8_BAD_CODE = 4,
  FCUDA_KV_FP8_BAD_RESULT = 8
};
// All functions require cudaMu and the ordinary private stream. They refuse
// capture. Return zero means submission accepted, NOT completed/numerically
// valid. Runtime/launch failures close shared admission; preflight argument,
// capture and already-closed admission refusals do not create a new fault.
// numerical_status is owned device int storage retained through a checked
// completion/readback. Begin only a fresh transaction or after the preceding
// status has been checked; clearing an unchecked failure is forbidden.
int fcuda_kv_fp8_begin_async(int *numerical_status);
int fcuda_kv_fp8_validate_async(const float *raw, const float *key,
    const float *value, int elements, int *numerical_status);
// Destination codes/scales/output must be unpublished disjoint staging.
// rows == positions * kv_heads. Native ABI uses contiguous rows of head_dim.
// Status is cumulative; no operation below clears a preceding numerical error.
// A nonzero numerical status invalidates the whole staged result. It does NOT
// by itself imply CUDA/device/global-stream failure.
int fcuda_kv_fp8_quantize_async(uint8_t *codes, float *scales,
    const float *src, int rows, int head_dim, int *numerical_status);
int fcuda_kv_fp8_widen_async(float *dst, const uint8_t *codes,
    const float *scales, int rows, int head_dim, int *numerical_status);
int fcuda_kv_fp8_attention_async(const float *q,
    const uint8_t *k, const float *k_scale,
    const uint8_t *v, const float *v_scale,
    float *out, int tokens, int prefix, int heads, int kv_heads,
    int head_dim, float attention_scale, int *numerical_status);
#ifdef __cplusplus
}
#endif
#endif
