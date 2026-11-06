#ifndef FAK_CUDA_KV_FP8_ATTENTION_OWNER_H
#define FAK_CUDA_KV_FP8_ATTENTION_OWNER_H
#include "cuda_kv_fp8_append.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { FCUDA_KV_FP8_PHASE_ATTENTION = 14 };
// All pointers stay rooted under cudaMu. Q is exact-base device-local F32
// activation storage. out is a distinct exact private workspace allocation.
// k contains all five private owner planes, including raw solely for alias and
// geometry validation. The numerical status is cumulative and is never cleared.
// Zero means accepted submission. Caller checks completion AND numerical status
// before dependent output access and fences every consumer before retirement.
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_attention_owned(
    const fcuda_kv_fp8_append_destination *k,
    const float *q, size_t q_bytes, float *out, size_t out_bytes,
    int tokens,int prefix,int heads,float scale,int *status);
#ifdef __cplusplus
}
#endif
#endif
