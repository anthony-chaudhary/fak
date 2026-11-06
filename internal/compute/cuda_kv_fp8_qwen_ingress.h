// SPDX-License-Identifier: Apache-2.0
#ifndef FAK_CUDA_KV_FP8_QWEN_INGRESS_H
#define FAK_CUDA_KV_FP8_QWEN_INGRESS_H
#include "cuda_kv_fp8_lifecycle.h"
#ifdef __cplusplus
extern "C" {
#endif
enum {
 FCUDA_KV_FP8_PHASE_QWEN_VALIDATE = 15,
 FCUDA_KV_FP8_PHASE_QWEN_NORM = 16,
 FCUDA_KV_FP8_PHASE_QWEN_ROPE = 17,
 FCUDA_KV_FP8_PHASE_QWEN_ADMIT = 18,
 FCUDA_KV_FP8_QWEN_TIMING_UNSUPPORTED = -76301
};
typedef struct fcuda_kv_fp8_qwen_stage {
 const float *input[5]; // Q, K, V, Q gain, K gain
 size_t input_bytes[5];
 float *work[4]; // normalized Q/K, rotated Q/K
 size_t work_bytes[4];
 int *status;
 int tokens, first, heads, rotary, producer;
 float epsilon;
 double theta;
} fcuda_kv_fp8_qwen_stage;
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_admit_owned(void);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_validate_owned(const fcuda_kv_fp8_qwen_stage *);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_norm_owned(const fcuda_kv_fp8_qwen_stage *);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_qwen_rope_owned(const fcuda_kv_fp8_qwen_stage *);
#ifdef __cplusplus
}
#endif
#endif
