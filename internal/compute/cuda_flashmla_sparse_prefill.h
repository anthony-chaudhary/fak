#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Pointer fields are device-only for launch, host-only for the synchronous bridge.
typedef struct {
 const uint16_t *q, *kv;
 const int32_t *indices, *lengths;
 const uint32_t *positions;
 const float *sink, *table;
 int queries, keys, topk, table_rows;
 float scale;
} fak_flashmla_input;
typedef struct { float *values, *max_logits, *lse; } fak_flashmla_output;
typedef struct {
 int device, major, minor, sms, driver, runtime, binary_version, ptx_version;
 char name[256]; unsigned char uuid[16];
} fak_flashmla_identity;
int fak_flashmla_prefill_run(const fak_flashmla_input*, const fak_flashmla_output*);
int fak_flashmla_prefill_host(const uint16_t*, const uint16_t*, const int32_t*,
 const int32_t*, const uint32_t*, const float*, const float*, int, int, int, int,
 float, float*, float*, float*);
int fak_flashmla_prefill_observe(uint64_t*, int*);
int fak_flashmla_prefill_kernel_info(fak_flashmla_identity*);
#ifdef __cplusplus
}
// All fourteen device bases are owned and checked by the generic bridge.
int fak_flashmla_prefill_launch(const fak_flashmla_input&, void **buffers, void *stream);
#endif
