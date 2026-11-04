#ifndef FAK_CUDA_KV_PRECISION_H
#define FAK_CUDA_KV_PRECISION_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int fcuda_kv_store_typed(void *dst, const float *src, int offset, int n, int dtype);
int fcuda_kv_load_typed(float *dst, const void *src, int n, int dtype);
int fcuda_kv_snapshot_load_checked(float *dst, const void *src, int n, int dtype);
int fcuda_kv_snapshot_d2h_checked(void *dst, const void *src, size_t bytes);
// Retirement is an ownership receipt, not a physical-free counter.
enum { FCUDA_KV_SNAPSHOT_RETAINED = 0, FCUDA_KV_SNAPSHOT_RETIRED = 1 };
int fcuda_kv_snapshot_release_checked(void *expected, size_t bytes, int *disposition);
int fcuda_kv_snapshot_release_failed(int status);
int fcuda_kv_attention_typed(const float *q, const void *k, const void *v, float *out, int tokens, int prefix, int heads, int kvheads, int hd, float scale, int dtype);
#ifdef __cplusplus
}
#endif
#endif
