#ifndef FAK_CUDA_KV_PRECISION_H
#define FAK_CUDA_KV_PRECISION_H
#ifdef __cplusplus
extern "C" {
#endif
int fcuda_kv_store_typed(void *dst, const float *src, int offset, int n, int dtype);
int fcuda_kv_load_typed(float *dst, const void *src, int n, int dtype);
int fcuda_kv_attention_typed(const float *q, const void *k, const void *v, float *out, int tokens, int prefix, int heads, int kvheads, int hd, float scale, int dtype);
#ifdef __cplusplus
}
#endif
#endif
