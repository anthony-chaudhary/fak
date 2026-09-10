#ifndef FAK_ROCM_BACKEND_H
#define FAK_ROCM_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every function returns 0 on success and a non-zero HIP or validation
 * status on failure. frocm_last_error is process-global under the Go backend's
 * mandatory operation mutex and remains valid until the next ABI call. The first backend is deliberately
 * one-device/one-stream; the Go adapter serializes calls. */
const char *frocm_last_error(void);
int frocm_init(int device, char *name, int name_len, char *gfx, int gfx_len,
               size_t *total_mem);
int frocm_mem_info(size_t *free_mem, size_t *total_mem);
int frocm_malloc(void **ptr, size_t bytes);
int frocm_free(void *ptr);
int frocm_h2d(void *dst, const void *src, size_t bytes);
int frocm_d2h(void *dst, const void *src, size_t bytes);
int frocm_d2d(void *dst, const void *src, size_t bytes);
int frocm_sync(void);
size_t frocm_host_transfer_bytes(void);
void frocm_host_transfer_reset(void);
size_t frocm_kernel_launches(void);
void frocm_kernel_launches_reset(void);

int frocm_matmul_f32(const float *w, const float *x, float *y,
                     int out, int in, int rows);
int frocm_q8_matmul_f32(const int8_t *codes, const float *scales,
                        const float *x, float *y, int out, int in,
                        int rows, int block);
int frocm_q4k_matmul_f32(const uint8_t *weights, const float *x, float *y,
                         int out, int in, int rows);
int frocm_q5k_matmul_f32(const uint8_t *weights, const float *x, float *y,
                         int out, int in, int rows);
int frocm_q6k_matmul_f32(const uint8_t *weights, const float *x, float *y,
                         int out, int in, int rows);

int frocm_rmsnorm_f32(const float *x, const float *weight, float *y,
                      int rows, int width, float eps);
int frocm_rope_f32(const float *src, float *dst, int pos, int heads,
                   int head_dim, double theta);
int frocm_swiglu_f32(const float *gate, const float *up, float *out, int n);
int frocm_add_f32(float *dst, const float *src, int n);
int frocm_add_bias_f32(float *dst, const float *bias, int rows, int width);
int frocm_attention_f32(const float *q, const float *k, const float *v,
                        float *out, int q_heads, int kv_heads, int positions,
                        int head_dim, int group_size, float scale);
int frocm_argmax_f32(const float *values, int n, int *index);

int frocm_kv_append_f32(float *keys, float *raw_keys, float *values,
                        const float *key, const float *raw_key,
                        const float *value, int slot, int row_width);
int frocm_kv_evict_f32(float *keys, float *raw_keys, float *values,
                       int positions, int from, int count, int kv_heads,
                       int head_dim, double theta);

#ifdef __cplusplus
}
#endif
#endif
