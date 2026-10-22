#ifndef FVK_V41_PACKED_ATTENTION_H
#define FVK_V41_PACKED_ATTENTION_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Private synchronous host-result bridge. No generic Tensor ABI. All pointers
 * are call-scoped; generation/owner validation must precede this entry. */
int fvk_v41_packed_attention_admission(uint64_t generation);
int fvk_v41_packed_attention(const void *source, size_t capacity, size_t initialized_rows,
    uint64_t generation, const uint32_t *rows, uint32_t count,
    const float *q, int32_t heads, const float *sink, int32_t has_sink,
    int32_t mode, float scale, float *output, uint32_t *status);
#ifdef __cplusplus
}
#endif
#endif
