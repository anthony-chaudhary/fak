#ifndef FVK_V41_PACKED_STORAGE_H
#define FVK_V41_PACKED_STORAGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Private prototype only. Serialized by vulkanMu. 2=geometry, 4=busy,
 * 3=unavailable/stale device; negative=sticky Vulkan failure. No batch flush. */
int fvk_v41_packed_state(uint64_t *generation);
int fvk_v41_packed_zero(void *dst, uint64_t generation, size_t capacity);
int fvk_v41_packed_upload(void *dst, uint64_t generation, size_t capacity,
                          size_t offset, const void *host, size_t bytes);
int fvk_v41_packed_copy(void *dst, size_t dst_capacity, size_t dst_offset,
                       const void *src, size_t src_capacity, size_t src_offset,
                       uint64_t generation, size_t bytes);
#ifdef __cplusplus
}
#endif
#endif
