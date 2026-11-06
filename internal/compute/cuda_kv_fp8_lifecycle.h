#ifndef FAK_CUDA_KV_FP8_LIFECYCLE_H
#define FAK_CUDA_KV_FP8_LIFECYCLE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum {
  FCUDA_KV_FP8_OWNER_PLANE = 0,
  FCUDA_KV_FP8_OWNER_WORKSPACE = 1,
  FCUDA_KV_FP8_PHASE_ADMIT = 1,
  FCUDA_KV_FP8_PHASE_ALLOCATE = 2,
  FCUDA_KV_FP8_PHASE_BEGIN = 3,
  FCUDA_KV_FP8_PHASE_COPY = 4,
  FCUDA_KV_FP8_PHASE_HOST_ALLOCATE = 5,
  FCUDA_KV_FP8_PHASE_STATUS_COPY = 6,
  FCUDA_KV_FP8_PHASE_FENCE = 7,
  FCUDA_KV_FP8_PHASE_HOST_RETIRE = 8,
  FCUDA_KV_FP8_PHASE_RETIRE = 9,
  FCUDA_KV_FP8_PHASE_ROTARY = 10,
  FCUDA_KV_FP8_PHASE_QUANTIZE = 11,
  FCUDA_KV_FP8_BOOKKEEPING_FAILED = -76101,
  FCUDA_KV_FP8_RECEIPT_FAILED = -76102
};
typedef struct fcuda_kv_fp8_lifecycle_result {
  int status;
  int terminal;
  int completed;
  int numerical_checked;
  uint32_t numerical;
  int retired;
  int phase;
} fcuda_kv_fp8_lifecycle_result;

// All functions require cudaMu, device 0 and an eager private-stream operation.
// The Go caller must retain every referenced owner through checked completion.
// Submission success never substitutes for completed/numerical_checked.
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_lifecycle_admit(void);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_lifecycle_failed(int status, int phase);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_allocate_owned(
    void **out, size_t bytes, int role);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_begin_owned(int *numerical_status);
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_copy_owned(
    void *dst, size_t dst_bytes, size_t dst_offset,
    const void *src, size_t src_bytes, size_t src_offset, size_t bytes);
// read_numerical=1 requires an exact owned 4-byte device status workspace.
// read_numerical=0 is ONLY a bit-copy fence for already-validated source images:
// status must be null and no numerical producer may be covered by that arm.
// Status D2H uses retained C-owned pinned host memory, never a stack/Go target.
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_finish_owned(
    const int *numerical_status, int read_numerical);
// Physical retirement only; no pool/cache sweep or shared allocation release.
// It fences outstanding readers before freeing and reports exact disposition.
fcuda_kv_fp8_lifecycle_result fcuda_kv_fp8_release_owned(
    void *expected, size_t bytes);
#ifdef __cplusplus
}
#endif
#endif
