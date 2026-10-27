#ifndef FAK_CUDA_MXFP8_CAST_H
#define FAK_CUDA_MXFP8_CAST_H
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

/* Explicit, standalone DeepGEMM recipe-(1,32,32) activation preparation.
 * NOT cuBLASLt's tiled scale-factor layout. No serving path enables this API.
 * F32 row-major [rows,cols] -> E4M3FN row-major [rows,cols], with one
 * ceil-power-of-two UE8M0 scale per 32 consecutive columns. Four scale bytes
 * form a little-endian uint32; words are column-major [rows,cols/128] with
 * stride (1,align(rows,4)). Padded rows are zero-filled. cols must be a multiple of128.
 * Numerical contract: amax=max(abs(block),1e-4), ratio=F32(amax*F32(1/448)),
 * scale=ceil_pow2(ratio), E4M3 round-nearest-even with finite saturation.
 * NaN/Inf input is rejected AFTER writing (all outputs then invalid; this is
 * not transactional). Negative zero is retained.
 */
enum fak_mxfp8_cast_status {
    FAK_MXFP8_CAST_OK = 0,
    FAK_MXFP8_CAST_SHAPE = 1,
    FAK_MXFP8_CAST_UNSUPPORTED = 2,
    FAK_MXFP8_CAST_BUFFER = 3,
    FAK_MXFP8_CAST_CAPTURE = 4,
    FAK_MXFP8_CAST_NONFINITE = 5,
    FAK_MXFP8_CAST_CUDA_ERROR_BASE = 100000
};
typedef struct fak_mxfp8_cast_layout {
    int rows, cols, scale_stride_words;
    size_t input_bytes, code_bytes, scale_bytes;
} fak_mxfp8_cast_layout;

/* Host-only checked sizing; no CUDA dependency and no writes on refusal.
 * The INT_MAX element cap also bounds grid indexing and every byte multiply.
 */
static inline int fak_mxfp8_cast_plan(int rows, int cols, fak_mxfp8_cast_layout *out) {
    if (!out || rows <= 0 || rows > INT_MAX-3 || cols <= 0 || cols % 128 != 0)
        return FAK_MXFP8_CAST_SHAPE;
    const size_t stride = ((size_t)rows + 3) & ~(size_t)3;
    if ((size_t)cols > (size_t)INT_MAX / stride || (size_t)rows * cols > SIZE_MAX / 4)
        return FAK_MXFP8_CAST_SHAPE;
    fak_mxfp8_cast_layout p;
    p.rows = rows; p.cols = cols; p.scale_stride_words = (int)stride;
    p.code_bytes = (size_t)rows * cols;
    p.input_bytes = p.code_bytes * sizeof(float);
    p.scale_bytes = stride * ((size_t)cols / 128) * sizeof(uint32_t);
    *out = p;
    return FAK_MXFP8_CAST_OK;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Runtime and compile-toolkit >=12.9, exact SM100/103. Not a GEMM capability. */
int fcuda_mxfp8_cast_supported(void);
/* Caller holds the backend CUDA serialization lock and supplies four disjoint
 * device allocations on the current device. Sizes include accessible tails;
 * pointer attributes verify device/type, not allocation extent: actual
 * accessible extents are a caller precondition, not proved by those sizes.
 * pointers must be aligned to 16 bytes (input/codes/scales), 4 bytes (status).
 * status is caller-owned 4-byte device scratch; never allocate in this call.
 * Refuses graph capture. Success fences g_stream and validates finite input.
 * On any error outputs MUST NOT be consumed. CUDA errors return BASE+cudaError;
 * if the drain fails, retain ALL buffers until an independently successful drain
 * or device teardown. No fallback, ownership transfer, or buffer freeing occurs.
 */
int fcuda_mxfp8_cast_f32(const float *input, size_t input_bytes,
                       uint8_t *codes, size_t code_bytes,
                       uint32_t *scales, size_t scale_bytes,
                       uint32_t *device_status, int rows, int cols);
#ifdef __cplusplus
}
#endif
#endif
