#ifndef FAK_CUDA_KV_FP8_ROTARY_H
#define FAK_CUDA_KV_FP8_ROTARY_H
#ifdef __cplusplus
extern "C" {
#endif
enum { FCUDA_KV_FP8_QWEN_PARTIAL_PANEL_V1 = 1 };
// Stages ONLY the moved suffix [end, old_positions) from normalized Kraw.
// Output token t is rerotated at absolute position from+t. Prefix codes/scales,
// kept raw, and V codes/scales are copied separately without requantization.
// Requires disjoint unpublished output, the exact bound producer identity, and
// native Qwen hd256/rotary64/4-KV-head geometry. Not a generic RoPE API.
// Submission success is NOT completion or numerical validity. Check both
// before quantization or any other dependent read of this staging buffer.
int fcuda_kv_fp8_partial_rope_stage_async(float *out, const float *raw,
    int old_positions, int from, int end, int heads, int head_dim,
    int rotary_dim, double theta, int producer, int *numerical_status);
#ifdef __cplusplus
}
#endif
#endif
