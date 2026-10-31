#pragma once
#include <stdint.h>
#include <stddef.h>

// Private qualification-only device-pointer ABI. No allocator, host fallback,
// default backend registration, or implicit stream synchronization.
// All pointers belong to the current device and remain live through stream drain.
// Indices are padded to topk, contain only -1 or in-range flat token positions,
// and lengths contain a single value in [0, topk]. The caller validates contents.
typedef struct fak_flashmla_decode_input {
    const uint16_t *query_bf16; // [64,512], contiguous
    const uint8_t *kv;         // [blocks,page_size,528], token-interleaved FP8
    const int32_t *indices;
    const int32_t *length;
    const float *sink;         // optional [64], unscaled
    const uint8_t *extra_kv;
    const int32_t *extra_indices;
    const int32_t *extra_length;
    uint16_t *output_bf16;     // fresh [64,512], no input overlap
    float *lse;               // fresh [64], no input overlap
    int blocks, page_size, topk;
    int extra_blocks, extra_page_size, extra_topk;
    float scale;
} fak_flashmla_decode_input;

// Caller-owned scratch. Exact byte extents are mandatory, and scratch may not
// alias inputs, outputs, or other scratch; owner retains it through stream drain.
typedef struct fak_flashmla_decode_workspace {
    void *metadata; size_t metadata_bytes;
    int32_t *num_splits; size_t num_splits_bytes;
    float *partial_lse; size_t partial_lse_bytes;
    float *partial_output; size_t partial_output_bytes;
} fak_flashmla_decode_workspace;
typedef struct fak_flashmla_decode_workspace_sizes {
    size_t metadata, num_splits, partial_lse, partial_output;
    int partitions;
} fak_flashmla_decode_workspace_sizes;
#ifdef __cplusplus
extern "C" {
#endif
// Query scratch for the current device before allocating. Partitions are the
// upstream head64 rule numSM/s_q (s_q=1), capped by refusal at 160, never silently
// reduced. Zero scratch when batch-invariant or topk+extra_topk<=640.
int fak_flashmla_decode_workspace_query(int topk, int extra_topk, int batch_invariant,
                                       fak_flashmla_decode_workspace_sizes *);
// Full copied scheduler -> real head64 kernel -> copied combine on same stream.
// Optional scratch only for split-KV. Output BF16 and natural-log LSE remain device-resident.
// ANY nonzero return may follow queued scheduler/kernel work. Retain ALL inputs,
// outputs and scratch until checked stream drain; do not automatically fall back.
// Failed drain requires owner quarantine, never free/reuse or host publication.
int fak_flashmla_decode_launch(const fak_flashmla_decode_input *,
                              const fak_flashmla_decode_workspace *, int batch_invariant, void *stream);
// No-split v41_h64 only. -1 contract, -2 device, -3 upstream exception.
// Success means submission only, not completion or validated numerical output.
int fak_flashmla_decode_nosplit_launch(const fak_flashmla_decode_input *, void *stream);
#ifdef __cplusplus
}
#endif
