/* DESIGN CONTRACT ONLY: no definitions, no callers, not included by any source.
 * Requires the lifecycle hooks enumerated in SEAM.md before implementation. */
#ifndef FVK_V41_PACKED_BATCH_PROPOSED_API_H
#define FVK_V41_PACKED_BATCH_PROPOSED_API_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t generation;
    uint64_t owner;
    uint64_t serial;
} fvk_v41_batch_identity;

enum fvk_v41_batch_phase {
    FVK_V41_BATCH_RESERVED = 0,
    FVK_V41_BATCH_RECORDING = 1,
    FVK_V41_BATCH_COMPLETE = 2,
    FVK_V41_BATCH_CANCELLED = 3,
    FVK_V41_BATCH_QUARANTINED = 4
};

/* Requires an already-recording checked step window, no inflight/restore/foreign
 * work, and healthy context. Does not flush, start, drain, or allocate commands.
 * Native-issued identity; caller must not fabricate or infer any component. */
int fvk_v41_batch_current(fvk_v41_batch_identity *out);

/* Requires exact generation/owner/serial. Returns immutable receipt for a
 * closed window, or RECORDING. A lookup failure is not completion.
 * Query must remain capable of reporting retained receipt after sticky failure,
 * without granting access to any Buffer or treating context as healthy. */
int fvk_v41_batch_query(fvk_v41_batch_identity id,
                        enum fvk_v41_batch_phase *phase, int *result);

/* Enrollment is internal to native packed recording functions, not a public
 * pointer registration API. Each recording returns a native-issued use ID;
 * source/destination references are pinned BEFORE commands are recorded.
 * Finish acknowledges exactly one terminal COMPLETE use after Go publication.
 * It NEVER frees a buffer directly. QUARANTINED uses cannot be acknowledged
 * through this normal path. Double acknowledgement must fail closed. */
int fvk_v41_batch_acknowledge(fvk_v41_batch_identity id, uint64_t use_id);
/* Explicit receipt retirement after no remaining uses; missing ID is error. */
int fvk_v41_batch_forget(fvk_v41_batch_identity id);

#ifdef __cplusplus
}
#endif
#endif
