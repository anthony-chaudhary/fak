#ifndef MG_INK2_DEVICE_H
#define MG_INK2_DEVICE_H

#include <stddef.h>
#include <stdint.h>

// One kernel argument, bound at its position in the argument list. A buffer is
// bound at offset and must hold length more bytes; buffer == 0 binds length
// bytes of the dispatch's constant block starting at offset.
typedef struct {
    uintptr_t buffer;
    size_t offset, length;
} mg_ink2_arg;

typedef struct {
    uint64_t submissions, dispatches, pending;
    // queue is commit to GPU start, tail is GPU end to the waiter resuming, and
    // kernel is the driver's scheduling time; last_gpu is the latest buffer's.
    double gpu_seconds, kernel_seconds, queue_seconds, tail_seconds, last_gpu_seconds;
} mg_ink2_stats;

void *mg_ink2_create(void);
void mg_ink2_close(void *dev);

void *mg_ink2_alloc(size_t bytes);
void mg_ink2_release(void *buffer);
void *mg_ink2_span(void *buffer, size_t *bytes);

// Both return an index, or -1 with the reason left for mg_ink2_error.
int mg_ink2_library(void *dev, const char *source, int precise);
int mg_ink2_pipeline(void *dev, int library, const char *function, uint32_t *width, uint32_t *max_threads);

// size is {x, y, z, threads per group x, y, z}; exact launches x*y*z threads
// instead of x*y*z threadgroups. Encodes only: nothing is submitted.
int mg_ink2_dispatch(void *dev, int pipeline, const mg_ink2_arg *args, int count,
                     const void *constants, const uint32_t *size, int exact);
void mg_ink2_barrier(void *dev);
// Submits the open command buffer and waits for it. A no-op when nothing is pending.
int mg_ink2_flush(void *dev);

// Gives every later dispatch its own compute pass with boundary timestamps.
// Returns 0, changing nothing, when the device cannot sample them.
int mg_ink2_profile_passes(void *dev);
// Resolves the passes of the last flushed command buffer into GPU seconds per
// dispatch, in encode order (negative where the sample is invalid), and
// returns how many there were.
int mg_ink2_profile_take(void *dev, double *out, int cap);

void mg_ink2_get_stats(void *dev, mg_ink2_stats *out);
size_t mg_ink2_error(void *dev, char *dst, size_t cap);

#endif
