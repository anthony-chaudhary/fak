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
    double gpu_seconds;
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

void mg_ink2_get_stats(void *dev, mg_ink2_stats *out);
size_t mg_ink2_error(void *dev, char *dst, size_t cap);

#endif
