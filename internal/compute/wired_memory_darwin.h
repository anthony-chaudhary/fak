/* wired_memory_darwin.h — programmatic macOS wired working-set memory locking
 * via mach_vm_wire and mlock fallback to eliminate swap stalls.
 */

#ifndef FAK_WIRED_MEMORY_DARWIN_H
#define FAK_WIRED_MEMORY_DARWIN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FAK_WIRE_METHOD_NONE = 0,
    FAK_WIRE_METHOD_MACH_VM_WIRE = 1,
    FAK_WIRE_METHOD_MLOCK = 2
} fak_wire_method_t;

typedef struct {
    uint64_t addr;
    uint64_t size;
    fak_wire_method_t method;
    int wired;          /* 1 if successfully wired, 0 otherwise */
    int error_code;     /* 0 on success, kern_return_t or errno on failure */
} fak_wire_receipt_t;

typedef struct {
    uint64_t recommended_max_working_set;
    uint64_t max_buffer_length;
    int has_unified_memory;
    uint64_t current_allocated_size;
} fak_darwin_memory_limits_t;

/* Queries Metal device memory limits and working-set guidance.
 * Returns 0 on success, or -1 if no Metal device is reachable.
 */
int fmetal_query_memory_limits(fak_darwin_memory_limits_t *limits);

/* Wires memory at [addr, addr+size) using mach_vm_wire (Rung 1) or mlock (Rung 2).
 * Records execution details into receipt. Returns 0 on success, or error code.
 */
int fmetal_wire_memory(uint64_t addr, uint64_t size, fak_wire_receipt_t *receipt);

/* Unwires previously wired memory.
 * Returns 0 on success, or error code.
 */
int fmetal_unwire_memory(uint64_t addr, uint64_t size, fak_wire_method_t method);

/* Queries whether memory at addr is wired and returns user_wired_count.
 * Returns 0 on success, or Mach/POSIX error code.
 */
int fmetal_is_memory_wired(uint64_t addr, int *is_wired, uint32_t *wired_count);

#ifdef __cplusplus
}
#endif

#endif /* FAK_WIRED_MEMORY_DARWIN_H */
