//go:build darwin && cgo

// wired_memory_darwin.m — programmatic macOS wired working-set memory locking
// via mach_vm_wire and mlock fallback to eliminate swap stalls.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/host_priv.h>
#include <mach/vm_region.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "wired_memory_darwin.h"

int fmetal_query_memory_limits(fak_darwin_memory_limits_t *limits) {
    if (!limits) {
        return -1;
    }
    memset(limits, 0, sizeof(*limits));

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            return -1;
        }
        limits->recommended_max_working_set = (uint64_t)[dev recommendedMaxWorkingSetSize];
        limits->max_buffer_length = (uint64_t)[dev maxBufferLength];
        limits->has_unified_memory = [dev hasUnifiedMemory] ? 1 : 0;
        limits->current_allocated_size = (uint64_t)[dev currentAllocatedSize];
#if !__has_feature(objc_arc)
        [dev release];
#endif
    }
    return 0;
}

int fmetal_wire_memory(uint64_t addr, uint64_t size, fak_wire_receipt_t *receipt) {
    if (!receipt) {
        return EINVAL;
    }
    memset(receipt, 0, sizeof(*receipt));
    receipt->addr = addr;
    receipt->size = size;
    receipt->method = FAK_WIRE_METHOD_NONE;
    receipt->wired = 0;
    receipt->error_code = 0;

    if (addr == 0 || size == 0) {
        receipt->error_code = EINVAL;
        return EINVAL;
    }

    size_t page_size = (size_t)getpagesize();
    if ((addr % page_size) != 0) {
        receipt->error_code = EINVAL;
        return EINVAL;
    }

    if (size > SIZE_MAX - page_size) {
        receipt->error_code = EINVAL;
        return EINVAL;
    }

    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    // Rung 1: Attempt mach_vm_wire with host_priv port
    mach_port_t host_priv = MACH_PORT_NULL;
    kern_return_t kr = host_get_host_priv_port(mach_host_self(), &host_priv);
    if (kr == KERN_SUCCESS && host_priv != MACH_PORT_NULL) {
        kern_return_t wkr = mach_vm_wire(
            host_priv,
            mach_task_self(),
            (mach_vm_address_t)addr,
            (mach_vm_size_t)aligned_size,
            VM_PROT_READ | VM_PROT_WRITE
        );
        // If read-write protection failed, attempt read-only (e.g. read-only mmap weights)
        if (wkr == KERN_PROTECTION_FAILURE) {
            wkr = mach_vm_wire(
                host_priv,
                mach_task_self(),
                (mach_vm_address_t)addr,
                (mach_vm_size_t)aligned_size,
                VM_PROT_READ
            );
        }
        mach_port_deallocate(mach_task_self(), host_priv);
        if (wkr == KERN_SUCCESS) {
            receipt->method = FAK_WIRE_METHOD_MACH_VM_WIRE;
            receipt->wired = 1;
            receipt->error_code = 0;
            return 0;
        }
    }

    // Rung 2: Fallback to POSIX mlock. On macOS XNU, unprivileged mlock
    // succeeds and wires memory pages in the task map.
    if (mlock((const void *)(uintptr_t)addr, aligned_size) == 0) {
        receipt->method = FAK_WIRE_METHOD_MLOCK;
        receipt->wired = 1;
        receipt->error_code = 0;
        return 0;
    }

    // Rung 3: Graceful failure recording in receipt if both fail.
    int err = errno ? errno : EPERM;
    receipt->method = FAK_WIRE_METHOD_NONE;
    receipt->wired = 0;
    receipt->error_code = err;
    return err;
}

int fmetal_unwire_memory(uint64_t addr, uint64_t size, fak_wire_method_t method) {
    if (addr == 0 || size == 0) {
        return EINVAL;
    }
    size_t page_size = (size_t)getpagesize();
    if ((addr % page_size) != 0) {
        return EINVAL;
    }
    if (size > SIZE_MAX - page_size) {
        return EINVAL;
    }
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    if (method == FAK_WIRE_METHOD_MACH_VM_WIRE) {
        mach_port_t host_priv = MACH_PORT_NULL;
        kern_return_t kr = host_get_host_priv_port(mach_host_self(), &host_priv);
        if (kr == KERN_SUCCESS && host_priv != MACH_PORT_NULL) {
            kern_return_t wkr = mach_vm_wire(
                host_priv,
                mach_task_self(),
                (mach_vm_address_t)addr,
                (mach_vm_size_t)aligned_size,
                VM_PROT_NONE
            );
            mach_port_deallocate(mach_task_self(), host_priv);
            if (wkr == KERN_SUCCESS) {
                return 0;
            }
            return (int)wkr;
        }
        // Fallback to munlock if host_priv could not be acquired during unwire
        if (munlock((const void *)(uintptr_t)addr, aligned_size) == 0) {
            return 0;
        }
        return errno ? errno : EINVAL;
    }

    if (method == FAK_WIRE_METHOD_MLOCK) {
        if (munlock((const void *)(uintptr_t)addr, aligned_size) == 0) {
            return 0;
        }
        return errno ? errno : EINVAL;
    }

    // Fallback: try munlock
    if (munlock((const void *)(uintptr_t)addr, aligned_size) == 0) {
        return 0;
    }
    return errno ? errno : EINVAL;
}

int fmetal_is_memory_wired(uint64_t addr, int *is_wired, uint32_t *wired_count) {
    if (is_wired) *is_wired = 0;
    if (wired_count) *wired_count = 0;

    if (addr == 0) {
        return EINVAL;
    }

    mach_vm_address_t region_addr = (mach_vm_address_t)addr;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    kern_return_t kr = mach_vm_region(
        mach_task_self(),
        &region_addr,
        &region_size,
        VM_REGION_BASIC_INFO_64,
        (vm_region_info_t)&info,
        &count,
        &object_name
    );
    if (object_name != MACH_PORT_NULL) {
        mach_port_deallocate(mach_task_self(), object_name);
    }
    if (kr != KERN_SUCCESS) {
        return (int)kr;
    }

    if (region_addr > addr || addr >= region_addr + region_size) {
        return ENOENT;
    }

    if (is_wired) {
        *is_wired = (info.user_wired_count > 0) ? 1 : 0;
    }
    if (wired_count) {
        *wired_count = (uint32_t)info.user_wired_count;
    }
    return 0;
}
