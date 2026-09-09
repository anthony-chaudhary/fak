//go:build darwin && cgo

package compute

/*
#cgo LDFLAGS: -framework Foundation -framework Metal
#include "wired_memory_darwin.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"syscall"
	"unsafe"
)

// WireMethod describes the mechanism used to lock memory pages into physical RAM.
type WireMethod int

const (
	// WireMethodNone indicates memory is not wired.
	WireMethodNone WireMethod = 0
	// WireMethodMachVMWire indicates memory was wired via mach_vm_wire (privileged).
	WireMethodMachVMWire WireMethod = 1
	// WireMethodMLock indicates memory was wired via POSIX mlock.
	WireMethodMLock WireMethod = 2
)

// String returns a human-readable name of the wire method.
func (m WireMethod) String() string {
	switch m {
	case WireMethodNone:
		return "none"
	case WireMethodMachVMWire:
		return "mach_vm_wire"
	case WireMethodMLock:
		return "mlock"
	default:
		return "unknown"
	}
}

// WiredMemoryReceipt records the result and provenance of a memory wiring operation.
type WiredMemoryReceipt struct {
	Addr      uintptr
	Size      uint64
	Method    WireMethod
	Wired     bool
	ErrorCode int
}

// DarwinMemoryLimits captures Metal and XNU working-set memory limits for macOS.
type DarwinMemoryLimits struct {
	RecommendedMaxWorkingSet uint64
	MaxBufferLength          uint64
	HasUnifiedMemory         bool
	CurrentAllocatedSize     uint64
}

var (
	// ErrWiredMemoryUnavailable indicates memory wiring is unavailable on this platform.
	ErrWiredMemoryUnavailable = errors.New("wired memory: unavailable on this platform or architecture")
	// ErrInvalidAddress indicates a null or invalid memory address was provided.
	ErrInvalidAddress = errors.New("wired memory: invalid or null memory address")
	// ErrInvalidSize indicates an invalid or zero size was provided.
	ErrInvalidSize = errors.New("wired memory: invalid memory size")
	// ErrNotPageAligned indicates the memory address is not aligned to the system page size.
	ErrNotPageAligned = errors.New("wired memory: memory address is not page aligned")
	// ErrWiringFailed indicates the kernel failed to wire the requested memory range.
	ErrWiringFailed = errors.New("wired memory: failed to wire memory")
	// ErrUnwiringFailed indicates the kernel failed to unwire the requested memory range.
	ErrUnwiringFailed = errors.New("wired memory: failed to unwire memory")
	// ErrUnmappedAddress indicates the queried address is not part of a mapped memory region.
	ErrUnmappedAddress = errors.New("wired memory: address not in mapped address space")
)

// DarwinWorkingSetLimits queries Metal device limits and working-set guidance.
func DarwinWorkingSetLimits() DarwinMemoryLimits {
	var cLimits C.fak_darwin_memory_limits_t
	ret := C.fmetal_query_memory_limits(&cLimits)
	if ret != 0 {
		return DarwinMemoryLimits{}
	}
	return DarwinMemoryLimits{
		RecommendedMaxWorkingSet: uint64(cLimits.recommended_max_working_set),
		MaxBufferLength:          uint64(cLimits.max_buffer_length),
		HasUnifiedMemory:         cLimits.has_unified_memory != 0,
		CurrentAllocatedSize:     uint64(cLimits.current_allocated_size),
	}
}

// RecommendedMaxWorkingSetSize returns Apple Silicon Metal's recommended working set size in bytes.
func RecommendedMaxWorkingSetSize() uint64 {
	return DarwinWorkingSetLimits().RecommendedMaxWorkingSet
}

// WireMemory locks the specified page-aligned virtual memory range [addr, addr+size)
// into physical RAM to eliminate swap stalls, using mach_vm_wire or falling back to mlock.
func WireMemory(addr uintptr, size uint64) (WiredMemoryReceipt, error) {
	receipt := WiredMemoryReceipt{
		Addr:   addr,
		Size:   size,
		Method: WireMethodNone,
		Wired:  false,
	}

	if addr == 0 {
		return receipt, ErrInvalidAddress
	}
	if size == 0 {
		return receipt, ErrInvalidSize
	}

	pageSize := uintptr(os.Getpagesize())
	if pageSize == 0 {
		pageSize = 16384
	}
	if addr%pageSize != 0 {
		return receipt, ErrNotPageAligned
	}

	var cReceipt C.fak_wire_receipt_t
	ret := C.fmetal_wire_memory(C.uint64_t(addr), C.uint64_t(size), &cReceipt)

	receipt.Method = WireMethod(cReceipt.method)
	receipt.Wired = cReceipt.wired != 0
	receipt.ErrorCode = int(cReceipt.error_code)

	if ret != 0 || !receipt.Wired {
		if receipt.ErrorCode != 0 {
			return receipt, fmt.Errorf("%w: system error %d (%s)", ErrWiringFailed, receipt.ErrorCode, syscall.Errno(receipt.ErrorCode).Error())
		}
		return receipt, ErrWiringFailed
	}

	return receipt, nil
}

// UnwireMemory unlocks a previously wired virtual memory range.
func UnwireMemory(addr uintptr, size uint64, method WireMethod) error {
	if addr == 0 {
		return ErrInvalidAddress
	}
	if size == 0 {
		return ErrInvalidSize
	}

	pageSize := uintptr(os.Getpagesize())
	if pageSize == 0 {
		pageSize = 16384
	}
	if addr%pageSize != 0 {
		return ErrNotPageAligned
	}

	ret := C.fmetal_unwire_memory(C.uint64_t(addr), C.uint64_t(size), C.fak_wire_method_t(method))
	if ret != 0 {
		return fmt.Errorf("%w: system error %d (%s)", ErrUnwiringFailed, int(ret), syscall.Errno(ret).Error())
	}
	return nil
}

// IsMemoryWired checks if the page containing addr is currently wired into physical RAM.
// It returns true if wired, along with the user_wired_count from Mach VM region info.
func IsMemoryWired(addr uintptr) (bool, uint32, error) {
	if addr == 0 {
		return false, 0, ErrInvalidAddress
	}

	var isWired C.int
	var wiredCount C.uint32_t

	ret := C.fmetal_is_memory_wired(C.uint64_t(addr), &isWired, &wiredCount)
	if ret != 0 {
		if ret == C.int(syscall.ENOENT) {
			return false, 0, ErrUnmappedAddress
		}
		return false, 0, fmt.Errorf("mach_vm_region query failed: code %d (%s)", int(ret), syscall.Errno(ret).Error())
	}

	return isWired != 0, uint32(wiredCount), nil
}

// WireModelSpan wires a model weights virtual memory span [addr, addr+size) upon model load,
// ensuring it is locked in unified RAM to eliminate swap stalls and memory compression.
// It returns a WiredMemoryReceipt and an idempotent teardown closure that unwires the memory
// upon model release.
func WireModelSpan(addr uintptr, size uint64) (WiredMemoryReceipt, func(), error) {
	receipt, err := WireMemory(addr, size)
	if err != nil {
		return receipt, func() {}, err
	}

	var once sync.Once
	teardown := func() {
		once.Do(func() {
			_ = UnwireMemory(addr, size, receipt.Method)
		})
	}
	return receipt, teardown, nil
}

// WireModelBuffer wires a byte slice (such as an mmap-backed model weights buffer) upon model load.
// It requires the buffer slice to be backed by a non-empty, page-aligned memory allocation.
// It returns a WiredMemoryReceipt and an idempotent teardown closure that unwires the memory
// upon model release.
func WireModelBuffer(buf []byte) (WiredMemoryReceipt, func(), error) {
	if len(buf) == 0 {
		return WiredMemoryReceipt{}, func() {}, ErrInvalidSize
	}
	addr := uintptr(unsafe.Pointer(&buf[0]))
	return WireModelSpan(addr, uint64(len(buf)))
}
