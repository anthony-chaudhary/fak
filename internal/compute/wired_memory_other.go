//go:build !darwin || !cgo

package compute

import "errors"

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

// DarwinWorkingSetLimits returns empty limits on non-darwin or non-cgo builds.
func DarwinWorkingSetLimits() DarwinMemoryLimits {
	return DarwinMemoryLimits{}
}

// RecommendedMaxWorkingSetSize returns 0 on non-darwin or non-cgo builds.
func RecommendedMaxWorkingSetSize() uint64 {
	return 0
}

// WireMemory returns ErrWiredMemoryUnavailable on non-darwin or non-cgo builds.
func WireMemory(addr uintptr, size uint64) (WiredMemoryReceipt, error) {
	return WiredMemoryReceipt{
		Addr:   addr,
		Size:   size,
		Method: WireMethodNone,
		Wired:  false,
	}, ErrWiredMemoryUnavailable
}

// UnwireMemory returns ErrWiredMemoryUnavailable on non-darwin or non-cgo builds.
func UnwireMemory(addr uintptr, size uint64, method WireMethod) error {
	return ErrWiredMemoryUnavailable
}

// IsMemoryWired returns ErrWiredMemoryUnavailable on non-darwin or non-cgo builds.
func IsMemoryWired(addr uintptr) (bool, uint32, error) {
	return false, 0, ErrWiredMemoryUnavailable
}

// WireModelSpan returns ErrWiredMemoryUnavailable on non-darwin or non-cgo builds.
func WireModelSpan(addr uintptr, size uint64) (WiredMemoryReceipt, func(), error) {
	return WiredMemoryReceipt{
		Addr:   addr,
		Size:   size,
		Method: WireMethodNone,
		Wired:  false,
	}, func() {}, ErrWiredMemoryUnavailable
}

// WireModelBuffer returns ErrWiredMemoryUnavailable on non-darwin or non-cgo builds.
func WireModelBuffer(buf []byte) (WiredMemoryReceipt, func(), error) {
	return WiredMemoryReceipt{
		Method: WireMethodNone,
		Wired:  false,
	}, func() {}, ErrWiredMemoryUnavailable
}
