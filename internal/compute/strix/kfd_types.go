// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
)

// UMAMappingMode represents the backing mode of a UMA zero-copy allocation.
type UMAMappingMode string

const (
	// UMAMappingModeNativeDRMKFD indicates allocation via Linux DRM GEM and KFD SVM ioctls.
	UMAMappingModeNativeDRMKFD UMAMappingMode = "native_drm_kfd"

	// UMAMappingModeSimulatedPinned indicates fallback 2MB-aligned pinned host memory simulation.
	UMAMappingModeSimulatedPinned UMAMappingMode = "simulated_pinned"
)

// UMAAllocationFlags specifies flags for UMA zero-copy allocations.
type UMAAllocationFlags uint32

const (
	// UMAFlagNone represents default allocation flags.
	UMAFlagNone UMAAllocationFlags = 0

	// UMAFlagUSWC marks memory as Uncached Speculative Write Combining (AMDGPU_GEM_CREATE_CPU_GTT_USWC).
	UMAFlagUSWC UMAAllocationFlags = 1 << 0

	// UMAFlag2MBHugepages enforces strict 2,097,152-byte boundary alignment.
	UMAFlag2MBHugepages UMAAllocationFlags = 1 << 1

	// UMAFlagNoEvict prohibits TTM page eviction under memory pressure (AMDGPU_GEM_CREATE_NO_EVICT).
	UMAFlagNoEvict UMAAllocationFlags = 1 << 2

	// UMAFlagCoherent marks the buffer for coherent CPU-GPU access across Infinity Fabric.
	UMAFlagCoherent UMAAllocationFlags = 1 << 3

	// UMAFlagSharedVirtual asserts unified address space (uintptr(HostPtr) == uintptr(DevPtr)).
	UMAFlagSharedVirtual UMAAllocationFlags = 1 << 4
)

// Has returns true if the specified flag is set.
func (f UMAAllocationFlags) Has(flag UMAAllocationFlags) bool {
	return f&flag == flag
}

// String returns a human-readable representation of allocation flags.
func (f UMAAllocationFlags) String() string {
	if f == UMAFlagNone {
		return "NONE"
	}
	var parts []string
	if f.Has(UMAFlagUSWC) {
		parts = append(parts, "USWC")
	}
	if f.Has(UMAFlag2MBHugepages) {
		parts = append(parts, "2MB_HUGEPAGE")
	}
	if f.Has(UMAFlagNoEvict) {
		parts = append(parts, "NO_EVICT")
	}
	if f.Has(UMAFlagCoherent) {
		parts = append(parts, "COHERENT")
	}
	if f.Has(UMAFlagSharedVirtual) {
		parts = append(parts, "SHARED_VA")
	}
	return fmt.Sprintf("%v", parts)
}

// KFDAcquireVMArgs matches struct kfd_ioctl_acquire_vm_args from Linux <linux/kfd_ioctl.h>.
type KFDAcquireVMArgs struct {
	DRMFD uint32 `json:"drm_fd"`
	GPUID uint32 `json:"gpu_id"`
}

// KFDMapMemoryToGPUFullArgs matches extended struct kfd_ioctl_map_memory_to_gpu_args.
type KFDMapMemoryToGPUFullArgs struct {
	Handle             uint64 `json:"handle"`
	DeviceIDsArrayAddr uint64 `json:"device_ids_array_address"`
	NDevices           uint32 `json:"n_devices"`
	NSuccess           uint32 `json:"n_success"`
}

// UMAZeroCopyConfig holds configuration for the UMA zero-copy allocator.
type UMAZeroCopyConfig struct {
	RenderPath     string             `json:"render_path"`
	KFDPath        string             `json:"kfd_path"`
	Alignment      int64              `json:"alignment"`
	Flags          UMAAllocationFlags `json:"flags"`
	ForceSimulated bool               `json:"force_simulated"`
}

// DefaultUMAZeroCopyConfig returns the production configuration targeting Strix Halo UMA.
func DefaultUMAZeroCopyConfig() *UMAZeroCopyConfig {
	return &UMAZeroCopyConfig{
		RenderPath: DefaultDRMRenderPath,
		KFDPath:    DefaultKFDPath,
		Alignment:  HugepageSize2MB,
		Flags:      UMAFlagUSWC | UMAFlag2MBHugepages | UMAFlagNoEvict | UMAFlagCoherent | UMAFlagSharedVirtual,
	}
}

// UMAZeroCopyTelemetry reports allocation counts, active buffers, and pointer identity assertions.
type UMAZeroCopyTelemetry struct {
	ActiveAllocations    int64          `json:"active_allocations"`
	TotalAllocatedBytes  int64          `json:"total_allocated_bytes"`
	TotalDeallocations   int64          `json:"total_deallocations"`
	IdentityChecksPassed int64          `json:"identity_checks_passed"`
	NativeAllocations    int64          `json:"native_allocations"`
	SimulatedAllocations int64          `json:"simulated_allocations"`
	ZeroCopyVerified     bool           `json:"zero_copy_verified"`
	PrimaryMode          UMAMappingMode `json:"primary_mode"`
}
