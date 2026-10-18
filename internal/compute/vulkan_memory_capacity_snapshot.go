//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"strings"
	"time"
)

// VulkanMemoryCapacityHeap keeps each real heap separate. BudgetKnown=false
// means both budget fields are unavailable, not zero free memory. Usage may
// exceed BudgetBytes under pressure. Heap totals must not be added to host RAM:
// this API does not establish whether they share the same physical memory.
type VulkanMemoryCapacityHeap struct {
	SizeBytes   uint64
	BudgetBytes uint64
	UsageBytes  uint64
	Flags       uint32
	BudgetKnown bool
}

// VulkanMemoryCapacityType maps a compatible memory type to its actual heap.
type VulkanMemoryCapacityType struct {
	PropertyFlags uint32
	HeapIndex     uint32
}

// VulkanMemoryCapacitySnapshot is one backend-atomic observation of current
// all-heap VkDeviceMemory reservations and driver heap budgets. It is not a
// whole-owner inventory, physical residency measure, allocation reservation or
// future-fit certificate. Guard excludes driver and CPU allocations; heap
// budgets include other processes and are volatile. Do not add Guard.LiveBytes
// to driver usage: their ownership domains overlap.
//
// The generation names the initialized native device lifetime, not this Go
// backend wrapper. The caller must retain the current live backend and prevent
// reinitialization; this query does not authenticate stale aliases. Queue flags
// report current metadata only and are not a successful completion fence.
// MaxMemoryAllocationSize=0 means that limit is unknown. No current allocation
// count is inferred from byte counters.
type VulkanMemoryCapacitySnapshot struct {
	DeviceGeneration         uint64
	DeviceName               string
	VendorID                 uint32
	DeviceID                 uint32
	DriverVersion            uint32
	APIVersion               uint32
	DeviceType               uint32
	StartedAt                time.Time
	CompletedAt              time.Time
	Guard                    VulkanMemoryGuardSnapshot
	MaxBufferBytes           uint64
	MaxStorageBufferRange    uint64
	MaxMemoryAllocationSize  uint64
	MaxMemoryAllocationCount uint32
	SubmissionStatus         int32
	Quarantined              bool
	BatchActive              bool
	InflightActive           bool
	RestoreActive            bool
	Heaps                    []VulkanMemoryCapacityHeap
	Types                    []VulkanMemoryCapacityType
}

// VulkanMemoryCapacitySnapshot makes one explicit pull query under vulkanMu.
// It does not allocate/free Vulkan storage, transfer, flush, fence or mutate
// ownership. It allocates bounded Go result slices and may call driver metadata
// queries. Backend fields are atomic against this process's Vulkan operations;
// budgets are not atomic against other processes. Invalid metadata yields no
// partial snapshot. Valid quarantine/busy flags are preserved, never cleared.
// Default execution never calls this optional diagnostic method.
func (v *vulkanBackend) VulkanMemoryCapacitySnapshot() (VulkanMemoryCapacitySnapshot, error) {
	if v == nil {
		return VulkanMemoryCapacitySnapshot{}, fmt.Errorf("compute: Vulkan memory capacity backend unavailable")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var raw C.fvk_memory_capacity_snapshot
	started := time.Now()
	status := C.fvk_memory_capacity_snapshot_read(&raw)
	completed := time.Now()
	if status != C.FVK_MEMORY_CAPACITY_OK {
		return VulkanMemoryCapacitySnapshot{}, fmt.Errorf("compute: Vulkan memory capacity query status %d", int(status))
	}
	if raw.heap_count == 0 || raw.heap_count > C.FVK_MEMORY_CAPACITY_MAX_HEAPS ||
		raw.type_count == 0 || raw.type_count > C.FVK_MEMORY_CAPACITY_MAX_TYPES {
		return VulkanMemoryCapacitySnapshot{}, fmt.Errorf("compute: invalid Vulkan memory capacity counts")
	}
	for _, flag := range []C.uint32_t{raw.guard_initialized, raw.guard_configured,
		raw.guard_configuration_valid, raw.guard_accounting_valid, raw.quarantined,
		raw.batch_active, raw.inflight_active, raw.restore_active} {
		if flag > 1 {
			return VulkanMemoryCapacitySnapshot{}, fmt.Errorf("compute: invalid Vulkan memory capacity flag")
		}
	}
	if raw.guard_initialized != 1 {
		return VulkanMemoryCapacitySnapshot{}, fmt.Errorf("compute: Vulkan memory accounting uninitialized")
	}
	s := VulkanMemoryCapacitySnapshot{
		DeviceGeneration: uint64(raw.device_generation),
		DeviceName:       strings.TrimSpace(C.GoString(&raw.device_name[0])),
		VendorID:         uint32(raw.vendor_id), DeviceID: uint32(raw.device_id),
		DriverVersion: uint32(raw.driver_version), APIVersion: uint32(raw.api_version),
		DeviceType: uint32(raw.device_type), StartedAt: started, CompletedAt: completed,
		Guard: VulkanMemoryGuardSnapshot{
			Scope: "all_vk_device_memory", LimitBytes: uint64(raw.limit_bytes),
			LiveBytes: uint64(raw.live_bytes), PeakBytes: uint64(raw.peak_bytes),
			RefusedAllocations: uint64(raw.refused_allocations),
			Configured:         raw.guard_configured != 0,
			ConfigurationValid: raw.guard_configuration_valid != 0,
			AccountingValid:    raw.guard_accounting_valid != 0,
		},
		MaxBufferBytes:           uint64(raw.max_buffer_bytes),
		MaxStorageBufferRange:    uint64(raw.max_storage_buffer_range),
		MaxMemoryAllocationSize:  uint64(raw.max_memory_allocation_size),
		MaxMemoryAllocationCount: uint32(raw.max_memory_allocation_count),
		SubmissionStatus:         int32(raw.submission_status), Quarantined: raw.quarantined != 0,
		BatchActive: raw.batch_active != 0, InflightActive: raw.inflight_active != 0,
		RestoreActive: raw.restore_active != 0,
		Heaps:         make([]VulkanMemoryCapacityHeap, int(raw.heap_count)),
		Types:         make([]VulkanMemoryCapacityType, int(raw.type_count)),
	}
	for i := range s.Heaps {
		h := raw.heaps[i]
		if h.budget_known > 1 {
			return VulkanMemoryCapacitySnapshot{}, fmt.Errorf("compute: invalid Vulkan heap budget flag")
		}
		s.Heaps[i] = VulkanMemoryCapacityHeap{uint64(h.size_bytes), uint64(h.budget_bytes),
			uint64(h.usage_bytes), uint32(h.flags), h.budget_known != 0}
	}
	for i := range s.Types {
		s.Types[i] = VulkanMemoryCapacityType{uint32(raw.types[i].property_flags), uint32(raw.types[i].heap_index)}
	}
	if err := validateVulkanMemoryCapacitySnapshot(s); err != nil {
		return VulkanMemoryCapacitySnapshot{}, err
	}
	return s, nil
}

func validateVulkanMemoryCapacitySnapshot(s VulkanMemoryCapacitySnapshot) error {
	fail := func(reason string) error {
		return fmt.Errorf("compute: invalid Vulkan memory capacity snapshot: %s", reason)
	}
	if s.DeviceGeneration == 0 || s.DeviceName == "" || s.StartedAt.IsZero() ||
		s.CompletedAt.Before(s.StartedAt) || s.MaxBufferBytes == 0 ||
		s.MaxStorageBufferRange == 0 || s.MaxMemoryAllocationCount == 0 {
		return fail("identity, interval or limits")
	}
	if s.Guard.Scope != "all_vk_device_memory" || !s.Guard.ConfigurationValid || !s.Guard.AccountingValid ||
		s.Guard.LiveBytes > s.Guard.PeakBytes ||
		(s.Guard.Configured && (s.Guard.LimitBytes == 0 || s.Guard.LiveBytes > s.Guard.LimitBytes)) ||
		(!s.Guard.Configured && s.Guard.LimitBytes != 0) {
		return fail("guard accounting")
	}
	if len(s.Heaps) == 0 || len(s.Heaps) > 16 || len(s.Types) == 0 || len(s.Types) > 32 {
		return fail("heap/type count")
	}
	for _, h := range s.Heaps {
		if h.SizeBytes == 0 || (!h.BudgetKnown && (h.BudgetBytes != 0 || h.UsageBytes != 0)) {
			return fail("heap")
		}
	}
	for _, t := range s.Types {
		if uint64(t.HeapIndex) >= uint64(len(s.Heaps)) {
			return fail("memory type heap")
		}
	}
	return nil
}
