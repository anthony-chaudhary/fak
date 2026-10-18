//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"time"
)

// VulkanBufferRequirementsProfile names exact flags=0, exclusive-sharing,
// pNext=nil buffer descriptions already used by this backend. It does not
// choose memory properties or a heap, and it does not model arena capacity.
type VulkanBufferRequirementsProfile uint32

const (
	VulkanBufferRequirementsStorage   VulkanBufferRequirementsProfile = 1
	VulkanBufferRequirementsTransfer  VulkanBufferRequirementsProfile = 2
	vulkanBufferRequirementsMaxProbes                                 = 16
)

// VulkanBufferRequirementsRequest describes one future buffer, not a tensor
// shape. Storage uses STORAGE_BUFFER|TRANSFER_SRC|TRANSFER_DST; Transfer uses
// TRANSFER_SRC|TRANSFER_DST. SizeBytes must already include caller-known logical
// padding. Both profiles are conservatively limited by the backend's current
// single-resource cap. This API does not infer geometric growth or lifetimes.
type VulkanBufferRequirementsRequest struct {
	SizeBytes uint64
	Profile   VulkanBufferRequirementsProfile
}

// VulkanBufferRequirementsRow preserves the exact driver requirements for the
// corresponding request. MemoryTypeBits requires a same-generation type/heap
// mapping; dedicated flags are facts, not an allocation or admission decision.
type VulkanBufferRequirementsRow struct {
	Request           VulkanBufferRequirementsRequest
	RequiredBytes     uint64
	Alignment         uint64
	MemoryTypeBits    uint32
	RequiresDedicated bool
	PrefersDedicated  bool
}

// VulkanBufferRequirementsSnapshot is a completed transient-object probe.
// There is no persistent Vulkan object or VkDeviceMemory allocation. Driver
// host allocation during vkCreateBuffer is possible and can fail. RequiredBytes
// is not resident memory, heap availability, arena reservation or a fit proof.
//
// The caller must retain the current live backend and prevent reinitialization.
// DeviceGeneration identifies the native lifetime; it does not authenticate a
// stale Go alias. Driver budget observations must be collected separately.
type VulkanBufferRequirementsSnapshot struct {
	DeviceGeneration uint64
	MemoryTypeCount  uint32
	StartedAt        time.Time
	CompletedAt      time.Time
	Rows             []VulkanBufferRequirementsRow
}

// VulkanBufferMemoryRequirements explicitly probes at most sixteen buffers.
// It TryLocks the complete forward owner, so active or reentrant execution is
// refused before any native call. Then it holds vulkanMu across validation,
// create/query/destroy and complete-or-empty publication. Direct unaudited
// backend callers still must honor the cooperative owner contract.
//
// The native probe refuses busy, quarantined or failed-submission state; it
// never flushes, fences, retries, binds memory, trims pools, clears faults or
// enables resident selection. Default execution does not call this method.
// Driver create errors return no partial results and no allocation-fit credit.
func (v *vulkanBackend) VulkanBufferMemoryRequirements(requests []VulkanBufferRequirementsRequest) (VulkanBufferRequirementsSnapshot, error) {
	if v == nil {
		return VulkanBufferRequirementsSnapshot{}, fmt.Errorf("compute: Vulkan requirements backend unavailable")
	}
	if err := validateVulkanBufferRequirementsRequests(requests); err != nil {
		return VulkanBufferRequirementsSnapshot{}, err
	}
	owner := v.ForwardExecutionOwner()
	if owner == nil || !owner.TryLock() {
		return VulkanBufferRequirementsSnapshot{}, fmt.Errorf("compute: Vulkan requirements execution owner busy")
	}
	defer owner.Unlock()
	if v.ForwardExecutionOwner() != owner {
		return VulkanBufferRequirementsSnapshot{}, fmt.Errorf("compute: Vulkan requirements execution owner changed")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var inputs [vulkanBufferRequirementsMaxProbes]C.fvk_buffer_requirements_request
	for i, request := range requests {
		inputs[i].size_bytes = C.uint64_t(request.SizeBytes)
		inputs[i].profile = C.uint32_t(request.Profile)
	}
	var raw C.fvk_buffer_requirements_snapshot
	var driver C.int32_t
	started := time.Now()
	status := C.fvk_buffer_requirements_probe(&inputs[0], C.uint32_t(len(requests)), &raw, &driver)
	completed := time.Now()
	if status != C.FVK_BUFFER_REQUIREMENTS_OK {
		return VulkanBufferRequirementsSnapshot{}, fmt.Errorf("compute: Vulkan requirements probe status %d, create VkResult %d", int(status), int32(driver))
	}
	if driver != 0 || int(raw.count) != len(requests) || raw.count > C.FVK_BUFFER_REQUIREMENTS_MAX_PROBES {
		return VulkanBufferRequirementsSnapshot{}, fmt.Errorf("compute: invalid Vulkan requirements result count/status")
	}
	result := VulkanBufferRequirementsSnapshot{
		DeviceGeneration: uint64(raw.device_generation), MemoryTypeCount: uint32(raw.memory_type_count),
		StartedAt: started, CompletedAt: completed,
		Rows: make([]VulkanBufferRequirementsRow, len(requests)),
	}
	for i := range result.Rows {
		row := raw.rows[i]
		if row.requires_dedicated > 1 || row.prefers_dedicated > 1 {
			return VulkanBufferRequirementsSnapshot{}, fmt.Errorf("compute: invalid Vulkan requirements dedicated flag")
		}
		result.Rows[i] = VulkanBufferRequirementsRow{
			Request:       VulkanBufferRequirementsRequest{uint64(row.requested_bytes), VulkanBufferRequirementsProfile(row.profile)},
			RequiredBytes: uint64(row.required_bytes), Alignment: uint64(row.alignment), MemoryTypeBits: uint32(row.memory_type_bits),
			RequiresDedicated: row.requires_dedicated != 0, PrefersDedicated: row.prefers_dedicated != 0,
		}
	}
	if err := validateVulkanBufferRequirementsSnapshot(requests, result); err != nil {
		return VulkanBufferRequirementsSnapshot{}, err
	}
	return result, nil
}

func validateVulkanBufferRequirementsRequests(requests []VulkanBufferRequirementsRequest) error {
	if len(requests) == 0 || len(requests) > vulkanBufferRequirementsMaxProbes {
		return fmt.Errorf("compute: invalid Vulkan requirements request count")
	}
	for _, request := range requests {
		if request.SizeBytes == 0 || request.SizeBytes > uint64(1<<63-1) ||
			(request.Profile != VulkanBufferRequirementsStorage && request.Profile != VulkanBufferRequirementsTransfer) {
			return fmt.Errorf("compute: invalid Vulkan requirements request")
		}
	}
	return nil
}

func validateVulkanBufferRequirementsSnapshot(requests []VulkanBufferRequirementsRequest, s VulkanBufferRequirementsSnapshot) error {
	if err := validateVulkanBufferRequirementsRequests(requests); err != nil {
		return err
	}
	if s.DeviceGeneration == 0 || s.MemoryTypeCount == 0 || s.MemoryTypeCount > 32 ||
		s.StartedAt.IsZero() || s.CompletedAt.Before(s.StartedAt) || len(s.Rows) != len(requests) {
		return fmt.Errorf("compute: invalid Vulkan requirements snapshot identity/count")
	}
	validTypes := ^uint32(0)
	if s.MemoryTypeCount < 32 {
		validTypes = uint32(1)<<s.MemoryTypeCount - 1
	}
	for i, row := range s.Rows {
		if row.Request != requests[i] || row.RequiredBytes < row.Request.SizeBytes ||
			row.Alignment == 0 || row.Alignment&(row.Alignment-1) != 0 ||
			row.MemoryTypeBits == 0 || row.MemoryTypeBits & ^validTypes != 0 {
			return fmt.Errorf("compute: invalid Vulkan requirements row %d", i)
		}
	}
	return nil
}
