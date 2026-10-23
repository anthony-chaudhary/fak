package agent

import "github.com/anthony-chaudhary/fak/internal/compute"

// DeviceWeightMemory is the device weight-memory ledger of a local planner's backend: the
// bump-only weight arena that holds immutable weights, and the paged-weight page-ins (an
// expert ring) that recycle evicted device buffers instead of taking arena bytes. Each half is
// present only when the backend reports it, so an unsupported backend renders as absent rather
// than as zeros.
type DeviceWeightMemory struct {
	Backend string `json:"backend"`

	Arena                  bool   `json:"arena"`
	ArenaMemoryAllocations uint64 `json:"arena_memory_allocations"`
	ArenaBufferBindings    uint64 `json:"arena_buffer_bindings"`
	ArenaReservedBytes     uint64 `json:"arena_reserved_bytes"`
	ArenaLiveBytes         uint64 `json:"arena_live_bytes"`
	ArenaPeakReservedBytes uint64 `json:"arena_peak_reserved_bytes"`

	PagedWeights        bool   `json:"paged_weights"`
	PagedWeightsReused  uint64 `json:"paged_weights_reused"`
	PagedWeightsFreshly uint64 `json:"paged_weights_fresh"`
}

// DeviceWeightMemoryReporter is implemented by local planners whose backend exposes the device
// weight-memory counters. ok=false omits the family.
type DeviceWeightMemoryReporter interface {
	DeviceWeightMemory() (DeviceWeightMemory, bool)
}

// Neutral readers take precedence independently for each counter family. Interface
// presence, not a nonzero value, distinguishes supported observations from absence.
type deviceWeightArenaCounters interface {
	WeightArenaCounters() (memoryAllocations, bufferBindings, reservedBytes, liveBytes, peakReservedBytes uint64)
}

type devicePagedWeightCounters interface {
	OffloadWeightCounters() (reused, fresh uint64)
}

type vulkanWeightArenaCounters interface {
	VulkanWeightArenaCounters() (memoryAllocations, bufferBindings, reservedBytes, liveBytes, peakReservedBytes uint64)
}

type vulkanPagedWeightCounters interface {
	VulkanOffloadWeightCounters() (reused, fresh uint64)
}

// DeviceWeightMemoryFromBackend reads the optional counter contracts a backend offers.
func DeviceWeightMemoryFromBackend(be compute.Backend) (DeviceWeightMemory, bool) {
	if be == nil {
		return DeviceWeightMemory{}, false
	}
	out := DeviceWeightMemory{Backend: be.Name()}
	if c, ok := be.(deviceWeightArenaCounters); ok {
		out.Arena = true
		out.ArenaMemoryAllocations, out.ArenaBufferBindings, out.ArenaReservedBytes, out.ArenaLiveBytes, out.ArenaPeakReservedBytes = c.WeightArenaCounters()
	} else if c, ok := be.(vulkanWeightArenaCounters); ok {
		out.Arena = true
		out.ArenaMemoryAllocations, out.ArenaBufferBindings, out.ArenaReservedBytes, out.ArenaLiveBytes, out.ArenaPeakReservedBytes = c.VulkanWeightArenaCounters()
	}
	if c, ok := be.(devicePagedWeightCounters); ok {
		out.PagedWeights = true
		out.PagedWeightsReused, out.PagedWeightsFreshly = c.OffloadWeightCounters()
	} else if c, ok := be.(vulkanPagedWeightCounters); ok {
		out.PagedWeights = true
		out.PagedWeightsReused, out.PagedWeightsFreshly = c.VulkanOffloadWeightCounters()
	}
	return out, out.Arena || out.PagedWeights
}

// DeviceWeightMemory implements DeviceWeightMemoryReporter for the planner's device backend.
func (p *InKernelPlanner) DeviceWeightMemory() (DeviceWeightMemory, bool) {
	if p == nil {
		return DeviceWeightMemory{}, false
	}
	return DeviceWeightMemoryFromBackend(p.backend)
}
