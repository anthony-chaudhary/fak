//go:build vulkan && (windows || linux) && cgo

package main

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// These optional metadata readers do not allocate Vulkan storage, transfer,
// submit, fence, or change residency. Reservation records already include backing
// flags; a second Backings query would introduce another observation window.
type serveVulkanReservationReader interface {
	VulkanTensorBufferReservations(compute.Tensor) ([]compute.VulkanBufferReservation, bool)
	VulkanQ4KHomeBufferReservations() ([]compute.VulkanBufferReservation, bool)
	VulkanQ4KStageBufferReservations() ([]compute.VulkanBufferReservation, bool)
	VulkanTransferStageBufferReservations() ([]compute.VulkanBufferReservation, bool)
}

type serveVulkanReservationTotals struct {
	allocations                        map[uint64]compute.VulkanBufferReservation
	reserved, deviceLocal, hostVisible uint64
	bindings                           uint64
	valid                              bool
}

func newServeVulkanReservationTotals() *serveVulkanReservationTotals {
	return &serveVulkanReservationTotals{allocations: make(map[uint64]compute.VulkanBufferReservation), valid: true}
}

// add deduplicates complete allocation requests globally across tensor bindings,
// homes and stages. Backing subsets overlap on UMA and must never be added.
func (s *serveVulkanReservationTotals) add(rows []compute.VulkanBufferReservation, available bool) {
	if !s.valid || !available {
		s.valid = false
		return
	}
	for _, r := range rows {
		if r.AllocationID == 0 || r.BufferBytes == 0 || r.ReservationBytes == 0 ||
			r.BindingOffset > r.ReservationBytes || r.BufferBytes > r.ReservationBytes-r.BindingOffset ||
			s.bindings == math.MaxUint64 {
			s.valid = false
			return
		}
		s.bindings++
		if old, seen := s.allocations[r.AllocationID]; seen {
			// Binding offsets, parts and requested flags may differ within an arena;
			// the actual allocation size and selected memory type/heap may not.
			if old.ReservationBytes != r.ReservationBytes || old.Backing.MemoryTypeIndex != r.Backing.MemoryTypeIndex ||
				old.Backing.PropertyFlags != r.Backing.PropertyFlags || old.Backing.HeapIndex != r.Backing.HeapIndex ||
				old.Backing.HeapFlags != r.Backing.HeapFlags {
				s.valid = false
				return
			}
			continue
		}
		if r.ReservationBytes > math.MaxUint64-s.reserved {
			s.valid = false
			return
		}
		s.allocations[r.AllocationID] = r
		s.reserved += r.ReservationBytes
		// Vulkan VkMemoryPropertyFlagBits: DEVICE_LOCAL=1, HOST_VISIBLE=2.
		if r.Backing.PropertyFlags&1 != 0 {
			s.deviceLocal += r.ReservationBytes
		}
		if r.Backing.PropertyFlags&2 != 0 {
			s.hostVisible += r.ReservationBytes
		}
	}
}

func (s *serveVulkanReservationTotals) text() string {
	if !s.valid {
		return "covered_reserved_bytes=unavailable backing_subsets=unavailable"
	}
	return fmt.Sprintf("covered_reserved_bytes=%d allocations=%d bindings=%d backing_device_local_reserved_bytes=%d backing_host_visible_reserved_bytes=%d (overlapping subsets)",
		s.reserved, len(s.allocations), s.bindings, s.deviceLocal, s.hostVisible)
}

// This is a bounded startup inventory, before serving sessions and demand-staged
// weights. It is not an atomic whole-backend snapshot or a residency/health probe.
func serveVulkanPostLoadReservations(m *fakmodel.Model, be compute.Backend, plan compute.MemoryPlan) (string, bool) {
	r, ok := be.(serveVulkanReservationReader)
	if !ok {
		return "", false
	}
	totals := newServeVulkanReservationTotals()
	weightsOK := m.VisitImmutableDeviceWeights(be, func(t compute.Tensor) bool {
		rows, available := r.VulkanTensorBufferReservations(t)
		totals.add(rows, available)
		return totals.valid
	})
	if !weightsOK {
		totals.valid = false
	}
	// The ordinary transfer stage is shim-global: query it exactly once, never
	// once per tensor, receiver, or transfer direction.
	homes, homesOK := r.VulkanQ4KHomeBufferReservations()
	totals.add(homes, homesOK)
	stage, stageOK := r.VulkanQ4KStageBufferReservations()
	totals.add(stage, stageOK)
	transfer, transferOK := r.VulkanTransferStageBufferReservations()
	totals.add(transfer, transferOK)
	var planned uint64
	plannedOK := true
	for _, demand := range plan {
		if !demand.DeviceScoped() {
			continue
		}
		if demand.Bytes < 0 || uint64(demand.Bytes) > math.MaxUint64-planned {
			plannedOK = false
			break
		}
		planned += uint64(demand.Bytes)
	}
	plannedText := "unavailable"
	if plannedOK {
		plannedText = fmt.Sprint(planned)
	}
	return fmt.Sprintf("phase=post-load-before-serving planned_device_bytes=%s; %s; owners_available: model_weights=%t q4k_homes=%t q4k_stage=%t transfer_stage=%t; physical_vram_observed_bytes=unknown; scope=model-owned immutable tensors+Q4K homes+Q4K stage+global transfer stage; excludes=session tensors,expert rings,KV,restore staging,scratch,pools,driver overhead; separate metadata reads, not atomic; reservation/backing flags do not prove physical residency",
		plannedText, totals.text(), weightsOK, homesOK, stageOK, transferOK), true
}
