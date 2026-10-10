//go:build vulkan && (windows || linux) && cgo

package main

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func serveReservationRow(id, size, offset, bytes uint64, flags uint32) compute.VulkanBufferReservation {
	return compute.VulkanBufferReservation{AllocationID: id, ReservationBytes: size, BindingOffset: offset, BufferBytes: bytes,
		Backing: compute.VulkanBufferBacking{PropertyFlags: flags, MemoryTypeIndex: 1, HeapIndex: 2}}
}

// fak-test:runtime fast est=5ms lane=default
// Metadata-only scalar controls; estimates unmeasured, no Vulkan runtime required by the body.
func TestServeVulkanReservationTotalsDeduplicateAcrossOwners(t *testing.T) {
	s := newServeVulkanReservationTotals()
	s.add([]compute.VulkanBufferReservation{serveReservationRow(1, 1024, 0, 128, 3)}, true)
	s.add([]compute.VulkanBufferReservation{serveReservationRow(1, 1024, 128, 128, 3), serveReservationRow(2, 512, 0, 256, 2)}, true)
	if !s.valid || s.reserved != 1536 || s.deviceLocal != 1024 || s.hostVisible != 1536 || len(s.allocations) != 2 || s.bindings != 3 {
		t.Fatalf("bad dedup/subsets: %+v", s)
	}
	if !strings.Contains(s.text(), "overlapping subsets") {
		t.Fatal(s.text())
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestServeVulkanReservationTotalsRejectPartialOrInconsistentEvidence(t *testing.T) {
	base := serveReservationRow(1, 1024, 0, 128, 3)
	for _, kind := range []string{"unavailable", "zero-id", "bounds", "size-conflict", "type-conflict", "overflow"} {
		t.Run(kind, func(t *testing.T) {
			s := newServeVulkanReservationTotals()
			s.add([]compute.VulkanBufferReservation{base}, true)
			r := base
			switch kind {
			case "unavailable":
				s.add(nil, false)
			case "zero-id":
				r.AllocationID = 0
				s.add([]compute.VulkanBufferReservation{r}, true)
			case "bounds":
				r.BindingOffset = 1024
				s.add([]compute.VulkanBufferReservation{r}, true)
			case "size-conflict":
				r.ReservationBytes = 2048
				s.add([]compute.VulkanBufferReservation{r}, true)
			case "type-conflict":
				r.Backing.MemoryTypeIndex++
				s.add([]compute.VulkanBufferReservation{r}, true)
			case "overflow":
				s.add([]compute.VulkanBufferReservation{serveReservationRow(2, math.MaxUint64, 0, 1, 1)}, true)
			}
			if s.valid || s.text() != "covered_reserved_bytes=unavailable backing_subsets=unavailable" {
				t.Fatal(s.text())
			}
		})
	}
	empty := newServeVulkanReservationTotals()
	empty.add(nil, true)
	if !empty.valid || empty.reserved != 0 {
		t.Fatal("known-empty lost")
	}
}

type serveReservationReader struct {
	compute.Backend
	calls [4]int
}

func (r *serveReservationReader) VulkanTensorBufferReservations(compute.Tensor) ([]compute.VulkanBufferReservation, bool) {
	r.calls[0]++
	return nil, true
}
func (r *serveReservationReader) VulkanQ4KHomeBufferReservations() ([]compute.VulkanBufferReservation, bool) {
	r.calls[1]++
	return []compute.VulkanBufferReservation{serveReservationRow(7, 1024, 0, 128, 3)}, true
}
func (r *serveReservationReader) VulkanQ4KStageBufferReservations() ([]compute.VulkanBufferReservation, bool) {
	r.calls[2]++
	return []compute.VulkanBufferReservation{serveReservationRow(7, 1024, 128, 128, 3)}, true
}
func (r *serveReservationReader) VulkanTransferStageBufferReservations() ([]compute.VulkanBufferReservation, bool) {
	r.calls[3]++
	return nil, true
}

// fak-test:runtime fast est=5ms lane=default
func TestServeVulkanPostLoadSummaryIsScopedAndQueriesGlobalStageOnce(t *testing.T) {
	r := &serveReservationReader{}
	text, ok := serveVulkanPostLoadReservations(&fakmodel.Model{}, r, compute.MemoryPlan{{Bytes: 4096, Scope: compute.MemoryScopeDevice}})
	if !ok || r.calls != [4]int{0, 1, 1, 1} {
		t.Fatalf("ok=%v calls=%v", ok, r.calls)
	}
	for _, want := range []string{"phase=post-load-before-serving", "planned_device_bytes=4096", "covered_reserved_bytes=1024", "physical_vram_observed_bytes=unknown", "excludes=session tensors,expert rings,KV,restore staging,scratch,pools,driver overhead", "not atomic"} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in %q", want, text)
		}
	}
	if _, ok := serveVulkanPostLoadReservations(&fakmodel.Model{}, nil, nil); ok {
		t.Fatal("unsupported backend reported Vulkan")
	}
}
