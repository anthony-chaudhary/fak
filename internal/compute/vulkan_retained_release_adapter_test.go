//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"strings"
	"testing"
)

func retainedAdapterFixture() VulkanRetainedReleaseSnapshot {
	return VulkanRetainedReleaseSnapshot{Status: VulkanRetainedOK, DeviceGeneration: 7, ReturnedUniqueAllocations: 1, ReturnedUniqueReservationBytes: 128, Records: []VulkanRetainedReleaseRecord{
		{AllocationID: 3, BufferBytes: 16, ReservationBytes: 128, DeviceGeneration: 7, ReleaseState: 1, BackingState: 1, Container: 1},
		{AllocationID: 3, BindingOffset: 64, BufferBytes: 16, ReservationBytes: 128, DeviceGeneration: 7, ReleaseState: 1, BackingState: 2, Container: 2},
	}}
}

// fak-test:runtime fast est=5ms lane=default
func TestVulkanRetainedReleaseAdapterValidation(t *testing.T) {
	s := retainedAdapterFixture()
	if got := validateVulkanRetainedReleaseSnapshot(s, 2, 2, 0, 1); got.Status != VulkanRetainedOK || got.ReturnedUniqueReservationBytes != 128 {
		t.Fatalf("dedup: %+v", got)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*VulkanRetainedReleaseSnapshot)
	}{
		{"unknown-status", func(s *VulkanRetainedReleaseSnapshot) { s.Status = 99 }},
		{"no-generation", func(s *VulkanRetainedReleaseSnapshot) { s.DeviceGeneration = 0 }},
		{"stale-row", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].DeviceGeneration = 6 }},
		{"invented-lease", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].LogicalLeaseGeneration = 1 }},
		{"zero-id", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].AllocationID = 0 }},
		{"zero-reservation", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].ReservationBytes = 0 }},
		{"offset-range", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].BindingOffset = 129 }},
		{"length-range", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].BufferBytes = 129 }},
		{"duplicate-binding", func(s *VulkanRetainedReleaseSnapshot) { s.Records[1].BindingOffset = 0 }},
		{"inconsistent-capacity", func(s *VulkanRetainedReleaseSnapshot) { s.Records[1].ReservationBytes = 256 }},
		{"release-state", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].ReleaseState = 0 }},
		{"container", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].Container = 4 }},
		{"backing", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].BackingState = 0 }},
		{"pool-deferred", func(s *VulkanRetainedReleaseSnapshot) { s.Records[1].BackingState = 1 }},
		{"deferred-pooled", func(s *VulkanRetainedReleaseSnapshot) { s.Records[0].BackingState = 2 }},
		{"count-total", func(s *VulkanRetainedReleaseSnapshot) { s.ReturnedUniqueAllocations = 2 }},
		{"byte-total", func(s *VulkanRetainedReleaseSnapshot) { s.ReturnedUniqueReservationBytes = 256 }},
		{"overflow", func(s *VulkanRetainedReleaseSnapshot) {
			s.Records[0].ReservationBytes = ^uint64(0)
			s.Records[1].AllocationID = 4
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := retainedAdapterFixture()
			tc.mutate(&s)
			got := validateVulkanRetainedReleaseSnapshot(s, 2, 2, 0, int(s.Status))
			if got.Status != VulkanRetainedInvalid || len(got.Records) != 0 || got.ReturnedUniqueReservationBytes != 0 || got.ReturnedUniqueAllocations != 0 {
				t.Fatalf("unsafe evidence: %+v", got)
			}
		})
	}
	for _, args := range [][4]int{{-1, 2, 0, 1}, {1025, 2, 0, 1}, {1, 2, 0, 1}, {2, 1, 0, 1}, {2, 2, 2, 1}, {2, 2, 0, 2}} {
		got := validateVulkanRetainedReleaseSnapshot(retainedAdapterFixture(), args[0], uint32(args[1]), uint32(args[2]), args[3])
		if got.Status != VulkanRetainedInvalid {
			t.Fatalf("args %v: %+v", args, got)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestVulkanRetainedReleaseAdapterDistinctions(t *testing.T) {
	for _, status := range []VulkanRetainedReleaseStatus{VulkanRetainedUnavailable, VulkanRetainedInvalid, VulkanRetainedStaleGeneration} {
		s := VulkanRetainedReleaseSnapshot{Status: status, DeviceGeneration: 7}
		got := validateVulkanRetainedReleaseSnapshot(s, 0, 0, 0, int(status))
		if got.Status != status {
			t.Fatalf("lost failure: %+v", got)
		}
		if !strings.Contains(got.Summary(), "returned_prefix_reserved_bytes=unavailable") {
			t.Fatal(got.Summary())
		}
		s.ReturnedUniqueReservationBytes = 1
		if validateVulkanRetainedReleaseSnapshot(s, 0, 0, 0, int(status)).Status != VulkanRetainedInvalid {
			t.Fatal("accepted failure totals")
		}
	}
	for _, truncated := range []uint32{0, 1} {
		got := validateVulkanRetainedReleaseSnapshot(VulkanRetainedReleaseSnapshot{Status: VulkanRetainedOK, DeviceGeneration: 7}, 0, 0, truncated, 1)
		if got.Status != VulkanRetainedOK || got.Truncated != (truncated == 1) {
			t.Fatalf("empty prefix: %+v", got)
		}
	}
	s := retainedAdapterFixture()
	s.Records[0].BackingState = 3
	s.Records[1].BackingState = 3
	got := validateVulkanRetainedReleaseSnapshot(s, 2, 2, 1, 1)
	for _, text := range []string{"truncated=true", "quarantined_bindings=2", "partial_scope=", "logical_lease_generation=unavailable", "physical_release=unknown", "prefix totals must not be added"} {
		if !strings.Contains(got.Summary(), text) {
			t.Fatalf("missing %s: %s", text, got.Summary())
		}
	}
	// Allocation identity is stable on reuse, with no fabricated lease generation.
	s = retainedAdapterFixture()
	s.Records = s.Records[:1]
	s.Records[0].BackingState = 2
	s.Records[0].Container = 2
	got = validateVulkanRetainedReleaseSnapshot(s, 1, 1, 0, 1)
	if got.Status != VulkanRetainedOK || got.Records[0].AllocationID != 3 || got.Records[0].LogicalLeaseGeneration != 0 {
		t.Fatalf("reuse: %+v", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestVulkanRetainedReleaseNativeCheckoutScope(t *testing.T) {
	for _, generation := range []uint64{0, 1, 2, ^uint64(0)} {
		s := retainedAdapterFixture()
		s.Records[0].NativeCheckoutGeneration = generation
		got := validateVulkanRetainedReleaseSnapshot(s, 2, 2, 0, 1)
		if got.Status != VulkanRetainedOK || got.Records[0].NativeCheckoutGeneration != generation || got.Records[0].LogicalLeaseGeneration != 0 {
			t.Fatalf("native checkout evidence: %+v", got)
		}
		for _, label := range []string{"native_checkout_scope=native-pool-only", "Go-transient-ownership=excluded", "stale-alias-validation=unavailable", "logical_lease_generation=unavailable"} {
			if !strings.Contains(got.Summary(), label) {
				t.Fatal(got.Summary())
			}
		}
		counts := "native_checkout_known_bindings=1 native_checkout_unavailable_bindings=1"
		if generation == 0 {
			counts = "native_checkout_known_bindings=0 native_checkout_unavailable_bindings=2"
		}
		if !strings.Contains(got.Summary(), counts) {
			t.Fatal(got.Summary())
		}
	}
}
