//go:build vulkan && (windows || linux) && cgo

package compute

import "fmt"

// VulkanRetainedReleaseLimit bounds both output capacity and native inspection work.
const VulkanRetainedReleaseLimit = 1024

type VulkanRetainedReleaseStatus uint32

const (
	VulkanRetainedOK              VulkanRetainedReleaseStatus = 1
	VulkanRetainedUnavailable     VulkanRetainedReleaseStatus = 2
	VulkanRetainedInvalid         VulkanRetainedReleaseStatus = 3
	VulkanRetainedStaleGeneration VulkanRetainedReleaseStatus = 4
)

// VulkanRetainedReleaseRecord identifies a retained binding, never a usable handle.
// NativeCheckoutGeneration compares native pool checkouts of the same device/allocation/
// binding across snapshots; zero means unavailable, including counter exhaustion.
// Go transient-pool ownership changes are excluded. This cannot validate stale aliases.
// LogicalLeaseGeneration is always zero (unavailable).
// ReleaseState 1 means release requested. BackingState is 1 deferred, 2 pool-retained,
// or 3 quarantined. Container is 1 deferred batch, 2 device pool, or 3 offload pool.
type VulkanRetainedReleaseRecord struct {
	AllocationID, BindingOffset, BufferBytes, ReservationBytes         uint64
	DeviceGeneration, LogicalLeaseGeneration, NativeCheckoutGeneration uint64
	ReleaseState, BackingState, Container                              uint32
}

// VulkanRetainedReleaseSnapshot covers only a bounded prefix of native deferred and
// pool containers. It excludes active owners, Go pools, stages and driver overhead.
// Totals deduplicate only returned allocations; never add them to owner totals.
// Truncated may be true with no records. Absence never proves physical release.
type VulkanRetainedReleaseSnapshot struct {
	Status                         VulkanRetainedReleaseStatus
	DeviceGeneration               uint64
	Records                        []VulkanRetainedReleaseRecord
	ReturnedUniqueAllocations      uint32
	ReturnedUniqueReservationBytes uint64
	Truncated                      bool
}

// validateVulkanRetainedReleaseSnapshot rejects malformed ABI evidence before publication.
func validateVulkanRetainedReleaseSnapshot(s VulkanRetainedReleaseSnapshot, capacity int, returned, truncated uint32, result int) VulkanRetainedReleaseSnapshot {
	invalid := VulkanRetainedReleaseSnapshot{Status: VulkanRetainedInvalid, DeviceGeneration: s.DeviceGeneration}
	if capacity < 0 || capacity > VulkanRetainedReleaseLimit || result != int(s.Status) ||
		s.Status < VulkanRetainedOK || s.Status > VulkanRetainedStaleGeneration || truncated > 1 ||
		returned > uint32(capacity) || int(returned) != len(s.Records) {
		return invalid
	}
	if s.Status != VulkanRetainedOK {
		if returned != 0 || truncated != 0 || s.ReturnedUniqueAllocations != 0 || s.ReturnedUniqueReservationBytes != 0 {
			return invalid
		}
		return VulkanRetainedReleaseSnapshot{Status: s.Status, DeviceGeneration: s.DeviceGeneration}
	}
	if s.DeviceGeneration == 0 {
		return invalid
	}
	allocations := make(map[uint64]uint64, len(s.Records))
	bindings := make(map[[2]uint64]bool, len(s.Records))
	var total uint64
	for _, r := range s.Records {
		if r.AllocationID == 0 || r.DeviceGeneration != s.DeviceGeneration || r.LogicalLeaseGeneration != 0 ||
			r.ReservationBytes == 0 || r.BindingOffset > r.ReservationBytes || r.BufferBytes > r.ReservationBytes-r.BindingOffset ||
			r.ReleaseState != 1 || r.Container < 1 || r.Container > 3 || r.BackingState < 1 || r.BackingState > 3 ||
			(r.BackingState != 3 && ((r.Container == 1 && r.BackingState != 1) || (r.Container != 1 && r.BackingState != 2))) {
			return invalid
		}
		key := [2]uint64{r.AllocationID, r.BindingOffset}
		if bindings[key] {
			return invalid
		}
		bindings[key] = true
		if prior, ok := allocations[r.AllocationID]; ok {
			if prior != r.ReservationBytes {
				return invalid
			}
		} else {
			if r.ReservationBytes > ^uint64(0)-total {
				return invalid
			}
			allocations[r.AllocationID] = r.ReservationBytes
			total += r.ReservationBytes
		}
	}
	if uint32(len(allocations)) != s.ReturnedUniqueAllocations || total != s.ReturnedUniqueReservationBytes {
		return invalid
	}
	s.Truncated = truncated != 0
	return s
}

// Summary labels prefix accounting and unavailable evidence for operator diagnostics.
func (s VulkanRetainedReleaseSnapshot) Summary() string {
	status := "invalid"
	switch s.Status {
	case VulkanRetainedOK:
		status = "ok"
	case VulkanRetainedUnavailable:
		status = "unavailable"
	case VulkanRetainedStaleGeneration:
		status = "stale-generation"
	case VulkanRetainedInvalid:
		status = "invalid"
	}
	counts := "returned_bindings=unavailable returned_unique_allocations=unavailable returned_prefix_reserved_bytes=unavailable"
	if s.Status == VulkanRetainedOK {
		var deferred, pooled, quarantined, checkoutKnown int
		for _, r := range s.Records {
			if r.NativeCheckoutGeneration != 0 {
				checkoutKnown++
			}
			switch r.BackingState {
			case 1:
				deferred++
			case 2:
				pooled++
			case 3:
				quarantined++
			}
		}
		counts = fmt.Sprintf("returned_bindings=%d returned_unique_allocations=%d returned_prefix_reserved_bytes=%d deferred_bindings=%d pool-retained bindings=%d quarantined_bindings=%d native_checkout_known_bindings=%d native_checkout_unavailable_bindings=%d", len(s.Records), s.ReturnedUniqueAllocations, s.ReturnedUniqueReservationBytes, deferred, pooled, quarantined, checkoutKnown, len(s.Records)-checkoutKnown)
	}
	return fmt.Sprintf("retained_release: status=%s device_generation=%d truncated=%t %s; partial_scope=native-deferred+native-pools; logical_lease_generation=unavailable native_checkout_scope=native-pool-only Go-transient-ownership=excluded stale-alias-validation=unavailable physical_release=unknown; prefix totals must not be added to owner totals", status, s.DeviceGeneration, s.Truncated, counts)
}
