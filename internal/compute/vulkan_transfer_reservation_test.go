//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"slices"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Injected value records test Go result shaping only. They do not establish C
// owner coherence or execute a native accessor with a synthetic buffer handle.
func TestVulkanTransferStageReservationsAllOrUnknown(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanTransferStageBufferReservations(); ok || got != nil {
		t.Fatalf("nil receiver = %v, %t; want unavailable", got, ok)
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	want := VulkanBufferReservation{
		BufferBytes: 64, ReservationBytes: 256, AllocationID: 17,
		Backing: VulkanBufferBacking{RequestedPropertyFlags: 6, MemoryTypeIndex: 2,
			PropertyFlags: 7, HeapIndex: 1, HeapFlags: 1},
	}
	read := func() (VulkanBufferReservation, bool) { return want, true }
	got, ok := vulkanTransferStageBufferReservationsLocked(read)
	expected := want
	expected.Backing.Part, expected.Backing.Chunk = "data", -1
	if !ok || len(got) != 1 || got[0] != expected {
		t.Fatalf("retained stage = %v, %t; want %+v", got, ok, expected)
	}
	got[0] = VulkanBufferReservation{}
	if again, ok := vulkanTransferStageBufferReservationsLocked(read); !ok || len(again) != 1 || again[0] != expected {
		t.Fatalf("copied stage = %v, %t; want unchanged value", again, ok)
	}
	for _, tc := range []struct {
		name   string
		record VulkanBufferReservation
		ok     bool
		empty  bool
	}{
		{name: "known empty", ok: true, empty: true},
		{name: "unavailable"},
		{name: "failed retained read", record: want},
		{name: "empty with reservation", record: VulkanBufferReservation{ReservationBytes: 256}, ok: true},
		{name: "empty with identity", record: VulkanBufferReservation{AllocationID: 17}, ok: true},
		{name: "empty with offset", record: VulkanBufferReservation{BindingOffset: 4}, ok: true},
		{name: "empty with backing", record: VulkanBufferReservation{Backing: want.Backing}, ok: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			got, ok := vulkanTransferStageBufferReservationsLocked(func() (VulkanBufferReservation, bool) {
				calls++
				return tc.record, tc.ok
			})
			if calls != 1 {
				t.Fatalf("reader called %d times; want one complete read", calls)
			}
			if tc.empty {
				if !ok || got == nil || len(got) != 0 {
					t.Fatalf("empty = %v, %t; want nonnil empty success", got, ok)
				}
			} else if ok || got != nil {
				t.Fatalf("unavailable = %v, %t; want nil, false", got, ok)
			}
		})
	}
}

// fak-test:runtime integration est=250ms lane=optin
// Run on a required physical Vulkan device with a matching rebuilt shim. This
// bounded upload/read fixture does not force initial emptiness or stage growth;
// init itself may have created the stage. A skipped run is not a witness.
func TestVulkanTransferStageBufferReservations(t *testing.T) {
	v := vk(t)
	values := []float32{3.25, -7.5, 11, 0.125}
	d := v.Upload(NewF32(Default(), []int{len(values)}, values), F32)
	defer v.Free(d)
	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("before reservation query: %v", err)
	}
	got, ok := v.VulkanTransferStageBufferReservations()
	if !ok || len(got) != 1 {
		t.Fatalf("transfer stage = %v, %t; want one complete reservation", got, ok)
	}
	want := got[0]
	const hostVisibleCoherent = uint32(0x2 | 0x4)
	if want.BufferBytes < uint64(len(values)*4) || want.ReservationBytes < want.BufferBytes ||
		want.AllocationID == 0 || want.BindingOffset != 0 ||
		want.Backing.Part != "data" || want.Backing.Chunk != -1 ||
		want.Backing.RequestedPropertyFlags != hostVisibleCoherent ||
		want.Backing.PropertyFlags&hostVisibleCoherent != hostVisibleCoherent ||
		want.Backing.WeightArenaBound || want.Backing.HostVisibleFallback {
		t.Fatalf("invalid ordinary transfer-stage reservation: %+v", want)
	}
	if backing, ok := v.VulkanTransferStageBufferBackings(); !ok || len(backing) != 1 ||
		backing[0].BufferBytes != want.BufferBytes || backing[0].Backing != want.Backing {
		t.Fatalf("reservation disagrees with backing observer: %v, %t; reservation=%+v", backing, ok, want)
	}
	got[0] = VulkanBufferReservation{}
	for _, receiver := range []*vulkanBackend{v, {}} {
		if again, ok := receiver.VulkanTransferStageBufferReservations(); !ok || len(again) != 1 || again[0] != want {
			t.Fatalf("shared retained owner = %v, %t; want copied unchanged reservation", again, ok)
		}
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("after reservation query: %v", err)
	}
	if before.Counters != after.Counters || !before.DeviceAllocationObserved ||
		!after.DeviceAllocationObserved || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
		t.Fatal("reservation queries changed transfer/dispatch counters or observed live allocation bytes")
	}
	if read := v.Read(d); !slices.Equal(read, values) {
		t.Fatalf("device read = %v; want %v", read, values)
	}
	if afterRead, ok := v.VulkanTransferStageBufferReservations(); !ok || len(afterRead) != 1 || afterRead[0] != want {
		t.Fatalf("D2H owner = %v, %t; want the same retained H2D/D2H reservation", afterRead, ok)
	}
	// Trim changes other owners/counters, so it is outside the nonmutation bracket.
	v.Trim()
	if afterTrim, ok := v.VulkanTransferStageBufferReservations(); !ok || len(afterTrim) != 1 || afterTrim[0] != want {
		t.Fatalf("trimmed owner = %v, %t; want the retained ordinary stage", afterTrim, ok)
	}
}
