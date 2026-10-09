//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=10ms lane=default
// These are Go ownership/control checks, not native or device qualification.
// Real Go-owned tokens are only compared by injected Go readers; they are never
// dereferenced as native buffers or passed to the C reservation accessor.
func TestVulkanQ4KOwnerReservationsAllOrUnknown(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanQ4KHomeBufferReservations(); ok || got != nil {
		t.Fatalf("nil home owner = %v, %t; want unavailable", got, ok)
	}
	if got, ok := absent.VulkanQ4KStageBufferReservations(); ok || got != nil {
		t.Fatalf("nil stage owner = %v, %t; want unavailable", got, ok)
	}
	empty := &vulkanBackend{}
	if got, ok := empty.VulkanQ4KHomeBufferReservations(); !ok || got == nil || len(got) != 0 {
		t.Fatalf("empty home owner = %v, %t; want known empty", got, ok)
	}
	if got, ok := empty.VulkanQ4KStageBufferReservations(); !ok || got == nil || len(got) != 0 {
		t.Fatalf("empty stage owner = %v, %t; want known empty", got, ok)
	}

	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var tokens [5]byte
	keyA := vulkanQ4KHomeKey{src: unsafe.Pointer(&tokens[0]), bytes: 16}
	keyB := vulkanQ4KHomeKey{src: unsafe.Pointer(&tokens[1]), bytes: 32}
	homeA := vulkanQ4KHome{ptr: unsafe.Pointer(&tokens[2]), bytes: 16}
	homeB := vulkanQ4KHome{ptr: unsafe.Pointer(&tokens[3]), bytes: 32}
	stage := unsafe.Pointer(&tokens[4])
	v := &vulkanBackend{
		homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: homeA, keyB: homeB}, homeBytes: 48,
		homeHits: 2, homeMisses: 3, homeBypasses: 4, homeCopied: 48,
		q4kStagePtr: stage, q4kStageBytes: 64, q4kStagedCalls: 5, q4kStagedBytes: 96, q4kStageFallbacks: 6,
		budgetBytes: 1024, dlUsed: 112, hostvisN: 7,
	}
	counters := func() [11]int64 {
		return [11]int64{v.homeBytes, v.homeHits, v.homeMisses, v.homeBypasses, v.homeCopied,
			v.q4kStageBytes, v.q4kStagedCalls, v.q4kStagedBytes, v.q4kStageFallbacks, v.budgetBytes, v.dlUsed}
	}
	before := counters()
	backing := VulkanBufferBacking{Part: "data", Chunk: -1, RequestedPropertyFlags: 1,
		MemoryTypeIndex: 2, PropertyFlags: 7, HeapIndex: 1, HeapFlags: 1, WeightArenaBound: true}
	wantA := VulkanBufferReservation{BufferBytes: 16, ReservationBytes: 128, AllocationID: 7, Backing: backing}
	wantB := VulkanBufferReservation{BufferBytes: 32, ReservationBytes: 128, AllocationID: 7, BindingOffset: 32, Backing: backing}
	wantStage := VulkanBufferReservation{BufferBytes: 64, ReservationBytes: 128, AllocationID: 9, Backing: backing}
	wantStage.Backing.WeightArenaBound = false
	wantStage.Backing.HostVisibleFallback = true
	calls := 0
	read := func(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferReservation, bool) {
		calls++
		if part != "data" || chunk != -1 {
			t.Fatalf("owner part/chunk = %q/%d", part, chunk)
		}
		switch ptr {
		case homeA.ptr:
			return wantA, true
		case homeB.ptr:
			return wantB, true
		case stage:
			return wantStage, true
		default:
			t.Fatalf("reader received a source key or unowned token")
			return VulkanBufferReservation{}, false
		}
	}
	got, ok := vulkanQ4KHomeBufferReservationsLocked(v.homes, v.homeBytes, read)
	if !ok || len(got) != 2 || calls != 2 {
		t.Fatalf("complete homes = %v, %t, reads=%d", got, ok, calls)
	}
	want := map[uint64]VulkanBufferReservation{16: wantA, 32: wantB}
	for _, record := range got {
		if expected, exists := want[record.BufferBytes]; !exists || record != expected {
			t.Fatalf("lost binding/shared-ID metadata: %+v", record)
		}
		delete(want, record.BufferBytes)
	}
	got[0] = VulkanBufferReservation{}
	if got, ok := vulkanQ4KStageBufferReservationsLocked(v.q4kStagePtr, v.q4kStageBytes, read); !ok || len(got) != 1 || got[0] != wantStage {
		t.Fatalf("retained disabled stage = %v, %t; want complete native metadata", got, ok)
	} else {
		got[0] = VulkanBufferReservation{}
	}

	// A failure after one successful home read must discard the first record,
	// regardless of map order. A native length mismatch is unavailable as well.
	for _, mismatch := range []bool{false, true} {
		calls = 0
		failSecond := func(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferReservation, bool) {
			record, ok := read(ptr, part, chunk)
			if calls == 2 {
				if mismatch {
					record.BufferBytes++
				} else {
					ok = false
				}
			}
			return record, ok
		}
		if got, ok := vulkanQ4KHomeBufferReservationsLocked(v.homes, v.homeBytes, failSecond); ok || got != nil || calls != 2 {
			t.Fatalf("incomplete homes (length mismatch=%t) = %v, %t, reads=%d", mismatch, got, ok, calls)
		}
		failStage := func(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferReservation, bool) {
			record, _ := read(ptr, part, chunk)
			if mismatch {
				record.BufferBytes++
			}
			return record, mismatch
		}
		if got, ok := vulkanQ4KStageBufferReservationsLocked(stage, 64, failStage); ok || got != nil {
			t.Fatalf("incomplete stage (length mismatch=%t) = %v, %t", mismatch, got, ok)
		}
	}

	calls = 0
	if got, ok := vulkanQ4KHomeBufferReservationsLocked(nil, 0, read); !ok || got == nil || len(got) != 0 || calls != 0 {
		t.Fatalf("retired homes = %v, %t, reads=%d", got, ok, calls)
	}
	if got, ok := vulkanQ4KStageBufferReservationsLocked(nil, 0, read); !ok || got == nil || len(got) != 0 || calls != 0 {
		t.Fatalf("retired stage = %v, %t, reads=%d", got, ok, calls)
	}
	for _, tc := range []struct {
		ptr   unsafe.Pointer
		bytes int64
	}{{nil, 64}, {nil, -1}, {stage, 0}, {stage, -1}} {
		if got, ok := vulkanQ4KStageBufferReservationsLocked(tc.ptr, tc.bytes, read); ok || got != nil || calls != 0 {
			t.Fatalf("inconsistent stage (%p, %d) = %v, %t, reads=%d", tc.ptr, tc.bytes, got, ok, calls)
		}
	}
	for _, tc := range []struct {
		name  string
		homes map[vulkanQ4KHomeKey]vulkanQ4KHome
		bytes int64
	}{
		{name: "negative counter", bytes: -1},
		{name: "empty with bytes", bytes: 1},
		{name: "nil source", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{{bytes: 16}: homeA}, bytes: 16},
		{name: "zero source length", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{{src: keyA.src}: homeA}, bytes: 16},
		{name: "nil home", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: {bytes: 16}}, bytes: 16},
		{name: "owner length mismatch", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: {ptr: homeA.ptr, bytes: 32}}, bytes: 32},
		{name: "counter too small", homes: v.homes, bytes: 47},
		{name: "counter too large", homes: v.homes, bytes: 49},
		{name: "duplicate handle", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: homeA, {src: keyB.src, bytes: 16}: homeA}, bytes: 32},
	} {
		if got, ok := vulkanQ4KHomeBufferReservationsLocked(tc.homes, tc.bytes, read); ok || got != nil {
			t.Fatalf("%s = %v, %t; want unavailable", tc.name, got, ok)
		}
	}
	if len(v.homes) != 2 || v.homes[keyA] != homeA || v.homes[keyB] != homeB ||
		v.q4kStagePtr != stage || v.q4kStage || v.hostvisN != 7 || counters() != before {
		t.Fatal("observations or result mutation changed owner records/counters")
	}
}
