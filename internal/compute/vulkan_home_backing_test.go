//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=10ms lane=default
// This exercises ownership/enumeration only. Synthetic handles stay in the Go
// reader; neither the C query nor a Vulkan allocation is used by this test.
func TestVulkanQ4KHomeBufferBackingsAllOrUnknown(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanQ4KHomeBufferBackings(); ok || got != nil {
		t.Fatalf("nil backend = %v, %t; want unavailable", got, ok)
	}
	if got, ok := (&vulkanBackend{}).VulkanQ4KHomeBufferBackings(); !ok || len(got) != 0 {
		t.Fatalf("empty owner cache = %v, %t; want known empty", got, ok)
	}

	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var handles [4]byte
	keyA := vulkanQ4KHomeKey{src: unsafe.Pointer(&handles[0]), bytes: 16}
	keyB := vulkanQ4KHomeKey{src: unsafe.Pointer(&handles[1]), bytes: 32}
	homeA := vulkanQ4KHome{ptr: unsafe.Pointer(&handles[2]), bytes: 16}
	homeB := vulkanQ4KHome{ptr: unsafe.Pointer(&handles[3]), bytes: 32}
	homes := map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: homeA, keyB: homeB}
	wantA := VulkanBufferBacking{
		Part: "data", Chunk: -1, RequestedPropertyFlags: 1,
		MemoryTypeIndex: 2, PropertyFlags: 7, HeapIndex: 1, HeapFlags: 1,
	}
	wantB := VulkanBufferBacking{
		Part: "data", Chunk: -1, RequestedPropertyFlags: 1,
		MemoryTypeIndex: 3, PropertyFlags: 6, HeapIndex: 1, HeapFlags: 1,
		HostVisibleFallback: true, WeightArenaBound: true,
	}
	calls := 0
	read := func(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferBacking, bool) {
		calls++
		if part != "data" || chunk != -1 {
			t.Fatalf("home part/chunk = %q/%d", part, chunk)
		}
		switch ptr {
		case homeA.ptr:
			return wantA, true
		case homeB.ptr:
			return wantB, true
		default:
			t.Fatalf("reader received a source key or an unowned handle")
			return VulkanBufferBacking{}, false
		}
	}
	got, ok := vulkanQ4KHomeBufferBackingsLocked(homes, 48, read)
	if !ok || len(got) != 2 || calls != 2 {
		t.Fatalf("complete inventory = %v, %t, reads=%d", got, ok, calls)
	}
	want := map[uint64]VulkanBufferBacking{16: wantA, 32: wantB}
	for _, item := range got {
		if backing, exists := want[item.BufferBytes]; !exists || item.Backing != backing {
			t.Fatalf("unexpected or duplicate home record: %+v", item)
		}
		delete(want, item.BufferBytes)
	}
	if len(want) != 0 {
		t.Fatalf("missing home records: %v", want)
	}
	if len(homes) != 2 || homes[keyA] != homeA || homes[keyB] != homeB {
		t.Fatal("inventory mutated cache ownership")
	}

	calls = 0
	failSecond := func(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferBacking, bool) {
		backing, _ := read(ptr, part, chunk)
		return backing, calls != 2
	}
	if got, ok := vulkanQ4KHomeBufferBackingsLocked(homes, 48, failSecond); ok || got != nil || calls != 2 {
		t.Fatalf("missing second backing = %v, %t, reads=%d; want no partial evidence", got, ok, calls)
	}

	cases := []struct {
		name  string
		homes map[vulkanQ4KHomeKey]vulkanQ4KHome
		bytes int64
	}{
		{name: "negative resident bytes", bytes: -1},
		{name: "empty cache with resident bytes", bytes: 1},
		{name: "nil source key", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{{bytes: 16}: homeA}, bytes: 16},
		{name: "zero source length", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{{src: keyA.src}: homeA}, bytes: 16},
		{name: "nil home handle", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: {bytes: 16}}, bytes: 16},
		{name: "mismatched home length", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: {ptr: homeA.ptr, bytes: 32}}, bytes: 32},
		{name: "resident bytes too small", homes: homes, bytes: 47},
		{name: "resident bytes too large", homes: homes, bytes: 49},
		{name: "duplicate owned handle", homes: map[vulkanQ4KHomeKey]vulkanQ4KHome{keyA: homeA, keyB: {ptr: homeA.ptr, bytes: 32}}, bytes: 48},
	}
	for _, tc := range cases {
		if got, ok := vulkanQ4KHomeBufferBackingsLocked(tc.homes, tc.bytes, read); ok || got != nil {
			t.Fatalf("%s: inconsistent cache = %v, %t; want unavailable", tc.name, got, ok)
		}
	}
}
