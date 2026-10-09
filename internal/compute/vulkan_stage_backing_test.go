//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=10ms lane=default
// Synthetic handles stay in the Go reader. This checks the ownership inventory
// contract, not C pointer validity, allocation selection or device execution.
func TestVulkanQ4KStageBufferBackingsAllOrUnknown(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanQ4KStageBufferBackings(); ok || got != nil {
		t.Fatalf("nil backend = %v, %t; want unavailable", got, ok)
	}
	if got, ok := (&vulkanBackend{}).VulkanQ4KStageBufferBackings(); !ok || len(got) != 0 {
		t.Fatalf("empty owner = %v, %t; want known empty", got, ok)
	}

	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var handle byte
	ptr := unsafe.Pointer(&handle)
	want := VulkanBufferBacking{
		Part: "data", Chunk: -1, RequestedPropertyFlags: 1,
		MemoryTypeIndex: 2, PropertyFlags: 7, HeapIndex: 1, HeapFlags: 1,
		HostVisibleFallback: true, WeightArenaBound: true,
	}
	calls := 0
	read := func(owned unsafe.Pointer, part string, chunk int) (VulkanBufferBacking, bool) {
		calls++
		if owned != ptr || part != "data" || chunk != -1 {
			t.Fatalf("reader received unexpected handle, part or chunk: %p, %q, %d", owned, part, chunk)
		}
		return want, true
	}
	got, ok := vulkanQ4KStageBufferBackingsLocked(ptr, 64, read)
	if !ok || len(got) != 1 || calls != 1 || got[0].BufferBytes != 64 || got[0].Backing != want {
		t.Fatalf("owned stage = %v, %t, reads=%d; want complete provenance", got, ok, calls)
	}
	// Results carry copied evidence, not the owner record or a persistent handle.
	got[0].BufferBytes = 0
	got[0].Backing = VulkanBufferBacking{}
	if again, ok := vulkanQ4KStageBufferBackingsLocked(ptr, 64, read); !ok || len(again) != 1 || again[0].BufferBytes != 64 || again[0].Backing != want {
		t.Fatalf("second observation = %v, %t; prior result mutation changed evidence", again, ok)
	}

	failRead := func(owned unsafe.Pointer, part string, chunk int) (VulkanBufferBacking, bool) {
		backing, _ := read(owned, part, chunk)
		return backing, false
	}
	if got, ok := vulkanQ4KStageBufferBackingsLocked(ptr, 64, failRead); ok || got != nil {
		t.Fatalf("missing backing = %v, %t; want unavailable without a record", got, ok)
	}

	calls = 0
	if got, ok := vulkanQ4KStageBufferBackingsLocked(nil, 0, read); !ok || len(got) != 0 || calls != 0 {
		t.Fatalf("retired stage = %v, %t, reads=%d; want known empty without reading a handle", got, ok, calls)
	}
	for _, tc := range []struct {
		name  string
		ptr   unsafe.Pointer
		bytes int64
	}{
		{name: "missing handle", bytes: 64},
		{name: "negative empty length", bytes: -1},
		{name: "zero live length", ptr: ptr},
		{name: "negative live length", ptr: ptr, bytes: -1},
	} {
		if got, ok := vulkanQ4KStageBufferBackingsLocked(tc.ptr, tc.bytes, read); ok || got != nil || calls != 0 {
			t.Fatalf("%s = %v, %t, reads=%d; want unavailable before reading a handle", tc.name, got, ok, calls)
		}
	}
}
