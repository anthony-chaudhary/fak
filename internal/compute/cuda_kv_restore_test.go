package compute

import (
	"testing"
	"unsafe"
)

// Host callbacks exercise ownership only; no CUDA runtime or hardware executes.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAKVRestoreRowAllocationFailureRetiresOldOwner(t *testing.T) {
	backing := new(float32)
	ptr := unsafe.Pointer(backing)
	old := ptr
	length, capacity := 2, 4
	failure := &DeviceAllocError{Bytes: 32, Site: "synthetic host restore", Class: MemoryKVCache}
	released := map[unsafe.Pointer]int{}
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		restoreCUDAKVAllocation(&ptr, &length, &capacity, 8, func() unsafe.Pointer {
			panic(failure)
		}, func(p unsafe.Pointer) { released[p]++ })
	}()
	// Mirror RestoreKVFromHost's deferred cleanup after the allocation panic.
	if ptr != nil {
		released[ptr]++
	}
	if recovered != failure {
		t.Fatalf("allocation panic identity changed: got %v", recovered)
	}
	if ptr != nil || length != 0 || capacity != 0 || released[old] != 1 {
		t.Fatalf("retired row remains owned: ptr=%p len=%d cap=%d releases=%d", ptr, length, capacity, released[old])
	}
}
