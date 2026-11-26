//go:build darwin && arm64 && cgo

package metalgemm

import (
	"syscall"
	"testing"
)

// fak-test: runtime=2s requires=metal
func TestZeroCopyFreedBuffersReturnDeviceMemoryAfterContents(t *testing.T) {
	if !Available() {
		t.Skip("requires a Metal device")
	}
	const size, rounds = 64 << 20, 8
	span, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(span)
	before := gdnCurrentAllocatedBytes()
	for range rounds {
		b, err := NewZeroCopyBuffer(span)
		if err != nil {
			t.Fatal(err)
		}
		if b.Contents() == nil {
			t.Fatal("no-copy buffer has no contents")
		}
		b.Free()
	}
	if grew := int64(gdnCurrentAllocatedBytes()) - int64(before); grew > size {
		t.Fatalf("%d MiB stayed allocated after %d rounds of a 64 MiB no-copy buffer wrapped, read and freed", grew>>20, rounds)
	}
}
