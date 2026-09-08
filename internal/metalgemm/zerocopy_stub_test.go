//go:build !(darwin && arm64 && cgo)

package metalgemm

import (
	"errors"
	"testing"
)

func TestZeroCopyStubReturnsMetalUnavailable(t *testing.T) {
	buffer, err := NewZeroCopyBuffer(make([]byte, 1))
	if buffer != nil {
		t.Fatalf("NewZeroCopyBuffer() buffer = %v, want nil", buffer)
	}
	if !errors.Is(err, ErrMetalUnavailable) {
		t.Fatalf("NewZeroCopyBuffer() error = %v, want %v", err, ErrMetalUnavailable)
	}

	weight, err := UploadZeroCopyQ4KTensor(make([]byte, 144), 0, 1, 256)
	if weight != nil {
		t.Fatalf("UploadZeroCopyQ4KTensor() weight = %v, want nil", weight)
	}
	if !errors.Is(err, ErrMetalUnavailable) {
		t.Fatalf("UploadZeroCopyQ4KTensor() error = %v, want %v", err, ErrMetalUnavailable)
	}
}

func TestZeroCopyStubHandlesRemainInert(t *testing.T) {
	buffer := &ZeroCopyBuffer{}
	if buffer.IsShared() {
		t.Fatal("ZeroCopyBuffer.IsShared() = true, want false")
	}
	if got := buffer.Length(); got != 0 {
		t.Fatalf("ZeroCopyBuffer.Length() = %d, want 0", got)
	}
	if got := buffer.Contents(); got != nil {
		t.Fatalf("ZeroCopyBuffer.Contents() = %v, want nil", got)
	}
	buffer.Free()

	weight := &Q4KWeight{Out: 1, In: 256}
	if weight.IsMetalSharedBuffer() {
		t.Fatal("Q4KWeight.IsMetalSharedBuffer() = true, want false")
	}
	var nilWeight *Q4KWeight
	if nilWeight.IsMetalSharedBuffer() {
		t.Fatal("nil Q4KWeight.IsMetalSharedBuffer() = true, want false")
	}
}
