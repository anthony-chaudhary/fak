//go:build !(darwin && arm64 && cgo)

package metalgemm

import (
	"errors"
	"testing"
)

func TestZeroCopyStubRemainsUnavailable(t *testing.T) {
	buffer, err := NewZeroCopyBuffer([]byte{0})
	if buffer != nil || !errors.Is(err, ErrMetalUnavailable) {
		t.Fatalf("NewZeroCopyBuffer() = (%v, %v), want (nil, ErrMetalUnavailable)", buffer, err)
	}

	weight, err := UploadZeroCopyQ4KTensor(make([]byte, 144), 0, 1, 256)
	if weight != nil || !errors.Is(err, ErrMetalUnavailable) {
		t.Fatalf("UploadZeroCopyQ4KTensor() = (%v, %v), want (nil, ErrMetalUnavailable)", weight, err)
	}

	inert := &Q4KWeight{Out: 1, In: 256}
	if inert.IsMetalSharedBuffer() {
		t.Fatal("stub Q4KWeight reported a shared Metal buffer")
	}
}
