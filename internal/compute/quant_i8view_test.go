package compute

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestI8ViewMatchesCopyWithoutAllocating(t *testing.T) {
	src := make([]int8, 256)
	for i := range src {
		src[i] = int8(i - 128)
	}
	if got, want := i8View(src), i8AsBytes(src); !bytes.Equal(got, want) {
		t.Fatalf("i8View bytes differ from i8AsBytes")
	}
	view := i8View(src)
	src[7] = 42
	if view[7] != 42 {
		t.Fatalf("i8View does not alias its source")
	}
	if allocs := testing.AllocsPerRun(10, func() { _ = i8View(src) }); allocs != 0 {
		t.Fatalf("i8View allocated %v times per call", allocs)
	}
	if i8View(nil) != nil {
		t.Fatalf("i8View(nil) must be nil")
	}
}
