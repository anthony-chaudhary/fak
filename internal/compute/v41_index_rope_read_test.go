package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexRoPEReadOwnership(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	for _, owned := range []bool{false, true} {
		legacy := &indexReadLegacy{values: []float32{1, 2, 5, 6}}
		var backend Backend = legacy
		if owned {
			backend = &indexReadOwned{indexReadLegacy: *legacy}
		}
		got, ok := ReadV41IndexRoPEOutput(backend, Tensor{}, input, 4, 2)
		var original []float32
		if owned {
			original = backend.(*indexReadOwned).values
		} else {
			original = legacy.values
		}
		if !ok || (&got[0] == &original[0]) != owned {
			t.Fatal("ownership", owned)
		}
	}
	base := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{1, 2, 5, 6}}}
	wrapped := &indexReadWrapper{indexReadOwned: base, replacement: []float32{1, 2, 7, 8}}
	got, ok := ReadV41IndexRoPEOutput(wrapped, Tensor{}, input, 4, 2)
	if !ok || wrapped.overrideCalls != 1 || base.calls != 0 || &got[0] == &wrapped.replacement[0] {
		t.Fatal("wrapper override")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexRoPEReadValidation(t *testing.T) {
	input := []float32{0, 2, 3, 4}
	cases := [][]float32{{0, 2, 3}, {float32(math.Copysign(0, -1)), 2, 3, 4}, {0, 2, float32(math.NaN()), 4}, {0, 2, 3, float32(math.Inf(1))}}
	for _, values := range cases {
		for _, owned := range []bool{false, true} {
			b := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: values}}
			var backend Backend = &b.indexReadLegacy
			if owned {
				backend = b
			}
			if got, ok := ReadV41IndexRoPEOutput(backend, Tensor{}, input, 4, 2); ok || got != nil || b.calls != 1 {
				t.Fatal("invalid result", owned, values)
			}
		}
	}
	b := &indexReadLegacy{values: []float32{0, 2, 3, 4}, failure: new(int)}
	defer func() {
		if recover() != b.failure || b.calls != 1 {
			t.Fatal("panic identity")
		}
	}()
	ReadV41IndexRoPEOutput(b, Tensor{}, input, 4, 2)
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexRoPEReadTailOverflow(t *testing.T) {
	for _, bits := range []uint32{0x7f7f7fff, 0x7f7f8000, 0xff7f7fff, 0xff7f8000} {
		value := math.Float32frombits(bits)
		for _, owned := range []bool{false, true} {
			b := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{1, 2, value, 4}}}
			var backend Backend = &b.indexReadLegacy
			if owned {
				backend = b
			}
			got, ok := ReadV41IndexRoPEOutput(backend, Tensor{}, []float32{1, 2, 3, 4}, 4, 2)
			want := bits&0x7fffffff < 0x7f7f8000
			if ok != want || (!ok && got != nil) || b.calls != 1 {
				t.Fatal("tail threshold", bits, owned, ok)
			}
		}
	}
	// Overflow is checked only in the rotated tail, as in the original callback.
	value := math.Float32frombits(0x7f7f8000)
	b := &indexReadLegacy{values: []float32{value, 2, 3, 4}}
	if _, ok := ReadV41IndexRoPEOutput(b, Tensor{}, b.values, 4, 2); !ok {
		t.Fatal("prefix overflow changed")
	}
}
