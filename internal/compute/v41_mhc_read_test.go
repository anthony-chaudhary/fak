package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestV41MHCCollapseReadOwnership(t *testing.T) {
	for _, owned := range []bool{false, true} {
		b := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{1, 2, 3, 4}}}
		var backend Backend = &b.indexReadLegacy
		if owned {
			backend = b
		}
		got, ok := ReadV41MHCCollapse(backend, Tensor{}, 4)
		if !ok || b.calls != 1 || (&got[0] == &b.values[0]) != owned {
			t.Fatal("ownership", owned)
		}
	}
	base := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{1, 2}}}
	wrapped := &indexReadWrapper{indexReadOwned: base, replacement: []float32{3, 4}}
	got, ok := ReadV41MHCCollapse(wrapped, Tensor{}, 2)
	if !ok || wrapped.overrideCalls != 1 || base.calls != 0 || &got[0] == &wrapped.replacement[0] {
		t.Fatal("wrapper")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41MHCCollapseReadValidation(t *testing.T) {
	for _, values := range [][]float32{{1}, {1, float32(math.NaN())}, {float32(math.Inf(-1)), 2}, {1, float32(math.Inf(1))}} {
		for _, owned := range []bool{false, true} {
			b := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: values}}
			var backend Backend = &b.indexReadLegacy
			if owned {
				backend = b
			}
			if got, ok := ReadV41MHCCollapse(backend, Tensor{}, 2); ok || got != nil || b.calls != 1 {
				t.Fatal("invalid", owned, values)
			}
		}
	}
	b := &indexReadLegacy{failure: new(int)}
	defer func() {
		if recover() != b.failure || b.calls != 1 {
			t.Fatal("transport")
		}
	}()
	ReadV41MHCCollapse(b, Tensor{}, 2)
}
