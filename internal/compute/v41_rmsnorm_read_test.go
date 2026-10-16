package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestV41RMSNormReadOwnershipAndDeferredFinite(t *testing.T) {
	for _, owned := range []bool{false, true} {
		b := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{1, float32(math.NaN())}}}
		var backend Backend = &b.indexReadLegacy
		if owned {
			backend = b
		}
		got, ok := ReadV41RMSNormOutput(backend, Tensor{}, 2)
		if !ok || b.calls != 1 || (&got[0] == &b.values[0]) != owned || !math.IsNaN(float64(got[1])) {
			t.Fatal("ownership or moved finite check", owned)
		}
	}
	base := &indexReadOwned{indexReadLegacy: indexReadLegacy{values: []float32{1, 2}}}
	wrapped := &indexReadWrapper{indexReadOwned: base, replacement: []float32{3, 4}}
	got, ok := ReadV41RMSNormOutput(wrapped, Tensor{}, 2)
	if !ok || wrapped.overrideCalls != 1 || base.calls != 0 || &got[0] == &wrapped.replacement[0] {
		t.Fatal("wrapper")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41RMSNormReadShapeAndTransport(t *testing.T) {
	b := &indexReadLegacy{values: []float32{1}}
	if got, ok := ReadV41RMSNormOutput(b, Tensor{}, 2); ok || got != nil || b.calls != 1 {
		t.Fatal("shape")
	}
	b.failure = new(int)
	defer func() {
		if recover() != b.failure || b.calls != 2 {
			t.Fatal("transport")
		}
	}()
	ReadV41RMSNormOutput(b, Tensor{}, 1)
}
