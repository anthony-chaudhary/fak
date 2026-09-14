package compute

import "testing"

// TestROCmFP4DtypeRegisteredInComputeRegistry witnesses that FP4 is a first-class,
// dispatchable compute Dtype: it has a storage width, a stable lowercase tag, is
// treated as quantized (needs a QuantSpec), and sits in the registry at a distinct
// ordinal from the F32 zero value.
func TestROCmFP4DtypeRegisteredInComputeRegistry(t *testing.T) {
	if got := FP4.Bytes(); got != 1 {
		t.Errorf("FP4.Bytes() = %d, want 1 (sub-byte formats round up)", got)
	}
	if got := FP4.String(); got != "fp4" {
		t.Errorf("FP4.String() = %q, want %q", got, "fp4")
	}
	if !FP4.Quantized() {
		t.Error("FP4.Quantized() = false, want true")
	}
	if FP4 == F32 {
		t.Error("FP4 must be distinct from F32")
	}
	if got := Dtype(0); got != F32 {
		t.Errorf("Dtype(0) = %s, want F32 (registry ordinal sanity)", got)
	}
}
