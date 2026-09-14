package compute

import "testing"

// TestROCmFP4DtypeAppendedWithoutShiftingRegistry is the witness for issue #13003's
// FP4 Dtype registration in the compute HAL. It proves FP4 is a first-class Dtype
// with the expected width/tag/quantized semantics, and that appending it to the
// enum did not shift any pre-existing dtype constant value.
//
// NOTE: the function is named ...AppendedWithoutShiftingRegistry rather than the
// issue's suggested ...RegisteredInComputeRegistry because a peer lane already
// declares TestROCmFP4DtypeRegisteredInComputeRegistry in fp4_dtype_test.go within
// this same package; two identical test funcs cannot compile. This file carries the
// stronger no-shift assertion the issue asked for.
func TestROCmFP4DtypeAppendedWithoutShiftingRegistry(t *testing.T) {
	// 1. FP4 is a defined, one-byte-per-code dtype.
	if got := FP4.Bytes(); got != 1 {
		t.Errorf("FP4.Bytes() = %d, want 1", got)
	}

	// 2. The lowercase tag is "fp4".
	if got := FP4.String(); got != "fp4" {
		t.Errorf("FP4.String() = %q, want %q", got, "fp4")
	}

	// 3. FP4 needs a QuantSpec to be interpreted.
	if !FP4.Quantized() {
		t.Error("FP4.Quantized() = false, want true")
	}

	// 4. FP4 is distinct and appended last, so prior numeric values are unchanged.
	if FP4 != IQ2_XXS+1 {
		t.Errorf("FP4 = %d, want IQ2_XXS+1 = %d (appended, not inserted)", FP4, IQ2_XXS+1)
	}
	distinct := []Dtype{F32, F16, BF16, Q8_0, I8, I4, FP8, Q4_K, Q5_K, Q6_K, Q2_0, Q2_K, IQ3_XXS, IQ3_S, IQ2_XXS}
	for _, d := range distinct {
		if d == FP4 {
			t.Errorf("FP4 collides with pre-existing dtype %s (%d)", d, d)
		}
	}
	if F32 != 0 {
		t.Errorf("F32 = %d, want 0 (prior constants must not shift)", F32)
	}
}
