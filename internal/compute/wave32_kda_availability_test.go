package compute

import (
	"testing"
)

func TestHasVectorizedDeltaNet_Availability(t *testing.T) {
	// 1. Default (unset) should be true
	t.Setenv("FAK_VECTORIZED_DELTANET", "")
	if !HasVectorizedDeltaNet() {
		t.Fatal("HasVectorizedDeltaNet must be true when FAK_VECTORIZED_DELTANET is empty/unset")
	}

	// 2. Explicitly disabled values
	disabledValues := []string{"0", "false", "FALSE", "False", "no", "NO", "off", "OFF", " 0 ", "  false  "}
	for _, val := range disabledValues {
		t.Setenv("FAK_VECTORIZED_DELTANET", val)
		if HasVectorizedDeltaNet() {
			t.Fatalf("HasVectorizedDeltaNet must be false when FAK_VECTORIZED_DELTANET=%q", val)
		}
	}

	// 3. Explicitly enabled values
	enabledValues := []string{"1", "true", "TRUE", "yes", "YES", "on", "ON"}
	for _, val := range enabledValues {
		t.Setenv("FAK_VECTORIZED_DELTANET", val)
		if !HasVectorizedDeltaNet() {
			t.Fatalf("HasVectorizedDeltaNet must be true when FAK_VECTORIZED_DELTANET=%q", val)
		}
	}
}
