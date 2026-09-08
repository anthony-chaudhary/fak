package compute

import (
	"testing"
)

func TestHasVectorizedDeltaNetEnvironment(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{{"", hasDeltaNetSIMD()}, {"1", hasDeltaNetSIMD()}, {"0", false}} {
		t.Setenv("FAK_VECTORIZED_DELTANET", tc.value)
		if got := HasVectorizedDeltaNet(); got != tc.want {
			t.Fatalf("FAK_VECTORIZED_DELTANET=%q: enabled=%v, want %v", tc.value, got, tc.want)
		}
	}
}

func TestHasVectorizedDeltaNet_Availability(t *testing.T) {
	// 1. Default (unset) reflects executable CPU and OS support.
	t.Setenv("FAK_VECTORIZED_DELTANET", "")
	if got, want := HasVectorizedDeltaNet(), hasDeltaNetSIMD(); got != want {
		t.Fatalf("HasVectorizedDeltaNet=%v, hardware/OS support=%v", got, want)
	}

	// 2. Explicitly disabled values
	disabledValues := []string{"0", "false", "FALSE", "False", "no", "NO", "off", "OFF", " 0 ", "  false  "}
	for _, val := range disabledValues {
		t.Setenv("FAK_VECTORIZED_DELTANET", val)
		if HasVectorizedDeltaNet() {
			t.Fatalf("HasVectorizedDeltaNet must be false when FAK_VECTORIZED_DELTANET=%q", val)
		}
	}

	// 3. Explicit enables permit detected support; they cannot synthesize it.
	enabledValues := []string{"1", "true", "TRUE", "yes", "YES", "on", "ON"}
	for _, val := range enabledValues {
		t.Setenv("FAK_VECTORIZED_DELTANET", val)
		if got, want := HasVectorizedDeltaNet(), hasDeltaNetSIMD(); got != want {
			t.Fatalf("FAK_VECTORIZED_DELTANET=%q: enabled=%v, hardware/OS support=%v", val, got, want)
		}
	}
}
