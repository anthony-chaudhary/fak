package compute

import "testing"

func TestHasVectorizedDeltaNetEnvironment(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{{"", true}, {"1", true}, {"0", false}} {
		t.Setenv("FAK_VECTORIZED_DELTANET", tc.value)
		if got := HasVectorizedDeltaNet(); got != tc.want {
			t.Fatalf("FAK_VECTORIZED_DELTANET=%q: enabled=%v, want %v", tc.value, got, tc.want)
		}
	}
}
