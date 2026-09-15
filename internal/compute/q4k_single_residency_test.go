package compute

import "testing"

// TestQ4KSingleResidencyRelease pins the pure release predicate: the host backing may be dropped
// ONLY when the operator knob is on AND the device shares physical RAM with the host (an
// "integrated:" APU tier). Every other tier keeps the host copy so a CPU fallback survives, and a
// wrong or empty tier can never release.
func TestQ4KSingleResidencyRelease(t *testing.T) {
	cases := []struct {
		name       string
		tier       string
		envEnabled bool
		want       bool
	}{
		{name: "integrated enabled", tier: "integrated:strix-halo", envEnabled: true, want: true},
		{name: "integrated disabled", tier: "integrated:strix-halo", envEnabled: false, want: false},
		{name: "discrete enabled", tier: "discrete:nvidia-4090", envEnabled: true, want: false},
		{name: "empty tier", tier: "", envEnabled: true, want: false},
		{name: "bare integrated prefix", tier: "integrated:", envEnabled: true, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Q4KSingleResidencyRelease(tc.tier, tc.envEnabled); got != tc.want {
				t.Fatalf("Q4KSingleResidencyRelease(%q, %v) = %v, want %v", tc.tier, tc.envEnabled, got, tc.want)
			}
		})
	}
}
