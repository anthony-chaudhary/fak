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

// TestQ4KSingleResidencyReleaseMode pins the FAK_Q4K_FREE_CPU arms: "1" keeps the integrated-only
// contract, "device" extends the release to a discrete device-memory tier, and nothing else arms.
//
// fak-test:runtime fast est=10ms
func TestQ4KSingleResidencyReleaseMode(t *testing.T) {
	cases := []struct {
		name         string
		env          string
		tier         string
		deviceMemory bool
		want         bool
	}{
		{name: "off discrete", env: "", tier: "sm_120", deviceMemory: true, want: false},
		{name: "one integrated", env: "1", tier: "integrated:strix-halo", deviceMemory: true, want: true},
		{name: "one discrete keeps", env: "1", tier: "sm_120", deviceMemory: true, want: false},
		{name: "device cuda", env: "device", tier: "sm_120", deviceMemory: true, want: true},
		{name: "device vulkan discrete", env: " Device ", tier: "discrete:rtx-5090", deviceMemory: true, want: true},
		{name: "device integrated", env: "device", tier: "integrated:strix-halo", deviceMemory: true, want: true},
		{name: "device host-memory backend", env: "device", tier: "scalar", deviceMemory: false, want: false},
		{name: "device empty tier", env: "device", tier: "", deviceMemory: true, want: false},
		{name: "unknown value", env: "yes", tier: "sm_120", deviceMemory: true, want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Q4KSingleResidencyReleaseMode(tc.tier, tc.deviceMemory, ParseQ4KSingleResidencyMode(tc.env)); got != tc.want {
				t.Fatalf("release(env=%q tier=%q dev=%v) = %v, want %v", tc.env, tc.tier, tc.deviceMemory, got, tc.want)
			}
		})
	}
}
