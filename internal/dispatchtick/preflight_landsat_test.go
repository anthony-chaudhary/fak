package dispatchtick

import "testing"

// landCleanInput is a preflight input that classifies SPAWN_OK with the land-contention
// term absent, so any non-OK verdict in these tests is attributable to LandContention.
// Kernel Alive/Target are nil, so the settle case never fires (the base stays SPAWN_OK
// independent of the kernel-settling classification).
func landCleanInput() PreflightInput {
	return PreflightInput{
		Workspace:  "repo",
		MaxWorkers: 4,
		Host:       HostCheck{Safe: true},
		Account:    AccountCheck{Available: true, Tag: "worker-a", Tier: 1, Model: "claude"},
		Kernel:     KernelCheck{},
		Seat:       SeatCheck{Total: nil},
		Resources:  HostResources{Cores: IntPtr(64), FreeRAMMB: IntPtr(128000), TotalThreads: IntPtr(1000)},
	}
}

func TestLandCleanInputPrecondition(t *testing.T) {
	if got := EvaluatePreflight(landCleanInput()); got.Verdict != PreflightOKVerdict {
		t.Fatalf("base verdict = %q, want %q so land-contention deltas are attributable", got.Verdict, PreflightOKVerdict)
	}
}

func TestLandContentionRefusesAboveHighWater(t *testing.T) {
	in := landCleanInput()
	in.LandContention = LandContention{Enabled: true, Signal: 0.95, HighWater: 0.8, LowWater: 0.5}
	got := EvaluatePreflight(in)
	if got.Verdict != PreflightRefuseLandSaturated {
		t.Fatalf("verdict = %q, want %q above the high-water mark", got.Verdict, PreflightRefuseLandSaturated)
	}
	if got.OK {
		t.Fatalf("OK = true, want a refusal above the high-water mark")
	}
}

func TestLandContentionAdmitsBelowLowWater(t *testing.T) {
	in := landCleanInput()
	// Below the low-water mark clears even a prior refusal (latch OFF).
	in.LandContention = LandContention{Enabled: true, Signal: 0.2, HighWater: 0.8, LowWater: 0.5, Prior: PreflightRefuseLandSaturated}
	got := EvaluatePreflight(in)
	if got.Verdict != PreflightOKVerdict {
		t.Fatalf("verdict = %q, want %q below the low-water mark", got.Verdict, PreflightOKVerdict)
	}
}

func TestLandContentionHoldsPriorRefusalInBand(t *testing.T) {
	in := landCleanInput()
	in.LandContention = LandContention{Enabled: true, Signal: 0.65, HighWater: 0.8, LowWater: 0.5, Prior: PreflightRefuseLandSaturated}
	got := EvaluatePreflight(in)
	if got.Verdict != PreflightRefuseLandSaturated {
		t.Fatalf("verdict = %q, want held prior refusal %q inside the hysteresis band", got.Verdict, PreflightRefuseLandSaturated)
	}
}

func TestLandContentionHoldsPriorAdmitInBand(t *testing.T) {
	in := landCleanInput()
	in.LandContention = LandContention{Enabled: true, Signal: 0.65, HighWater: 0.8, LowWater: 0.5, Prior: ""}
	got := EvaluatePreflight(in)
	if got.Verdict != PreflightOKVerdict {
		t.Fatalf("verdict = %q, want held prior admit %q inside the hysteresis band", got.Verdict, PreflightOKVerdict)
	}
}

func TestLandContentionDisabledIsByteIdentical(t *testing.T) {
	// Knob OFF: even a signal far above the high-water mark (and a prior refusal) must
	// leave the verdict, reason, and OK bit exactly as the base with no term at all.
	base := EvaluatePreflight(landCleanInput())
	in := landCleanInput()
	in.LandContention = LandContention{Enabled: false, Signal: 9.9, HighWater: 0.8, LowWater: 0.5, Prior: PreflightRefuseLandSaturated}
	got := EvaluatePreflight(in)
	if got.Verdict != base.Verdict || got.Reason != base.Reason || got.OK != base.OK {
		t.Fatalf("disabled result = {%q %q %v}, want byte-identical base {%q %q %v}",
			got.Verdict, got.Reason, got.OK, base.Verdict, base.Reason, base.OK)
	}
	if base.Verdict != PreflightOKVerdict {
		t.Fatalf("base verdict = %q, want SPAWN_OK precondition", base.Verdict)
	}
}

func TestLandSaturatedTrigger(t *testing.T) {
	cases := []struct {
		name string
		c    LandContention
		want bool
	}{
		{"disabled ignores a high signal", LandContention{Enabled: false, Signal: 5, HighWater: 1, LowWater: 0.5}, false},
		{"above high latches on", LandContention{Enabled: true, Signal: 0.9, HighWater: 0.8, LowWater: 0.5}, true},
		{"at high latches on", LandContention{Enabled: true, Signal: 0.8, HighWater: 0.8, LowWater: 0.5}, true},
		{"below low latches off", LandContention{Enabled: true, Signal: 0.4, HighWater: 0.8, LowWater: 0.5}, false},
		{"band holds prior refuse", LandContention{Enabled: true, Signal: 0.6, HighWater: 0.8, LowWater: 0.5, Prior: PreflightRefuseLandSaturated}, true},
		{"band holds prior admit", LandContention{Enabled: true, Signal: 0.6, HighWater: 0.8, LowWater: 0.5, Prior: ""}, false},
		{"at low is in band, holds prior refuse", LandContention{Enabled: true, Signal: 0.5, HighWater: 0.8, LowWater: 0.5, Prior: PreflightRefuseLandSaturated}, true},
	}
	for _, tc := range cases {
		if got := landSaturated(tc.c); got != tc.want {
			t.Fatalf("%s: landSaturated = %v, want %v", tc.name, got, tc.want)
		}
	}
}
