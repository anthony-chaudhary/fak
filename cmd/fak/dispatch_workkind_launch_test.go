package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchTierLaunchProfileWorkKind is the SHELL-wiring witness for the label-free
// "project management runs on fable by default" rung: it proves the tick-wide work kind
// (dispatchTickOptions.WorkKind) actually threads through dispatchTierLaunchProfile into
// dispatchtick.LaunchProfileForDispatch. The pure decision is unit-tested in the leaf
// (internal/dispatchtick.TestLaunchProfileForDispatch*); this guards that the cmd/fak
// caller passes the work kind (not a dropped arg) and still honors the claude-only +
// FLEET_TIER_LAUNCH gates.
func TestDispatchTierLaunchProfileWorkKind(t *testing.T) {
	t.Setenv("FLEET_TIER_LAUNCH", "1")

	t.Run("unlabelled PM tick -> fable PM bucket", func(t *testing.T) {
		prof, bucket := dispatchTierLaunchProfile("claude", nil, dispatchtick.WorkKindProjectManagement)
		if prof == nil || *prof != dispatchtick.ProfileFableXHigh || bucket != dispatchtick.BucketPM {
			t.Fatalf("got prof=%v bucket=%q, want fable+xhigh / BucketPM", prof, bucket)
		}
	})

	t.Run("engineering tick keeps the seat default", func(t *testing.T) {
		prof, bucket := dispatchTierLaunchProfile("claude", nil, "engineering")
		if prof != nil || bucket != "" {
			t.Fatalf("got prof=%v bucket=%q, want nil (seat default)", prof, bucket)
		}
	})

	t.Run("a valid hard tier label wins over the PM tick", func(t *testing.T) {
		prof, bucket := dispatchTierLaunchProfile("claude",
			[]string{"tier/T0-required", "tier/T0-optimal"}, dispatchtick.WorkKindProjectManagement)
		if prof == nil || *prof != dispatchtick.ProfileOpusUltracode || bucket != dispatchtick.BucketHard {
			t.Fatalf("got prof=%v bucket=%q, want opus+ultracode / BucketHard", prof, bucket)
		}
	})

	t.Run("non-claude backend ignores the uplift", func(t *testing.T) {
		if prof, _ := dispatchTierLaunchProfile("opencode", nil, dispatchtick.WorkKindProjectManagement); prof != nil {
			t.Fatalf("got prof=%v, want nil for a non-claude backend", prof)
		}
	})
}

// TestDispatchTierLaunchProfileKnobOff confirms the FLEET_TIER_LAUNCH gate still fences the
// whole seam: with the knob off, even a coordination work kind leaves the seat default, so a
// default fleet tick is byte-identical to before this rung.
func TestDispatchTierLaunchProfileKnobOff(t *testing.T) {
	t.Setenv("FLEET_TIER_LAUNCH", "")
	if prof, bucket := dispatchTierLaunchProfile("claude", nil, dispatchtick.WorkKindProjectManagement); prof != nil || bucket != "" {
		t.Fatalf("knob off: got prof=%v bucket=%q, want nil (seat default)", prof, bucket)
	}
}
