package relay

import "testing"

// TestRotationCapHoldsTheNPlusFirstWithinTheHour is the done-condition for issue
// #1892: with a cap of N, the (N+1)th proposed rotation inside the same wall-clock
// hour is held with RELAY_ROTATION_CAPPED instead of allowed.
func TestRotationCapHoldsTheNPlusFirstWithinTheHour(t *testing.T) {
	c := RotationCap{MaxPerHour: 3}
	for i, now := range []int64{0, 600, 1200} {
		allow, reason := c.Admit(now)
		if !allow || reason != "" {
			t.Fatalf("Admit #%d at t=%d = (%v, %q), want (true, \"\")", i+1, now, allow, reason)
		}
	}
	allow, reason := c.Admit(1800)
	if allow {
		t.Fatalf("4th Admit within the hour = allow=true, want held")
	}
	if reason != ReasonRotationCapped {
		t.Fatalf("4th Admit reason = %q, want %q", reason, ReasonRotationCapped)
	}
}

// TestRotationCapAllowsAfterWindowSlides fills the cap, then admits at a time more
// than an hour after the OLDEST accepted rotation: the oldest ages out, the new
// rotation is allowed, and CountInWindow reflects the prune.
func TestRotationCapAllowsAfterWindowSlides(t *testing.T) {
	c := RotationCap{MaxPerHour: 2}
	for _, now := range []int64{100, 200} {
		if allow, reason := c.Admit(now); !allow || reason != "" {
			t.Fatalf("fill Admit at t=%d = (%v, %q), want (true, \"\")", now, allow, reason)
		}
	}
	// t=3701 is >3600s after the oldest (t=100): 100 ages out, 200 stays.
	allow, reason := c.Admit(3701)
	if !allow || reason != "" {
		t.Fatalf("Admit after window slide = (%v, %q), want (true, \"\")", allow, reason)
	}
	// 200 (still inside the hour) + the newly accepted 3701.
	if got := c.CountInWindow(); got != 2 {
		t.Fatalf("CountInWindow after prune+accept = %d, want 2", got)
	}
}

// TestRotationCapDisabledNeverHolds proves the zero-value / MaxPerHour<=0 cap is a
// pass-through: an unset policy never holds.
func TestRotationCapDisabledNeverHolds(t *testing.T) {
	c := RotationCap{MaxPerHour: 0}
	for i := int64(0); i < 50; i++ {
		allow, reason := c.Admit(i * 10)
		if !allow || reason != "" {
			t.Fatalf("disabled Admit #%d = (%v, %q), want (true, \"\")", i+1, allow, reason)
		}
	}
	if got := c.CountInWindow(); got != 0 {
		t.Fatalf("disabled cap recorded %d rotations, want 0 (pass-through never records)", got)
	}
}

// TestRotationCapHeldRotationIsNotRecorded proves a held rotation consumes no slot:
// once at the ceiling, held Admits do not grow CountInWindow.
func TestRotationCapHeldRotationIsNotRecorded(t *testing.T) {
	c := RotationCap{MaxPerHour: 2}
	c.Admit(0)
	c.Admit(10)
	if got := c.CountInWindow(); got != 2 {
		t.Fatalf("CountInWindow at ceiling = %d, want 2", got)
	}
	for _, now := range []int64{20, 30, 40} {
		if allow, _ := c.Admit(now); allow {
			t.Fatalf("Admit at t=%d over the ceiling = allow=true, want held", now)
		}
		if got := c.CountInWindow(); got != 2 {
			t.Fatalf("CountInWindow after held Admit at t=%d = %d, want 2 (held rotation must not be recorded)", now, got)
		}
	}
}
