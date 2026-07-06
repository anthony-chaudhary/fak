package relay

import "testing"

// safe is the safe-point verdict (all three sub-conditions hold); unsafe is any
// non-safe point. The state machine only ever consults SafePoint.IsSafe(), so these
// two values exercise both sides of the fire guard.
var (
	safe   = SafePoint{NoInFlightTurn: true, TreeGreenOrParked: true, NextActionExpressible: true}
	unsafe = SafePoint{NoInFlightTurn: false, TreeGreenOrParked: true, NextActionExpressible: true}
)

// TestArmFireDrivesSoftMarkToArmedThenFires is the #1889 done-condition witness: drive
// soft-mark -> armed, hold across unsafe boundaries, and assert the fire lands only at
// a safe point.
func TestArmFireDrivesSoftMarkToArmedThenFires(t *testing.T) {
	var af ArmFire

	// Zero value is a ready, disarmed controller.
	if got := af.State(); got != RotationDisarmed {
		t.Fatalf("zero value State() = %q, want %q", got, RotationDisarmed)
	}

	// Below the soft mark: keep working, never arm — even at a safe point.
	if got := af.Step(false, safe); got != RotationDisarmed {
		t.Fatalf("Step(soft=false, safe) = %q, want %q (a safe point alone must not arm)", got, RotationDisarmed)
	}

	// Soft mark crosses while NOT at a safe point: arm, but do not fire.
	if got := af.Step(true, unsafe); got != RotationArmed {
		t.Fatalf("Step(soft=true, unsafe) = %q, want %q", got, RotationArmed)
	}

	// Armed and still unsafe across several boundaries: hold. Arming is sticky — even a
	// soft mark reading false again must not un-arm.
	for i := 0; i < 3; i++ {
		if got := af.Step(false, unsafe); got != RotationArmed {
			t.Fatalf("hold #%d Step(soft=false, unsafe) = %q, want %q (armed must be sticky)", i, got, RotationArmed)
		}
	}

	// The next safe point after arming fires the rotation.
	if got := af.Step(false, safe); got != RotationFired {
		t.Fatalf("Step(_, safe) while armed = %q, want %q", got, RotationFired)
	}
}

// TestArmFireNeverFiresAtUnsafePoint is the invariant the whole two-phase design
// exists to guarantee: across every reachable (softMark, safe) sequence, the machine
// only ever transitions INTO RotationFired on a boundary where the SafePoint is safe.
func TestArmFireNeverFiresAtUnsafePoint(t *testing.T) {
	steps := []struct {
		soft bool
		sp   SafePoint
	}{
		{false, unsafe}, {true, unsafe}, {false, unsafe}, {true, unsafe},
		{false, unsafe}, {false, unsafe}, {true, safe}, {false, unsafe},
	}
	var af ArmFire
	prev := af.State()
	for i, s := range steps {
		got := af.Step(s.soft, s.sp)
		if got == RotationFired && prev != RotationFired && !s.sp.IsSafe() {
			t.Fatalf("step %d fired at an UNSAFE point (soft=%v): mid-action rotation must be impossible", i, s.soft)
		}
		prev = got
	}
}

// TestArmFireArmsAndFiresSameBoundary covers the "next safe point is now" case: if the
// soft mark crosses while the leg is already at a safe point, arming and firing collapse
// into one boundary — and the fire still happens only because the point is safe.
func TestArmFireArmsAndFiresSameBoundary(t *testing.T) {
	var af ArmFire
	if got := af.Step(true, safe); got != RotationFired {
		t.Fatalf("Step(soft=true, safe) from disarmed = %q, want %q", got, RotationFired)
	}
}

// TestArmFireIdempotentAfterFire proves a fired leg is terminal — it never re-rotates,
// whatever later boundaries report. This underwrites idempotent restart (#1897): a done
// relay stays done.
func TestArmFireIdempotentAfterFire(t *testing.T) {
	var af ArmFire
	af.Step(true, safe) // -> fired
	for _, s := range []struct {
		soft bool
		sp   SafePoint
	}{{true, safe}, {false, unsafe}, {true, unsafe}} {
		if got := af.Step(s.soft, s.sp); got != RotationFired {
			t.Fatalf("Step(%v,%v) after fire = %q, want %q (fired is terminal)", s.soft, s.sp.IsSafe(), got, RotationFired)
		}
	}
	if !af.Fired() || af.Armed() {
		t.Fatalf("after fire: Fired()=%v Armed()=%v, want true/false", af.Fired(), af.Armed())
	}
}

// TestArmFireReasonTokens ties each phase to its closed reason token so a future floor
// reads RELAY_ARMED / RELAY_ROTATED, never prose.
func TestArmFireReasonTokens(t *testing.T) {
	var af ArmFire
	if got := af.Reason(); got != "" {
		t.Fatalf("disarmed Reason() = %q, want empty", got)
	}
	af.Step(true, unsafe) // -> armed
	if got := af.Reason(); got != "RELAY_ARMED" {
		t.Fatalf("armed Reason() = %q, want RELAY_ARMED", got)
	}
	af.Step(false, safe) // -> fired
	if got := af.Reason(); got != "RELAY_ROTATED" {
		t.Fatalf("fired Reason() = %q, want RELAY_ROTATED", got)
	}
}
