package relay

import "testing"

// TestInFlightGuardBlocksMidTurnAllowsAtBoundary is rung E2's witness (issue #1881):
// on an otherwise-safe point, GuardRotation must REFUSE the rotation while a tool
// call/decode is in flight (stamping the closed ReasonInFlight token) and PERMIT it
// once the turn has reached its boundary. This is the "never rotate mid-decode"
// assertion the Done condition names — the guard blocks a safe point mid-turn and
// allows it at the boundary.
func TestInFlightGuardBlocksMidTurnAllowsAtBoundary(t *testing.T) {
	// An otherwise-safe point: tree green/parked and next action expressible. The
	// in-flight axis is derived by the guard from the boundary signal, not read here,
	// so NoInFlightTurn is deliberately left zero to prove the guard sets it.
	base := SafePoint{TreeGreenOrParked: true, NextActionExpressible: true}

	// Mid-turn (turnInFlight=true): a tool call may be mid-decode -> refuse.
	if v := GuardRotation(base, true); v.Permit {
		t.Fatalf("mid-turn rotation permitted; want refused")
	} else if v.Reason != ReasonInFlight {
		t.Fatalf("mid-turn refusal reason = %q, want %q", v.Reason, ReasonInFlight)
	}

	// At the boundary (turnInFlight=false): no tool call in flight -> permit.
	if v := GuardRotation(base, false); !v.Permit {
		t.Fatalf("boundary rotation refused (%q); want permitted", v.Reason)
	} else if v.Reason != "" {
		t.Fatalf("permitted verdict carried a reason %q; want empty", v.Reason)
	}
}

// TestInFlightGuardDefersToSafePoint proves rung E2 does not weaken rung E1: at the
// boundary, with no tool call in flight, the guard still refuses a rotation when a
// different SafePoint axis fails — a dirty tree — and stamps ReasonNotAtSafePoint
// rather than silently permitting or mislabelling it as an in-flight refusal.
func TestInFlightGuardDefersToSafePoint(t *testing.T) {
	dirtyTree := SafePoint{TreeGreenOrParked: false, NextActionExpressible: true}
	if v := GuardRotation(dirtyTree, false); v.Permit {
		t.Fatalf("rotation permitted on a dirty tree at the boundary; want refused")
	} else if v.Reason != ReasonNotAtSafePoint {
		t.Fatalf("boundary-but-unsafe refusal reason = %q, want %q", v.Reason, ReasonNotAtSafePoint)
	}

	midThought := SafePoint{TreeGreenOrParked: true, NextActionExpressible: false}
	if v := GuardRotation(midThought, false); v.Permit {
		t.Fatalf("rotation permitted mid-thought at the boundary; want refused")
	} else if v.Reason != ReasonNotAtSafePoint {
		t.Fatalf("boundary-but-unsafe refusal reason = %q, want %q", v.Reason, ReasonNotAtSafePoint)
	}
}
