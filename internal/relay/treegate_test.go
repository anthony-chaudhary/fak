package relay

import "testing"

// TestTreeGateDirtyUnparkedFailsGreenOrParkedPasses is rung E3's witness (issue #1882) and
// states the Done condition literally: a dirty NON-PARKED tree fails the check, and a green
// tree or an explicitly parked one passes. The SafePoint handed in leaves TreeGreenOrParked
// at its zero value on purpose — the gate must DERIVE that axis from the tree evidence, so a
// permit here proves the derivation rather than a trusted field.
func TestTreeGateDirtyUnparkedFailsGreenOrParkedPasses(t *testing.T) {
	// Otherwise-safe: no tool call in flight, next action expressible. Only the tree axis
	// is in question, and it is left zero so the gate has to set it.
	base := SafePoint{NoInFlightTurn: true, NextActionExpressible: true}

	// A dirty, non-parked tree: a half-written commit sits at the candidate safe point.
	dirty := TreeStatus{DirtyPaths: []string{"internal/relay/codec.go"}}
	if v := TreeGate(base, dirty); v.Permit {
		t.Fatalf("rotation permitted on a dirty non-parked tree; want refused")
	} else if v.Reason != ReasonTreeDirty {
		t.Fatalf("dirty-tree refusal reason = %q, want %q", v.Reason, ReasonTreeDirty)
	}

	// A green tree: nothing half-written at all.
	green := TreeStatus{}
	if v := TreeGate(base, green); !v.Permit {
		t.Fatalf("rotation refused on a green tree (%q); want permitted", v.Reason)
	} else if v.Reason != "" {
		t.Fatalf("permitted verdict carried a reason %q; want empty", v.Reason)
	}

	// An explicitly parked tree: dirty, but every dirty path parked at a committable
	// boundary. The spine's disjunction ("green OR explicitly parked") must pass here.
	parked := TreeStatus{
		DirtyPaths:  []string{"internal/relay/codec.go", "internal/relay/baton.go"},
		ParkedPaths: []string{"internal/relay/codec.go", "internal/relay/baton.go"},
	}
	if v := TreeGate(base, parked); !v.Permit {
		t.Fatalf("rotation refused on an explicitly parked tree (%q); want permitted", v.Reason)
	}

	// Partially parked is NOT parked: one unparked dirty path is still a half-written
	// commit, so the disjunction fails as a whole rather than per-path.
	partial := TreeStatus{
		DirtyPaths:  []string{"internal/relay/codec.go", "internal/relay/baton.go"},
		ParkedPaths: []string{"internal/relay/codec.go"},
	}
	if v := TreeGate(base, partial); v.Permit {
		t.Fatalf("rotation permitted with one unparked dirty path; want refused")
	} else if v.Reason != ReasonTreeDirty {
		t.Fatalf("partially-parked refusal reason = %q, want %q", v.Reason, ReasonTreeDirty)
	}

	// A parked path that is not dirty is inert — it must not fabricate a dirty tree.
	inert := TreeStatus{ParkedPaths: []string{"internal/relay/codec.go"}}
	if v := TreeGate(base, inert); !v.Permit {
		t.Fatalf("rotation refused on a green tree with an inert park (%q); want permitted", v.Reason)
	}
}

// TestTreeGateDerivesAxisRatherThanTrustingCaller pins the rung's actual content: the caller's
// TreeGreenOrParked field is IGNORED in both directions. A caller asserting a green tree
// cannot talk the gate past a half-written commit, and a caller asserting a dirty tree cannot
// block a rotation the evidence permits.
func TestTreeGateDerivesAxisRatherThanTrustingCaller(t *testing.T) {
	// Caller LIES green; the evidence says a dirty unparked path. Evidence wins.
	lyingGreen := SafePoint{NoInFlightTurn: true, TreeGreenOrParked: true, NextActionExpressible: true}
	dirty := TreeStatus{DirtyPaths: []string{"internal/relay/codec.go"}}
	if v := TreeGate(lyingGreen, dirty); v.Permit {
		t.Fatalf("caller's TreeGreenOrParked=true overrode dirty evidence; want refused")
	} else if v.Reason != ReasonTreeDirty {
		t.Fatalf("refusal reason = %q, want %q", v.Reason, ReasonTreeDirty)
	}

	// Caller says dirty; the evidence says green. Evidence wins again.
	staleDirty := SafePoint{NoInFlightTurn: true, TreeGreenOrParked: false, NextActionExpressible: true}
	if v := TreeGate(staleDirty, TreeStatus{}); !v.Permit {
		t.Fatalf("stale TreeGreenOrParked=false blocked a green tree (%q); want permitted", v.Reason)
	}
}

// TestTreeGateDefersToSafePoint proves rung E3 does not weaken rung E1: with a green tree, the
// gate still refuses a rotation when a DIFFERENT SafePoint axis fails — a tool call in flight
// or a mid-thought next action — and stamps ReasonNotAtSafePoint rather than mislabelling the
// refusal as a dirty tree or silently permitting it.
func TestTreeGateDefersToSafePoint(t *testing.T) {
	green := TreeStatus{}

	inFlight := SafePoint{NoInFlightTurn: false, NextActionExpressible: true}
	if v := TreeGate(inFlight, green); v.Permit {
		t.Fatalf("rotation permitted mid-turn on a green tree; want refused")
	} else if v.Reason != ReasonNotAtSafePoint {
		t.Fatalf("in-flight refusal reason = %q, want %q", v.Reason, ReasonNotAtSafePoint)
	}

	midThought := SafePoint{NoInFlightTurn: true, NextActionExpressible: false}
	if v := TreeGate(midThought, green); v.Permit {
		t.Fatalf("rotation permitted mid-thought on a green tree; want refused")
	} else if v.Reason != ReasonNotAtSafePoint {
		t.Fatalf("mid-thought refusal reason = %q, want %q", v.Reason, ReasonNotAtSafePoint)
	}
}
