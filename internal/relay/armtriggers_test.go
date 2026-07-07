package relay

import "testing"

// TestArmTriggersEachAxisIndependentlyArms is the #1890 done-condition witness: assert
// that EACH Envelope axis independently arms rotation at its soft mark. For every axis
// in turn we drive only that axis past the mark (all others left unbounded) and require
// the fold to arm, naming that axis — and drive it just below the mark and require it
// not to arm. This is the "each axis independently arms at its soft mark" assertion the
// issue's Done condition names, done as a real per-axis equality check.
func TestArmTriggersEachAxisIndependentlyArms(t *testing.T) {
	const softMark = 0.6
	tr := ArmTriggers{SoftMark: softMark}
	unbounded := AxisUsage{} // cap 0 => never arms

	// Each entry places the axis-under-test in its slot; the helper builds the four
	// args with the other three unbounded, so a positive arm can ONLY come from this
	// axis — that is what makes the assertion "independent".
	cases := []struct {
		name string
		axis Axis
		// place puts `u` in this axis's argument slot and unbounded in the rest.
		place func(u AxisUsage) (context, turns, wall, spend AxisUsage)
	}{
		{"context", AxisContext, func(u AxisUsage) (AxisUsage, AxisUsage, AxisUsage, AxisUsage) {
			return u, unbounded, unbounded, unbounded
		}},
		{"turns", AxisTurns, func(u AxisUsage) (AxisUsage, AxisUsage, AxisUsage, AxisUsage) {
			return unbounded, u, unbounded, unbounded
		}},
		{"wall", AxisWall, func(u AxisUsage) (AxisUsage, AxisUsage, AxisUsage, AxisUsage) {
			return unbounded, unbounded, u, unbounded
		}},
		{"spend", AxisSpend, func(u AxisUsage) (AxisUsage, AxisUsage, AxisUsage, AxisUsage) {
			return unbounded, unbounded, unbounded, u
		}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// At the mark exactly (600/1000 == 0.6): arms, and names THIS axis.
			at := AxisUsage{Used: 600, Cap: 1000}
			gotAxis, armed := tr.Cross(c.place(at))
			if !armed {
				t.Fatalf("axis %s at the soft mark (0.6) did not arm; want armed", c.name)
			}
			if gotAxis != c.axis {
				t.Fatalf("axis %s armed but reported %q; want %q", c.name, gotAxis, c.axis)
			}
			if !tr.Crossed(c.place(at)) {
				t.Fatalf("axis %s: Crossed() disagreed with Cross()", c.name)
			}

			// Just below the mark (599/1000 < 0.6): must NOT arm — the axis is
			// independent, so nothing else can arm it either.
			below := AxisUsage{Used: 599, Cap: 1000}
			if gotAxis, armed := tr.Cross(c.place(below)); armed {
				t.Fatalf("axis %s below the mark armed on %q; want disarmed", c.name, gotAxis)
			}
		})
	}
}

// TestArmTriggersNonBinaryMarkBoundary guards the float-rounding hazard at the soft
// mark: for a mark whose product with the cap is not binary-exact (0.55 * 100), a run
// AT exactly the mark must arm. The algebraically-equal multiply form (used >=
// mark*cap) rounds 0.55*100 up to 55.0000000000000071 and would REFUSE to arm at
// 55/100 — the boundary must track the stated decimal fraction, so the division form is
// the contract this pins. 54/100 (< 0.55) must stay disarmed on both forms.
func TestArmTriggersNonBinaryMarkBoundary(t *testing.T) {
	tr := ArmTriggers{SoftMark: 0.55}
	atMark := AxisUsage{Used: 55, Cap: 100} // exactly 0.55
	if ax, armed := tr.Cross(atMark, AxisUsage{}, AxisUsage{}, AxisUsage{}); !armed || ax != AxisContext {
		t.Fatalf("55/100 at the 0.55 mark => (%q,%v); want (%q,true) — the boundary must track the stated fraction, not float(mark*cap)", ax, armed, AxisContext)
	}
	belowMark := AxisUsage{Used: 54, Cap: 100} // 0.54 < 0.55
	if ax, armed := tr.Cross(belowMark, AxisUsage{}, AxisUsage{}, AxisUsage{}); armed {
		t.Fatalf("54/100 below the 0.55 mark armed on %q; want disarmed", ax)
	}
}

// TestArmTriggersUnboundedAndZeroNeverArm proves the "0 = no opinion" convention: an
// axis with a non-positive cap (unbounded / not stated) never arms however large its
// used counter reads, and an all-unbounded envelope never arms at all. A surprise
// rotation on an axis the user never bounded is the failure this guards.
func TestArmTriggersUnboundedAndZeroNeverArm(t *testing.T) {
	tr := ArmTriggers{SoftMark: 0.5}

	// Huge used, zero cap on every axis: still no arm.
	huge := AxisUsage{Used: 1 << 40, Cap: 0}
	if ax, armed := tr.Cross(huge, huge, huge, huge); armed {
		t.Fatalf("unbounded axes armed on %q with cap=0; want never", ax)
	}

	// Negative cap is also unbounded (mirrors Budget's Unbounded=-1 convention).
	neg := AxisUsage{Used: 1000, Cap: -1}
	if _, armed := tr.Cross(neg, neg, neg, neg); armed {
		t.Fatal("negative cap treated as bounded; want unbounded (never arms)")
	}

	// Bounded cap but zero used: nothing consumed yet, no arm.
	fresh := AxisUsage{Used: 0, Cap: 1000}
	if _, armed := tr.Cross(fresh, fresh, fresh, fresh); armed {
		t.Fatal("zero-used bounded axes armed; want disarmed until the mark is crossed")
	}
}

// TestArmTriggersInvalidSoftMarkInert proves the soft mark is fail-closed policy data:
// the zero value and any out-of-range fraction arm on nothing, so a leg with no
// configured rotation policy simply never rotates on a budget axis.
func TestArmTriggersInvalidSoftMarkInert(t *testing.T) {
	over := AxisUsage{Used: 1_000_000, Cap: 1000} // way past any real mark
	for _, mark := range []float64{0, -0.2, 1.5} {
		tr := ArmTriggers{SoftMark: mark}
		if ax, armed := tr.Cross(over, over, over, over); armed {
			t.Fatalf("soft mark %v (invalid) armed on %q; want inert", mark, ax)
		}
	}
	// The zero-value ArmTriggers is inert by the same rule.
	var zero ArmTriggers
	if zero.Crossed(over, over, over, over) {
		t.Fatal("zero-value ArmTriggers armed; want inert")
	}
}

// TestArmTriggersContextIsPrimary pins the spine's priority order: when several axes
// cross on the same boundary, the PRIMARY context axis is the one reported. This is
// what lets the RELAY_ARMED reason name a single deterministic arming axis rather than
// an arbitrary one.
func TestArmTriggersContextIsPrimary(t *testing.T) {
	tr := ArmTriggers{SoftMark: 0.5}
	crossed := AxisUsage{Used: 900, Cap: 1000} // 0.9, past the 0.5 mark

	// All four cross at once: context wins.
	if ax, armed := tr.Cross(crossed, crossed, crossed, crossed); !armed || ax != AxisContext {
		t.Fatalf("all axes crossed => (%q,%v); want (%q,true)", ax, armed, AxisContext)
	}

	// Context below the mark, turns/wall/spend above: the first crossed in priority
	// order (turns) wins, not an arbitrary one.
	ctxLow := AxisUsage{Used: 100, Cap: 1000} // 0.1, below
	if ax, armed := tr.Cross(ctxLow, crossed, crossed, crossed); !armed || ax != AxisTurns {
		t.Fatalf("context low, rest crossed => (%q,%v); want (%q,true)", ax, armed, AxisTurns)
	}
}

// TestArmTriggersFeedsArmFire closes the loop with the G2 consumer: the bool this rung
// produces is exactly the softMarkCrossed input ArmFire.Step wants, and driving one
// into the other arms then fires the two-phase machine. This is the wiring the #1889
// header promised #1890 would supply.
func TestArmTriggersFeedsArmFire(t *testing.T) {
	tr := ArmTriggers{SoftMark: 0.6}
	var af ArmFire

	// Below the mark: the fold says disarmed, so the state machine stays disarmed even
	// at a safe point.
	below := AxisUsage{Used: 500, Cap: 1000}
	if got := af.Step(tr.Crossed(below, AxisUsage{}, AxisUsage{}, AxisUsage{}), safe); got != RotationDisarmed {
		t.Fatalf("below mark drove state to %q; want %q", got, RotationDisarmed)
	}

	// Cross the mark at an unsafe point: arm, do not fire.
	over := AxisUsage{Used: 800, Cap: 1000}
	if got := af.Step(tr.Crossed(over, AxisUsage{}, AxisUsage{}, AxisUsage{}), unsafe); got != RotationArmed {
		t.Fatalf("crossed mark at unsafe point drove state to %q; want %q", got, RotationArmed)
	}

	// Next safe point fires.
	if got := af.Step(false, safe); got != RotationFired {
		t.Fatalf("safe point while armed drove state to %q; want %q", got, RotationFired)
	}
}
