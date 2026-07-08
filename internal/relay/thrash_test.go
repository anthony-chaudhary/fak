package relay

import "testing"

// Rung J3 (issue #1905): the adversarial thrash witness. Thrash — rotating so often
// that no leg makes progress — is a named failure of its own (the spine, "Thrash").
// Two rungs mitigate it: G4 hysteresis (hysteresis.go) withholds a re-arm until
// verified progress has moved, and G6 the no-progress escape (noprogress.go) halts the
// relay after K empty legs. This test drives the trigger to FIRE every leg with no
// verified progress and asserts, together, that hysteresis holds the re-arm and
// RELAY_NO_PROGRESS stops the spin — the composed behavior no per-rung unit test shows.

// TestThrashHysteresisAndNoProgressStopTheSpin is the #1905 done-condition witness: a
// relay whose soft mark crosses every leg but that makes NO verified progress must not
// re-arm past the first leg (hysteresis) and must halt after K empty legs
// (RELAY_NO_PROGRESS).
func TestThrashHysteresisAndNoProgressStopTheSpin(t *testing.T) {
	const (
		minProgress = 1 // G4: a re-arm needs >= 1 new verified step
		maxEmpty    = 3 // G6: halt after 3 consecutive empty legs
	)
	h := ArmHysteresis{MinSteps: minProgress}
	esc := NoProgressEscape{MaxEmptyLegs: maxEmpty}

	// The worst case for thrash: the trigger fires every leg AND every boundary is a
	// safe point, so nothing but hysteresis stands between the spin and a rotation each
	// leg. Yet no leg makes verified progress.
	stuck := verifiedSteps(0)

	rotations := 0
	haltedAt := -1
	for leg := 0; leg < 10; leg++ {
		// A fresh ArmFire per leg (a leg fires at most once, terminal). Hysteresis gates
		// the soft mark the G2 machine sees: a re-arm is only offered when progress
		// permits it, so the spin cannot arm-fire every leg.
		var af ArmFire
		mayReArm := h.MayArm(stuck)
		if af.Step(mayReArm, safe) == RotationFired {
			rotations++
			h.NoteArmed(stuck)
		}
		// The leg completes with no verified progress; fold it into the escape.
		halt, reason := esc.ObserveLeg(stuck)
		if halt {
			if reason != ReasonNoProgress {
				t.Fatalf("halt reason = %q, want %q", reason, ReasonNoProgress)
			}
			haltedAt = leg
			break
		}
	}

	// Hysteresis: only the FIRST leg rotated; every later leg was refused a re-arm
	// because no verified progress was made. No thrash.
	if rotations != 1 {
		t.Fatalf("hysteresis failed: %d rotations under a no-progress spin, want exactly 1 (the first)", rotations)
	}
	// The escape stopped the spin, and it did so after exactly K empty legs (legs
	// indexed 0..K-1, halting on the K-th observation).
	if haltedAt < 0 {
		t.Fatalf("no-progress escape never halted the spin")
	}
	if haltedAt != maxEmpty-1 {
		t.Fatalf("halted at leg index %d, want %d (the K=%d-th empty leg)", haltedAt, maxEmpty-1, maxEmpty)
	}
}

// TestThrashProgressingRelayReArmsAndNeverHalts is the positive control: the SAME two
// rungs must NOT interfere with a relay that is actually making verified progress — it
// re-arms every leg and never trips the no-progress escape. This proves the mitigation
// is specific to thrash, not a blanket brake on rotation.
func TestThrashProgressingRelayReArmsAndNeverHalts(t *testing.T) {
	const (
		minProgress = 1
		maxEmpty    = 3
	)
	h := ArmHysteresis{MinSteps: minProgress}
	esc := NoProgressEscape{MaxEmptyLegs: maxEmpty}

	rotations := 0
	for leg := 0; leg < 10; leg++ {
		// Each leg advances the verified progress cursor by one step (1, 2, 3, ...).
		progress := verifiedSteps(leg + 1)
		var af ArmFire
		if af.Step(h.MayArm(progress), safe) == RotationFired {
			rotations++
			h.NoteArmed(progress)
		}
		if halt, _ := esc.ObserveLeg(progress); halt {
			t.Fatalf("a progressing relay halted on RELAY_NO_PROGRESS at leg %d", leg)
		}
	}
	if rotations != 10 {
		t.Fatalf("a progressing relay rotated %d times, want 10 (every leg re-arms on real progress)", rotations)
	}
}
