package relay

import (
	"fmt"
	"testing"
)

// verifiedSteps builds a ProgressVerified reading carrying n distinct verified
// progress steps — the ledger-recorded shape ArmHysteresis measures movement against.
func verifiedSteps(n int) VerifiedProgress {
	steps := make([]ProgressStep, n)
	for i := range steps {
		steps[i] = ProgressStep{Ref: fmt.Sprintf("sha-%02d", i)}
	}
	return VerifiedProgress{Verdict: ProgressVerified, Steps: steps}
}

// unknownProgress is the fail-closed reading: the ledger could not be verified, so no
// movement can be proven.
var unknownProgress = VerifiedProgress{Verdict: ProgressUnknown}

// TestHysteresisRefusesReArmWithoutVerifiedProgress is the #1891 done-condition
// witness: a leg with no verified progress cannot re-arm, while a re-arm backed by
// enough verified movement is permitted.
func TestHysteresisRefusesReArmWithoutVerifiedProgress(t *testing.T) {
	h := ArmHysteresis{MinSteps: 2}

	// The first arm is always permitted — there is no prior arm to thrash against.
	if !h.MayArm(verifiedSteps(1)) {
		t.Fatalf("first arm must be permitted regardless of progress")
	}
	h.NoteArmed(verifiedSteps(1)) // baseline is now 1 verified step

	// A re-arm with no further verified movement (still 1 step) must be refused: this
	// is exactly the thrash the gate exists to stop.
	if h.MayArm(verifiedSteps(1)) {
		t.Fatalf("re-arm with no verified movement must be refused")
	}
	// One extra step is still below the MinSteps=2 threshold: refuse.
	if h.MayArm(verifiedSteps(2)) {
		t.Fatalf("re-arm with movement below MinSteps must be refused (1 < 2)")
	}
	// Two or more new verified steps clear the threshold: permit.
	if !h.MayArm(verifiedSteps(3)) {
		t.Fatalf("re-arm with movement >= MinSteps must be permitted (2 >= 2)")
	}

	// After acting on the permitted re-arm, the baseline advances and the bar resets.
	h.NoteArmed(verifiedSteps(3))
	if got := h.Baseline(); got != 3 {
		t.Fatalf("Baseline() = %d, want 3 after arming at 3 verified steps", got)
	}
	if h.MayArm(verifiedSteps(4)) {
		t.Fatalf("after re-arm at 3 steps, a single new step (4) is below MinSteps and must be refused")
	}
	if !h.MayArm(verifiedSteps(5)) {
		t.Fatalf("two new steps beyond the advanced baseline must be permitted (5-3 >= 2)")
	}
}

// TestHysteresisUnknownProgressFailsClosed pins the fail-closed edge: once armed, a
// reading whose progress cannot be verified is treated as no movement and can never
// permit a re-arm — a relay that cannot prove it advanced cannot thrash.
func TestHysteresisUnknownProgressFailsClosed(t *testing.T) {
	h := ArmHysteresis{MinSteps: 1}
	if !h.MayArm(verifiedSteps(0)) {
		t.Fatalf("first arm must be permitted")
	}
	h.NoteArmed(verifiedSteps(0))

	if h.MayArm(unknownProgress) {
		t.Fatalf("an unverifiable reading must never permit a re-arm (fail closed)")
	}
	// An unknown reading must not move the baseline either.
	h.NoteArmed(unknownProgress)
	if got := h.Baseline(); got != 0 {
		t.Fatalf("Baseline() = %d, want 0; an unverifiable arm must not move the bar", got)
	}
	// A later verified reading with real movement still clears the original baseline.
	if !h.MayArm(verifiedSteps(1)) {
		t.Fatalf("verified movement after an unknown reading must be measured against the last verified baseline")
	}
}

// TestHysteresisDisabledPermitsEveryArm pins that the zero/unset policy never refuses:
// MinSteps <= 0 is an unconfigured gate, so every arm — including a re-arm with no
// movement at all — is permitted.
func TestHysteresisDisabledPermitsEveryArm(t *testing.T) {
	for _, min := range []int{0, -1} {
		h := ArmHysteresis{MinSteps: min}
		if !h.MayArm(verifiedSteps(0)) {
			t.Fatalf("MinSteps=%d: first arm must be permitted", min)
		}
		h.NoteArmed(verifiedSteps(0))
		if !h.MayArm(unknownProgress) {
			t.Fatalf("MinSteps=%d: a disabled gate must permit every arm, even unverifiable", min)
		}
	}
}
