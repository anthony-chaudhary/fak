// Rung G4 (issue #1891): anti-thrash hysteresis — a leg may re-arm only after the
// relay has shown a minimum VERIFIED progress-cursor movement since the last arm.
//
// The spine (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Thrash") names
// rotating-without-progress as a failure in its own right: a relay whose soft mark
// keeps crossing could arm, fire, and rotate every leg while getting nothing done,
// burning fresh windows on spin. G2 (armfire.go) deliberately "holds no hysteresis
// (that is G4, #1891)"; this rung is that missing gate. It sits in front of the arm
// transition: even when the G3 soft mark has crossed, arming is withheld until
// verified progress has advanced at least MinSteps beyond the last arm's baseline.
//
// Verified, never claimed. Movement is measured only against the D3 ledger-verified
// progress read (progress.go, VerifiedProgress) — the ProgressStep count the intent
// ledger ACTUALLY records, never a number a leg asserted. An unverifiable read
// (ProgressUnknown) fails closed: it is treated as no movement, so a relay that
// cannot PROVE it advanced cannot re-arm and thrash. This mirrors the no-`claimed`
// invariant the baton and progress rungs enforce.
//
// Pure state machine, like its G2 sibling. It reads no clock, does no I/O, and holds
// only the baseline step count carried across legs; the caller supplies each verified
// progress reading and records when an arm is acted on. Composition with ArmFire is a
// later floor's job — this rung emits the gate, it does not wire it.
package relay

// ArmHysteresis is the anti-thrash gate for re-arming across relay legs. The zero
// value is an unconfigured, non-gating hysteresis (MinSteps 0 permits every arm — an
// unset policy never refuses, matching armtriggers' "unset SoftMark never arms"). Set
// MinSteps to the minimum verified progress-step movement a re-arm requires.
type ArmHysteresis struct {
	// MinSteps is the minimum number of NEW verified progress steps that must appear
	// since the last arm before a re-arm is permitted. <= 0 disables the gate.
	MinSteps int

	// armedOnce records whether a first arm has been noted; the first arm is always
	// permitted (there is no prior arm to thrash against).
	armedOnce bool
	// baseline is the verified progress-step count captured at the last noted arm — the
	// high-water mark a re-arm's movement is measured against.
	baseline int
}

// MayArm reports whether a leg may arm now, given the current verified progress. The
// first arm is always permitted (there is no prior arm to thrash against). A re-arm is
// permitted only when the reading is ProgressVerified AND its step count has advanced
// at least MinSteps beyond the last arm's baseline; an unverifiable read
// (ProgressUnknown) fails closed and refuses, so a relay that cannot prove movement
// cannot re-arm. With MinSteps <= 0 the gate is disabled and every arm is permitted.
func (h *ArmHysteresis) MayArm(now VerifiedProgress) bool {
	if h.MinSteps <= 0 {
		return true
	}
	if !h.armedOnce {
		return true
	}
	if now.Verdict != ProgressVerified {
		return false
	}
	return len(now.Steps)-h.baseline >= h.MinSteps
}

// NoteArmed records that an arm was acted on at the given verified progress, moving
// the baseline that future re-arms are measured against. The caller invokes it when it
// acts on a permitted arm (MayArm returned true). A ProgressUnknown reading, or one
// that has not advanced past the current baseline, leaves the baseline unchanged: the
// next verified read is measured against the last known high-water mark, never reset
// to a number that was never verified or that would move the bar backward.
func (h *ArmHysteresis) NoteArmed(at VerifiedProgress) {
	h.armedOnce = true
	if at.Verdict == ProgressVerified && len(at.Steps) > h.baseline {
		h.baseline = len(at.Steps)
	}
}

// Baseline returns the current verified high-water step count re-arms are measured
// against — the operator/debug read of how far the last permitted arm had advanced.
func (h *ArmHysteresis) Baseline() int { return h.baseline }
