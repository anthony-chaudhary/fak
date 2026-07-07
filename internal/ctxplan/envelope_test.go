package ctxplan

import "testing"

// TestDefaultEnvelopesNeverTargetRawWindow is the doctrine's core acceptance: no default envelope
// treats the provider's advertised hard cap as the target resident budget without a WITNESSED
// provenance. Every shipped default row is a bounded, labeled prior — target held below the modeled
// effective ceiling, not at the raw window.
func TestDefaultEnvelopesNeverTargetRawWindow(t *testing.T) {
	for _, e := range DefaultEnvelopes() {
		if e.Provenance == "" {
			t.Errorf("%s/%s: default envelope must carry a provenance label", e.ModelPattern, e.TaskClass)
		}
		if e.RawWindowTarget() {
			t.Errorf("%s/%s: default target %d treats the raw window as the resident budget without a witness (cap %d, MECW %d, provenance %s)",
				e.ModelPattern, e.TaskClass, e.TargetResidentTokens, e.HardContextCap, e.MaxEffectiveTokens, e.Provenance)
		}
		if e.TargetResidentTokens >= e.HardContextCap {
			t.Errorf("%s/%s: target %d must stay below the hard cap %d", e.ModelPattern, e.TaskClass, e.TargetResidentTokens, e.HardContextCap)
		}
	}
}

// TestEnvelopeTargetClampsToEffectiveEnvelope: Target() is chosen from the effective envelope —
// floored at the minimum viable evidence set and ceiled at min(cap-reserve, MECW) — never the raw
// hard cap.
func TestEnvelopeTargetClampsToEffectiveEnvelope(t *testing.T) {
	// A caller that (wrongly) sets the target to the raw window is clamped down to the effective
	// ceiling, and the output reserve is subtracted before the cap applies.
	e := EffectiveContextEnvelope{
		HardContextCap:          200000,
		OutputReserve:           32000,
		MinViableEvidenceTokens: 4000,
		TargetResidentTokens:    200000,
		MaxEffectiveTokens:      32000,
		Provenance:              ProvenanceModeled,
	}
	if got, want := e.SafeCap(), 32000; got != want {
		t.Fatalf("SafeCap = %d, want %d (MECW below cap-reserve)", got, want)
	}
	if got := e.Target(); got != 32000 {
		t.Errorf("Target = %d, want 32000 (clamped to the effective ceiling, not the raw window)", got)
	}
	// A target below the evidence floor is raised to MVC.
	e.TargetResidentTokens = 100
	if got := e.Target(); got != 4000 {
		t.Errorf("Target = %d, want 4000 (floored at MinViableEvidenceTokens)", got)
	}
}

// TestRawWindowTargetDetectsUnwitnessedCap: an envelope that pins the target at the effective
// ceiling with no witness is flagged; a WITNESSED row at the same size is not.
func TestRawWindowTargetDetectsUnwitnessedCap(t *testing.T) {
	bad := EffectiveContextEnvelope{HardContextCap: 200000, OutputReserve: 0, TargetResidentTokens: 200000, MaxEffectiveTokens: 200000, Provenance: ProvenanceModeled}
	if !bad.RawWindowTarget() {
		t.Errorf("unwitnessed target at the raw window must be flagged as raw-window-target debt")
	}
	good := bad
	good.Provenance = ProvenanceWitnessed
	if good.RawWindowTarget() {
		t.Errorf("a WITNESSED same-task envelope at that size is a measured default, not debt")
	}
	unlabeled := bad
	unlabeled.Provenance = ""
	if !unlabeled.RawWindowTarget() {
		t.Errorf("an unlabeled default (no provenance) must be flagged as raw-window-target debt")
	}
}

// TestDefaultBudgetBoundsFromEnvelope: the seed spectrum's ceiling is sourced from the generic
// envelope's Target (a bounded resident view), and the unconstrained default is unchanged at
// [512, 8192] — the caller's contract stays stable while it is now derived from the envelope.
func TestDefaultBudgetBoundsFromEnvelope(t *testing.T) {
	b := DefaultBudgetBounds()
	if b.Floor != 512 {
		t.Errorf("default floor = %d, want 512", b.Floor)
	}
	if b.Ceil != 8192 {
		t.Errorf("default ceil = %d, want 8192 (from GenericTurnEnvelope().Target())", b.Ceil)
	}
	if b.Ceil >= GenericTurnEnvelope().HardContextCap {
		t.Errorf("default ceil %d must stay far below the hard cap %d", b.Ceil, GenericTurnEnvelope().HardContextCap)
	}
}
