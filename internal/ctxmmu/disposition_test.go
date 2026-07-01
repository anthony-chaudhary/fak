package ctxmmu_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestDispositionNotMintedFromSingleObservation is the issue's own fixture
// (#1598), verbatim: "I am tired today" alone must NOT mint "user prefers short
// answers". No Evidence is offered, so the gate must refuse — never silently
// drop (it must return a typed outcome) and never silently promote.
func TestDispositionNotMintedFromSingleObservation(t *testing.T) {
	obs := ctxmmu.Observation{Text: "I am tired today", Source: "user"}
	want := ctxmmu.Disposition{Trait: "user prefers short answers"}

	out := ctxmmu.GateDisposition(obs, want, ctxmmu.Evidence{})

	if out.Minted() {
		t.Fatalf("GateDisposition minted %+v from a single unsupported observation, want refused", out.Disposition)
	}
	if out.Kind != ctxmmu.OutcomeRefusedUnsupported {
		t.Fatalf("Kind = %v, want OutcomeRefusedUnsupported", out.Kind)
	}
	if out.Reason == "" {
		t.Fatalf("refusal must carry a non-empty Reason (typed outcome, not a silent drop)")
	}
	if out.Disposition != (ctxmmu.Disposition{}) {
		t.Fatalf("refused outcome must not carry a populated Disposition, got %+v", out.Disposition)
	}
}

// TestDispositionRequiresEvidence sweeps every EvidenceKind the closed
// vocabulary defines against a single situational observation, proving the
// gate's default is refuse and that only explicit, non-empty support flips it.
func TestDispositionRequiresEvidence(t *testing.T) {
	obs := ctxmmu.Observation{Text: "I am tired today", Source: "user"}
	want := ctxmmu.Disposition{Trait: "user prefers short answers"}

	cases := []struct {
		name       string
		ev         ctxmmu.Evidence
		wantMinted bool
	}{
		{"zero_value_evidence", ctxmmu.Evidence{}, false},
		{"explicit_none", ctxmmu.Evidence{Kind: ctxmmu.EvidenceNone}, false},
		{"confirmed_but_blank_text", ctxmmu.Evidence{Kind: ctxmmu.EvidenceUserConfirmed, Confirmed: "  "}, false},
		{"pattern_but_blank", ctxmmu.Evidence{Kind: ctxmmu.EvidenceEstablishedPattern, Pattern: ""}, false},
		{"one_corroborator_short_of_bar", ctxmmu.Evidence{
			Kind:          ctxmmu.EvidenceCorroboration,
			Corroborating: []ctxmmu.Observation{{Text: "I'd rather you keep it brief today too"}},
		}, false},
		{"user_confirmed", ctxmmu.Evidence{Kind: ctxmmu.EvidenceUserConfirmed, Confirmed: "yes, in general keep answers short"}, true},
		{"established_pattern", ctxmmu.Evidence{Kind: ctxmmu.EvidenceEstablishedPattern, Pattern: "prior session durable store: user prefers short answers"}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := ctxmmu.GateDisposition(obs, want, tc.ev)
			if out.Minted() != tc.wantMinted {
				t.Fatalf("Minted() = %v, want %v (outcome=%+v)", out.Minted(), tc.wantMinted, out)
			}
			if !tc.wantMinted && out.Reason == "" {
				t.Fatalf("refusal must carry a Reason")
			}
		})
	}
}

// TestDispositionMintedWithCorroboration proves the corroboration path: a
// single observation is still refused, but once independent corroborating
// observations reach ctxmmu.MinCorroboration, the SAME trait mints — and the
// minted Disposition carries the durable class.
func TestDispositionMintedWithCorroboration(t *testing.T) {
	obs := ctxmmu.Observation{Text: "I am tired today, keep it short", Source: "user"}
	want := ctxmmu.Disposition{Trait: "user prefers short answers"}

	// Below the bar: only one corroborator.
	short := ctxmmu.Evidence{
		Kind: ctxmmu.EvidenceCorroboration,
		Corroborating: []ctxmmu.Observation{
			{Text: "please just give me the short version"},
		},
	}
	if out := ctxmmu.GateDisposition(obs, want, short); out.Minted() {
		t.Fatalf("minted with only 1 corroborator (< MinCorroboration=%d), want refused", ctxmmu.MinCorroboration)
	} else if out.Kind != ctxmmu.OutcomeRefusedInsufficientCorroboration {
		t.Fatalf("Kind = %v, want OutcomeRefusedInsufficientCorroboration", out.Kind)
	}

	// At the bar: MinCorroboration independent corroborators.
	sufficient := ctxmmu.Evidence{
		Kind: ctxmmu.EvidenceCorroboration,
		Corroborating: []ctxmmu.Observation{
			{Text: "please just give me the short version"},
			{Text: "can you trim your answers down, I don't need the detail"},
		},
	}
	out := ctxmmu.GateDisposition(obs, want, sufficient)
	if !out.Minted() {
		t.Fatalf("did not mint with %d independent corroborators (>= MinCorroboration=%d): %+v",
			len(sufficient.Corroborating), ctxmmu.MinCorroboration, out)
	}
	if out.Kind != ctxmmu.OutcomeMinted {
		t.Fatalf("Kind = %v, want OutcomeMinted", out.Kind)
	}
	if out.Disposition.Trait != want.Trait {
		t.Fatalf("minted Disposition.Trait = %q, want %q", out.Disposition.Trait, want.Trait)
	}
	if out.Disposition.Durability != ctxmmu.DurabilityDurable {
		t.Fatalf("minted Disposition.Durability = %q, want %q", out.Disposition.Durability, ctxmmu.DurabilityDurable)
	}
}

// TestDispositionCorroborationDoesNotDoubleCount proves the triggering
// observation and duplicate/blank candidates cannot be laundered into meeting
// MinCorroboration — otherwise a caller could "corroborate" a claim with copies
// of the very same single remark, defeating the whole point of the bar.
func TestDispositionCorroborationDoesNotDoubleCount(t *testing.T) {
	obs := ctxmmu.Observation{Text: "I am tired today", Source: "user"}
	want := ctxmmu.Disposition{Trait: "user prefers short answers"}

	ev := ctxmmu.Evidence{
		Kind: ctxmmu.EvidenceCorroboration,
		Corroborating: []ctxmmu.Observation{
			{Text: "I am tired today"}, // same as the triggering observation
			{Text: "  "},               // blank
			{Text: "I am tired today"}, // duplicate of the triggering observation again
		},
	}
	out := ctxmmu.GateDisposition(obs, want, ev)
	if out.Minted() {
		t.Fatalf("minted from re-counted/blank pseudo-corroborators, want refused: %+v", out)
	}
}

// TestDispositionOutcomeNeverAmbiguous checks the typed-outcome contract
// itself: a refused outcome never carries a populated Disposition (so a caller
// cannot accidentally use a zero-value/partial trait as if it had been
// approved), and a minted outcome always does.
func TestDispositionOutcomeNeverAmbiguous(t *testing.T) {
	obs := ctxmmu.Observation{Text: "I am tired today"}
	want := ctxmmu.Disposition{Trait: "user prefers short answers"}

	refused := ctxmmu.GateDisposition(obs, want, ctxmmu.Evidence{})
	if refused.Disposition.Trait != "" {
		t.Fatalf("refused outcome leaked a Disposition.Trait: %+v", refused)
	}

	minted := ctxmmu.GateDisposition(obs, want, ctxmmu.Evidence{Kind: ctxmmu.EvidenceUserConfirmed, Confirmed: "yes always"})
	if minted.Disposition.Trait != want.Trait {
		t.Fatalf("minted outcome missing the Disposition trait: %+v", minted)
	}
}
