package sessionreset

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// issue #1583: the objective pin must survive BuildSeed/reset carryover unchanged, or the
// host must get a visible, typed refusal/query rather than a silent drop/rewrite. Test
// names match the witness regex `Objective` per the issue's exact command:
//   go test ./internal/ctxplan ./internal/sessionreset -run Objective

// TestObjectivePinSurvivesReset proves the end-to-end #1583 contract: pin the objective for
// a session, carry it across a reset whose transcript still opens with the SAME objective,
// and CarryObjective must report ObjectivePreserved with identical PinID/Digest.
func TestObjectivePinSurvivesReset(t *testing.T) {
	in := Input{Messages: sampleTranscript()}
	before := PinObjective("run-1", in)
	if before.IsZero() {
		t.Fatal("PinObjective must not produce a zero pin on a transcript with a user line")
	}

	// The fresh session after a reset is seeded from the SAME logical transcript (a
	// faithful carryover never changes what the objective WAS).
	after, decision := CarryObjective(before, "run-1", in)

	if decision.Outcome != ctxplan.ObjectivePreserved {
		t.Fatalf("a faithful carryover must be Preserved, got %q (%s)", decision.Outcome, decision.Reason)
	}
	if decision.Outcome.Refusal() {
		t.Error("ObjectivePreserved must not be a Refusal outcome")
	}
	if after.PinID != before.PinID || after.Digest != before.Digest {
		t.Fatalf("pin identity/digest must be stable across the reset: before=%+v after=%+v", before, after)
	}
}

// TestObjectivePinReplanPreservesIdentity proves that RepinObjective threads the PRIOR
// PinID forward across a call boundary that mimics a ctxplan replan followed by a session
// reset — the identity is preserved BY CONSTRUCTION, not by luck of matching text.
func TestObjectivePinReplanPreservesIdentity(t *testing.T) {
	in := Input{Messages: sampleTranscript()}
	first := PinObjective("run-42", in)

	// Simulate two further replans/resets in a row, chaining prior -> next each time.
	second := RepinObjective(first, "run-42", in)
	third := RepinObjective(second, "run-42", in)

	if second.PinID != first.PinID || third.PinID != first.PinID {
		t.Fatalf("PinID must be threaded forward unchanged: %q -> %q -> %q", first.PinID, second.PinID, third.PinID)
	}
	if second.Digest != first.Digest || third.Digest != first.Digest {
		t.Fatalf("Digest must stay stable when the objective text is unchanged: %q -> %q -> %q",
			first.Digest, second.Digest, third.Digest)
	}
}

// TestObjectivePinDropIsVisibleNotSilent proves the "done condition" from #1583 directly:
// a reset that produces NO objective at all (an empty transcript, as if the carryover
// pipeline lost it) must not be silently accepted — CarryObjective must report a Refusal
// outcome.
func TestObjectivePinDropIsVisibleNotSilent(t *testing.T) {
	before := PinObjective("run-7", Input{Messages: sampleTranscript()})
	if before.IsZero() {
		t.Fatal("test setup: before pin must be non-zero")
	}

	// A fresh session Input with no user line at all: the extractive re-derivation finds
	// nothing, so RepinObjective would produce a pin with the same PinID but EMPTY text —
	// simulating the "objective text quietly vanished" carryover bug.
	emptyIn := Input{Messages: []Msg{{Role: "system", Content: "fresh boot, no history"}}}
	after, decision := CarryObjective(before, "run-7", emptyIn)

	if decision.Outcome != ctxplan.ObjectiveDrifted {
		t.Fatalf("an objective that silently emptied out under the same id must be Drifted, got %q (%s)",
			decision.Outcome, decision.Reason)
	}
	if !decision.Outcome.Refusal() {
		t.Fatal("a drifted/dropped objective must be a Refusal outcome — #1583's done condition")
	}
	if after.Text != "" {
		t.Fatalf("test setup sanity: expected the re-derived text to be empty, got %q", after.Text)
	}
}

// TestObjectivePinRewriteIsVisibleNotSilent proves the content-drift half of the done
// condition: a reset whose re-derived objective text genuinely changed (a different first
// user line) under the SAME identity must reconcile as Drifted, a Refusal outcome — not be
// silently accepted as a normal carryover.
func TestObjectivePinRewriteIsVisibleNotSilent(t *testing.T) {
	original := Input{Messages: sampleTranscript()}
	before := PinObjective("run-9", original)

	rewritten := Input{Messages: []Msg{
		{Role: "system", Content: "You are a helpful coding assistant for the fak repo."},
		{Role: "user", Content: "Totally different objective now."},
	}}
	_, decision := CarryObjective(before, "run-9", rewritten)

	if decision.Outcome != ctxplan.ObjectiveDrifted {
		t.Fatalf("a genuinely different re-derived objective under the same id must be Drifted, got %q (%s)",
			decision.Outcome, decision.Reason)
	}
	if !decision.Outcome.Refusal() {
		t.Fatal("ObjectiveDrifted must be a Refusal outcome")
	}
}

// TestObjectivePinFirstSessionIsEstablished proves CarryObjective distinguishes "this is
// the first pin ever" (no prior objective) from a carryover: calling it with a zero prior
// must report ObjectiveEstablished, not Preserved or a Refusal.
func TestObjectivePinFirstSessionIsEstablished(t *testing.T) {
	in := Input{Messages: sampleTranscript()}
	_, decision := CarryObjective(ctxplan.ObjectivePin{}, "run-1", in)
	if decision.Outcome != ctxplan.ObjectiveEstablished {
		t.Fatalf("a first pin with no prior must be Established, got %q (%s)", decision.Outcome, decision.Reason)
	}
	if decision.Outcome.Refusal() {
		t.Error("ObjectiveEstablished must not be a Refusal outcome")
	}
}

// TestObjectivePinContributorRendersIdentityAndDigest proves the seed-facing half of the
// wiring: a registered objectiveContributor folds the pin's id+digest into BuildSeed's
// recap, so the fresh session's carryover text is self-describing and a downstream
// CarryObjective check can be corroborated by eye.
func TestObjectivePinContributorRendersIdentityAndDigest(t *testing.T) {
	in := Input{Messages: sampleTranscript()}
	pin := PinObjective("run-1", in)

	p, ok := NewObjectivePinContributor(pin).Contribute(in)
	if !ok {
		t.Fatal("objectiveContributor must fire for a non-zero pin")
	}
	if !strings.Contains(p.Text, pin.PinID) {
		t.Fatalf("rendered part must name the pin id: %q", p.Text)
	}
	if p.Meta["pin_id"] != pin.PinID || p.Meta["digest"] != pin.Digest {
		t.Fatalf("Meta must carry the exact pin id/digest for programmatic comparison: %+v", p.Meta)
	}
	if p.Order != ObjectivePinOrder {
		t.Fatalf("objectiveContributor Order = %d, want %d", p.Order, ObjectivePinOrder)
	}
}

// TestObjectivePinContributorDeclinesWithoutAPin proves importing this file changes no
// default behavior: a zero-value objectiveContributor (the state before any host calls
// RegisterObjectivePin) always declines.
func TestObjectivePinContributorDeclinesWithoutAPin(t *testing.T) {
	_, ok := NewObjectivePinContributor(ctxplan.ObjectivePin{}).Contribute(Input{Messages: sampleTranscript()})
	if ok {
		t.Fatal("objectiveContributor must decline for the zero pin")
	}
}

// TestObjectivePinRegisterJoinsSeedWithoutBreakingDefaults proves RegisterObjectivePin is
// additive: after registering a live pin contributor, BuildSeed's recap contains the
// objective-pin line AND the four built-in contributors still fire (mirrors
// TestThirdPartyRegisterJoinsTheFold's additivity check).
func TestObjectivePinRegisterJoinsSeedWithoutBreakingDefaults(t *testing.T) {
	in := Input{Messages: sampleTranscript()}
	pin := PinObjective("run-1", in)
	RegisterObjectivePin(pin)

	seed := BuildSeed(in)
	if !strings.Contains(seed.Recap, pin.PinID) {
		t.Fatalf("seed recap missing the objective pin id: %q", seed.Recap)
	}
	if !strings.Contains(seed.Recap, "Objective pin") {
		t.Fatalf("seed recap missing the objective pin section: %q", seed.Recap)
	}
	// The default contributors (durability_facts / task_distill / warm_prefix /
	// verbatim_tail) must still all fire alongside the new one.
	if !strings.Contains(seed.Recap, "Where we are") {
		t.Fatal("registering the objective pin contributor must not suppress task_distill")
	}
	if !strings.Contains(seed.Recap, "Most recent exchange") {
		t.Fatal("registering the objective pin contributor must not suppress verbatim_tail")
	}
}

// TestObjectivePinReportRendersRefusalFlag proves ReportObjectiveCarryover surfaces the
// Refusal bit alongside the outcome string, so a host logging surface does not have to
// re-derive Outcome.Refusal() itself.
func TestObjectivePinReportRendersRefusalFlag(t *testing.T) {
	before := PinObjective("run-1", Input{Messages: sampleTranscript()})
	_, dropped := CarryObjective(before, "run-1", Input{})
	report := ReportObjectiveCarryover(dropped)
	if !report.Refusal {
		t.Fatalf("report must flag a dropped/drifted decision as a refusal: %+v", report)
	}
	if report.Outcome != string(dropped.Outcome) {
		t.Fatalf("report outcome = %q, want %q", report.Outcome, dropped.Outcome)
	}
	if report.String() == "" {
		t.Error("String() must render a non-empty operator summary")
	}
}

// TestObjectivePinCarryoverIsReplayable proves CarryObjective's underlying reconciliation
// is deterministic end-to-end through the sessionreset wiring: the same (before, pinID, in)
// always reconciles to the identical decision.
func TestObjectivePinCarryoverIsReplayable(t *testing.T) {
	before := PinObjective("run-1", Input{Messages: sampleTranscript()})
	in := Input{Messages: sampleTranscript()}

	_, first := CarryObjective(before, "run-1", in)
	for i := 0; i < 5; i++ {
		_, got := CarryObjective(before, "run-1", in)
		if got != first {
			t.Fatalf("CarryObjective must be deterministic: iteration %d got %+v, want %+v", i, got, first)
		}
	}
}
