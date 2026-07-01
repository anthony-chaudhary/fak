package ctxplan

import "testing"

// issue #1583: the OBJECTIVE PIN must survive a ctxplan replan or a session reset without
// being silently dropped or rewritten. Each test isolates one ReconcileObjective outcome
// (plus the log's replay/summary/explain surfaces), matching the pagefault_test.go /
// assumption_test.go style already in this package. Test names match the witness regex
// `Objective` per the issue's exact command:
//   go test ./internal/ctxplan ./internal/sessionreset -run Objective

// TestObjectivePinSurvivesReset is the headline "ordinary reset" case: an unchanged pin
// carried across a reset must reconcile as ObjectivePreserved, and its PinID/Digest must be
// byte-identical before and after.
func TestObjectivePinSurvivesReset(t *testing.T) {
	before := NewObjectivePin("obj-1", "ship the fak ctxplan objective pin", 3)
	after := before // a faithful carryover: identical pin threaded through

	d := ReconcileObjective(before, after)
	if d.Outcome != ObjectivePreserved {
		t.Fatalf("an unchanged pin across a reset must be Preserved, got %q (%s)", d.Outcome, d.Reason)
	}
	if d.PinID != before.PinID {
		t.Errorf("decision must echo the pin identity, got %q want %q", d.PinID, before.PinID)
	}
	if after.PinID != before.PinID || after.Digest != before.Digest {
		t.Fatalf("pin identity/digest must be stable across the reset: before=%+v after=%+v", before, after)
	}
	if d.Reason == "" {
		t.Error("a decision must carry a non-empty, operator-readable reason")
	}
}

// TestObjectivePinFirstPinIsEstablishedNotPreserved: pinning an objective for the first
// time (no prior pin) is a distinct outcome from surviving a reset, so an audit can tell
// "turn one" from "carried over."
func TestObjectivePinFirstPinIsEstablishedNotPreserved(t *testing.T) {
	after := NewObjectivePin("obj-1", "first objective", 0)
	d := ReconcileObjective(ObjectivePin{}, after)
	if d.Outcome != ObjectiveEstablished {
		t.Fatalf("a first pin (no prior) must be Established, got %q (%s)", d.Outcome, d.Reason)
	}
}

// TestObjectivePinNeverPinnedStaysPreserved: no objective before, none after, is trivially
// "preserved" (there was nothing to lose) — distinct from Established, which requires a
// real after pin.
func TestObjectivePinNeverPinnedStaysPreserved(t *testing.T) {
	d := ReconcileObjective(ObjectivePin{}, ObjectivePin{})
	if d.Outcome != ObjectivePreserved {
		t.Fatalf("no pin before or after must be Preserved (nothing to lose), got %q", d.Outcome)
	}
}

// TestObjectivePinDropReportsRefusal: a reset that silently loses the pin (after is zero
// but before was not) must reconcile as ObjectiveDropped, and Dropped must be a Refusal —
// the visible signal a caller is required to surface rather than proceed silently.
func TestObjectivePinDropReportsRefusal(t *testing.T) {
	before := NewObjectivePin("obj-1", "keep this objective alive", 1)
	d := ReconcileObjective(before, ObjectivePin{})
	if d.Outcome != ObjectiveDropped {
		t.Fatalf("losing a prior pin must be Dropped, got %q (%s)", d.Outcome, d.Reason)
	}
	if !d.Outcome.Refusal() {
		t.Error("ObjectiveDropped must be a Refusal outcome — a caller must not proceed silently")
	}
}

// TestObjectivePinRepinUnderNewIdentityIsDropped: a reset that mints a fresh PinID for
// what is supposed to be the same objective — with no reconciling link back to the
// original — must be treated as dropping the original pin, not silently renaming it.
func TestObjectivePinRepinUnderNewIdentityIsDropped(t *testing.T) {
	before := NewObjectivePin("obj-1", "ship the objective pin", 1)
	after := NewObjectivePin("obj-2", "ship the objective pin", 1) // same text, different identity
	d := ReconcileObjective(before, after)
	if d.Outcome != ObjectiveDropped {
		t.Fatalf("a re-pin under a new identity must be Dropped, got %q (%s)", d.Outcome, d.Reason)
	}
	if !d.Outcome.Refusal() {
		t.Error("an identity change with no reconciling link must be a Refusal outcome")
	}
}

// TestObjectivePinContentRewriteIsDrifted: the subtler failure — same PinID (claimed
// identity preserved) but different Text/Digest, i.e. a silent rewrite. Drifted must also
// be a Refusal so a caller cannot trust PinID equality alone.
func TestObjectivePinContentRewriteIsDrifted(t *testing.T) {
	before := NewObjectivePin("obj-1", "refund the customer for order 42", 5)
	after := NewObjectivePin("obj-1", "refund the customer for order 99", 5) // paraphrased/rewritten
	d := ReconcileObjective(before, after)
	if d.Outcome != ObjectiveDrifted {
		t.Fatalf("a content rewrite under a preserved identity must be Drifted, got %q (%s)", d.Outcome, d.Reason)
	}
	if !d.Outcome.Refusal() {
		t.Error("ObjectiveDrifted must be a Refusal outcome — content drift is as dangerous as a drop")
	}
	if before.PinID != after.PinID {
		t.Fatalf("this test's premise requires identical PinID, got %q vs %q", before.PinID, after.PinID)
	}
}

// TestObjectivePinCorruptDigestQueriesUser: an after-pin whose stored Digest does not match
// its own PinID+Text (corrupt/tampered persisted state, or a hand-built pin that skipped
// NewObjectivePin) must not be silently trusted as preserved OR flagged as drift — it must
// query the user, mirroring PageFaultQueryUser's "ask rather than assume" posture.
func TestObjectivePinCorruptDigestQueriesUser(t *testing.T) {
	before := NewObjectivePin("obj-1", "the real objective", 2)
	corrupt := before
	corrupt.Digest = "not-a-real-digest"
	if corrupt.Verify() {
		t.Fatal("test setup: corrupt pin must fail Verify")
	}
	d := ReconcileObjective(before, corrupt)
	if d.Outcome != ObjectiveQueryUser {
		t.Fatalf("a corrupt after-pin must query the user, got %q (%s)", d.Outcome, d.Reason)
	}
	if !d.Outcome.Refusal() {
		t.Error("ObjectiveQueryUser must be a Refusal outcome — it blocks silent continuation")
	}
}

// TestObjectivePinIsClosedVocabulary walks every ReconcileObjective branch this file
// exercises and asserts each result is a member of the closed outcome vocabulary — the
// same "never an unrecognized decision" property pagefault_test.go's analogous test checks.
func TestObjectivePinIsClosedVocabulary(t *testing.T) {
	a := NewObjectivePin("obj-a", "objective a", 0)
	b := NewObjectivePin("obj-b", "objective b", 0)
	rewritten := NewObjectivePin("obj-a", "objective a, rewritten", 0)
	corrupt := a
	corrupt.Digest = "corrupt"

	cases := []struct {
		name          string
		before, after ObjectivePin
	}{
		{"none-none", ObjectivePin{}, ObjectivePin{}},
		{"none-established", ObjectivePin{}, a},
		{"dropped", a, ObjectivePin{}},
		{"preserved", a, a},
		{"repin-dropped", a, b},
		{"drifted", a, rewritten},
		{"corrupt-query", a, corrupt},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := ReconcileObjective(c.before, c.after)
			if !ValidObjectiveOutcome(d.Outcome) {
				t.Fatalf("%s: outcome %q is not in the closed vocabulary", c.name, d.Outcome)
			}
			if d.Reason == "" {
				t.Errorf("%s: decision must carry a non-empty reason", c.name)
			}
		})
	}
}

// TestObjectivePinReconciliationIsReplayable is the pure-function determinism witness:
// the same (before, after) pair must always reproduce the identical decision, mirroring
// TestDecidePageFaultDecisionIsReplayable.
func TestObjectivePinReconciliationIsReplayable(t *testing.T) {
	before := NewObjectivePin("obj-1", "objective text", 4)
	after := NewObjectivePin("obj-1", "objective text, changed", 4)

	first := ReconcileObjective(before, after)
	for i := 0; i < 5; i++ {
		got := ReconcileObjective(before, after)
		if got != first {
			t.Fatalf("ReconcileObjective must be deterministic: iteration %d got %+v, want %+v", i, got, first)
		}
	}
}

// TestObjectivePinLogAppendAndReplay exercises the persisted-state half of the contract:
// Append records a typed transition per reconciliation, and Replay recomputes every entry
// from its own stored (before, after) pair with zero divergence, mirroring
// TestPageFaultLogAppendAndReplay.
func TestObjectivePinLogAppendAndReplay(t *testing.T) {
	var log ObjectiveLog

	p1 := NewObjectivePin("obj-1", "first objective", 0)
	d1 := log.Append(ObjectivePin{}, p1)
	if d1.Outcome != ObjectiveEstablished {
		t.Fatalf("first append must be Established, got %q", d1.Outcome)
	}

	p1survived := p1
	d2 := log.Append(p1, p1survived)
	if d2.Outcome != ObjectivePreserved {
		t.Fatalf("second append (unchanged) must be Preserved, got %q", d2.Outcome)
	}

	p1rewritten := NewObjectivePin("obj-1", "first objective, rewritten", 1)
	d3 := log.Append(p1survived, p1rewritten)
	if d3.Outcome != ObjectiveDrifted {
		t.Fatalf("third append (content changed) must be Drifted, got %q", d3.Outcome)
	}

	entries := log.Entries()
	if len(entries) != 3 {
		t.Fatalf("expected 3 logged entries, got %d", len(entries))
	}

	latest, ok := log.Latest()
	if !ok || latest.Digest != p1rewritten.Digest {
		t.Fatalf("Latest must return the most recent after-pin, got %+v ok=%v", latest, ok)
	}

	verdicts, allMatch := log.Replay()
	if !allMatch {
		t.Fatalf("a log built from pure inputs must replay with zero divergence: %+v", verdicts)
	}
	if len(verdicts) != 3 {
		t.Fatalf("expected 3 replay verdicts, got %d", len(verdicts))
	}
	for i, v := range verdicts {
		if v.Diverged {
			t.Errorf("verdict %d unexpectedly diverged: stored=%q recomputed=%q", i, v.Stored, v.Recomputed)
		}
	}

	summary := log.Summary()
	if summary.Established != 1 || summary.Preserved != 1 || summary.Drifted != 1 {
		t.Fatalf("unexpected summary: %+v", summary)
	}
	if summary.Clean() {
		t.Error("a summary containing a Drifted entry must not be Clean")
	}

	if explain := log.Explain(); explain == "" {
		t.Error("Explain must render a non-empty operator report")
	}
}

// TestObjectivePinReplayDetectsDrift is the tamper/regression detector: an entry whose
// stored Decision no longer matches what ReconcileObjective(Before, After) recomputes must
// be reported DIVERGED, mirroring TestPageFaultLogReplayDetectsDivergence.
func TestObjectivePinReplayDetectsDrift(t *testing.T) {
	var log ObjectiveLog
	before := NewObjectivePin("obj-1", "objective", 0)
	log.Append(ObjectivePin{}, before) // Established
	log.Append(before, before)         // Preserved

	// Tamper with the second entry's stored decision directly (simulating a corrupted
	// persisted log / a stale entry from a since-changed reconciliation policy).
	tampered := log.entries[1]
	tampered.Decision.Outcome = ObjectiveDropped
	log.entries[1] = tampered

	verdicts, allMatch := log.Replay()
	if allMatch {
		t.Fatal("a tampered entry must be detected as diverged, not silently pass")
	}
	if len(verdicts) != 2 {
		t.Fatalf("expected 2 verdicts, got %d", len(verdicts))
	}
	if !verdicts[1].Diverged {
		t.Error("the tampered entry (index 1) must be reported diverged")
	}
	if verdicts[1].Recomputed != ObjectivePreserved {
		t.Errorf("recomputed outcome must reflect the true reconciliation (Preserved), got %q", verdicts[1].Recomputed)
	}
	if verdicts[0].Diverged {
		t.Error("the untouched entry (index 0) must not be reported diverged")
	}
}

// TestObjectivePinDigestStableAcrossStepChange: Step is audit metadata only — it must not
// participate in the Digest, so re-stamping a pin's step (e.g. a replan renumbering turns)
// without touching its text never looks like drift.
func TestObjectivePinDigestStableAcrossStepChange(t *testing.T) {
	p1 := NewObjectivePin("obj-1", "stable objective text", 3)
	p2 := NewObjectivePin("obj-1", "stable objective text", 99)
	if p1.Digest != p2.Digest {
		t.Fatalf("Digest must not depend on Step: %q vs %q", p1.Digest, p2.Digest)
	}
	d := ReconcileObjective(p1, p2)
	if d.Outcome != ObjectivePreserved {
		t.Fatalf("a step-only change must still be Preserved, got %q", d.Outcome)
	}
}

// TestObjectivePinRehashUpdatesDigestUnderSameIdentity: Rehash is the explicit,
// caller-driven path for a legitimate content edit — it must produce a pin that still
// Verifies and whose Digest reflects the new text.
func TestObjectivePinRehashUpdatesDigestUnderSameIdentity(t *testing.T) {
	p := NewObjectivePin("obj-1", "original text", 0)
	orig := p.Digest
	p.Text = "revised text"
	p = p.Rehash()
	if p.Digest == orig {
		t.Fatal("Rehash must change the digest when the text changed")
	}
	if !p.Verify() {
		t.Fatal("a freshly rehashed pin must Verify")
	}
	if p.PinID != "obj-1" {
		t.Fatalf("Rehash must not change PinID, got %q", p.PinID)
	}
}

// TestObjectivePinIsZero checks the zero-value sentinel used throughout ReconcileObjective.
func TestObjectivePinIsZero(t *testing.T) {
	if !(ObjectivePin{}).IsZero() {
		t.Error("the zero-value ObjectivePin must report IsZero")
	}
	if NewObjectivePin("obj-1", "text", 0).IsZero() {
		t.Error("a well-formed pin must not report IsZero")
	}
}
