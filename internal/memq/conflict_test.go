package memq

import "testing"

// TestConflictingPreferenceCannotSilentlyOverwrite pins the #1597 done condition
// verbatim: "a conflicting preference fixture cannot silently overwrite or merge
// without a typed resolution." Two explicit-consent promotions of the SAME subject
// (same CellID) with different content digests, recorded in the same scope (same
// step/role), must resolve to an explicit ConflictPreferNewer decision that names the
// winner and carries a reason — never a bare "the second Add wins" with no record of
// the fact that a conflict happened at all.
func TestConflictingPreferenceCannotSilentlyOverwrite(t *testing.T) {
	older := PromotionRecord{
		CellID:     "pref-1",
		SourceSpan: SourceSpan{Step: 1, Role: "user", Descriptor: "coffee preference", Digest: "digest-tea"},
		Durability: "durable",
		Consent:    ConsentExplicit,
		Producer:   "user",
		Seq:        0,
	}
	newer := PromotionRecord{
		CellID:     "pref-1",
		SourceSpan: SourceSpan{Step: 1, Role: "user", Descriptor: "coffee preference", Digest: "digest-coffee"},
		Durability: "durable",
		Consent:    ConsentExplicit,
		Producer:   "user",
		Seq:        1,
	}

	d := DetectFactConflict(older, newer)

	if d.Outcome != ConflictPreferNewer {
		t.Fatalf("conflicting same-scope explicit-consent facts: want outcome %q, got %q (reason=%s)", ConflictPreferNewer, d.Outcome, d.Reason)
	}
	if d.Winner.SourceSpan.Digest != "digest-coffee" {
		t.Fatalf("winner must be the newer (higher-Seq) record, got digest %q", d.Winner.SourceSpan.Digest)
	}
	if d.Reason == "" {
		t.Fatalf("decision must carry an operator-readable reason, got empty")
	}
	// The resolution must be a TYPED decision, not a silent merge: the losing record's
	// content must still be recoverable from the decision (A/B echoed back), so a
	// caller/audit surface can see what was overwritten and why.
	if d.A.SourceSpan.Digest != "digest-tea" || d.B.SourceSpan.Digest != "digest-coffee" {
		t.Fatalf("decision must echo back both inputs verbatim in call order, got A=%q B=%q", d.A.SourceSpan.Digest, d.B.SourceSpan.Digest)
	}
}

// TestConflictAskUserWhenConsentCannotSettleIt proves the second named outcome: when
// consent alone cannot resolve which side should win (one or both sides are not
// ConsentExplicit), the decision must be ConflictAskUser rather than guessing a winner.
func TestConflictAskUserWhenConsentCannotSettleIt(t *testing.T) {
	a := PromotionRecord{
		CellID:     "pref-2",
		SourceSpan: SourceSpan{Step: 2, Role: "assistant", Descriptor: "timezone", Digest: "digest-pst"},
		Durability: "durable",
		Consent:    ConsentInferred,
		Seq:        0,
	}
	b := PromotionRecord{
		CellID:     "pref-2",
		SourceSpan: SourceSpan{Step: 3, Role: "assistant", Descriptor: "timezone", Digest: "digest-est"},
		Durability: "durable",
		Consent:    ConsentUnknown,
		Seq:        1,
	}

	d := DetectFactConflict(a, b)

	if d.Outcome != ConflictAskUser {
		t.Fatalf("consent cannot settle the disagreement: want outcome %q, got %q (reason=%s)", ConflictAskUser, d.Outcome, d.Reason)
	}
	if d.Winner != (PromotionRecord{}) {
		t.Fatalf("ConflictAskUser must never pick a winner, got %+v", d.Winner)
	}
	if d.Reason == "" {
		t.Fatalf("decision must carry an operator-readable reason, got empty")
	}
}

// TestConflictKeepBothScopedForDistinctContexts proves the third named outcome: two
// explicit-consent facts about the same descriptor but recorded in genuinely different
// scopes (different step AND role) are retained as versioned/scoped facts rather than
// forced into a single winner.
func TestConflictKeepBothScopedForDistinctContexts(t *testing.T) {
	home := PromotionRecord{
		CellID:     "pref-3",
		SourceSpan: SourceSpan{Step: 1, Role: "user-home", Descriptor: "beverage preference", Digest: "digest-tea"},
		Durability: "durable",
		Consent:    ConsentExplicit,
		Seq:        0,
	}
	work := PromotionRecord{
		CellID:     "pref-3",
		SourceSpan: SourceSpan{Step: 9, Role: "user-work", Descriptor: "beverage preference", Digest: "digest-coffee"},
		Durability: "durable",
		Consent:    ConsentExplicit,
		Seq:        1,
	}

	d := DetectFactConflict(home, work)

	if d.Outcome != ConflictKeepBothScoped {
		t.Fatalf("distinct-scope explicit facts: want outcome %q, got %q (reason=%s)", ConflictKeepBothScoped, d.Outcome, d.Reason)
	}
	if d.Winner != (PromotionRecord{}) {
		t.Fatalf("ConflictKeepBothScoped must never discard a side by picking a winner, got %+v", d.Winner)
	}
}

// TestConflictNoneForDifferentSubjects proves DetectFactConflict never manufactures a
// conflict between two records that are not actually about the same thing.
func TestConflictNoneForDifferentSubjects(t *testing.T) {
	a := PromotionRecord{CellID: "cell-a", SourceSpan: SourceSpan{Descriptor: "favorite color", Digest: "d1"}, Consent: ConsentExplicit}
	b := PromotionRecord{CellID: "cell-b", SourceSpan: SourceSpan{Descriptor: "favorite food", Digest: "d2"}, Consent: ConsentExplicit}

	d := DetectFactConflict(a, b)
	if d.Outcome != ConflictNone {
		t.Fatalf("different subjects must never conflict, got %q (reason=%s)", d.Outcome, d.Reason)
	}
}

// TestConflictNoneForIdenticalContent proves a re-affirmation (same digest) is not
// treated as a disagreement, even though the records share a subject.
func TestConflictNoneForIdenticalContent(t *testing.T) {
	a := PromotionRecord{CellID: "cell-c", SourceSpan: SourceSpan{Digest: "same-digest"}, Consent: ConsentExplicit, Seq: 0}
	b := PromotionRecord{CellID: "cell-c", SourceSpan: SourceSpan{Digest: "same-digest"}, Consent: ConsentExplicit, Seq: 1}

	d := DetectFactConflict(a, b)
	if d.Outcome != ConflictNone {
		t.Fatalf("identical content under the same subject is a re-affirmation, not a conflict: got %q (reason=%s)", d.Outcome, d.Reason)
	}
}

// TestDetectFactConflictSymmetric proves argument order never changes the resolved
// Outcome or Winner — a caller cannot get a different answer by calling it "the other
// way around."
func TestDetectFactConflictSymmetric(t *testing.T) {
	older := PromotionRecord{
		CellID:     "pref-sym",
		SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "digest-a"},
		Consent:    ConsentExplicit,
		Seq:        0,
	}
	newer := PromotionRecord{
		CellID:     "pref-sym",
		SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "digest-b"},
		Consent:    ConsentExplicit,
		Seq:        1,
	}

	fwd := DetectFactConflict(older, newer)
	rev := DetectFactConflict(newer, older)

	if fwd.Outcome != rev.Outcome {
		t.Fatalf("outcome must be symmetric regardless of argument order: fwd=%q rev=%q", fwd.Outcome, rev.Outcome)
	}
	if fwd.Winner.SourceSpan.Digest != rev.Winner.SourceSpan.Digest {
		t.Fatalf("winner must be symmetric regardless of argument order: fwd=%q rev=%q", fwd.Winner.SourceSpan.Digest, rev.Winner.SourceSpan.Digest)
	}
}

// TestDetectFactConflictsScansPairs proves the batch entry point surfaces every
// actionable pair from a PromotionLedger.For-shaped slice (the reuse the issue calls
// for) and skips non-conflicting pairs.
func TestDetectFactConflictsScansPairs(t *testing.T) {
	recs := []PromotionRecord{
		{CellID: "pref-4", SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "d1"}, Consent: ConsentExplicit, Seq: 0},
		{CellID: "pref-4", SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "d1"}, Consent: ConsentExplicit, Seq: 1}, // re-affirmation, no conflict
		{CellID: "pref-4", SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "d2"}, Consent: ConsentExplicit, Seq: 2}, // conflicts with both prior
	}

	decisions := DetectFactConflicts(recs)
	if len(decisions) != 2 {
		t.Fatalf("want 2 actionable conflict decisions (rec[0]-rec[2], rec[1]-rec[2]), got %d: %+v", len(decisions), decisions)
	}
	for _, d := range decisions {
		if d.Outcome != ConflictPreferNewer {
			t.Fatalf("want ConflictPreferNewer for same-scope explicit conflicts, got %q", d.Outcome)
		}
		if d.Winner.Seq != 2 {
			t.Fatalf("want the highest-Seq record (seq=2) to win every pair it's in, got seq=%d", d.Winner.Seq)
		}
	}
}

// TestPromotionLedgerConflicts proves the ledger-integrated entry point: reusing
// PromotionLedger's OWN storage/lookup (For(cellID)) rather than a second ledger, a
// caller can ask the ledger directly whether a cell's own promotion history conflicts
// with itself.
func TestPromotionLedgerConflicts(t *testing.T) {
	l := NewPromotionLedger()
	l.Record(PromotionRecord{
		CellID:     "pref-5",
		SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "digest-tea"},
		Durability: DurabilityDurable,
		Consent:    ConsentExplicit,
	})
	l.Record(PromotionRecord{
		CellID:     "pref-5",
		SourceSpan: SourceSpan{Step: 1, Role: "user", Digest: "digest-coffee"},
		Durability: DurabilityDurable,
		Consent:    ConsentExplicit,
	})

	decisions := l.Conflicts("pref-5")
	if len(decisions) != 1 {
		t.Fatalf("want exactly 1 conflict decision for pref-5's own history, got %d", len(decisions))
	}
	if decisions[0].Outcome != ConflictPreferNewer {
		t.Fatalf("want ConflictPreferNewer, got %q (reason=%s)", decisions[0].Outcome, decisions[0].Reason)
	}

	if got := l.Conflicts("no-such-cell"); got != nil {
		t.Fatalf("unknown cell must yield nil (no conflicts, no history), got %+v", got)
	}
}

// TestValidConflictOutcomeFailsClosed pins ValidConflictOutcome/String's fail-closed
// posture for a corrupt or foreign value, mirroring recall.StaleFactOutcome's own test.
func TestValidConflictOutcomeFailsClosed(t *testing.T) {
	var bogus ConflictOutcome = "not_a_real_outcome"
	if ValidConflictOutcome(bogus) {
		t.Fatalf("bogus outcome must not be reported valid")
	}
	if got := bogus.String(); got != "unknown(not_a_real_outcome)" {
		t.Fatalf("String() must fail closed and name the foreign value, got %q", got)
	}
	var unset ConflictOutcome
	if got := unset.String(); got != "(unset)" {
		t.Fatalf("zero-value String() = %q, want %q", got, "(unset)")
	}
}
