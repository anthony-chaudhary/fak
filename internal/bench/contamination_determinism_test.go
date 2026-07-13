package bench

import (
	"bytes"
	"testing"
)

// TestContaminationDeterministic pins the "independently replayed environment"
// half of #4571's witness: the audit report is a pure function of its input with
// no clock and no map iteration, so re-running it on the same corpus produces
// byte-identical JSON. That is what lets a reviewer regenerate the committed golden
// in a clean checkout and get the same artifact — the replay is reproducible, not
// merely plausible.
func TestContaminationDeterministic(t *testing.T) {
	corpus := DefaultContaminationCorpus()
	first, err := AuditContamination(corpus).JSON()
	if err != nil {
		t.Fatalf("first JSON: %v", err)
	}
	second, err := AuditContamination(corpus).JSON()
	if err != nil {
		t.Fatalf("second JSON: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("audit is non-deterministic: two runs on the same corpus differ")
	}
}

// TestContaminationDedupeKeepsFirst pins the dedupe semantics the acceptance
// criterion names ("known duplicates are flagged and excluded from confirmatory
// claims"): the FIRST occurrence of a content hash stays eligible, and only the
// later occurrence is flagged as a duplicate that points back at the first. This is
// the standard "keep one representative" dedupe — flagging both would throw away a
// legitimately-usable case, flagging neither would double-count it.
func TestContaminationDedupeKeepsFirst(t *testing.T) {
	base := DefaultContaminationCorpus()[0] // a fully-evidenced admitted case
	dupe := base
	dupe.ID = base.ID + "-copy" // same ContentHash, different ID

	r := AuditContamination([]ContaminationCase{base, dupe})
	if r.Verdict != VerdictContaminationRisk {
		t.Fatalf("verdict = %q, want %q (a duplicate is a risk)", r.Verdict, VerdictContaminationRisk)
	}
	// The first occurrence is the sole eligible case.
	if len(r.ConfirmatoryEligible) != 1 || r.ConfirmatoryEligible[0] != base.ID {
		t.Fatalf("eligible = %v, want only the first occurrence %q", r.ConfirmatoryEligible, base.ID)
	}
	// The later occurrence is the sole exclusion, flagged as a duplicate of the first.
	if len(r.Excluded) != 1 {
		t.Fatalf("excluded = %+v, want exactly the later duplicate", r.Excluded)
	}
	ex := r.Excluded[0]
	if ex.Case != dupe.ID || ex.Status != StatusDuplicate || ex.DuplicateOf != base.ID {
		t.Fatalf("excluded[0] = %+v, want %q duplicate_of %q", ex, dupe.ID, base.ID)
	}
}
