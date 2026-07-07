package rsiloop

import (
	"path/filepath"
	"testing"
)

// TestPromoteSkillsKeepsOnlyWitnessed is the batch-gate witness: given a mix of a
// skill that beats its held corpus and one that regresses it, PromoteSkills promotes
// ONLY the witnessed one and journals the reverted one into the curator ledger under
// ReasonUnwitnessed — the read-path can then answer why it was not promoted. This is
// the honesty property at the skill-creation entry point: an unmeasured skill in the
// batch cannot slip into the promoted set.
func TestPromoteSkillsKeepsOnlyWitnessed(t *testing.T) {
	l, err := OpenCuratorLedger(filepath.Join(t.TempDir(), "curator.jsonl"))
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}

	base := DefaultSkillFixtureCorpus()
	good := SkillCandidate{
		Skill:      "beats-corpus",
		TruthClean: true,
		Corpus: base.WithCandidateScores(map[string]float64{
			"grep-scoped-symbol":       0.90,
			"refactor-preserves-tests": 0.93,
			"commit-by-explicit-path":  0.85,
			"witness-before-claim":     0.94,
		}, nil),
	}
	bad := SkillCandidate{
		Skill:      "regresses-corpus",
		TruthClean: true,
		Corpus: base.WithCandidateScores(map[string]float64{
			"grep-scoped-symbol":       0.30,
			"refactor-preserves-tests": 0.40,
			"commit-by-explicit-path":  0.20,
			"witness-before-claim":     0.50,
		}, nil),
	}

	promoted, err := PromoteSkills([]SkillCandidate{good, bad}, l)
	if err != nil {
		t.Fatalf("PromoteSkills: %v", err)
	}
	if len(promoted) != 1 || promoted[0].Skill != "beats-corpus" {
		t.Fatalf("promoted set = %+v, want only beats-corpus", promoted)
	}

	// The reverted skill is journaled with the structured reason; the promoted one is
	// not (it entered the active set — there is nothing to revert).
	if r, gone := l.Why("regresses-corpus"); !gone || r.Kind != ReasonUnwitnessed {
		t.Fatalf("Why(regresses-corpus) = %+v gone=%v, want ReasonUnwitnessed/gone", r, gone)
	}
	if _, gone := l.Why("beats-corpus"); gone {
		t.Fatalf("a promoted skill must not be journaled as gone")
	}

	// A nil ledger still gates (journaling is the audit trail, not the gate).
	got, err := PromoteSkills([]SkillCandidate{good, bad}, nil)
	if err != nil {
		t.Fatalf("PromoteSkills nil-ledger: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("nil-ledger promoted %d, want 1", len(got))
	}
}
