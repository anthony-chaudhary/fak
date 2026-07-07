package rsiloop

import "testing"

// TestDefaultSkillFixtureCorpusGatesPromotion exercises the frozen held benchmark
// end to end through PromoteSkill: a skill that lifts every held task above baseline
// is promoted, one that regresses the corpus is refused with ReasonUnwitnessed, and
// an un-run candidate (no measured scores) cannot win — the same honesty property,
// now against the committed fixture corpus rather than ad-hoc inline data.
func TestDefaultSkillFixtureCorpusGatesPromotion(t *testing.T) {
	base := DefaultSkillFixtureCorpus()
	if len(base.Tasks) == 0 {
		t.Fatalf("default fixture corpus is empty")
	}

	// A skill that beats baseline on every held task, suite-green + truth-clean.
	improved := base.WithCandidateScores(map[string]float64{
		"grep-scoped-symbol":       0.90,
		"refactor-preserves-tests": 0.92,
		"commit-by-explicit-path":  0.88,
		"witness-before-claim":     0.95,
	}, nil)
	if got := PromoteSkill(SkillCandidate{Skill: "beats-corpus", Corpus: improved, TruthClean: true}); !got.Promoted {
		t.Fatalf("a skill beating every held task was not promoted (before=%v after=%v)", got.Witness.Before, got.Witness.After)
	}

	// A skill that regresses the corpus is refused as unwitnessed.
	regressed := base.WithCandidateScores(map[string]float64{
		"grep-scoped-symbol":       0.40,
		"refactor-preserves-tests": 0.50,
		"commit-by-explicit-path":  0.30,
		"witness-before-claim":     0.60,
	}, nil)
	got := PromoteSkill(SkillCandidate{Skill: "regresses-corpus", Corpus: regressed, TruthClean: true})
	if got.Promoted || got.Reason.Kind != ReasonUnwitnessed {
		t.Fatalf("a corpus-regressing skill: promoted=%v reason=%+v, want refused/ReasonUnwitnessed", got.Promoted, got.Reason)
	}

	// An un-run candidate (no candidate scores) cannot promote: unmeasured -> no gain.
	if got := PromoteSkill(SkillCandidate{Skill: "unrun", Corpus: base.WithCandidateScores(nil, nil), TruthClean: true}); got.Promoted {
		t.Fatalf("an un-run candidate (CandidateScore 0) was promoted; unmeasured must not win")
	}
}
