package workerworktree

import "testing"

func TestVerifyAppliedDisambiguationRejectsStaleBaseCollision(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()
	calls := 0
	readDisambiguation = func(repo, tree string) DisambiguationWitness {
		calls++
		w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, FamilyCoverage: map[string]float64{"loop": 100}}
		if calls == 3 {
			w.SemanticValid = false
			w.CriticalClean = false
			w.Detail = "duplicate canonical row collision"
		}
		return w
	}
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if ok || got.PostApply.SemanticValid || got.PostApply.Detail == "" {
		t.Fatalf("collision accepted: %+v", got)
	}
}
func TestVerifyAppliedDisambiguationRejectsConcurrentCorpusGap(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()
	calls := 0
	readDisambiguation = func(repo, tree string) DisambiguationWitness {
		calls++
		w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, CoverageDebt: 0, FamilyCoverage: map[string]float64{"loop": 100}}
		if calls == 3 {
			w.Coverage = 99
			w.CoverageDebt = 1
			w.FamilyCoverage["loop"] = 90
			w.Detail = "loop: newtoken unpositioned"
		}
		return w
	}
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if ok || got.PostApply.CoverageDebt != 1 {
		t.Fatalf("gap accepted: %+v", got)
	}
}
func TestVerifyAppliedDisambiguationRecordsThreeWitnesses(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()
	readDisambiguation = func(repo, tree string) DisambiguationWitness {
		return DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, FamilyCoverage: map[string]float64{"loop": 100}}
	}
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if !ok || got.Before.Tree != "HEAD" || got.Worktree.Tree != "HEAD" || got.PostApply.Tree != "candidate" {
		t.Fatalf("witnesses=%+v ok=%v", got, ok)
	}
}

// TestVerifyAppliedDisambiguationFreshnessIsNonRegression pins the #5359 fix: freshness gates
// land as a NON-REGRESSION, not an absolute bar. When HEAD is already concept-stale (a peer
// left a generated artifact un-regenerated), a diff that regresses nothing must still land even
// though its post tree is also stale; only turning a fresh HEAD stale is refused. The stub is
// called in order before/worktree/post, so call 1 sets before.Fresh and call 3 sets post.Fresh
// while every other witness field stays clean and identical (no coverage regression), isolating
// freshness as the only variable.
func TestVerifyAppliedDisambiguationFreshnessIsNonRegression(t *testing.T) {
	cases := []struct {
		name        string
		beforeFresh bool
		postFresh   bool
		wantOK      bool
	}{
		{"stale HEAD, still stale post -> admitted", false, false, true},
		{"stale HEAD, fresh post -> admitted", false, true, true},
		{"fresh HEAD, fresh post -> admitted", true, true, true},
		{"fresh HEAD, stale post -> refused (regression)", true, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			old := readDisambiguation
			defer func() { readDisambiguation = old }()
			calls := 0
			readDisambiguation = func(repo, tree string) DisambiguationWitness {
				calls++
				w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, CoverageDebt: 0, FamilyCoverage: map[string]float64{"loop": 100}}
				if calls == 1 {
					w.Fresh = tc.beforeFresh
				}
				if calls == 3 {
					w.Fresh = tc.postFresh
				}
				return w
			}
			_, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
			if ok != tc.wantOK {
				t.Fatalf("freshness non-regression: got ok=%v want %v (before.Fresh=%v post.Fresh=%v)", ok, tc.wantOK, tc.beforeFresh, tc.postFresh)
			}
		})
	}
}
