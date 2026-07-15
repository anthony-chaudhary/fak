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
