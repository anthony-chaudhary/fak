package guardaccuracy

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestSeedCorpusIsHealthyAndBalanced pins the baseline: the real classifier
// classifies every seed row correctly (0 FP, 0 FN) and the corpus carries both
// polarities well above the non-vacuous floor. If a change to the reversibility
// classifier regresses any seeded case, this test AND the scorecard go red.
func TestSeedCorpusIsHealthyAndBalanced(t *testing.T) {
	rows := SeedCorpus()
	if len(rows) == 0 {
		t.Fatal("seed corpus is empty")
	}
	res := Fold(rows)
	if len(res.FalsePositive) != 0 {
		t.Fatalf("seed corpus has false positives (classifier over-blocked): %v", res.FalsePositive)
	}
	if len(res.FalseNegative) != 0 {
		t.Fatalf("seed corpus has false negatives (classifier under-blocked): %v", res.FalseNegative)
	}
	if res.BenignRows < minPolarity || res.DangerousRows < minPolarity {
		t.Fatalf("seed corpus not balanced: %d benign, %d dangerous (need >= %d each)", res.BenignRows, res.DangerousRows, minPolarity)
	}
	if res.Correct != res.Total {
		t.Fatalf("seed corpus correct=%d != total=%d", res.Correct, res.Total)
	}

	p := BuildScorecard("")
	if !p.OK {
		t.Fatalf("seed scorecard should be OK, got verdict=%q reason=%q", p.Verdict, p.Reason)
	}
	if p.Corpus[DebtKey] != 0 {
		t.Fatalf("seed scorecard debt = %v, want 0", p.Corpus[DebtKey])
	}
}

// TestFoldDetectsFalsePositiveAndNegative is the non-vacuity witness: it injects
// one row whose benign expectation the classifier escalates (a false positive)
// and one row whose dangerous expectation the classifier leaves reversible (a
// false negative), and proves BOTH surface as debt. A rubber-stamp fold that
// always returns OK cannot pass this.
func TestFoldDetectsFalsePositiveAndNegative(t *testing.T) {
	rows := append([]CorpusRow{}, SeedCorpus()...)
	rows = append(rows,
		CorpusRow{ // real push, but MIS-labeled benign -> the guard escalates -> FP
			Name: "injected-fp", Tool: "Bash",
			Args:   map[string]any{"command": "git push origin main"},
			Expect: "reversible",
		},
		CorpusRow{ // read-only, but MIS-labeled dangerous -> the guard passes it -> FN
			Name: "injected-fn", Tool: "Bash",
			Args:   map[string]any{"command": "go test ./..."},
			Expect: "outward-facing",
		},
	)
	res := Fold(rows)
	if len(res.FalsePositive) != 1 {
		t.Fatalf("want exactly 1 false positive, got %d: %v", len(res.FalsePositive), res.FalsePositive)
	}
	if len(res.FalseNegative) != 1 {
		t.Fatalf("want exactly 1 false negative, got %d: %v", len(res.FalseNegative), res.FalseNegative)
	}

	p := BuildScorecardFromRows("", rows)
	if p.OK {
		t.Fatal("scorecard must not be OK when a FP and FN are present")
	}
	if p.Finding != "guard_accuracy_debt" {
		t.Fatalf("finding = %q, want guard_accuracy_debt", p.Finding)
	}
	// A false negative is the heaviest defect: it must lead the worst-first action.
	if !strings.Contains(p.NextAction, "no_false_negatives") {
		t.Fatalf("worst-first action should name no_false_negatives, got %q", p.NextAction)
	}
	debt, _ := p.Corpus[DebtKey].(int)
	if debt < 2 {
		t.Fatalf("debt = %v, want >= 2 (one FP + one FN)", p.Corpus[DebtKey])
	}
}

// TestVacuousCorpusRefusesToCertify proves the honesty gate: a corpus missing a
// polarity cannot exhibit one of the two failure modes, so a green rate is not
// evidence and the fold refuses it.
func TestVacuousCorpusRefusesToCertify(t *testing.T) {
	benignOnly := []CorpusRow{
		{Name: "a", Tool: "Bash", Args: map[string]any{"command": "go test ./..."}, Expect: "reversible"},
		{Name: "b", Tool: "Bash", Args: map[string]any{"command": "ls -la"}, Expect: "reversible"},
	}
	p := BuildScorecardFromRows("", benignOnly)
	if p.OK {
		t.Fatal("a single-polarity corpus must not certify OK")
	}
	if !strings.Contains(p.NextAction, "corpus_nonvacuous") {
		t.Fatalf("vacuous corpus should name corpus_nonvacuous first, got %q", p.NextAction)
	}
}

// TestBuildScorecardIsDeterministic pins the same-corpus-same-payload contract:
// the payload is a pure function of the corpus + the classifier.
func TestBuildScorecardIsDeterministic(t *testing.T) {
	a, err := json.Marshal(BuildScorecard("/repo"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(BuildScorecard("/repo"))
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(b) {
		t.Fatalf("BuildScorecard is not deterministic:\n%s\n%s", a, b)
	}
}
