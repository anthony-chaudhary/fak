package operatorresolve

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

type benchRunner struct{}

func (benchRunner) Run(_ context.Context, _ string, args ...string) ([]byte, error) {
	if len(args) > 0 && args[0] == "status" {
		return []byte(" M internal/operatorresolve/operatorresolve.go\n"), nil
	}
	if len(args) > 0 && args[0] == "log" {
		return []byte("abc1234 prior commit\n"), nil
	}
	if len(args) >= 3 && args[0] == "ls-files" && args[1] == "--" {
		return []byte(args[2] + "\n"), nil
	}
	return nil, nil
}

// BenchmarkOperatorResolve measures end-to-end question resolution throughput
// across multiple oracles and candidate options.
func BenchmarkOperatorResolve(b *testing.B) {
	runner := benchRunner{}
	resolver := Resolver{
		Oracles: []Oracle{
			GitIsolationOracle{Runner: runner},
			TrackedArtifactOracle{Runner: runner},
		},
	}
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "How should I isolate this commit from peer edits?",
		Options: []operatorquestion.Option{
			{Label: "Commit explicit owned paths", Rationale: "leave peer-dirty files untouched"},
			{Label: "Wait for a clean tree", Rationale: "delay until peers finish"},
			{Label: "Rebase onto origin/main", Rationale: "align branch state"},
		},
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verdict, err := resolver.Resolve(ctx, q)
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if verdict.Disposition == "" {
			b.Fatal("unexpected empty disposition")
		}
	}
}

// BenchmarkScorecardAxisOracle measures Pareto dominance evaluation across
// multiple scorecard dimensions and competing options.
func BenchmarkScorecardAxisOracle(b *testing.B) {
	oracle := ScorecardAxisOracle{
		Axes: []ScorecardAxis{
			{Name: "quality-score", Scores: map[string]float64{"OptionA": 95, "OptionB": 80, "OptionC": 70}, Witness: "quality-ledger@r10"},
			{Name: "operator-heaviness-score", Scores: map[string]float64{"OptionA": 2, "OptionB": 5, "OptionC": 8}, LowerIsBetter: true, Witness: "heaviness-ledger@r4"},
			{Name: "steerability-score", Scores: map[string]float64{"OptionA": 90, "OptionB": 85, "OptionC": 60}, Witness: "steerability-ledger@r8"},
		},
		Reversible: map[string]bool{"OptionA": true, "OptionB": true, "OptionC": true},
	}
	q := operatorquestion.OperatorQuestion{
		Kind: operatorquestion.ChooseApproach,
		Options: []operatorquestion.Option{
			{Label: "OptionA"},
			{Label: "OptionB"},
			{Label: "OptionC"},
		},
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ev, ok, err := oracle.Inspect(ctx, q, q.Options[0])
		if err != nil {
			b.Fatalf("Inspect failed: %v", err)
		}
		if !ok || ev.Score != 100 {
			b.Fatalf("unexpected evidence: ok=%v score=%d", ok, ev.Score)
		}
	}
}

// BenchmarkTrackedArtifactOracle measures path extraction and git tracking verification
// for create-vs-edit option framing.
func BenchmarkTrackedArtifactOracle(b *testing.B) {
	oracle := TrackedArtifactOracle{Runner: benchRunner{}}
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "Should I create a new file or edit the existing one?",
		Options: []operatorquestion.Option{
			{Label: "Edit `internal/operatorresolve/operatorresolve.go`", Rationale: "extend existing"},
			{Label: "Create `internal/operatorresolve/operatorresolve_v2.go`", Rationale: "new file"},
		},
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ev, ok, err := oracle.Inspect(ctx, q, q.Options[0])
		if err != nil {
			b.Fatalf("Inspect failed: %v", err)
		}
		if !ok || ev.Score == 0 {
			b.Fatalf("unexpected inspect result: ok=%v score=%d", ok, ev.Score)
		}
	}
}

// BenchmarkGitIsolationOracle measures inspect throughput for peer-dirty commit isolation.
func BenchmarkGitIsolationOracle(b *testing.B) {
	oracle := GitIsolationOracle{Runner: benchRunner{}}
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "How should I isolate this commit?",
		Options: []operatorquestion.Option{
			{Label: "Commit explicit owned paths", Rationale: "owned paths"},
			{Label: "Wait", Rationale: "wait"},
		},
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ev, ok, err := oracle.Inspect(ctx, q, q.Options[0])
		if err != nil {
			b.Fatalf("Inspect failed: %v", err)
		}
		if !ok || ev.Score == 0 {
			b.Fatalf("unexpected inspect result: ok=%v score=%d", ok, ev.Score)
		}
	}
}

// BenchmarkAuthorityFork measures detection latency for policy and authority escalations.
func BenchmarkAuthorityFork(b *testing.B) {
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "Which product priority and release policy should guide this decision?",
		Options: []operatorquestion.Option{
			{Label: "Maintain backward compatibility", Rationale: "release stability"},
			{Label: "Deprecate legacy interfaces", Rationale: "cleaner architecture"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !authorityFork(q) {
			b.Fatal("expected authority fork to return true")
		}
	}
}
