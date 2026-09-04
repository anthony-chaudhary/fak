package planresolve

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

type staticOracles struct {
	tree      OracleResult
	direction OracleResult
	done      OracleResult
}

func (s staticOracles) TreeDisjoint(context.Context, []string) (OracleResult, error) {
	return s.tree, nil
}

func (s staticOracles) DirectionAllowed(context.Context, []string) (OracleResult, error) {
	return s.direction, nil
}

func (s staticOracles) DoneVerifiable(context.Context, string) (OracleResult, error) {
	return s.done, nil
}

func BenchmarkPlanResolve(b *testing.B) {
	ctx := context.Background()
	q := operatorquestion.OperatorQuestion{
		Kind: operatorquestion.PlanApproval,
		Plan: &operatorquestion.Plan{
			FileTree:      []string{"internal/planresolve/**", "internal/operatorquestion/**"},
			DoneCriterion: "dos verify plans/operator.md phase-1",
			Steps: []operatorquestion.PlanStep{
				{Text: "inspect policy", Tool: "Read", Args: map[string]any{"path": "policy.json"}},
				{Text: "check schema", Tool: "Read", Args: map[string]any{"path": "schema.json"}},
			},
		},
	}
	oracles := staticOracles{
		tree:      OracleResult{OK: true, Witness: "dos arbitrate: DISJOINT"},
		direction: OracleResult{OK: true, Witness: "architest: direction allowed"},
		done:      OracleResult{OK: true, Witness: "dos verify: criterion parseable"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Resolve(ctx, q, oracles)
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if v.Disposition != AutoApprove {
			b.Fatalf("unexpected disposition: %s", v.Disposition)
		}
	}
}

func BenchmarkPlanResolveTreeCollision(b *testing.B) {
	ctx := context.Background()
	q := operatorquestion.OperatorQuestion{
		Kind: operatorquestion.PlanApproval,
		Plan: &operatorquestion.Plan{
			FileTree:      []string{"internal/planresolve/**"},
			DoneCriterion: "dos verify plans/operator.md phase-1",
			Steps: []operatorquestion.PlanStep{
				{Text: "inspect policy", Tool: "Read", Args: map[string]any{"path": "policy.json"}},
			},
		},
	}
	oracles := staticOracles{
		tree: OracleResult{OK: false, Reason: ReasonTreeCollision, Witness: "lease conflict"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Resolve(ctx, q, oracles)
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if v.Disposition != AutoRefuse || v.Reason != ReasonTreeCollision {
			b.Fatalf("unexpected verdict: %+v", v)
		}
	}
}

func BenchmarkPlanResolveIrreversibleEscalate(b *testing.B) {
	ctx := context.Background()
	q := operatorquestion.OperatorQuestion{
		Kind: operatorquestion.PlanApproval,
		Plan: &operatorquestion.Plan{
			FileTree:      []string{"internal/planresolve/**"},
			DoneCriterion: "dos verify plans/operator.md phase-1",
			Steps: []operatorquestion.PlanStep{
				{Text: "deploy change", Tool: "Bash", Args: map[string]any{"command": "git push"}},
			},
		},
	}
	oracles := staticOracles{
		tree:      OracleResult{OK: true, Witness: "dos arbitrate: DISJOINT"},
		direction: OracleResult{OK: true, Witness: "architest: direction allowed"},
		done:      OracleResult{OK: true, Witness: "dos verify: criterion parseable"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Resolve(ctx, q, oracles)
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if v.Disposition != Escalate || v.Reason != ReasonIrreversibleUnwitnessed {
			b.Fatalf("unexpected verdict: %+v", v)
		}
	}
}

func BenchmarkPlanResolveMultiStep(b *testing.B) {
	ctx := context.Background()
	steps := make([]operatorquestion.PlanStep, 20)
	for i := 0; i < len(steps); i++ {
		if i%2 == 0 {
			steps[i] = operatorquestion.PlanStep{
				Text: fmt.Sprintf("read step %d", i),
				Tool: "Read",
				Args: map[string]any{"path": fmt.Sprintf("file_%d.go", i)},
			}
		} else {
			steps[i] = operatorquestion.PlanStep{
				Text:    fmt.Sprintf("mutate step %d", i),
				Tool:    "Bash",
				Args:    map[string]any{"command": fmt.Sprintf("make step_%d", i)},
				Witness: fmt.Sprintf("witness token #%d", i),
			}
		}
	}

	q := operatorquestion.OperatorQuestion{
		Kind: operatorquestion.PlanApproval,
		Plan: &operatorquestion.Plan{
			FileTree:      []string{"internal/planresolve/**"},
			DoneCriterion: "dos verify plans/operator.md phase-1",
			Steps:         steps,
		},
	}
	oracles := staticOracles{
		tree:      OracleResult{OK: true, Witness: "dos arbitrate: DISJOINT"},
		direction: OracleResult{OK: true, Witness: "architest: direction allowed"},
		done:      OracleResult{OK: true, Witness: "dos verify: criterion parseable"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Resolve(ctx, q, oracles)
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if v.Disposition != AutoApprove || len(v.Steps) != len(steps) {
			b.Fatalf("unexpected verdict: %+v", v)
		}
	}
}

func BenchmarkPlanResolveParallel(b *testing.B) {
	ctx := context.Background()
	q := operatorquestion.OperatorQuestion{
		Kind: operatorquestion.PlanApproval,
		Plan: &operatorquestion.Plan{
			FileTree:      []string{"internal/planresolve/**"},
			DoneCriterion: "dos verify plans/operator.md phase-1",
			Steps: []operatorquestion.PlanStep{
				{Text: "read policy", Tool: "Read", Args: map[string]any{"path": "policy.json"}},
			},
		},
	}
	oracles := staticOracles{
		tree:      OracleResult{OK: true, Witness: "dos arbitrate: DISJOINT"},
		direction: OracleResult{OK: true, Witness: "architest: direction allowed"},
		done:      OracleResult{OK: true, Witness: "dos verify: criterion parseable"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v, err := Resolve(ctx, q, oracles)
			if err != nil {
				b.Fatalf("Resolve failed: %v", err)
			}
			if v.Disposition != AutoApprove {
				b.Fatalf("unexpected disposition: %s", v.Disposition)
			}
		}
	})
}
