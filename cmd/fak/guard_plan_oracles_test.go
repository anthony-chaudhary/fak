package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
	"github.com/anthony-chaudhary/fak/internal/planresolve"
)

type planRunnerStub struct {
	output []byte
	err    error
	calls  [][]string
}

func (s *planRunnerStub) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	s.calls = append(s.calls, append([]string{name}, args...))
	return s.output, s.err
}

func TestGuardPlanTreeOracleUsesReadOnlyDOSArbitration(t *testing.T) {
	stub := &planRunnerStub{output: []byte(`{"outcome":"refuse","reason":"overlaps live lane"}`), err: errors.New("exit status 1")}
	got, err := (guardPlanOracleSet{runner: stub}).TreeDisjoint(context.Background(), []string{"cmd/fak/**"})
	if err != nil || got.OK || got.Reason != planresolve.ReasonTreeCollision {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	call := strings.Join(stub.calls[0], " ")
	if !strings.Contains(call, "dos arbitrate") || strings.Contains(call, "--force") {
		t.Fatalf("call=%q", call)
	}
}

func TestGuardPlanDirectionCoreLockRefusesFrozenABIWithoutCommand(t *testing.T) {
	stub := &planRunnerStub{}
	got, err := (guardPlanOracleSet{runner: stub}).DirectionAllowed(context.Background(), []string{"internal/abi/**"})
	if err != nil || got.OK || got.Reason != planresolve.ReasonWrongDirection {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if len(stub.calls) != 0 {
		t.Fatalf("core lock invoked commands: %v", stub.calls)
	}
}

func TestGuardPlanDoneOracleRequiresRegisteredWitness(t *testing.T) {
	oracle := guardPlanOracleSet{runner: &planRunnerStub{}}
	ok, err := oracle.DoneVerifiable(context.Background(), "go test ./internal/policy -run TestFloor")
	if err != nil || !ok.OK {
		t.Fatalf("registered=%+v err=%v", ok, err)
	}
	no, err := oracle.DoneVerifiable(context.Background(), "looks correct")
	if err != nil || no.OK || no.Reason != planresolve.ReasonDoneUnverifiable {
		t.Fatalf("unregistered=%+v err=%v", no, err)
	}
}

func TestGuardPlanResolverInstalledEndToEnd(t *testing.T) {
	old := guardOperatorQuestionPlanResolver
	t.Cleanup(func() { guardOperatorQuestionPlanResolver = old })
	if old == nil {
		t.Fatal("production plan resolver is nil")
	}
	guardOperatorQuestionPlanResolver = func(ctx context.Context, q operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Resolve(ctx, q, guardPlanOracleSet{runner: &scriptedPlanRunner{}})
	}
	path := writePlanTranscript(t, operatorquestion.Plan{
		FileTree:      []string{"internal/policy/**"},
		Steps:         []operatorquestion.PlanStep{{Text: "edit policy", Tool: "write", Args: map[string]any{"path": "internal/policy/x.go"}}},
		DoneCriterion: "go test ./internal/policy",
	})
	var stderr strings.Builder
	_, disp, _, ran := runGuardOperatorQuestionGate(&stderr, "enforce", path)
	if !ran || disp != stopDispOperatorQuestionResolved || !strings.Contains(stderr.String(), string(planresolve.ReasonApproved)) {
		t.Fatalf("ran=%v disp=%v stderr=%q", ran, disp, stderr.String())
	}
}

func TestGuardPlanResolverCollidingPlanRefusesEndToEnd(t *testing.T) {
	old := guardOperatorQuestionPlanResolver
	t.Cleanup(func() { guardOperatorQuestionPlanResolver = old })
	guardOperatorQuestionPlanResolver = func(ctx context.Context, q operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Resolve(ctx, q, guardPlanOracleSet{runner: &collisionPlanRunner{}})
	}
	path := writePlanTranscript(t, operatorquestion.Plan{
		FileTree:      []string{"cmd/fak/**"},
		Steps:         []operatorquestion.PlanStep{{Text: "edit guard", Tool: "write", Args: map[string]any{"path": "cmd/fak/guard.go"}}},
		DoneCriterion: "go test ./cmd/fak",
	})
	var stderr strings.Builder
	_, disp, _, ran := runGuardOperatorQuestionGate(&stderr, "enforce", path)
	if !ran || disp != stopDispOperatorQuestionBlocked || !strings.Contains(stderr.String(), string(planresolve.ReasonTreeCollision)) {
		t.Fatalf("ran=%v disp=%v stderr=%q", ran, disp, stderr.String())
	}
}

type collisionPlanRunner struct{}

func (*collisionPlanRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	if name == "dos" {
		return []byte(`{"outcome":"refuse","reason":"overlaps live cmd lease"}`), errors.New("exit status 1")
	}
	return []byte("ok"), nil
}

func TestGuardPlanOracleErrorFailsOpenToOperator(t *testing.T) {
	old := guardOperatorQuestionPlanResolver
	t.Cleanup(func() { guardOperatorQuestionPlanResolver = old })
	guardOperatorQuestionPlanResolver = func(context.Context, operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Verdict{}, errors.New("oracle transport unavailable")
	}
	path := writePlanTranscript(t, operatorquestion.Plan{
		FileTree: []string{"internal/policy/**"}, Steps: []operatorquestion.PlanStep{{Text: "edit", Tool: "write"}},
		DoneCriterion: "go test ./internal/policy",
	})
	var stderr strings.Builder
	_, disp, _, ran := runGuardOperatorQuestionGate(&stderr, "enforce", path)
	if !ran || disp != stopDispOperatorQuestionEscalate || !strings.Contains(stderr.String(), "PLAN_ORACLE_ERROR") {
		t.Fatalf("ran=%v disp=%v stderr=%q", ran, disp, stderr.String())
	}
}

type scriptedPlanRunner struct{}

func (*scriptedPlanRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	if name == "dos" {
		return []byte(`{"outcome":"acquire","reason":"disjoint"}`), nil
	}
	return []byte("ok"), nil
}

func TestGuardPlanResolverEscalatesOnlyIrreversibleUnwitnessedStep(t *testing.T) {
	old := guardOperatorQuestionPlanResolver
	t.Cleanup(func() { guardOperatorQuestionPlanResolver = old })
	guardOperatorQuestionPlanResolver = func(ctx context.Context, q operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Resolve(ctx, q, guardPlanOracleSet{runner: &scriptedPlanRunner{}})
	}
	path := writePlanTranscript(t, operatorquestion.Plan{FileTree: []string{"internal/policy/**"}, Steps: []operatorquestion.PlanStep{
		{Text: "edit", Tool: "write", Args: map[string]any{"path": "internal/policy/x.go"}},
		{Text: "delete remote", Tool: "delete_record", Args: map[string]any{"id": "1"}},
	}, DoneCriterion: "go test ./internal/policy"})
	var stderr strings.Builder
	_, disp, _, ran := runGuardOperatorQuestionGate(&stderr, "enforce", path)
	if !ran || disp != stopDispOperatorQuestionEscalate || !strings.Contains(stderr.String(), string(planresolve.ReasonIrreversibleUnwitnessed)) {
		t.Fatalf("ran=%v disp=%v stderr=%q", ran, disp, stderr.String())
	}
}

func writePlanTranscript(t *testing.T, plan operatorquestion.Plan) string {
	t.Helper()
	payload := map[string]any{
		"plan":           "structured plan",
		"file_tree":      plan.FileTree,
		"steps":          plan.Steps,
		"done_criterion": plan.DoneCriterion,
	}
	rec := map[string]any{"type": "assistant", "message": map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "tool_use", "name": "ExitPlanMode", "input": payload}}}}
	line, _ := json.Marshal(rec)
	path := filepath.Join(t.TempDir(), "transcript.jsonl")
	if err := os.WriteFile(path, append(line, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
