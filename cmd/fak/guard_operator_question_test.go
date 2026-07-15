package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
	"github.com/anthony-chaudhary/fak/internal/operatorresolve"
	"github.com/anthony-chaudhary/fak/internal/planresolve"
)

func writeOperatorGateTranscript(t *testing.T, harness string, authority bool) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "transcript.jsonl")
	question := "How should I isolate this commit?"
	options := `[{"label":"Commit explicit owned paths","description":"owned files only"},{"label":"Wait","description":"wait for peers"}]`
	tool := "AskUserQuestion"
	input := `{"questions":[{"header":"Isolation","multiSelect":false,"question":"` + question + `","options":` + options + `}]}`
	if harness == "codex" {
		tool = "functions.request_user_input"
		input = `{"questions":[{"id":"isolation","header":"Isolation","question":"` + question + `","options":` + options + `}]}`
	}
	if authority {
		input = `{"questions":[{"header":"Priority","multiSelect":false,"question":"Which product priority should win?","options":[{"label":"Reliability","description":"policy choice"},{"label":"Launch","description":"release priority"}]}]}`
	}
	line := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"` + tool + `","input":` + input + `}]}}` + "\n"
	if err := os.WriteFile(path, []byte(line), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func withOperatorResolver(t *testing.T) {
	t.Helper()
	old := guardOperatorQuestionClarifyResolver
	guardOperatorQuestionClarifyResolver = operatorresolve.Resolver{Oracles: []operatorresolve.Oracle{operatorresolve.OracleFunc{
		OracleName: "git-readonly",
		InspectFn: func(_ context.Context, _ operatorquestion.OperatorQuestion, option operatorquestion.Option) (operatorresolve.Evidence, bool, error) {
			if strings.Contains(option.Label, "explicit owned paths") {
				return operatorresolve.Evidence{Claim: "path-scoped commit preserves peer dirt", Score: 10}, true, nil
			}
			return operatorresolve.Evidence{Claim: "no isolation gain", Score: 0}, true, nil
		},
	}}}
	t.Cleanup(func() { guardOperatorQuestionClarifyResolver = old })
}

func TestOperatorQuestionGateEnforceBlocksWithWitnessedAnswerForClaudeAndCodex(t *testing.T) {
	withOperatorResolver(t)
	for _, harness := range []string{"claude", "codex"} {
		t.Run(harness, func(t *testing.T) {
			var stderr bytes.Buffer
			exit, disp, gotHarness, fired := runGuardOperatorQuestionGate(&stderr, guardPreCompactModeEnforce, writeOperatorGateTranscript(t, harness, false))
			if !fired || exit != 0 || disp != stopDispOperatorQuestionResolved || gotHarness != harness {
				t.Fatalf("fired=%v exit=%d disp=%s harness=%s stderr=%s", fired, exit, disp, gotHarness, stderr.String())
			}
			if !strings.Contains(stderr.String(), "harness="+harness) || !strings.Contains(stderr.String(), "Commit explicit owned paths") {
				t.Fatalf("missing harness/action evidence: %s", stderr.String())
			}
		})
	}
}

func TestOperatorQuestionGateHumanResidualAllowsTypedEscalation(t *testing.T) {
	withOperatorResolver(t)
	var stderr bytes.Buffer
	exit, disp, _, fired := runGuardOperatorQuestionGate(&stderr, guardPreCompactModeEnforce, writeOperatorGateTranscript(t, "claude", true))
	if !fired || exit != 0 || disp != stopDispOperatorQuestionEscalate {
		t.Fatalf("fired=%v exit=%d disp=%s stderr=%s", fired, exit, disp, stderr.String())
	}
	if !strings.Contains(stderr.String(), string(choicetriage.HumanResidual)) && !strings.Contains(stderr.String(), "AUTHORITY_FORK") {
		t.Fatalf("missing typed escalation: %s", stderr.String())
	}
}

func TestOperatorQuestionGateResolvableFindingBlocks(t *testing.T) {
	old := guardOperatorQuestionClarifyResolver
	guardOperatorQuestionClarifyResolver = operatorresolve.Resolver{}
	t.Cleanup(func() { guardOperatorQuestionClarifyResolver = old })
	path := writeOperatorGateTranscript(t, "claude", false)
	exit, disp, _, fired := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeEnforce, path)
	if !fired || exit != 2 || disp != stopDispOperatorQuestionBlocked {
		t.Fatalf("fired=%v exit=%d disp=%s", fired, exit, disp)
	}
}

func TestOperatorQuestionGateWarnAndOffNeverActuate(t *testing.T) {
	withOperatorResolver(t)
	path := writeOperatorGateTranscript(t, "claude", false)
	if exit, disp, _, fired := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardOperatorDirectedModeWarn, path); !fired || exit != 0 || disp != stopDispOperatorQuestionResolved {
		t.Fatalf("warn fired=%v exit=%d disp=%s", fired, exit, disp)
	}
	if exit, _, _, fired := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeOff, path); fired || exit != 0 {
		t.Fatalf("off fired=%v exit=%d", fired, exit)
	}
	if got := guardOperatorDirectedEffectiveMode(guardPreCompactModeEnforce, true, true, false); got != guardOperatorDirectedModeWarn {
		t.Fatalf("attended enforce cap=%q", got)
	}
}

func TestOperatorQuestionPlanDispositionMapping(t *testing.T) {
	old := guardOperatorQuestionPlanResolver
	t.Cleanup(func() { guardOperatorQuestionPlanResolver = old })
	q := operatorquestion.OperatorQuestion{Kind: operatorquestion.PlanApproval, Harness: "codex", Plan: &operatorquestion.Plan{Steps: []operatorquestion.PlanStep{{Text: "inspect", Tool: "Read"}}}}
	guardOperatorQuestionPlanResolver = func(context.Context, operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Verdict{Disposition: planresolve.AutoApprove, Reason: planresolve.ReasonApproved}, nil
	}
	if disp, reason, _ := adjudicateOperatorQuestion(q); disp != stopDispOperatorQuestionResolved || reason != string(planresolve.ReasonApproved) {
		t.Fatalf("approved disp=%s reason=%s", disp, reason)
	}
	guardOperatorQuestionPlanResolver = func(context.Context, operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Verdict{Disposition: planresolve.AutoRefuse, Reason: planresolve.ReasonTreeCollision, Appeal: "fak complain"}, nil
	}
	if disp, reason, _ := adjudicateOperatorQuestion(q); disp != stopDispOperatorQuestionBlocked || reason != string(planresolve.ReasonTreeCollision) {
		t.Fatalf("refused disp=%s reason=%s", disp, reason)
	}
	guardOperatorQuestionPlanResolver = func(context.Context, operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
		return planresolve.Verdict{Disposition: planresolve.Escalate, Reason: planresolve.ReasonIrreversibleUnwitnessed}, nil
	}
	if disp, _, _ := adjudicateOperatorQuestion(q); disp != stopDispOperatorQuestionEscalate {
		t.Fatalf("escalated disp=%s", disp)
	}
}
