package main

import (
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
	"github.com/anthony-chaudhary/fak/internal/operatorresolve"
	"github.com/anthony-chaudhary/fak/internal/planresolve"
)

const (
	stopDispOperatorQuestionResolved guardStopDisposition = "OPERATOR_QUESTION_RESOLVED"
	stopDispOperatorQuestionBlocked  guardStopDisposition = "OPERATOR_QUESTION_BLOCKED"
	stopDispOperatorQuestionEscalate guardStopDisposition = "OPERATOR_QUESTION_HUMAN_RESIDUAL"
)

type commandReadOnlyRunner struct{}

func (commandReadOnlyRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

var guardOperatorQuestionClarifyResolver = operatorresolve.Resolver{
	Oracles: []operatorresolve.Oracle{operatorresolve.GitIsolationOracle{Runner: commandReadOnlyRunner{}}},
}

// guardOperatorQuestionPlanResolver is installed in production, not only tests. The
// concrete oracle set lives beside the guard so PLAN_APPROVAL cannot silently fall back
// to PLAN_ORACLES_UNAVAILABLE in a real session.
var guardOperatorQuestionPlanResolver = resolveGuardOperatorQuestionPlan

func runGuardOperatorQuestionGate(stderr io.Writer, rawMode, transcriptPath string) (int, guardStopDisposition, string, bool) {
	mode, err := normalizeGuardOperatorDirectedMode(rawMode)
	if err != nil || mode == guardPreCompactModeOff {
		return 0, "", "", false
	}
	q, found, err := operatorquestion.LastFromTranscriptAny(transcriptPath)
	if err != nil || !found {
		return 0, "", "", false
	}
	disp, reason, action := adjudicateOperatorQuestion(q)
	fmt.Fprintf(stderr, "fak guard Stop: operator-question harness=%s kind=%s disposition=%s reason=%s action=%s\n", q.Harness, q.Kind, disp, reason, action)
	if mode != guardPreCompactModeEnforce {
		return 0, disp, q.Harness, true
	}
	switch disp {
	case stopDispOperatorQuestionResolved:
		fmt.Fprintf(stderr, "fak guard Stop: auto-resolved from witness: %s\n", action)
		return 0, disp, q.Harness, true
	case stopDispOperatorQuestionBlocked:
		return 2, disp, q.Harness, true
	default:
		return 0, disp, q.Harness, true
	}
}

func adjudicateOperatorQuestion(q operatorquestion.OperatorQuestion) (guardStopDisposition, string, string) {
	switch q.Kind {
	case operatorquestion.Clarify, operatorquestion.ChooseApproach:
		v, err := guardOperatorQuestionClarifyResolver.Resolve(context.Background(), q)
		if err != nil {
			return stopDispOperatorQuestionBlocked, "ORACLE_ERROR", "investigate the resolver error in a fresh context"
		}
		switch v.Disposition {
		case choicetriage.TakeObvious:
			return stopDispOperatorQuestionResolved, string(v.Reason), v.Action
		case choicetriage.HumanResidual:
			return stopDispOperatorQuestionEscalate, string(v.Reason), v.Action
		default:
			return stopDispOperatorQuestionBlocked, string(v.Reason), v.Action
		}
	case operatorquestion.PlanApproval:
		if guardOperatorQuestionPlanResolver == nil {
			return stopDispOperatorQuestionEscalate, "PLAN_ORACLES_UNAVAILABLE", "route the structured plan to the plan-content oracle set"
		}
		v, err := guardOperatorQuestionPlanResolver(context.Background(), q)
		if err != nil {
			return stopDispOperatorQuestionEscalate, "PLAN_ORACLE_ERROR", err.Error()
		}
		switch v.Disposition {
		case planresolve.AutoApprove:
			return stopDispOperatorQuestionResolved, string(v.Reason), "proceed with the oracle-approved plan"
		case planresolve.AutoRefuse:
			return stopDispOperatorQuestionBlocked, string(v.Reason), v.Appeal
		default:
			return stopDispOperatorQuestionEscalate, string(v.Reason), strings.TrimSpace(v.Appeal)
		}
	default:
		return stopDispOperatorQuestionEscalate, "HUMAN_RESIDUAL", "route the authority question to the operator"
	}
}
