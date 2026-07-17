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

// guardStopHookOperatorQuestionEnvMode is the resolved evidence-first operator-question gate mode the
// guard installer injects into the Stop-hook child. It is the DEDICATED dial for the native
// ExitPlanMode/AskUserQuestion adjudication rung (runGuardOperatorQuestionGate) — plan-approval and
// clarify/choose-approach redirection — split out from the linguistic --operator-directed sensor so
// an operator can tune the two rungs independently. Like the sibling hardware-gate it INHERITS the
// resolved operator-directed posture as its install-time default (both are operator-absence-capped
// headless gates: guardOperatorDirectedEffectiveMode has already applied the attended cap), so a
// session that tunes neither is byte-for-byte unchanged from before the dial split out. An operator
// who wants the plan/question rung on a different posture than the prose sensor sets
// FAK_GUARD_OPERATOR_QUESTION_MODE / --operator-question on the child.
const guardStopHookOperatorQuestionEnvMode = "FAK_GUARD_OPERATOR_QUESTION_MODE"

// normalizeGuardOperatorQuestionMode canonicalizes the operator-question gate mode, mirroring
// normalizeGuardOperatorDirectedMode: empty/whitespace defaults to warn (the soak default),
// off/shadow/enforce reuse the guardPreCompactMode* vocabulary, and an unknown value is an error the
// caller treats as fail-open (a bad mode string must never block a stop).
func normalizeGuardOperatorQuestionMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardOperatorDirectedModeWarn:
		return guardOperatorDirectedModeWarn, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --operator-question mode %q (want off, shadow, warn, or enforce)", mode)
	}
}

// guardOperatorQuestionNormalizedOrWarn is the total form of normalizeGuardOperatorQuestionMode for
// the env-injection boundary: it never returns an error, pinning a bad string to the warn soak
// default rather than accidentally to enforce.
func guardOperatorQuestionNormalizedOrWarn(mode string) string {
	normalized, err := normalizeGuardOperatorQuestionMode(mode)
	if err != nil {
		return guardOperatorDirectedModeWarn
	}
	return normalized
}

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

func runGuardOperatorQuestionGate(stderr io.Writer, rawMode, transcriptPath, sessionID string, answerOut ...*string) (int, guardStopDisposition, string, bool) {
	mode, err := normalizeGuardOperatorQuestionMode(rawMode)
	if err != nil || mode == guardPreCompactModeOff {
		return 0, "", "", false
	}
	q, found, err := operatorquestion.LastFromTranscriptAny(transcriptPath)
	if err != nil || !found {
		return 0, "", "", false
	}
	disp, reason, action := adjudicateOperatorQuestion(q)
	if len(answerOut) > 0 && answerOut[0] != nil {
		*answerOut[0] = action
	}
	fmt.Fprintf(stderr, "fak guard Stop: operator-question harness=%s kind=%s disposition=%s reason=%s action=%s\n", q.Harness, q.Kind, disp, reason, action)
	// FLAG THE OPERATOR on a genuine HUMAN_RESIDUAL — a structured native question no oracle could
	// auto-resolve (authority fork, plan-approval escalate, oracle-unavailable). Without this the
	// residual was left as the stderr prose above and never picked up by anyone. Active in warn and
	// enforce; silent in shadow (pure observe) and off (returned above). Best-effort / fail-open: the
	// sink no-ops when there is no session thread to post to, so it can never block the stop.
	if disp == stopDispOperatorQuestionEscalate && mode != guardPreCompactModeShadow {
		guardOperatorQuestionEscalationSink(sessionID, q, reason, action)
	}
	if mode != guardPreCompactModeEnforce {
		return 0, disp, q.Harness, true
	}
	switch disp {
	case stopDispOperatorQuestionResolved:
		fmt.Fprintf(stderr, "fak guard Stop: auto-resolved from witness: %s\n", action)
		return 2, disp, q.Harness, true
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
