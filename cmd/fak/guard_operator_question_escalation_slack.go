package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

const guardOperatorQuestionEscalationSlackSource = guardSessionThreadSource + ":operator-question-escalate"

// guardOperatorQuestionEscalationSink is the seam runGuardOperatorQuestionGate calls to FLAG THE
// OPERATOR when a native operator question (ExitPlanMode / AskUserQuestion) folds to a genuine
// HUMAN_RESIDUAL — a structured authority/approval wall no oracle could auto-resolve. Before this
// wiring the residual was left as stderr prose that routed nowhere (the "empty prose, never picked
// up by the operator" gap). It defaults to the real Slack enqueue and is a package var so tests can
// observe the routed payload without a live outbox — matching the existing resolver seams in
// guard_operator_question.go ([[guardOperatorQuestionClarifyResolver]] / *PlanResolver).
var guardOperatorQuestionEscalationSink = enqueueGuardOperatorQuestionEscalationFailOpen

// enqueueGuardOperatorQuestionEscalationFailOpen routes an operator-question HUMAN_RESIDUAL to the
// guarded session's existing Slack thread. Best-effort like its operator-directed sibling (a missing
// token/identity/session/thread is a clean no-op), but it builds the notification from the STRUCTURED
// question rather than the linguistic transcript, so the operator sees the actual prompt, the choices,
// why no oracle could resolve it, and the residual next action.
func enqueueGuardOperatorQuestionEscalationFailOpen(sessionID string, q operatorquestion.OperatorQuestion, reason, action string) {
	seed := sessionID + "\x00" + q.Harness + "\x00" + string(q.Kind) + "\x00" + strings.TrimSpace(q.Question) + "\x00" + strings.TrimSpace(reason)
	enqueueGuardSessionThreadNotice(sessionID, seed, "guard-oq-escalate-", guardOperatorQuestionEscalationSlackText(q, reason, action), guardOperatorQuestionEscalationSlackSource)
}

// guardOperatorQuestionEscalationSlackText renders the operator-facing escalation: the structured
// question, the choices the operator is being asked to pick between, why no oracle could resolve it
// (reason), and the residual next action. Each field defaults defensively so a sparse question still
// produces an actionable notice rather than an empty one.
func guardOperatorQuestionEscalationSlackText(q operatorquestion.OperatorQuestion, reason, action string) string {
	title := strings.TrimSpace(q.Question)
	if title == "" {
		title = "(native operator question — no prompt text)"
	}
	kind := strings.TrimSpace(string(q.Kind))
	if kind == "" {
		kind = "operator-question"
	}
	harness := strings.TrimSpace(q.Harness)
	if harness == "" {
		harness = "unknown"
	}
	b := &strings.Builder{}
	fmt.Fprintf(b, ":warning: *OPERATOR_QUESTION · HUMAN_RESIDUAL* · `%s/%s`\n%s", harness, kind, title)
	if opts := guardOperatorQuestionOptionLabels(q); opts != "" {
		fmt.Fprintf(b, "\noptions: %s", opts)
	}
	if r := strings.TrimSpace(reason); r != "" {
		fmt.Fprintf(b, "\n> reason: %s", r)
	}
	if a := strings.TrimSpace(action); a != "" {
		fmt.Fprintf(b, "\n> next: %s", a)
	}
	return b.String()
}

// guardOperatorQuestionOptionLabels joins the choice labels the operator is being asked to decide
// between, so the notice is actionable without opening the transcript.
func guardOperatorQuestionOptionLabels(q operatorquestion.OperatorQuestion) string {
	labels := make([]string, 0, len(q.Options))
	for _, o := range q.Options {
		if l := strings.TrimSpace(o.Label); l != "" {
			labels = append(labels, l)
		}
	}
	return strings.Join(labels, " / ")
}
