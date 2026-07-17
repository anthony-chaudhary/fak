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
			exit, disp, gotHarness, fired := runGuardOperatorQuestionGate(&stderr, guardPreCompactModeEnforce, writeOperatorGateTranscript(t, harness, false), "")
			if !fired || exit != 2 || disp != stopDispOperatorQuestionResolved || gotHarness != harness {
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
	exit, disp, _, fired := runGuardOperatorQuestionGate(&stderr, guardPreCompactModeEnforce, writeOperatorGateTranscript(t, "claude", true), "")
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
	exit, disp, _, fired := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeEnforce, path, "")
	if !fired || exit != 2 || disp != stopDispOperatorQuestionBlocked {
		t.Fatalf("fired=%v exit=%d disp=%s", fired, exit, disp)
	}
}

func TestOperatorQuestionGateWarnAndOffNeverActuate(t *testing.T) {
	withOperatorResolver(t)
	path := writeOperatorGateTranscript(t, "claude", false)
	if exit, disp, _, fired := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardOperatorDirectedModeWarn, path, ""); !fired || exit != 0 || disp != stopDispOperatorQuestionResolved {
		t.Fatalf("warn fired=%v exit=%d disp=%s", fired, exit, disp)
	}
	if exit, _, _, fired := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeOff, path, ""); fired || exit != 0 {
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

// TestInstallGuardStopHookInjectsOperatorQuestionEnv pins that the operator-question mode is threaded
// into the Stop-hook child as its OWN dial (FAK_GUARD_OPERATOR_QUESTION_MODE), split from the prose
// operator-directed sensor but INHERITING the resolved operator-directed posture as its default (both
// are operator-absent-capped headless gates), and that a bad string is pinned to warn (advisory),
// never enforce. Mirrors TestInstallGuardStopHookInjectsHardwareGateEnv — the sibling inherited gate.
func TestInstallGuardStopHookInjectsOperatorQuestionEnv(t *testing.T) {
	dir := t.TempDir()
	_, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, guardPreCompactModeEnforce)
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	got := ""
	for _, kv := range env {
		if kv[0] == guardStopHookOperatorQuestionEnvMode {
			got = kv[1]
		}
	}
	if got != guardPreCompactModeEnforce {
		t.Fatalf("operator-question env = %q, want enforce (inherited from operator-directed posture)", got)
	}
	// A bad operator-directed mode string pins the inherited operator-question dial to warn, not enforce.
	_, env2, _, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, "loud")
	if err != nil {
		t.Fatalf("install(bad mode): %v", err)
	}
	got2 := ""
	for _, kv := range env2 {
		if kv[0] == guardStopHookOperatorQuestionEnvMode {
			got2 = kv[1]
		}
	}
	if got2 != guardOperatorDirectedModeWarn {
		t.Fatalf("bad-mode operator-question env = %q, want warn", got2)
	}
}

// TestOperatorQuestionGateFlagsOperatorOnHumanResidual pins the fix for the "questions thing left as
// empty prose, never picked up by the operator" gap: when a native question folds to HUMAN_RESIDUAL
// the gate must FLAG THE OPERATOR (route the escalation), not merely emit stderr. Routes in warn and
// enforce (a structured genuine escalation), stays silent in shadow (pure observe) and off, and never
// fires for an auto-RESOLVED question.
func TestOperatorQuestionGateFlagsOperatorOnHumanResidual(t *testing.T) {
	var gotSession, gotReason string
	var gotQ operatorquestion.OperatorQuestion
	var called int
	old := guardOperatorQuestionEscalationSink
	guardOperatorQuestionEscalationSink = func(sessionID string, q operatorquestion.OperatorQuestion, reason, action string) {
		called++
		gotSession, gotReason, gotQ = sessionID, reason, q
	}
	t.Cleanup(func() { guardOperatorQuestionEscalationSink = old })
	withOperatorResolver(t)

	// authority=true → no option carries decisive evidence → HUMAN_RESIDUAL (see the escalate test).
	residual := writeOperatorGateTranscript(t, "claude", true)

	// enforce: a genuine residual flags the operator with the session + structured question.
	called, gotSession = 0, ""
	if _, disp, _, _ := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeEnforce, residual, "sess-enforce"); disp != stopDispOperatorQuestionEscalate {
		t.Fatalf("enforce disp=%s want escalate", disp)
	}
	if called != 1 || gotSession != "sess-enforce" || gotReason == "" || gotQ.Harness != "claude" {
		t.Fatalf("enforce residual must flag operator: called=%d session=%q reason=%q harness=%q", called, gotSession, gotReason, gotQ.Harness)
	}

	// warn: still routes — a native question folding to a residual is a real escalation, not a soak.
	called, gotSession = 0, ""
	if _, disp, _, _ := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardOperatorDirectedModeWarn, residual, "sess-warn"); disp != stopDispOperatorQuestionEscalate {
		t.Fatalf("warn disp=%s want escalate", disp)
	}
	if called != 1 || gotSession != "sess-warn" {
		t.Fatalf("warn residual must flag operator: called=%d session=%q", called, gotSession)
	}

	// shadow: pure observe — never actuates an operator notice.
	called = 0
	runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeShadow, residual, "sess-shadow")
	if called != 0 {
		t.Fatalf("shadow must not flag operator, called=%d", called)
	}

	// off: gate does not fire at all.
	called = 0
	runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeOff, residual, "sess-off")
	if called != 0 {
		t.Fatalf("off must not flag operator, called=%d", called)
	}

	// An auto-RESOLVED question (authority=false, decisive witness) is not an escalation → no flag.
	called = 0
	resolved := writeOperatorGateTranscript(t, "claude", false)
	if _, disp, _, _ := runGuardOperatorQuestionGate(&bytes.Buffer{}, guardPreCompactModeEnforce, resolved, "sess-resolved"); disp != stopDispOperatorQuestionResolved {
		t.Fatalf("resolved disp=%s", disp)
	}
	if called != 0 {
		t.Fatalf("auto-resolved question must not flag operator, called=%d", called)
	}
}

func TestOperatorQuestionResolvedCarriesAnswerForNextContinuation(t *testing.T) {
	withOperatorResolver(t)
	var stderr bytes.Buffer
	var answer string
	exit, disp, harness, fired := runGuardOperatorQuestionGate(&stderr, guardPreCompactModeEnforce, writeOperatorGateTranscript(t, "claude", false), "sess-answer", &answer)
	if !fired || exit != 2 || disp != stopDispOperatorQuestionResolved || harness != "claude" {
		t.Fatalf("resolved gate = fired %v exit %d disp %s harness %q", fired, exit, disp, harness)
	}
	if !strings.Contains(answer, "Commit explicit owned paths") {
		t.Fatalf("answer = %q, want selected witnessed option", answer)
	}
}

// TestOperatorQuestionEscalationSlackTextCarriesQuestionAndReason pins that the operator actually
// receives the actionable payload — the prompt, the choices, why no oracle resolved it, and the next
// action — rather than an empty notice.
func TestOperatorQuestionEscalationSlackTextCarriesQuestionAndReason(t *testing.T) {
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Harness:  "codex",
		Question: "Which product priority should win?",
		Options:  []operatorquestion.Option{{Label: "Reliability"}, {Label: "Launch"}},
	}
	text := guardOperatorQuestionEscalationSlackText(q, "AUTHORITY_FORK", "route the authority question to the operator")
	for _, want := range []string{"HUMAN_RESIDUAL", "codex", "Which product priority should win?", "Reliability", "Launch", "AUTHORITY_FORK", "route the authority question to the operator"} {
		if !strings.Contains(text, want) {
			t.Fatalf("escalation text missing %q:\n%s", want, text)
		}
	}
	// A sparse question still yields a non-empty, defaulted notice rather than blank prose.
	if got := guardOperatorQuestionEscalationSlackText(operatorquestion.OperatorQuestion{}, "", ""); !strings.Contains(got, "HUMAN_RESIDUAL") || !strings.Contains(got, "operator-question") {
		t.Fatalf("sparse question should still render an actionable notice: %q", got)
	}
}
