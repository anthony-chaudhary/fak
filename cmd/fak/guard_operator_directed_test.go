package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/headlesslint"
)

// guard_operator_directed_test.go — the ACT half of the "stopped to ask a human" gate:
// runGuardOperatorDirectedGate and its install/resolution wiring. The sensor half
// (applyHeadlessLintSignal / readGuardStopTranscript) is pinned in
// guard_stops_headlesslint_test.go; these tests pin the enforce/warn/shadow ladder, the
// operator-absent cap, the HUMAN_RESIDUAL escalation carve-out, and the disposition rollups.

// TestNormalizeGuardOperatorDirectedMode covers the closed vocabulary: empty defaults to the warn
// soak, every named rung round-trips, and an unknown value is an error the callers treat as fail-open.
func TestNormalizeGuardOperatorDirectedMode(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", guardOperatorDirectedModeWarn, false},
		{"  ", guardOperatorDirectedModeWarn, false},
		{"WARN", guardOperatorDirectedModeWarn, false},
		{"off", guardPreCompactModeOff, false},
		{"Shadow", guardPreCompactModeShadow, false},
		{"enforce", guardPreCompactModeEnforce, false},
		{"loud", "", true},
	} {
		got, err := normalizeGuardOperatorDirectedMode(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("normalize(%q): want error, got %q", tc.in, got)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("normalize(%q) = %q, %v; want %q, nil", tc.in, got, err, tc.want)
		}
	}
	// The env-boundary total form never errors and pins a bad string to warn (advisory), never enforce.
	if got := guardOperatorDirectedNormalizedOrWarn("loud"); got != guardOperatorDirectedModeWarn {
		t.Errorf("guardOperatorDirectedNormalizedOrWarn(bad) = %q, want warn", got)
	}
	if got := guardOperatorDirectedNormalizedOrWarn("enforce"); got != guardPreCompactModeEnforce {
		t.Errorf("guardOperatorDirectedNormalizedOrWarn(enforce) = %q, want enforce", got)
	}
}

// TestGuardOperatorDirectedEffectiveMode pins the operator-absent cap AND the operator-presence axis
// that governs it (#4951). An ATTENDED interactive child (a human is present to answer) must never
// have a stop BLOCKED by this gate, so an unset default goes off and an explicit enforce is capped to
// warn; a headless child — OR an interactive child an orchestrator marked unattended (no responsive
// human, a fleet/remote-driven session) — keeps the configured mode verbatim and can reach enforce.
// The invariant preserved is "enforce ⇒ operator absent", now satisfied by headless OR unattended, so
// acting on enforce can never silence a real human question. Mirrors [[TestGuardTaskHandoffEffectiveMode]].
func TestGuardOperatorDirectedEffectiveMode(t *testing.T) {
	for _, tc := range []struct {
		name               string
		configured         string
		explicitlySet      bool
		childInteractive   bool
		operatorUnattended bool
		want               string
	}{
		// Default on an attended interactive child: fully off (no friction where a human can answer).
		{"default interactive -> off", guardOperatorDirectedModeWarn, false, true, false, guardPreCompactModeOff},
		// Default headless: the shipped warn soak is kept.
		{"default headless -> keep warn", guardOperatorDirectedModeWarn, false, false, false, guardOperatorDirectedModeWarn},
		// Explicit enforce on an attended interactive child is capped to warn — surface, never block.
		{"explicit enforce interactive -> warn", guardPreCompactModeEnforce, true, true, false, guardOperatorDirectedModeWarn},
		// Explicit enforce headless: the whole point — a headless run reaches enforce.
		{"explicit enforce headless -> enforce", guardPreCompactModeEnforce, true, false, false, guardPreCompactModeEnforce},
		// Explicit shadow/off are honored as-given on either child.
		{"explicit shadow interactive -> shadow", guardPreCompactModeShadow, true, true, false, guardPreCompactModeShadow},
		{"explicit off interactive -> off", guardPreCompactModeOff, true, true, false, guardPreCompactModeOff},
		{"explicit warn headless -> warn", guardOperatorDirectedModeWarn, true, false, false, guardOperatorDirectedModeWarn},
		// A bad configured string fails safe to warn, then the cap applies (interactive+unset -> off).
		{"bad string interactive unset -> off", "loud", false, true, false, guardPreCompactModeOff},
		{"bad string headless -> warn", "loud", false, false, false, guardOperatorDirectedModeWarn},

		// #4951 operator-presence axis: an interactive child marked UNATTENDED is treated like headless
		// (the operator is absent), so the attended cap is lifted and the full ladder is first-class.
		// The whole point of the gap fix: an operator-driven interactive session reaches enforce.
		{"unattended interactive explicit enforce -> enforce", guardPreCompactModeEnforce, true, true, true, guardPreCompactModeEnforce},
		// Unattended + unset keeps the warn soak (headless-equivalent default), NOT the attended off.
		{"unattended interactive default -> warn", guardOperatorDirectedModeWarn, false, true, true, guardOperatorDirectedModeWarn},
		// Unattended honors shadow verbatim, same as headless.
		{"unattended interactive shadow -> shadow", guardPreCompactModeShadow, true, true, true, guardPreCompactModeShadow},
		// The unattended flag is inert on a headless child (already operator-absent) — no double effect.
		{"unattended headless enforce -> enforce", guardPreCompactModeEnforce, true, false, true, guardPreCompactModeEnforce},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := guardOperatorDirectedEffectiveMode(tc.configured, tc.explicitlySet, tc.childInteractive, tc.operatorUnattended)
			if got != tc.want {
				t.Errorf("guardOperatorDirectedEffectiveMode(%q, set=%v, interactive=%v, unattended=%v) = %q, want %q",
					tc.configured, tc.explicitlySet, tc.childInteractive, tc.operatorUnattended, got, tc.want)
			}
		})
	}
}

// TestGuardOperatorUnattendedEnv pins the operator-presence env grammar: only 1|true|yes|on (any
// case, trimmed) mark the interactive session operator-driven; everything else — unset, empty, "0",
// "off", junk — leaves the attended cap in force, so an ordinary `fak guard -- claude` is unchanged.
func TestGuardOperatorUnattendedEnv(t *testing.T) {
	for _, tc := range []struct {
		val  string
		want bool
	}{
		{"1", true}, {"true", true}, {"TRUE", true}, {"Yes", true}, {" on ", true},
		{"", false}, {"0", false}, {"off", false}, {"no", false}, {"maybe", false},
	} {
		t.Setenv(guardOperatorUnattendedEnv, tc.val)
		if got := guardOperatorUnattended(); got != tc.want {
			t.Errorf("guardOperatorUnattended() with %q = %v, want %v", tc.val, got, tc.want)
		}
	}
}

// resolvableDirected is a transcript whose final turn asked a human an OBVIOUS-action question
// (PERMISSION_ASK -> TAKE_OBVIOUS): the resolvable case enforce should auto-continue.
func resolvableDirected() *guardStopTranscript {
	return &guardStopTranscript{
		Read: true, OperatorDirected: true, OperatorDirectedCount: 1,
		OperatorDirectedClass:       "PERMISSION_ASK",
		OperatorDirectedDisposition: string(choicetriage.TakeObvious),
		OperatorDirectedResolve:     "push the change yourself and state that you assumed approval.",
	}
}

// residualDirected is a transcript whose final turn hit a genuine authority wall
// (HUMAN_RESIDUAL): the case enforce should ROUTE as an escalation, not re-prompt.
func residualDirected() *guardStopTranscript {
	return &guardStopTranscript{
		Read: true, OperatorDirected: true, OperatorDirectedCount: 1,
		OperatorDirectedClass:       "CONFIRMATION_WAIT",
		OperatorDirectedDisposition: string(choicetriage.HumanResidual),
		OperatorDirectedResolve:     "record the pending sign-off as a blocking escalation for the operator to route.",
	}
}

// TestRunGuardOperatorDirectedGate pins the ladder decisions directly off the gate function.
func TestRunGuardOperatorDirectedGate(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mode      string
		tr        *guardStopTranscript
		wantExit  int
		wantDisp  guardStopDisposition
		wantFired bool
		wantSub   string // stderr substring (empty = don't check)
	}{
		// off / nil / clean transcript are inert: the gate does not fire, the caller proceeds clean.
		{"off is inert", guardPreCompactModeOff, resolvableDirected(), 0, "", false, ""},
		{"nil transcript", guardPreCompactModeEnforce, nil, 0, "", false, ""},
		{"clean final turn", guardPreCompactModeEnforce, &guardStopTranscript{Read: true}, 0, "", false, ""},
		{"bad mode fails open", "loud", resolvableDirected(), 0, "", false, ""},
		// shadow / warn observe only: allow the stop (exit 0), record the observe disposition.
		{"shadow allows", guardPreCompactModeShadow, resolvableDirected(), 0, stopDispOperatorDirectedShadow, true, "shadow"},
		{"warn allows + soaks", guardOperatorDirectedModeWarn, resolvableDirected(), 0, stopDispOperatorDirectedWarn, true, "soak mode"},
		// enforce BLOCKS a resolvable ask (exit 2) and feeds the remediation back to the model.
		{"enforce blocks resolvable", guardPreCompactModeEnforce, resolvableDirected(), 2, stopDispOperatorDirectedContinue, true, "no operator to answer"},
		// enforce BLOCKS a premature surrender (exit 2) and directs the worker not to give up.
		{"enforce blocks premature surrender", guardPreCompactModeEnforce, &guardStopTranscript{
			Read:                    true,
			OperatorDirected:        true,
			OperatorDirectedCount:   1,
			OperatorDirectedClass:   string(headlesslint.PrematureSurrender),
			OperatorDirectedResolve: "take the obvious next action yourself and state the assumption you made.",
		}, 2, stopDispOperatorDirectedContinue, true, "prematurely surrendered"},
		// enforce ALLOWS a HUMAN_RESIDUAL ask and routes it as a typed escalation (exit 0).
		{"enforce escalates residual", guardPreCompactModeEnforce, residualDirected(), 0, stopDispOperatorDirectedEscalate, true, "HUMAN_RESIDUAL"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr strings.Builder
			exit, disp, fired := runGuardOperatorDirectedGate(&stderr, tc.mode, tc.tr)
			if exit != tc.wantExit || disp != tc.wantDisp || fired != tc.wantFired {
				t.Fatalf("gate(%q) = exit %d disp %q fired %v; want %d/%q/%v",
					tc.mode, exit, disp, fired, tc.wantExit, tc.wantDisp, tc.wantFired)
			}
			if tc.wantSub != "" && !strings.Contains(stderr.String(), tc.wantSub) {
				t.Errorf("stderr missing %q:\n%s", tc.wantSub, stderr.String())
			}
		})
	}
}

// TestGuardStopDispositionKindOperatorDirected pins the rollups of the four new dispositions: the
// enforce continue rolls up as a continue, the routed escalation as a clean stop (a legitimate,
// human-owned conclusion), and the warn/shadow observers as shadow.
func TestGuardStopDispositionKindOperatorDirected(t *testing.T) {
	for _, tc := range []struct {
		disp guardStopDisposition
		want guardStopKind
	}{
		{stopDispOperatorDirectedContinue, stopKindContinue},
		{stopDispOperatorDirectedEscalate, stopKindClean},
		{stopDispOperatorDirectedWarn, stopKindShadow},
		{stopDispOperatorDirectedShadow, stopKindShadow},
	} {
		if got := guardStopDispositionKind(tc.disp); got != tc.want {
			t.Errorf("kind(%q) = %q, want %q", tc.disp, got, tc.want)
		}
	}
}

// stopHookQuietGauge is a gauge server with all counters at zero, so runGuardStopHook reaches the
// clean-stop (stage=allow) branch where the operator-directed gate runs.
func stopHookQuietGauge(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestRunGuardStopHookOperatorDirectedEnforceBlocks is the end-to-end enforce path: a clean stop
// whose final turn asks a human an obvious-action question is BLOCKED (exit 2) so the agent gets the
// choicetriage remediation and keeps working instead of hanging on an unanswerable question.
func TestRunGuardStopHookOperatorDirectedEnforceBlocks(t *testing.T) {
	srv := stopHookQuietGauge(t)
	dir := t.TempDir()
	fixture := filepath.Join(dir, "asked.jsonl")
	writeStopTranscriptFixture(t, fixture,
		`{"type":"user","message":{"role":"user","content":"fix the handler"}}`,
		`{"type":"assistant","message":{"role":"assistant","content":"Fixed the handler. Do you want me to push the changes?"}}`,
	)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"transcript_path":"`+filepath.ToSlash(fixture)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-directed", guardPreCompactModeEnforce,
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (headless operator-directed stop is blocked); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"no operator to answer", "Do not stop to ask", "no allowed path"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("continue guidance missing %q:\n%s", want, stderr.String())
		}
	}
}

// TestRunGuardStopHookOperatorDirectedEscalates is the end-to-end HUMAN_RESIDUAL carve-out: a final
// turn that hit a genuine authority wall is ALLOWED to stop (exit 0) and routed as a typed
// escalation rather than re-prompted into a loop.
func TestRunGuardStopHookOperatorDirectedEscalates(t *testing.T) {
	srv := stopHookQuietGauge(t)
	dir := t.TempDir()
	fixture := filepath.Join(dir, "authwall.jsonl")
	writeStopTranscriptFixture(t, fixture,
		`{"type":"assistant","message":{"role":"assistant","content":"Waiting for your approval to release the build."}}`,
	)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"transcript_path":"`+filepath.ToSlash(fixture)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-directed", guardPreCompactModeEnforce,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (a genuine authority wall is a legitimate stop); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "HUMAN_RESIDUAL") {
		t.Errorf("escalation line missing HUMAN_RESIDUAL:\n%s", stderr.String())
	}
}

// TestRunGuardStopHookOperatorDirectedWarnSoaks proves the shipped default rung: warn prints the
// would-enforce remediation for the operator but still ALLOWS the stop (exit 0), so the pathology
// can soak before promotion to enforce.
func TestRunGuardStopHookOperatorDirectedWarnSoaks(t *testing.T) {
	srv := stopHookQuietGauge(t)
	dir := t.TempDir()
	fixture := filepath.Join(dir, "asked.jsonl")
	writeStopTranscriptFixture(t, fixture,
		`{"type":"assistant","message":{"role":"assistant","content":"Done. Do you want me to push it?"}}`,
	)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"transcript_path":"`+filepath.ToSlash(fixture)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-directed", guardOperatorDirectedModeWarn,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (warn soak allows the stop); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "[warn]") || !strings.Contains(stderr.String(), "soak mode") {
		t.Errorf("warn soak line missing:\n%s", stderr.String())
	}
}

// TestInstallGuardStopHookInjectsOperatorDirectedEnv pins that the resolved gate mode is threaded
// into the Stop-hook child as an env var, so the install-time operator-absent cap is authoritative.
func TestInstallGuardStopHookInjectsOperatorDirectedEnv(t *testing.T) {
	dir := t.TempDir()
	_, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, guardPreCompactModeEnforce)
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	got := ""
	for _, kv := range env {
		if kv[0] == guardStopHookOperatorDirectedEnvMode {
			got = kv[1]
		}
	}
	if got != guardPreCompactModeEnforce {
		t.Fatalf("operator-directed env = %q, want enforce", got)
	}
	// A bad mode string is pinned to warn (advisory), never accidentally enforce.
	_, env2, _, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, "loud")
	if err != nil {
		t.Fatalf("install(bad mode): %v", err)
	}
	got2 := ""
	for _, kv := range env2 {
		if kv[0] == guardStopHookOperatorDirectedEnvMode {
			got2 = kv[1]
		}
	}
	if got2 != guardOperatorDirectedModeWarn {
		t.Fatalf("bad-mode operator-directed env = %q, want warn", got2)
	}
}

// TestGuardOperatorDirectedContinueInjectsResolve pins the #3883 acceptance criterion that a BLOCKED
// operator-directed stop injects the choicetriage Resolve text FOR THE TOP FINDING back to the model
// — the actionable per-finding remediation, not a generic nudge — and covers the defensive fallbacks
// the message helpers use when the sensor left Class/Resolve blank. The ladder decisions are pinned by
// [[TestRunGuardOperatorDirectedGate]]; this pins the model-facing REMEDIATION CONTENT the enforce
// rung feeds, so the redirect-to-operator guidance is a usable instruction in all cases (a fired-but-
// unclassified row must still yield a real continue message, never an empty "()" or a dangling
// "Instead:  Then finish").
func TestGuardOperatorDirectedContinueInjectsResolve(t *testing.T) {
	// The exact per-finding choicetriage Resolve text — and the finding's Class — flow into the
	// enforce continue message. This is what makes the redirect actionable rather than a generic hint.
	resolvable := resolvableDirected()
	var stderr strings.Builder
	exit, disp, fired := runGuardOperatorDirectedGate(&stderr, guardPreCompactModeEnforce, resolvable)
	if exit != 2 || disp != stopDispOperatorDirectedContinue || !fired {
		t.Fatalf("enforce resolvable = exit %d disp %q fired %v; want 2/continue/true", exit, disp, fired)
	}
	for _, want := range []string{resolvable.OperatorDirectedResolve, resolvable.OperatorDirectedClass} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("continue message missing top-finding %q:\n%s", want, stderr.String())
		}
	}

	// Defensive fallbacks: a fired row the sensor left unclassified (blank Class + blank Resolve) must
	// still produce a usable continue message — the generic "operator-directed" label and the generic
	// take-the-obvious-action remediation — never an empty parenthetical or a dangling "Instead:".
	blank := &guardStopTranscript{Read: true, OperatorDirected: true, OperatorDirectedCount: 1}
	var stderr2 strings.Builder
	exit2, _, fired2 := runGuardOperatorDirectedGate(&stderr2, guardPreCompactModeEnforce, blank)
	if exit2 != 2 || !fired2 {
		t.Fatalf("enforce blank-classification = exit %d fired %v; want 2/true", exit2, fired2)
	}
	msg := stderr2.String()
	if !strings.Contains(msg, "(operator-directed)") {
		t.Errorf("blank Class did not fall back to the generic label:\n%s", msg)
	}
	if !strings.Contains(msg, "take the obvious next action yourself") {
		t.Errorf("blank Resolve did not fall back to the generic remediation:\n%s", msg)
	}
	if strings.Contains(msg, "()") || strings.Contains(msg, "Instead:  ") {
		t.Errorf("blank fields produced an empty parenthetical or dangling clause:\n%s", msg)
	}
}

// TestGuardOperatorDirectedPrematureSurrender verifies that PrematureSurrender outputs the premature surrender continue message.
func TestGuardOperatorDirectedPrematureSurrender(t *testing.T) {
	tr := &guardStopTranscript{
		Read:                    true,
		OperatorDirected:        true,
		OperatorDirectedCount:   1,
		OperatorDirectedClass:   string(headlesslint.PrematureSurrender),
		OperatorDirectedResolve: "run the affected test suite to verify changes",
	}
	var stderr strings.Builder
	exit, disp, fired := runGuardOperatorDirectedGate(&stderr, guardPreCompactModeEnforce, tr)
	if exit != 2 || disp != stopDispOperatorDirectedContinue || !fired {
		t.Fatalf("enforce premature surrender = exit %d disp %q fired %v; want 2/continue/true", exit, disp, fired)
	}
	wantMsg := fmt.Sprintf("fak guard Stop: this turn prematurely surrendered (%s) without completing the task or verifying deliverables. Do not give up. Instead: %s Then finish the turn. If a protected boundary genuinely blocks the last step, note it on one line (`no allowed path: <reason>`) and stop cleanly — that is a complete outcome.",
		string(headlesslint.PrematureSurrender), "run the affected test suite to verify changes")
	if strings.TrimSpace(stderr.String()) != wantMsg {
		t.Errorf("got stderr %q, want %q", strings.TrimSpace(stderr.String()), wantMsg)
	}
}
