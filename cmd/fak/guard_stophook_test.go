package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func readGuardStopRecordsForTest(path string) ([]guardStopRecord, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return jsonlledger.Parse(string(content), func(r guardStopRecord) bool {
		return r.Schema == guardStopRecordSchema
	}), nil
}

func TestGuardRefusalRestatementCapStopsThirdNudge(t *testing.T) {
	root := t.TempDir()
	transcriptPath := filepath.Join(root, "session.jsonl")
	statePath := filepath.Join(root, "restatement.json")
	message := "I cannot perform the requested destructive action safely. A human must resolve this blocker."
	write := func(msg string) {
		t.Helper()
		line := fmt.Sprintf(`{"type":"assistant","message":{"content":[{"type":"text","text":%q}]}}`+"\n", msg)
		if err := os.WriteFile(transcriptPath, []byte(line), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(message)
	for attempt := 1; attempt <= refusalRestatementCap+1; attempt++ {
		decision := refusalRestatementCheck(refusalRestatementInput{
			SessionID:      "session-5018",
			TranscriptPath: transcriptPath,
			StatePath:      statePath,
			Head:           "same-commit",
		})
		if attempt <= refusalRestatementCap && decision.Blocked {
			t.Fatalf("attempt %d blocked early: %+v", attempt, decision)
		}
		if attempt == refusalRestatementCap+1 {
			if !decision.Blocked || decision.Reason != refusalNeedsHumanReason {
				t.Fatalf("attempt %d = %+v, want terminal %s", attempt, decision, refusalNeedsHumanReason)
			}
		}
		write("  " + message + "  ")
	}
}

func TestGuardRefusalRestatementProgressResetsCap(t *testing.T) {
	root := t.TempDir()
	transcriptPath := filepath.Join(root, "session.jsonl")
	statePath := filepath.Join(root, "restatement.json")
	line := `{"type":"assistant","message":{"content":[{"type":"text","text":"I cannot proceed safely; a human must resolve this blocker."}]}}` + "\n"
	if err := os.WriteFile(transcriptPath, []byte(line), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, head := range []string{"a", "a", "b", "b"} {
		if got := refusalRestatementCheck(refusalRestatementInput{SessionID: "s", TranscriptPath: transcriptPath, StatePath: statePath, Head: head}); got.Blocked {
			t.Fatalf("head %q unexpectedly blocked: %+v", head, got)
		}
	}
}
func TestGuardStopHookDecision(t *testing.T) {
	// Default ladder: warn=3, final=7, max=9.
	const (
		warn  = guardStopHookDefaultWarn  // 3
		final = guardStopHookDefaultFinal // 7
		max   = guardStopHookDefaultMax   // 9
	)
	for _, tc := range []struct {
		name        string
		consecutive int
		warnAt      int
		finalAt     int
		maxN        int
		mode        string
		wantExit    int
		wantBlock   bool
		wantStage   guardStopHookStage
	}{
		// off never blocks, but still reports the rung it WOULD be at.
		{"off-never-blocks", 5, warn, final, max, guardPreCompactModeOff, 0, false, guardStopHookWarn},
		// enforce: a clean completion (rung 0) allows; the three continue rungs all block (exit 2).
		{"enforce-clean-completion", 0, warn, final, max, guardPreCompactModeEnforce, 0, false, guardStopHookAllow},
		{"enforce-nudge-low", 1, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
		{"enforce-nudge-high", 2, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
		{"enforce-warn-low", 3, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookWarn},
		{"enforce-warn-high", 6, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookWarn},
		{"enforce-final-low", 7, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		{"enforce-final-at-max", 9, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		// > max is the bounded give-up: allow the stop (exit 0) so a stuck model cannot loop forever.
		{"enforce-give-up-above-max", 10, warn, final, max, guardPreCompactModeEnforce, 0, false, guardStopHookGiveUp},
		// shadow always allows (exit 0) but reports the would-be block + rung.
		{"shadow-would-block", 1, warn, final, max, guardPreCompactModeShadow, 0, true, guardStopHookNudge},
		{"shadow-clean", 0, warn, final, max, guardPreCompactModeShadow, 0, false, guardStopHookAllow},
		{"shadow-give-up", 10, warn, final, max, guardPreCompactModeShadow, 0, false, guardStopHookGiveUp},
		// Normalization clamps an INVERTED config so the rungs cannot invert: warn=5 clamps to
		// max=4, final=2 pulls up to warn -> warn=final=max=4.
		{"normalize-inverted-final", 4, 5, 2, 4, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		{"normalize-inverted-giveup", 5, 5, 2, 4, guardPreCompactModeEnforce, 0, false, guardStopHookGiveUp},
		// A zero/garbage max falls back to the default ladder rather than wedging.
		{"normalize-zero-max", 1, 100, 1, 0, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exit, block, stage := guardStopHookDecision(tc.consecutive, tc.warnAt, tc.finalAt, tc.maxN, tc.mode)
			if exit != tc.wantExit || block != tc.wantBlock || stage != tc.wantStage {
				t.Fatalf("decision(c=%d,w=%d,f=%d,m=%d,%q) = exit %d block %v stage %s, want exit %d block %v stage %s",
					tc.consecutive, tc.warnAt, tc.finalAt, tc.maxN, tc.mode, exit, block, stage, tc.wantExit, tc.wantBlock, tc.wantStage)
			}
		})
	}
}

func TestNormalizeGuardStopHookModeDefaultsEnforce(t *testing.T) {
	got, err := normalizeGuardStopHookMode("")
	if err != nil || got != guardPreCompactModeEnforce {
		t.Fatalf("normalize(\"\") = %q, %v; want enforce", got, err)
	}
	if _, err := normalizeGuardStopHookMode("bogus"); err == nil {
		t.Fatalf("normalize(bogus) should error")
	}
}

func TestParseGuardStopHookConsecutive(t *testing.T) {
	n, err := parseGuardStopHookConsecutive("# HELP x\nfak_guard_deny_all_consecutive 2\n")
	if err != nil || n != 2 {
		t.Fatalf("parse = %d, %v; want 2", n, err)
	}
	if _, err := parseGuardStopHookConsecutive("fak_guard_deny_all_stops_total 5\n"); err == nil {
		t.Fatalf("missing gauge must error (so the hook fails open, not silently treats 0)")
	}
}

func TestParseGuardStopHookSignalsReadsToolFeedback(t *testing.T) {
	signals, err := parseGuardStopHookSignals(strings.Join([]string{
		"# HELP x",
		"fak_guard_deny_all_consecutive 0",
		"fak_guard_tool_feedback_consecutive 4",
	}, "\n"))
	if err != nil {
		t.Fatalf("parse signals: %v", err)
	}
	if signals.DenyAllConsecutive != 0 || signals.ToolFeedbackConsecutive != 4 {
		t.Fatalf("signals = %+v, want deny_all=0 tool_feedback=4", signals)
	}
}

func TestNormalizeSameStop(t *testing.T) {
	for _, tc := range []struct {
		in, warn, final, stop int
	}{
		{6, 3, 5, 6},  // default: warn=stop-3, final=stop-1
		{9, 6, 8, 9},  // larger depth keeps the -3/-1 spread
		{2, 1, 1, 2},  // small depth clamps warn/final up to 1 (no inversion)
		{3, 1, 2, 3},  // warn clamps to 1, final=2
		{1, 3, 5, 6},  // < 2 falls back to the default (would otherwise stop on first deny-all)
		{0, 3, 5, 6},  // 0 falls back to the default
		{-4, 3, 5, 6}, // negative falls back to the default
	} {
		warn, final, stop := normalizeSameStop(tc.in)
		if warn != tc.warn || final != tc.final || stop != tc.stop {
			t.Fatalf("normalizeSameStop(%d) = (%d,%d,%d), want (%d,%d,%d)", tc.in, warn, final, stop, tc.warn, tc.final, tc.stop)
		}
		if !(warn >= 1 && warn <= final && final < stop) {
			t.Fatalf("normalizeSameStop(%d) violated warn<=final<stop invariant: (%d,%d,%d)", tc.in, warn, final, stop)
		}
	}
}

func TestGuardStopHookSameDecision(t *testing.T) {
	// Default give-up depth 6 -> warn=3, final=5, give-up at 6.
	const stop = guardStopHookSameStopDefault // 6
	for _, tc := range []struct {
		name      string
		same      int
		stop      int
		mode      string
		wantExit  int
		wantBlock bool
		wantStage guardStopHookStage
	}{
		{"off-never-blocks", 6, stop, guardPreCompactModeOff, 0, false, guardStopHookGiveUp},
		{"clean-completion", 0, stop, guardPreCompactModeEnforce, 0, false, guardStopHookAllow},
		// A varied session pins same=1 forever: it rides the NUDGE rung and is NEVER given up.
		{"varied-nudge", 1, stop, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
		{"nudge-high", 2, stop, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
		{"warn-low", 3, stop, guardPreCompactModeEnforce, 2, true, guardStopHookWarn},
		{"warn-high", 4, stop, guardPreCompactModeEnforce, 2, true, guardStopHookWarn},
		{"final", 5, stop, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		// AT the give-up depth the session stands down (exit 0), so a true repeated same issue
		// cannot loop forever.
		{"give-up-at-depth", 6, stop, guardPreCompactModeEnforce, 0, false, guardStopHookGiveUp},
		{"give-up-above-depth", 20, stop, guardPreCompactModeEnforce, 0, false, guardStopHookGiveUp},
		// shadow always allows (exit 0) but reports the would-be block + rung.
		{"shadow-would-block", 3, stop, guardPreCompactModeShadow, 0, true, guardStopHookWarn},
		{"shadow-give-up", 6, stop, guardPreCompactModeShadow, 0, false, guardStopHookGiveUp},
		// A garbage give-up depth falls back to the default ladder rather than stopping on turn 1.
		{"garbage-stop-falls-back", 1, 0, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exit, block, stage := guardStopHookSameDecision(tc.same, tc.stop, tc.mode)
			if exit != tc.wantExit || block != tc.wantBlock || stage != tc.wantStage {
				t.Fatalf("sameDecision(same=%d,stop=%d,%q) = exit %d block %v stage %s, want exit %d block %v stage %s",
					tc.same, tc.stop, tc.mode, exit, block, stage, tc.wantExit, tc.wantBlock, tc.wantStage)
			}
		})
	}
}

func TestParseGuardStopHookSignalsReadsSameConsecutive(t *testing.T) {
	// Gauge present: the value is parsed AND marked seen (so the hook keys on it).
	signals, err := parseGuardStopHookSignals(strings.Join([]string{
		"fak_guard_deny_all_consecutive 5",
		"fak_guard_deny_all_same_consecutive 2",
	}, "\n"))
	if err != nil {
		t.Fatalf("parse signals: %v", err)
	}
	if signals.DenyAllSameConsecutive != 2 || !signals.DenyAllSameConsecutiveSeen {
		t.Fatalf("signals = %+v, want same=2 seen=true", signals)
	}
	// Gauge absent (older gateway): not seen, so the hook falls back to the blind ladder.
	old, err := parseGuardStopHookSignals("fak_guard_deny_all_consecutive 5\n")
	if err != nil {
		t.Fatalf("parse old signals: %v", err)
	}
	if old.DenyAllSameConsecutiveSeen {
		t.Fatalf("older gateway must leave DenyAllSameConsecutiveSeen false, got %+v", old)
	}
}

func TestReadStopHookActive(t *testing.T) {
	if !readStopHookActive(strings.NewReader(`{"stop_hook_active":true,"session_id":"s"}`)) {
		t.Fatalf("stop_hook_active true not parsed")
	}
	if readStopHookActive(strings.NewReader(`{"stop_hook_active":false}`)) {
		t.Fatalf("stop_hook_active false misread as true")
	}
	if readStopHookActive(strings.NewReader("not json")) {
		t.Fatalf("garbage stdin must read as false")
	}
	if readStopHookActive(nil) {
		t.Fatalf("nil stdin must read as false")
	}
}

func TestRunGuardStopHookEnforceBlocksOnDenyAll(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--max", "3",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (block the unchosen stop)", code)
	}
	if !strings.Contains(stderr.String(), "ALLOWED alternative") {
		t.Fatalf("stderr should carry the continuation instruction: %q", stderr.String())
	}
	// The nudge must TEACH a reshape (self-improve) instead of a bare stall: shell
	// SELF_MODIFY is target-scoped, so the actionable fix is to move or isolate the
	// guarded write target (fak#1917), not to retry the refused command unchanged.
	for _, want := range []string{"RESHAPING", "SELF_MODIFY", "guarded write target"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("nudge missing reshape guidance %q: %s", want, stderr.String())
		}
	}
	// Clarity: the nudge must point the model at the ACTUAL refusal reason in the
	// in-band note (`Tool (REASON/DISPOSITION)`) rather than asserting one hard-coded
	// reason — a deny-all can cite any code, and SELF_MODIFY guidance is wrong for a
	// TRUST_VIOLATION or SECRET_EXFIL refusal.
	if !strings.Contains(stderr.String(), "REASON/DISPOSITION") {
		t.Fatalf("nudge should tell the model to read the reason code from the in-band note: %s", stderr.String())
	}
	// Permissive: reshaping is not the only sanctioned outcome. A genuine (TERMINAL)
	// block must not send the model looping on reworded retries, and the honest
	// wrap-up (`no allowed path: <reason>`) must be named as a clean outcome, not a
	// failure — co-equal with taking an allowed alternative.
	for _, want := range []string{"TERMINAL", "no allowed path"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("nudge missing the sanctioned clean-stop path %q: %s", want, stderr.String())
		}
	}
}

// TestRunGuardStopHookVariedSessionNeverGivenUp is the behavioral crux of the same-issue fix:
// a session with a HIGH blind deny-all count but a DIFFERENT refusal each turn (same-issue gauge
// pinned at 1) must be CONTINUED, never given up — the exact false-give-up the old blind ladder
// caused. Blind consecutive is 50 (far above --max 3), yet the hook keys on the same-issue gauge
// and blocks the stop (exit 2) because the session is exploring, not spinning.
func TestRunGuardStopHookVariedSessionNeverGivenUp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Join([]string{
			"fak_guard_deny_all_consecutive 50",
			"fak_guard_deny_all_same_consecutive 1",
		}, "\n")))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--max", "3", // blind max the OLD logic would have given up on (50 > 3)
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (a varied session must be continued, never given up on the blind count); stderr=%s", code, stderr.String())
	}
}

// TestRunGuardStopHookSameIssueGivesUp is the other half: a TRUE repeated same refusal (same-issue
// gauge at the default give-up depth) stands down (exit 0), and the operator line names the
// identical refused action so the give-up is legible as a genuine spin, not exploration.
func TestRunGuardStopHookSameIssueGivesUp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Join([]string{
			"fak_guard_deny_all_consecutive 6",
			"fak_guard_deny_all_same_consecutive 6", // == default give-up depth
		}, "\n")))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (a true repeated same issue stands down); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"IDENTICAL refused action", "same-issue give-up", "FRESH block each turn is never stopped"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("same-issue give-up line missing %q: %s", want, stderr.String())
		}
	}
}

// TestRunGuardStopHookSameIssueFinalNamesRepeat pins the escalated in-band guidance: at the final
// same-issue rung the model is told it has proposed the IDENTICAL action and to take a DIFFERENT
// one, not to retry — the whole reason a repeated deny-all should change tack.
func TestRunGuardStopHookSameIssueFinalNamesRepeat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Join([]string{
			"fak_guard_deny_all_consecutive 5",
			"fak_guard_deny_all_same_consecutive 5", // final rung at default depth 6
		}, "\n")))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (final rung still continues); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"IDENTICAL refused action", "last auto-continue", "DIFFERENT action"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("same-issue final guidance missing %q: %s", want, stderr.String())
		}
	}
}

func TestRunGuardStopHookContinuesRetryableToolFeedbackPastDenyAllBound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Join([]string{
			"fak_guard_deny_all_consecutive 0",
			"fak_guard_tool_feedback_consecutive 4",
		}, "\n")))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--max", "3",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (retryable tool feedback continues even past hard deny-all max); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"retryable tool-call feedback", "not a session stop", "Fix the JSON/arguments/tool shape"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("tool-feedback guidance missing %q: %s", want, stderr.String())
		}
	}
}

func TestRunGuardStopHookEnforceAllowsWhenNoDenyAll(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	code := runGuardStopHook(ioDiscard{}, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (a clean completion is a real stop)", code)
	}
}

func TestRunGuardStopHookBlocksCleanStopWithoutTaskHandoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", filepath.Join(t.TempDir(), "missing-handoff.json"),
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (block clean stop until handoff exists); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "task handoff required") || !strings.Contains(stderr.String(), taskmgr.SchemaHandoff) {
		t.Fatalf("stderr should tell the agent how to write the handoff: %q", stderr.String())
	}
}

func TestRunGuardStopHookAllowsValidTaskHandoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()
	path := filepath.Join(t.TempDir(), "handoff.json")
	writeGuardStopHookHandoff(t, path, []taskmgr.HandoffNextStep{{
		Key:    "guard-test/follow-up",
		Title:  "Follow up after guarded task",
		Body:   "Pick up the remaining validation work.",
		Reason: "The completed task left a concrete verification rung.",
	}}, "")

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", path,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for valid handoff; stderr=%s", code, stderr.String())
	}
}

// TestRunGuardStopHookHandoffStandsDownWhenStopHookActive is the bound half of #A2: the task-handoff
// gate no longer blocks forever. When the harness is ALREADY re-firing this Stop hook because we
// blocked last turn (stop_hook_active) and the handoff is STILL missing, the gate stands down and
// allows the stop instead of spinning the harness with a demand it already made.
func TestRunGuardStopHookHandoffStandsDownWhenStopHookActive(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"stop_hook_active": true}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", filepath.Join(t.TempDir(), "missing-handoff.json"),
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stand down after a prior block); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"stop_hook_active", "standing down"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("handoff give-up guidance missing %q: %s", want, stderr.String())
		}
	}
}

// TestRunGuardStopHookGivesUpToolFeedbackPastOwnBound is the bound half of #A6: the retryable
// tool-feedback continue no longer runs forever. Past its own (separate, generous) ceiling the hook
// stands down and ALLOWS the stop instead of holding the turn open every turn. The ceiling is
// independent of the deny-all --max, so a lower --max cannot cut a legitimate multi-turn repair short.
func TestRunGuardStopHookGivesUpToolFeedbackPastOwnBound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Join([]string{
			"fak_guard_deny_all_consecutive 0",
			"fak_guard_tool_feedback_consecutive 4",
		}, "\n")))
	}))
	defer srv.Close()
	t.Setenv(guardStopHookEnvToolFeedbackMax, "3") // feedback 4 > bound 3 -> stand down

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (tool-feedback continue stands down past its own bound); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"standing down", "exceeded the continue bound", guardStopHookEnvToolFeedbackMax} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("tool-feedback give-up guidance missing %q: %s", want, stderr.String())
		}
	}
}

// TestRunGuardStopHookSignalIsCleanForCleanStopOnSameIssueGateway is #A7: a clean completion is
// labeled "clean", never "same-issue", even when the gateway EMITS the same-issue gauge at 0.
// Before the fix the label keyed on gauge-presence alone, so every clean stop on a current gateway
// was mis-recorded as a same-issue stop, poisoning the folded ledger's signal tally.
func TestRunGuardStopHookSignalIsCleanForCleanStopOnSameIssueGateway(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Join([]string{
			"fak_guard_deny_all_consecutive 0",
			"fak_guard_deny_all_same_consecutive 0", // gauge SEEN (current gateway) but zero repeats
		}, "\n")))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeShadow, // shadow logs the resolved signal label to stderr
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (shadow always allows); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "signal=clean") {
		t.Fatalf("clean stop should be labeled signal=clean: %s", stderr.String())
	}
	if strings.Contains(stderr.String(), "signal=same-issue") {
		t.Fatalf("clean stop must NOT be mislabeled same-issue: %s", stderr.String())
	}
}

func TestRunGuardStopHookDenyAllPrecedesTaskHandoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", filepath.Join(t.TempDir(), "missing-handoff.json"),
		"--max", "3",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "ALLOWED alternative") {
		t.Fatalf("deny-all guidance should take precedence, got %q", stderr.String())
	}
	if strings.Contains(stderr.String(), "task handoff required") {
		t.Fatalf("task handoff should not mask deny-all guidance: %q", stderr.String())
	}
}

func TestRunGuardStopHookEnforceGivesUpAboveBound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 9\n"))
	}))
	defer srv.Close()

	code := runGuardStopHook(ioDiscard{}, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--max", "3",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (above the retry bound, stop looping)", code)
	}
}

func TestRunGuardStopHookShadowAllowsButLogs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeShadow,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("shadow exit = %d, want 0", code)
	}
	if !strings.Contains(stderr.String(), "would auto-continue") {
		t.Fatalf("shadow should log the would-be continue: %q", stderr.String())
	}
}

func TestRunGuardStopHookFailsOpenWhenGaugeUnavailable(t *testing.T) {
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", "http://127.0.0.1:1/metrics",
		"--timeout", "1ms",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want fail-open 0 (a Stop hook must never wedge the agent)", code)
	}
	if !strings.Contains(stderr.String(), "allowing stop") {
		t.Fatalf("stderr = %q, want fail-open log", stderr.String())
	}
}

func TestRunGuardStopHookFailOpenWritesUnifiedNextRefusal(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "guard-stops.jsonl")
	t.Setenv(guardStopsLedgerEnv, ledger)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", "http://127.0.0.1:1/metrics",
		"--timeout", "1ms",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want fail-open 0", code)
	}
	records, err := readGuardStopRecordsForTest(ledger)
	if err != nil || len(records) != 1 {
		t.Fatalf("records = %+v err=%v", records, err)
	}
	next := records[0].Next
	if next == nil || next.Applied || next.Move.Kind != sessionctl.MoveHalt || next.Move.Render != sessionctl.RenderStop {
		t.Fatalf("next = %+v, want unapplied halt->stop fail-open witness", next)
	}
	if !strings.Contains(next.Refusal, "fail-open") {
		t.Fatalf("refusal = %q, want fail-open", next.Refusal)
	}
}

func TestRunGuardStopHookResolvedQuestionReopensWithAnswerWitness(t *testing.T) {
	withOperatorResolver(t)
	ledger := filepath.Join(t.TempDir(), "guard-stops.jsonl")
	t.Setenv(guardStopsLedgerEnv, ledger)
	transcript := writeOperatorGateTranscript(t, "claude", false)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()
	var stderr strings.Builder
	payload := `{"session_id":"sess-answer","transcript_path":"` + filepath.ToSlash(transcript) + `"}`
	code := runGuardStopHook(&stderr, strings.NewReader(payload), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-question", guardPreCompactModeEnforce,
	})
	if code != 2 {
		t.Fatalf("exit = %d, want resolved-answer continuation 2; stderr=%s", code, stderr.String())
	}
	records, err := readGuardStopRecordsForTest(ledger)
	if err != nil || len(records) != 1 {
		t.Fatalf("records = %+v err=%v", records, err)
	}
	next := records[0].Next
	if next == nil || !next.Applied || next.Move.Kind != sessionctl.MoveContinue || next.Move.Render != sessionctl.RenderReopen {
		t.Fatalf("next = %+v, want applied continue->reopen answer witness", next)
	}
	if !strings.Contains(next.Move.Payload, "Resolved operator answer: Commit explicit owned paths") {
		t.Fatalf("next payload = %q, want resolved answer", next.Move.Payload)
	}
	if next.Move.Session != sessionctl.SessionInteractive {
		t.Fatalf("next session class = %s, want interactive Stop-hook authority", next.Move.Session)
	}
}

func TestRunGuardStopHookContinueWritesUnifiedReopen(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "guard-stops.jsonl")
	t.Setenv(guardStopsLedgerEnv, ledger)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{"--mode", guardPreCompactModeEnforce, "--metrics-url", srv.URL + "/metrics"})
	if code != 2 {
		t.Fatalf("exit = %d, want continue 2", code)
	}
	records, err := readGuardStopRecordsForTest(ledger)
	if err != nil || len(records) != 1 {
		t.Fatalf("records = %+v err=%v", records, err)
	}
	next := records[0].Next
	if next == nil || !next.Applied || next.Move.Kind != sessionctl.MoveContinue || next.Move.Render != sessionctl.RenderReopen {
		t.Fatalf("next = %+v, want applied continue->reopen witness", next)
	}
	if next.Move.Payload == "" {
		t.Fatal("continue witness lost stderr guidance payload")
	}
}

func TestRunGuardStopHookOffIsNoOp(t *testing.T) {
	code := runGuardStopHook(ioDiscard{}, strings.NewReader("{}"), []string{"--mode", guardPreCompactModeOff})
	if code != 0 {
		t.Fatalf("off exit = %d, want 0", code)
	}
}

// TestInstallGuardStopHookMergesIntoPreCompactSettings is the load-bearing wiring test: when the
// PreCompact hook already wrote a --settings file, the Stop hook MERGES into it (both hooks
// present, a SINGLE --settings on the command) rather than injecting a second --settings that
// would clobber the first.
func TestInstallGuardStopHookMergesIntoPreCompactSettings(t *testing.T) {
	dir := t.TempDir()
	fakBin := filepath.Join(dir, "fak.exe")

	command, _, pcInstall, err := installGuardPreCompactHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeShadow, "http://127.0.0.1:4567", fakBin, dir)
	if err != nil || !pcInstall.Applied {
		t.Fatalf("precompact install: applied=%v err=%v", pcInstall.Applied, err)
	}

	command, env, stopInstall, err := installGuardStopHookAt(
		command, guardPreCompactModeEnforce, "http://127.0.0.1:4567", fakBin, "", pcInstall.SettingsPath, 3, 7, 9, 6, guardPreCompactModeOff)
	if err != nil || !stopInstall.Applied {
		t.Fatalf("stop install: applied=%v err=%v", stopInstall.Applied, err)
	}
	if stopInstall.SettingsPath != pcInstall.SettingsPath {
		t.Fatalf("stop hook wrote a different settings file (%s) than precompact (%s) — must merge into one",
			stopInstall.SettingsPath, pcInstall.SettingsPath)
	}
	if n := strings.Count(strings.Join(command, "\x00"), "--settings"); n != 1 {
		t.Fatalf("command has %d --settings flags, want exactly 1: %v", n, command)
	}
	if stopInstall.WarnAt != 3 || stopInstall.FinalAt != 7 || stopInstall.Max != 9 || stopInstall.SameStop != 6 {
		t.Fatalf("ladder = warn %d final %d max %d same-stop %d, want 3/7/9/6", stopInstall.WarnAt, stopInstall.FinalAt, stopInstall.Max, stopInstall.SameStop)
	}
	var sawMode, sawWarn, sawFinal, sawMax, sawSame bool
	for _, kv := range env {
		switch {
		case kv[0] == guardStopHookEnvMode && kv[1] == guardPreCompactModeEnforce:
			sawMode = true
		case kv[0] == guardStopHookEnvWarn && kv[1] == "3":
			sawWarn = true
		case kv[0] == guardStopHookEnvFinal && kv[1] == "7":
			sawFinal = true
		case kv[0] == guardStopHookEnvMax && kv[1] == "9":
			sawMax = true
		case kv[0] == guardStopHookEnvSameStop && kv[1] == "6":
			sawSame = true
		}
	}
	if !sawMode || !sawWarn || !sawFinal || !sawMax || !sawSame {
		t.Fatalf("missing stop-hook env: mode=%v warn=%v final=%v max=%v same-stop=%v from %v", sawMode, sawWarn, sawFinal, sawMax, sawSame, env)
	}

	// The single settings file now carries BOTH hooks.
	data, err := os.ReadFile(stopInstall.SettingsPath)
	if err != nil {
		t.Fatalf("read merged settings: %v", err)
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("unmarshal merged settings: %v\n%s", err, data)
	}
	if len(settings.Hooks["PreCompact"]) != 1 {
		t.Fatalf("merged file lost the PreCompact hook: %s", data)
	}
	stop := settings.Hooks["Stop"]
	if len(stop) != 1 || len(stop[0].Hooks) != 1 {
		t.Fatalf("merged file missing the Stop hook: %s", data)
	}
	if stop[0].Matcher != "" {
		t.Fatalf("Stop hook must carry no matcher, got %q", stop[0].Matcher)
	}
	if got := stop[0].Hooks[0].Args; len(got) != 1 || got[0] != "guard-stophook" {
		t.Fatalf("Stop hook args = %v, want [guard-stophook]", got)
	}
}

// TestInstallGuardStopHookCreatesOwnSettingsWhenPreCompactOff covers the path where PreCompact is
// off: the Stop hook writes its own settings file and injects the single --settings itself.
func TestInstallGuardStopHookCreatesOwnSettingsWhenPreCompactOff(t *testing.T) {
	dir := t.TempDir()
	command, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, guardPreCompactModeOff)
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	if command[1] != "--settings" || command[2] != install.SettingsPath {
		t.Fatalf("stop hook did not inject its own --settings: %v", command)
	}
	if got := strings.Join(command[3:], "\x00"); got != strings.Join([]string{"-p", "hi"}, "\x00") {
		t.Fatalf("user args changed: %v", command)
	}
	if len(env) == 0 {
		t.Fatalf("expected stop-hook env vars")
	}
}

func TestInstallGuardStopHookInjectsTaskHandoffEnv(t *testing.T) {
	dir := t.TempDir()
	handoffPath := filepath.Join(dir, "handoff.json")
	_, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, guardPreCompactModeOff,
		guardTaskHandoffConfig{Mode: guardPreCompactModeEnforce, File: handoffPath, Repo: "owner/repo", Live: true})
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	got := map[string]string{}
	for _, kv := range env {
		got[kv[0]] = kv[1]
	}
	if got[guardTaskHandoffEnvMode] != guardPreCompactModeEnforce ||
		got[guardTaskHandoffEnvFile] != handoffPath ||
		got[guardTaskHandoffFileEnv] != handoffPath ||
		got[guardTaskHandoffEnvRepo] != "owner/repo" ||
		got[guardTaskHandoffEnvLive] != "1" {
		t.Fatalf("task handoff env missing: %+v", got)
	}
}

func TestInstallGuardStopHookSkipsOffAndNonClaude(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mode    string
		command []string
	}{
		{"off", guardPreCompactModeOff, []string{"claude"}},
		{"non-claude", guardPreCompactModeEnforce, []string{"codex"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			command, env, install, err := installGuardStopHookAt(tc.command, tc.mode, "http://127.0.0.1:4567", "fak", dir, "", 3, 7, 9, 6, guardPreCompactModeOff)
			if err != nil {
				t.Fatalf("install: %v", err)
			}
			if install.Applied {
				t.Fatalf("hook applied unexpectedly: %+v", install)
			}
			if len(env) != 0 {
				t.Fatalf("env = %v, want none", env)
			}
			if strings.Join(command, "\x00") != strings.Join(tc.command, "\x00") {
				t.Fatalf("command changed: %v -> %v", tc.command, command)
			}
		})
	}
}

func writeGuardStopHookHandoff(t *testing.T, path string, steps []taskmgr.HandoffNextStep, noNext string) {
	t.Helper()
	h := taskmgr.Handoff{
		Schema:       taskmgr.SchemaHandoff,
		CurrentState: "The guarded task completed and the remaining state is documented.",
		Task: taskmgr.HandoffTask{
			TaskID: "guard-test",
			Title:  "guard test",
			State:  taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{
				VerifiedState: taskmgr.VerifiedDone,
				Source:        "test",
				SHA:           "deadbeef",
			},
		},
		NextSteps:        steps,
		NoNextStepReason: noNext,
	}
	b, err := json.MarshalIndent(h, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}
