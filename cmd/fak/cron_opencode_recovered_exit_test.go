package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestCronOpenCodeRecoveredExitHelper mirrors an OpenCode v2 run that hit one
// retried provider.transport error, finished its final step with reason
// "stop", and still exited 1. FAK_CRON_RECOVERED_FINAL picks the final event.
func TestCronOpenCodeRecoveredExitHelper(t *testing.T) {
	if os.Getenv("FAK_CRON_RECOVERED_HELPER") != "1" {
		return
	}
	_, _ = os.Stdout.WriteString(`{"type":"step_start","sessionID":"ses_recov_1","part":{"type":"step-start"}}` + "\n")
	_, _ = os.Stdout.WriteString(`{"type":"text","sessionID":"ses_recov_1","part":{"type":"text","text":"done"}}` + "\n")
	switch os.Getenv("FAK_CRON_RECOVERED_FINAL") {
	case "stop":
		_, _ = os.Stdout.WriteString(`{"type":"step_finish","sessionID":"ses_recov_1","part":{"type":"step-finish","reason":"stop"}}` + "\n")
	case "tool-calls":
		_, _ = os.Stdout.WriteString(`{"type":"step_finish","sessionID":"ses_recov_1","part":{"type":"step-finish","reason":"tool-calls"}}` + "\n")
	}
	code := 1
	if os.Getenv("FAK_CRON_RECOVERED_EXIT") == "3" {
		code = 3
	}
	os.Exit(code)
}

func runRecoveredExitHelper(t *testing.T, final, exit string) OpenCodeRunReceipt {
	t.Helper()
	var stdout, stderr bytes.Buffer
	receipt, err := RunScheduledOpenCode(ScheduledOpenCodeOptions{
		Job:         "job-recovered-exit",
		Ledger:      filepath.Join(t.TempDir(), "recovered.jsonl"),
		Timeout:     30 * time.Second,
		Command:     []string{os.Args[0], "-test.run=^TestCronOpenCodeRecoveredExitHelper$"},
		Env:         []string{"FAK_CRON_RECOVERED_HELPER=1", "FAK_CRON_RECOVERED_FINAL=" + final, "FAK_CRON_RECOVERED_EXIT=" + exit},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	})
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	return receipt
}

func TestCronOpenCodeRecoveredExitScoresFinalStopSucceeded(t *testing.T) {
	r := runRecoveredExitHelper(t, "stop", "1")
	if r.Outcome != "succeeded" || r.ExitCode != 0 || r.RecoveredExitCode != 1 {
		t.Fatalf("outcome=%q exit=%d recovered=%d, want succeeded/0/1", r.Outcome, r.ExitCode, r.RecoveredExitCode)
	}
	if r.SessionID != "ses_recov_1" || r.FailureTail != "" || r.FailureClass != "" {
		t.Fatalf("session=%q tail=%q class=%q, want ses_recov_1 with no failure fields", r.SessionID, r.FailureTail, r.FailureClass)
	}
}

func TestCronOpenCodeRecoveredExitKeepsUnfinishedRunFailed(t *testing.T) {
	for _, tc := range []struct{ final, exit string }{
		{"tool-calls", "1"}, // last step wanted more tool calls: not finished
		{"none", "1"},       // last event is text, no step_finish
		{"stop", "3"},       // only exit 1 is the v2 recovered-step exit
	} {
		r := runRecoveredExitHelper(t, tc.final, tc.exit)
		if r.Outcome != "failed" || r.RecoveredExitCode != 0 || r.ExitCode == 0 {
			t.Fatalf("final=%s exit=%s: outcome=%q exit=%d recovered=%d, want failed with raw exit", tc.final, tc.exit, r.Outcome, r.ExitCode, r.RecoveredExitCode)
		}
	}
}

func TestCronOpenCodeFinalStepStop(t *testing.T) {
	for in, want := range map[string]bool{
		`{"type":"step_finish","part":{"reason":"stop"}}` + "\r\n\n":                     true,
		`{"type":"step_finish","part":{"reason":"error"}}`:                               false,
		"{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}\n{\"type\":\"text\"}": false,
		"not json": false,
		"":         false,
	} {
		if got := cronOpenCodeFinalStepStop(in); got != want {
			t.Errorf("cronOpenCodeFinalStepStop(%q)=%v, want %v", in, got, want)
		}
	}
	if cronOpenCodeRecoveredExit(1, "", "completed", `{"type":"step_finish","part":{"reason":"stop"}}`) {
		t.Error("no session id must not recover")
	}
	if cronOpenCodeRecoveredExit(1, "ses_x", "progress_silence", `{"type":"step_finish","part":{"reason":"stop"}}`) {
		t.Error("a timed-out attempt must not recover")
	}
}
