package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestCronOpenCodeFailureTailHelper is a child that creates a session and then
// exits nonzero with a stderr cause, mirroring an OpenCode v2 run that
// completed a session yet exited 1.
func TestCronOpenCodeFailureTailHelper(t *testing.T) {
	if os.Getenv("FAK_CRON_FAILURE_TAIL_HELPER") != "1" {
		return
	}
	_, _ = os.Stdout.WriteString(`{"session_id": "ses_tail_123"}` + "\n")
	_, _ = os.Stderr.WriteString("boom-after-session: provider stream closed\n")
	os.Exit(3)
}

func TestCronOpenCodeFailureTailKeepsCauseWhenSessionStarted(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_failure_tail.jsonl")
	var stdout, stderr bytes.Buffer
	receipt, err := RunScheduledOpenCode(ScheduledOpenCodeOptions{
		Job:         "job-failure-tail",
		Ledger:      ledger,
		Timeout:     30 * time.Second,
		Command:     []string{os.Args[0], "-test.run=^TestCronOpenCodeFailureTailHelper$"},
		Env:         []string{"FAK_CRON_FAILURE_TAIL_HELPER=1"},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	})
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Outcome != "failed" || receipt.SessionID != "ses_tail_123" {
		t.Fatalf("outcome=%q session=%q, want failed/ses_tail_123", receipt.Outcome, receipt.SessionID)
	}
	if receipt.StartupFailed || receipt.StartError != "" {
		t.Fatalf("session started, so startup fields must stay empty: %+v", receipt)
	}
	if !strings.Contains(receipt.FailureTail, "boom-after-session") {
		t.Fatalf("failure_tail=%q, want child stderr cause", receipt.FailureTail)
	}
	if len(receipt.FailureTail) > cronOpenCodeFailureTailBytes {
		t.Fatalf("failure_tail len=%d exceeds bound %d", len(receipt.FailureTail), cronOpenCodeFailureTailBytes)
	}
	rows, err := cronReadOpenCodeReceipts(ledger)
	if err != nil || len(rows) != 1 || !strings.Contains(rows[0].FailureTail, "boom-after-session") {
		t.Fatalf("ledger row lost failure_tail: rows=%+v err=%v", rows, err)
	}
}

func TestCronOpenCodeFailureTailAbsentOnSuccess(t *testing.T) {
	var stdout, stderr bytes.Buffer
	receipt, err := RunScheduledOpenCode(ScheduledOpenCodeOptions{
		Job:         "job-failure-tail-ok",
		Ledger:      filepath.Join(t.TempDir(), "ok.jsonl"),
		Timeout:     30 * time.Second,
		Command:     []string{os.Args[0], "-test.run=^TestCronOpenCodeSessionHelper$"},
		Env:         []string{"FAK_CRON_SESSION_HELPER=1"},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	})
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Outcome != "succeeded" || receipt.FailureTail != "" {
		t.Fatalf("outcome=%q failure_tail=%q, want succeeded with no tail", receipt.Outcome, receipt.FailureTail)
	}
}
