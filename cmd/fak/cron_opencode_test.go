package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestCronOpenCodeSuccess(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_success.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo {\"session_id\": \"ses_test_success_99\"}"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo '{\"session_id\": \"ses_test_success_99\"}'"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:         "job-success",
		Ledger:      ledger,
		Interval:    15 * time.Minute,
		Timeout:     5 * time.Second,
		RunID:       "run-success-001",
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("RunScheduledOpenCode error: %v (stderr: %s)", err, stderr.String())
	}

	if receipt.Schema != cronOpenCodeRunSchema {
		t.Errorf("expected schema %q, got %q", cronOpenCodeRunSchema, receipt.Schema)
	}
	if receipt.RunID != "run-success-001" {
		t.Errorf("expected run_id 'run-success-001', got %q", receipt.RunID)
	}
	if receipt.SessionID != "ses_test_success_99" {
		t.Errorf("expected session_id 'ses_test_success_99', got %q", receipt.SessionID)
	}
	if receipt.ExitCode != 0 {
		t.Errorf("expected exit_code 0, got %d", receipt.ExitCode)
	}
	if receipt.Outcome != "succeeded" {
		t.Errorf("expected outcome 'succeeded', got %q", receipt.Outcome)
	}
	if receipt.StartedAt == "" {
		t.Errorf("expected non-empty started_at")
	}
	if receipt.EndedAt == "" {
		t.Errorf("expected non-empty ended_at")
	}
	if _, err := time.Parse(time.RFC3339, receipt.StartedAt); err != nil {
		t.Errorf("started_at is not valid RFC3339: %v", err)
	}
	if _, err := time.Parse(time.RFC3339, receipt.EndedAt); err != nil {
		t.Errorf("ended_at is not valid RFC3339: %v", err)
	}
	if receipt.DurationMS < 0 {
		t.Errorf("expected non-negative duration_ms, got %d", receipt.DurationMS)
	}
	if receipt.WitnessRef != nil {
		t.Errorf("expected witness_ref to be nil, got %v", *receipt.WitnessRef)
	}

	// Verify serialized JSON contains explicitly null witness_ref
	rawJSON, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	if !strings.Contains(string(rawJSON), `"witness_ref":null`) {
		t.Errorf("expected json to contain '\"witness_ref\":null', got: %s", string(rawJSON))
	}

	// Verify ledger
	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].RunID != "run-success-001" || receipts[0].Outcome != "succeeded" {
		t.Errorf("unexpected receipt in ledger: %+v", receipts[0])
	}
}

func TestCronOpenCodeFailure(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_fail.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "exit 42"}
	} else {
		cmdArgs = []string{"sh", "-c", "exit 42"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:         "job-fail",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.Outcome != "failed" {
		t.Errorf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.ExitCode != 42 {
		t.Errorf("expected true child exit_code 42, got %d", receipt.ExitCode)
	}

	// Verify ledger recorded failure
	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].Outcome != "failed" || receipts[0].ExitCode != 42 {
		t.Errorf("ledger record mismatch: %+v", receipts[0])
	}
}

func TestCronOpenCodeSessionIDExtraction(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "JSON_session_id",
			input:    `{"session_id": "ses_01ABC123"}`,
			expected: "ses_01ABC123",
		},
		{
			name:     "JSON_sessionId",
			input:    `{"sessionId": "ses_02DEF456"}`,
			expected: "ses_02DEF456",
		},
		{
			name:     "JSON_sessionID",
			input:    `{"sessionID": "ses_03GHI789"}`,
			expected: "ses_03GHI789",
		},
		{
			name:     "JSON_session_string",
			input:    `{"session": "ses_04JKL012"}`,
			expected: "ses_04JKL012",
		},
		{
			name:     "JSON_session_object_id",
			input:    `{"session": {"id": "ses_05MNO345"}}`,
			expected: "ses_05MNO345",
		},
		{
			name:     "JSON_session_object_session_id",
			input:    `{"session": {"session_id": "ses_06PQR678"}}`,
			expected: "ses_06PQR678",
		},
		{
			name:     "JSON_session_object_sessionId",
			input:    `{"session": {"sessionId": "ses_07STU901"}}`,
			expected: "ses_07STU901",
		},
		{
			name:     "JSON_id_matching_ses",
			input:    `{"id": "ses_08VWX234", "name": "worker-1"}`,
			expected: "ses_08VWX234",
		},
		{
			name:     "JSON_id_not_matching_ses",
			input:    `{"id": "task_12345", "status": "ok"}`,
			expected: "",
		},
		{
			name:     "JSON_nested_data_wrapper",
			input:    `{"type": "event", "data": {"session_id": "ses_nested_01"}}`,
			expected: "ses_nested_01",
		},
		{
			name:     "PlainText_ses_prefix",
			input:    `2026-09-07T14:22:00Z [INFO] connected to session ses_plain_887766 cleanly`,
			expected: "ses_plain_887766",
		},
		{
			name:     "PlainText_session_id_colon",
			input:    `OpenCode daemon initialized: session_id: "s-custom-alpha-beta"`,
			expected: "s-custom-alpha-beta",
		},
		{
			name:     "PlainText_sessionId_equals",
			input:    `Worker started with sessionId=my-custom-uuid-1234`,
			expected: "my-custom-uuid-1234",
		},
		{
			name:     "PlainText_session_equals",
			input:    `[debug] session=session-val-42`,
			expected: "session-val-42",
		},
		{
			name: "MultiLine_Stream",
			input: strings.Join([]string{
				"Starting OpenCode runner...",
				`{"status": "initializing"}`,
				`{"type": "session_join", "session_id": "ses_multiline_stream_55"}`,
				"Processing turn 1...",
			}, "\n"),
			expected: "ses_multiline_stream_55",
		},
		{
			name:     "Empty_Input",
			input:    "",
			expected: "",
		},
		{
			name:     "Null_Session",
			input:    "session: null",
			expected: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := extractOpenCodeSessionID(tc.input)
			if actual != tc.expected {
				t.Errorf("extractOpenCodeSessionID(%q) = %q, want %q", tc.input, actual, tc.expected)
			}
		})
	}
}

func TestCronOpenCodeTimeout(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_timeout.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"powershell", "-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 4"}
	} else {
		cmdArgs = []string{"sleep", "4"}
	}

	killInvoked := false
	origKill := cronRunKillTree
	cronRunKillTree = func(pid int) (bool, string) {
		killInvoked = true
		return origKill(pid)
	}
	defer func() { cronRunKillTree = origKill }()

	opts := ScheduledOpenCodeOptions{
		Job:         "job-timeout",
		Ledger:      ledger,
		Timeout:     100 * time.Millisecond,
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.Outcome != "timeout" {
		t.Errorf("expected outcome 'timeout', got %q", receipt.Outcome)
	}
	if !killInvoked {
		t.Errorf("expected process tree kill to be invoked on timeout")
	}

	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].Outcome != "timeout" {
		t.Errorf("expected ledger receipt outcome 'timeout', got %q", receipts[0].Outcome)
	}
}

func TestCronOpenCodeCASDeduplication(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_dedup.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo first-run"}
	} else {
		cmdArgs = []string{"echo", "first-run"}
	}

	// 1. Initial fire at 10:05 with 1h interval -> slot is 10:00:00Z
	opts1 := ScheduledOpenCodeOptions{
		Job:         "job-dedup-test",
		Ledger:      ledger,
		Interval:    time.Hour,
		Timeout:     5 * time.Second,
		At:          "2026-09-07T10:05:00Z",
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt1, err := RunScheduledOpenCode(opts1)
	if err != nil {
		t.Fatalf("initial run error: %v", err)
	}
	if receipt1.ExitCode != 0 {
		t.Fatalf("initial run exit_code = %d, want 0", receipt1.ExitCode)
	}
	if receipt1.Outcome != "succeeded" {
		t.Fatalf("initial run outcome = %q, want 'succeeded'", receipt1.Outcome)
	}

	stdout.Reset()
	stderr.Reset()

	// 2. Duplicate fire at 10:45 with 1h interval -> maps to SAME slot (10:00:00Z)
	var cmdArgs2 []string
	if runtime.GOOS == "windows" {
		cmdArgs2 = []string{"cmd", "/c", "echo duplicate-should-not-run"}
	} else {
		cmdArgs2 = []string{"echo", "duplicate-should-not-run"}
	}

	opts2 := ScheduledOpenCodeOptions{
		Job:         "job-dedup-test",
		Ledger:      ledger,
		Interval:    time.Hour,
		Timeout:     5 * time.Second,
		At:          "2026-09-07T10:45:00Z",
		Command:     cmdArgs2,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt2, err := RunScheduledOpenCode(opts2)
	if err != nil {
		t.Fatalf("duplicate run error: %v", err)
	}
	if receipt2.ExitCode != cronExitDeduped {
		t.Fatalf("expected dedup exit code %d (cronExitDeduped), got %d", cronExitDeduped, receipt2.ExitCode)
	}

	// Verify fires ledger recorded dedup
	fires, err := cronReadFires(ledger)
	if err != nil {
		t.Fatalf("cronReadFires error: %v", err)
	}
	dedupCount := 0
	firedCount := 0
	for _, f := range fires {
		if f.Outcome == cronOutcomeDeduped {
			dedupCount++
		}
		if f.Outcome == cronOutcomeFired {
			firedCount++
		}
	}
	if firedCount != 1 {
		t.Errorf("expected 1 fired record, got %d", firedCount)
	}
	if dedupCount != 1 {
		t.Errorf("expected 1 dedup fire record, got %d", dedupCount)
	}

	// Verify that the duplicate did NOT append an executed run receipt
	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Errorf("expected exactly 1 run receipt in ledger, got %d", len(receipts))
	}
}

func TestCronOpenCodeCLIEntryPoint(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_cli.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo session_id: ses_cli_join_77"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo 'session_id: ses_cli_join_77'"}
	}

	argv := append([]string{
		"opencode",
		"--job", "cli-test-job",
		"--ledger", ledger,
		"--interval", "20m",
		"--timeout", "5s",
		"--run-id", "cli-run-id-99",
		"--",
	}, cmdArgs...)

	code := runCron(&stdout, &stderr, argv)
	if code != 0 {
		t.Fatalf("runCron failed with exit code %d (stderr: %s)", code, stderr.String())
	}

	outStr := strings.TrimSpace(stdout.String())
	if outStr == "" {
		t.Fatalf("expected stdout to contain receipt JSON, got empty")
	}

	var receipt OpenCodeRunReceipt
	if err := json.Unmarshal([]byte(outStr), &receipt); err != nil {
		t.Fatalf("failed to unmarshal stdout receipt JSON: %v (raw=%q)", err, outStr)
	}

	if receipt.Schema != cronOpenCodeRunSchema {
		t.Errorf("expected schema %q, got %q", cronOpenCodeRunSchema, receipt.Schema)
	}
	if receipt.RunID != "cli-run-id-99" {
		t.Errorf("expected run_id 'cli-run-id-99', got %q", receipt.RunID)
	}
	if receipt.SessionID != "ses_cli_join_77" {
		t.Errorf("expected session_id 'ses_cli_join_77', got %q", receipt.SessionID)
	}
	if receipt.Outcome != "succeeded" {
		t.Errorf("expected outcome 'succeeded', got %q", receipt.Outcome)
	}
	if receipt.ExitCode != 0 {
		t.Errorf("expected exit_code 0, got %d", receipt.ExitCode)
	}
	if receipt.WitnessRef != nil {
		t.Errorf("expected witness_ref nil, got %v", *receipt.WitnessRef)
	}
}

func TestCronOpenCodePassthrough(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_passthrough.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo live-child-stream"}
	} else {
		cmdArgs = []string{"echo", "live-child-stream"}
	}

	argv := append([]string{
		"opencode",
		"--job", "passthrough-job",
		"--ledger", ledger,
		"--timeout", "5s",
		"--passthrough",
		"--",
	}, cmdArgs...)

	code := runCron(&stdout, &stderr, argv)
	if code != 0 {
		t.Fatalf("runCron failed with exit code %d (stderr: %s)", code, stderr.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "live-child-stream") {
		t.Errorf("expected stdout to contain teed child stream 'live-child-stream', got: %s", outStr)
	}
	if !strings.Contains(outStr, `"schema": "fak-opencode-run/1"`) {
		t.Errorf("expected stdout to contain receipt JSON, got: %s", outStr)
	}
}

func TestCronOpenCodeUntilExpiration(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_until_expired.jsonl")
	var stdout, stderr bytes.Buffer

	cmdArgs := []string{"echo", "should-not-run"}

	// Set deadline in the past relative to --at
	atTime := time.Date(2026, 9, 16, 12, 0, 0, 0, time.UTC)
	untilTime := time.Date(2026, 9, 16, 10, 0, 0, 0, time.UTC)

	opts := ScheduledOpenCodeOptions{
		Job:         "job-expired",
		Ledger:      ledger,
		Interval:    1 * time.Hour,
		At:          atTime.Format(time.RFC3339),
		Until:       untilTime.Format(time.RFC3339),
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.Outcome != "expired" {
		t.Errorf("expected outcome 'expired', got %q", receipt.Outcome)
	}
	if receipt.ExitCode != 0 {
		t.Errorf("expected exit_code 0, got %d", receipt.ExitCode)
	}
	if receipt.DurationMS != 0 {
		t.Errorf("expected duration_ms 0, got %d", receipt.DurationMS)
	}

	// Verify ledger recorded expired receipt
	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].Outcome != "expired" {
		t.Errorf("expected ledger outcome 'expired', got %q", receipts[0].Outcome)
	}
}

func TestCronOpenCodeSessionHelper(t *testing.T) {
	if os.Getenv("FAK_CRON_SESSION_HELPER") != "1" {
		return
	}
	if _, err := os.Stdout.WriteString(`{"session_id": "ses_active_123"}` + "\n"); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}

func TestCronOpenCodeUntilActive(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_until_active.jsonl")
	var stdout, stderr bytes.Buffer

	cmdArgs := []string{os.Args[0], "-test.run=^TestCronOpenCodeSessionHelper$"}

	// Set deadline in the future relative to --at
	atTime := time.Date(2026, 9, 9, 12, 0, 0, 0, time.UTC)
	untilTime := time.Date(2026, 9, 16, 12, 0, 0, 0, time.UTC)

	opts := ScheduledOpenCodeOptions{
		Job:         "job-active",
		Ledger:      ledger,
		Interval:    1 * time.Hour,
		At:          atTime.Format(time.RFC3339),
		Until:       untilTime.Format(time.RFC3339),
		Command:     cmdArgs,
		Env:         []string{"FAK_CRON_SESSION_HELPER=1"},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.Outcome != "succeeded" {
		t.Errorf("expected outcome 'succeeded', got %q (exit_code: %d, stderr: %s)", receipt.Outcome, receipt.ExitCode, stderr.String())
	}
	if receipt.SessionID != "ses_active_123" {
		t.Errorf("expected session_id 'ses_active_123', got %q", receipt.SessionID)
	}
}

func TestCronOpenCodeAllowsLongTimeout(t *testing.T) {
	if got := cronOpenCodeEffectiveTimeout(45 * time.Minute); got != 45*time.Minute {
		t.Fatalf("expected 45m, got %v", got)
	}
	if got := cronOpenCodeEffectiveTimeout(0); got != 45*time.Minute {
		t.Fatalf("expected default 45m, got %v", got)
	}
	if got := cronOpenCodeEffectiveTimeout(3 * time.Hour); got != 2*time.Hour {
		t.Fatalf("expected clamped 2h, got %v", got)
	}
}

func TestCronOpenCodeStartupFailureNoSession(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_startup_fail.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo boom-child-error 1>&2 & exit 7"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo boom-child-error 1>&2; exit 7"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:         "job-startup-fail",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.SessionID != "" {
		t.Errorf("expected empty session_id for startup failure, got %q", receipt.SessionID)
	}
	if !receipt.StartupFailed {
		t.Errorf("expected startup_failed true, got false (receipt: %+v)", receipt)
	}
	if receipt.StartError == "" {
		t.Errorf("expected non-empty start_error containing child stderr, got empty")
	}
	if !strings.Contains(receipt.StartError, "boom-child-error") {
		t.Errorf("expected start_error to contain child stderr 'boom-child-error', got %q", receipt.StartError)
	}
	if receipt.Outcome != "failed" {
		t.Errorf("expected outcome 'failed', got %q", receipt.Outcome)
	}

	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if !receipts[0].StartupFailed {
		t.Errorf("expected ledger receipt startup_failed true, got false: %+v", receipts[0])
	}
	if receipts[0].StartError == "" {
		t.Errorf("expected ledger receipt non-empty start_error, got empty")
	}
	if receipts[0].Outcome != "failed" {
		t.Errorf("expected ledger receipt outcome 'failed', got %q", receipts[0].Outcome)
	}
}

func TestCronOpenCodeZeroExitNoSessionNotStartupFailure(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_zero_no_session.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo ran-but-no-work"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo ran-but-no-work"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:         "job-zero-no-session",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.SessionID != "" {
		t.Errorf("expected empty session_id, got %q", receipt.SessionID)
	}
	if receipt.ExitCode != 0 {
		t.Errorf("expected exit_code 0, got %d", receipt.ExitCode)
	}
	if receipt.StartupFailed {
		t.Errorf("expected startup_failed false for zero-exit no-session run, got true (receipt: %+v)", receipt)
	}
	if receipt.StartError != "" {
		t.Errorf("expected empty start_error for zero-exit no-session run, got %q", receipt.StartError)
	}
}

func TestCronOpenCodeSuccessNoStartupFailure(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_success_no_startup_fail.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo {\"session_id\": \"ses_regression_guard_11\"}"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo '{\"session_id\": \"ses_regression_guard_11\"}'"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:         "job-success-no-startup-fail",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("RunScheduledOpenCode error: %v (stderr: %s)", err, stderr.String())
	}

	if receipt.SessionID != "ses_regression_guard_11" {
		t.Errorf("expected session_id 'ses_regression_guard_11', got %q", receipt.SessionID)
	}
	if receipt.StartupFailed {
		t.Errorf("expected startup_failed false on successful run, got true (receipt: %+v)", receipt)
	}
	if receipt.StartError != "" {
		t.Errorf("expected empty start_error on successful run, got %q", receipt.StartError)
	}
}

func TestCronOpenCodePreflightMissingBinary(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_missing_binary.jsonl")
	var stdout, stderr bytes.Buffer

	opts := ScheduledOpenCodeOptions{
		Job:         "job-missing-binary",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     []string{"fak-nonexistent-binary-xyz-zzz"},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err == nil {
		t.Fatalf("expected non-nil error for missing binary pre-flight probe")
	}

	if !receipt.StartupFailed {
		t.Errorf("expected startup_failed true, got false (receipt: %+v)", receipt)
	}
	if receipt.ExitCode != 2 {
		t.Errorf("expected exit_code 2, got %d", receipt.ExitCode)
	}
	if !strings.Contains(receipt.StartError, "command not found") {
		t.Errorf("expected start_error to mention 'command not found', got %q", receipt.StartError)
	}
}

func TestCronOpenCodeExecFailureSurfacesRunErrWhenOutputEmpty(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_exec_failure.jsonl")
	var stdout, stderr bytes.Buffer

	// A path binary that cannot be exec'd emits no child output, so the only
	// diagnostic is runErr; the receipt must still carry a non-empty start_error.
	missing := filepath.Join(t.TempDir(), "definitely-missing-opencode-binary")
	opts := ScheduledOpenCodeOptions{
		Job:         "job-exec-failure",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     []string{missing},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if !receipt.StartupFailed {
		t.Errorf("expected startup_failed true for a non-executable path binary, got false: %+v", receipt)
	}
	if receipt.StartError == "" {
		t.Errorf("expected non-empty start_error from runErr when child output is empty")
	}
	if receipt.SessionID != "" {
		t.Errorf("expected empty session_id, got %q", receipt.SessionID)
	}
}

// TestCronOpenCodeCrashHelper is a deterministic stub child for the Bun-crash
// retry tests (#1316). It is gated by FAK_CRON_CRASH_HELPER so it is inert in a
// normal `go test` run, and it records its invocation count in a counter file so
// the parent's re-exec can advance the attempt sequence.
func TestCronOpenCodeCrashHelper(t *testing.T) {
	if os.Getenv("FAK_CRON_CRASH_HELPER") != "1" {
		return
	}
	mode := os.Getenv("FAK_CRON_CRASH_MODE")
	counter := os.Getenv("FAK_CRON_CRASH_COUNTER")
	n := 0
	if counter != "" {
		if b, err := os.ReadFile(counter); err == nil {
			n, _ = strconv.Atoi(strings.TrimSpace(string(b)))
		}
		n++
		_ = os.WriteFile(counter, []byte(strconv.Itoa(n)), 0o644)
	}
	crash := func() {
		fmt.Fprintln(os.Stderr, "panic(thread 11): Segmentation fault at address 0x0")
		fmt.Fprintln(os.Stderr, "oh no: Bun has crashed. This indicates a bug in Bun, not your code.")
		fmt.Fprintln(os.Stderr, "Bun v1.3.14 (baseline) Windows x64")
		fmt.Fprintln(os.Stderr, "https://bun.report/1.3.14")
		os.Exit(3)
	}
	switch mode {
	case "crash-then-ok":
		if n <= 1 {
			crash()
		}
		fmt.Println(`{"session_id": "ses_retry_ok"}`)
		os.Exit(0)
	case "crash-always":
		crash()
	case "exit3-nocrash":
		fmt.Fprintln(os.Stderr, "real opencode error: invalid flag")
		os.Exit(3)
	}
	os.Exit(0)
}

func cronOpenCodeCrashHelperCommand(t *testing.T, mode string) ([]string, []string) {
	dir := t.TempDir()
	counter := filepath.Join(dir, "attempts")
	cmdArgs := []string{os.Args[0], "-test.run=^TestCronOpenCodeCrashHelper$"}
	env := []string{
		"FAK_CRON_CRASH_HELPER=1",
		"FAK_CRON_CRASH_MODE=" + mode,
		"FAK_CRON_CRASH_COUNTER=" + counter,
	}
	t.Cleanup(func() { _ = os.Remove(counter) })
	return cmdArgs, env
}

func TestCronOpenCodeBunCrashRetryRecovers(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_crash_retry.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "crash-then-ok")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-crash-retry",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 1,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v (stderr: %s)", err, stderr.String())
	}
	if receipt.Outcome != "succeeded" {
		t.Errorf("expected outcome 'succeeded' after recovered crash, got %q (exit=%d)", receipt.Outcome, receipt.ExitCode)
	}
	if receipt.Attempts != 2 {
		t.Errorf("expected attempts==2, got %d", receipt.Attempts)
	}
	if !receipt.CrashSignature {
		t.Errorf("expected crash_signature true, got false (receipt: %+v)", receipt)
	}
	if !receipt.CrashRecovered {
		t.Errorf("expected crash_recovered true, got false (receipt: %+v)", receipt)
	}
	if receipt.SessionID != "ses_retry_ok" {
		t.Errorf("expected session_id 'ses_retry_ok', got %q", receipt.SessionID)
	}

	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected exactly 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].Outcome != "succeeded" || receipts[0].Attempts != 2 {
		t.Errorf("ledger receipt mismatch: %+v", receipts[0])
	}
}

func TestCronOpenCodeNonCrashExitNotRetried(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_nocrash.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "exit3-nocrash")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-nocrash",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 2,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Outcome != "failed" {
		t.Errorf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.ExitCode != 3 {
		t.Errorf("expected exit_code 3, got %d", receipt.ExitCode)
	}
	if receipt.Attempts != 1 {
		t.Errorf("expected no retry on non-crash failure (attempts==1), got %d", receipt.Attempts)
	}
	if receipt.CrashSignature {
		t.Errorf("expected crash_signature false, got true")
	}
	if receipt.CrashRecovered {
		t.Errorf("expected crash_recovered false, got true")
	}
}

func TestCronOpenCodeBunCrashBoundedRetries(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_crash_bounded.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "crash-always")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-crash-bounded",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 2,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Outcome != "failed" {
		t.Errorf("expected outcome 'failed' after exhausting retries, got %q", receipt.Outcome)
	}
	if receipt.Attempts != 3 {
		t.Errorf("expected attempts==3 (1 + 2 retries), got %d", receipt.Attempts)
	}
	if !receipt.CrashSignature {
		t.Errorf("expected crash_signature true, got false")
	}
	if receipt.CrashRecovered {
		t.Errorf("expected crash_recovered false when all attempts crash, got true")
	}
}

func TestCronOpenCodeCrashRetriesZeroDisables(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_crash_zero.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "crash-then-ok")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-crash-zero",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 0,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Outcome != "failed" {
		t.Errorf("expected outcome 'failed' with --crash-retries 0, got %q", receipt.Outcome)
	}
	if receipt.ExitCode != 3 {
		t.Errorf("expected exit_code 3, got %d", receipt.ExitCode)
	}
	if receipt.Attempts != 1 {
		t.Errorf("expected attempts==1 when retries disabled, got %d", receipt.Attempts)
	}
	if !receipt.CrashSignature {
		t.Errorf("expected crash_signature true, got false")
	}
}

func TestCronOpenCodeBunCrashSignatureDetector(t *testing.T) {
	if cronOpenCodeBunCrashSignature("") {
		t.Errorf("empty output must not be a crash signature")
	}
	if cronOpenCodeBunCrashSignature("exit status 3") {
		t.Errorf("bare non-zero exit must not be a crash signature")
	}
	if !cronOpenCodeBunCrashSignature("panic(thread 1): Segmentation fault at address 0x10\noh no: Bun has crashed. This indicates a bug in Bun, not your code.") {
		t.Errorf("expected Bun crash signature to be detected")
	}
	if !cronOpenCodeBunCrashSignature("Segmentation fault\nsee https://bun.report/1.3.14") {
		t.Errorf("expected bun.report link alone to count")
	}
}

// --- Provider-failure classification (#1866): TRANSIENT-vs-ONGOING producer half ---
//
// A failed/timeout receipt MUST carry failure_class ("transient" | "ongoing"); a
// succeeded receipt MUST omit it entirely (byte-identical to the pre-#1866 shape).

// cronOpenCodeFailureClassCommand builds a cross-platform child that exits
// non-zero after writing `stderrText` (best-effort, via the shell) so a failure
// run deterministically surfaces that text in the bounded receipt output.
func cronOpenCodeFailureClassCommand(stderrText string) []string {
	if runtime.GOOS == "windows" {
		return []string{"cmd", "/c", "echo " + stderrText + " 1>&2 & exit 1"}
	}
	return []string{"sh", "-c", "echo " + strconv.Quote(stderrText) + " 1>&2; exit 1"}
}

// cronRunFailureClassChild executes a failing child whose stderr carries
// stderrText, and returns the receipt (asserting the run itself is a harness
// success so the classification assertions stay meaningful).
func cronRunFailureClassChild(t *testing.T, job, stderrText string) OpenCodeRunReceipt {
	t.Helper()
	ledger := filepath.Join(t.TempDir(), "opencode_failure_class.jsonl")
	var stdout, stderr bytes.Buffer

	opts := ScheduledOpenCodeOptions{
		Job:         job,
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     cronOpenCodeFailureClassCommand(stderrText),
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v (stderr: %s)", err, stderr.String())
	}
	if receipt.Outcome != "failed" {
		t.Fatalf("expected outcome 'failed', got %q (exit=%d, stderr=%s)", receipt.Outcome, receipt.ExitCode, stderr.String())
	}

	// The classification must survive the JSONL round-trip, not just live on the
	// in-memory struct: read the ledger back and cross-check.
	receipts, readErr := cronReadOpenCodeReceipts(ledger)
	if readErr != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", readErr)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].FailureClass != receipt.FailureClass {
		t.Errorf("ledger failure_class %q != in-memory %q (round-trip drift)", receipts[0].FailureClass, receipt.FailureClass)
	}
	return receipt
}

func TestCronOpenCodeFailureClassTransient500(t *testing.T) {
	receipt := cronRunFailureClassChild(t, "job-class-500", "provider error: HTTP 500 Internal Server Error")
	if receipt.Outcome != "failed" {
		t.Fatalf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.FailureClass != "transient" {
		t.Errorf("expected failure_class 'transient' for HTTP 500, got %q (receipt: %+v)", receipt.FailureClass, receipt)
	}
}

func TestCronOpenCodeFailureClassTransient429(t *testing.T) {
	receipt := cronRunFailureClassChild(t, "job-class-429", "rate limited: status 429 Too Many Requests")
	if receipt.Outcome != "failed" {
		t.Fatalf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.FailureClass != "transient" {
		t.Errorf("expected failure_class 'transient' for HTTP 429, got %q", receipt.FailureClass)
	}
}

func TestCronOpenCodeFailureClassTransientConnectionReset(t *testing.T) {
	receipt := cronRunFailureClassChild(t, "job-class-reset", "read tcp 10.0.0.1:443: connection reset by peer")
	if receipt.Outcome != "failed" {
		t.Fatalf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.FailureClass != "transient" {
		t.Errorf("expected failure_class 'transient' for connection reset, got %q", receipt.FailureClass)
	}
}

func TestCronOpenCodeFailureClassOngoingPausedOrg(t *testing.T) {
	receipt := cronRunFailureClassChild(t, "job-class-paused", "HTTP 405 Method Not Allowed: organization is paused")
	if receipt.Outcome != "failed" {
		t.Fatalf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.FailureClass != "ongoing" {
		t.Errorf("expected failure_class 'ongoing' for paused org (HTTP 405), got %q", receipt.FailureClass)
	}
}

func TestCronOpenCodeFailureClassOngoingUnclassified(t *testing.T) {
	// The critical invariant: a failure the classifier cannot positively identify
	// as a bounded provider blip MUST default to "ongoing" (conservative), never
	// "transient".
	receipt := cronRunFailureClassChild(t, "job-class-unclassified", "invalid flag: --nonexistent-option")
	if receipt.Outcome != "failed" {
		t.Fatalf("expected outcome 'failed', got %q", receipt.Outcome)
	}
	if receipt.FailureClass != "ongoing" {
		t.Errorf("expected conservative failure_class 'ongoing' for unclassifiable failure, got %q", receipt.FailureClass)
	}
}

func TestCronOpenCodeFailureClassOmittedOnSuccess(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_class_success.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo {\"session_id\": \"ses_class_ok\"}"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo '{\"session_id\": \"ses_class_ok\"}'"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:         "job-class-success",
		Ledger:      ledger,
		Timeout:     5 * time.Second,
		Command:     cmdArgs,
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("RunScheduledOpenCode error: %v (stderr: %s)", err, stderr.String())
	}
	if receipt.Outcome != "succeeded" {
		t.Fatalf("expected outcome 'succeeded', got %q", receipt.Outcome)
	}
	if receipt.FailureClass != "" {
		t.Errorf("expected empty FailureClass on success, got %q", receipt.FailureClass)
	}

	// The success receipt must be byte-identical to the pre-#1866 schema: the
	// additive omitempty field contributes ZERO bytes when empty.
	rawJSON, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	if strings.Contains(string(rawJSON), "failure_class") {
		t.Errorf("succeeded receipt JSON must NOT contain 'failure_class', got: %s", string(rawJSON))
	}

	// Independent of field presence, an empty-string-typed field landing in the
	// ledger must also serialize without the key (round-trip read-back).
	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("expected 1 receipt in ledger, got %d", len(receipts))
	}
	if receipts[0].FailureClass != "" {
		t.Errorf("expected ledger succeeded receipt FailureClass empty, got %q", receipts[0].FailureClass)
	}
}

func TestCronClassifyFailureClass(t *testing.T) {
	tests := []struct {
		name     string
		outcome  string
		output   string
		expected string
	}{
		{name: "empty_outcome", outcome: "", output: "HTTP 500", expected: ""},
		{name: "succeeded", outcome: "succeeded", output: "HTTP 500 Internal Server Error", expected: ""},
		{name: "expired_not_a_failure_class", outcome: "expired", output: "", expected: ""},
		{name: "deduped_not_a_failure_class", outcome: "deduped", output: "", expected: ""},

		{name: "failed_http_500", outcome: "failed", output: "provider error: HTTP 500 Internal Server Error", expected: "transient"},
		{name: "failed_http_429", outcome: "failed", output: "rate limited: status 429 Too Many Requests", expected: "transient"},
		{name: "failed_http_503", outcome: "failed", output: "HTTP/1.1 503 Service Unavailable", expected: "transient"},
		{name: "timeout_http_502", outcome: "timeout", output: "bad gateway: HTTP 502", expected: "transient"},
		{name: "failed_status_code_eq_504", outcome: "failed", output: "status_code=504", expected: "transient"},
		{name: "failed_connection_reset", outcome: "failed", output: "read tcp: connection reset by peer", expected: "transient"},

		{name: "failed_paused_org_405", outcome: "failed", output: "HTTP 405 Method Not Allowed: organization is paused", expected: "ongoing"},
		{name: "failed_paused_word_only", outcome: "failed", output: "this account is paused", expected: "ongoing"},
		{name: "failed_unrecognized_stderr", outcome: "failed", output: "invalid flag: --nope", expected: "ongoing"},
		{name: "failed_empty_output", outcome: "failed", output: "", expected: "ongoing"},
		{name: "timeout_empty_output", outcome: "timeout", output: "", expected: "ongoing"},

		// Conservative-default guard: a bare "500" with NO HTTP/status context (for
		// example a JSON body carrying a numeric field) must NOT be read as a
		// transient provider blip. The regex requires an explicit status context
		// token, so this stays "ongoing".
		{name: "bare_500_without_context_is_ongoing", outcome: "failed", output: `{"latency_ms": 500, "tokens": 123}`, expected: "ongoing"},
		{name: "bare_429_without_context_is_ongoing", outcome: "failed", output: `{"retry_after": 429}`, expected: "ongoing"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := cronClassifyFailureClass(tc.outcome, tc.output)
			if got != tc.expected {
				t.Errorf("cronClassifyFailureClass(%q, %q) = %q, want %q", tc.outcome, tc.output, got, tc.expected)
			}
		})
	}
}
