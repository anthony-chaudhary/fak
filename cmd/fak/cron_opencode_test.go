package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"runtime"
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

func TestCronOpenCodeUntilActive(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_until_active.jsonl")
	var stdout, stderr bytes.Buffer

	cmdArgs := []string{"echo", `{"session_id": "ses_active_123"}`}

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
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}

	if receipt.Outcome != "succeeded" {
		t.Errorf("expected outcome 'succeeded', got %q", receipt.Outcome)
	}
	if receipt.SessionID != "ses_active_123" {
		t.Errorf("expected session_id 'ses_active_123', got %q", receipt.SessionID)
	}
}
