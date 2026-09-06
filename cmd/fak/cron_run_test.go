package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCronRunRealExecution(t *testing.T) {
	// 1. Executes a real subprocess, verifies outcome "ran", ledger records, and receipt
	t.Run("SubprocessExecutionAndReceipt", func(t *testing.T) {
		ledger := filepath.Join(t.TempDir(), "exec_receipt.jsonl")
		var stdout, stderr bytes.Buffer

		var cmdArgs []string
		if runtime.GOOS == "windows" {
			cmdArgs = []string{"cmd", "/c", "echo witness-exec"}
		} else {
			cmdArgs = []string{"echo", "witness-exec"}
		}

		argv := append([]string{
			"run",
			"--job", "witness-job-1",
			"--ledger", ledger,
			"--interval", "30m",
			"--timeout", "5s",
			"--at", "2026-09-06T14:00:00Z",
			"--json",
			"--",
		}, cmdArgs...)

		code := runCron(&stdout, &stderr, argv)
		if code != 0 {
			t.Fatalf("runCron failed with exit code %d: stderr=%s", code, stderr.String())
		}

		outStr := stdout.String()
		jsonStart := strings.Index(outStr, "{")
		if jsonStart < 0 {
			t.Fatalf("no JSON object found in stdout: %q", outStr)
		}
		var receipt cronRunReceipt
		if err := json.Unmarshal([]byte(outStr[jsonStart:]), &receipt); err != nil {
			t.Fatalf("failed to parse json receipt: %v (raw=%q)", err, outStr[jsonStart:])
		}
		if receipt.Status != cronRunStatusRan {
			t.Errorf("expected status %q, got %q", cronRunStatusRan, receipt.Status)
		}
		if receipt.ExitCode != 0 {
			t.Errorf("expected exit_code 0, got %d", receipt.ExitCode)
		}
		if receipt.Job != "witness-job-1" {
			t.Errorf("expected job 'witness-job-1', got %q", receipt.Job)
		}

		// Verify ledger
		runs, err := cronReadRuns(ledger)
		if err != nil || len(runs) != 1 {
			t.Fatalf("expected 1 run in ledger, got %d (err: %v)", len(runs), err)
		}
		if runs[0].Status != cronRunStatusRan || runs[0].ExitCode != 0 {
			t.Errorf("unexpected run in ledger: %+v", runs[0])
		}
		fires, err := cronReadFires(ledger)
		if err != nil || len(fires) != 1 {
			t.Fatalf("expected 1 fire in ledger, got %d (err: %v)", len(fires), err)
		}
		if fires[0].Outcome != cronOutcomeFired {
			t.Errorf("expected fire outcome 'fired', got %q", fires[0].Outcome)
		}
	})

	// 2. Proves duplicate suppression within the same slot boundary
	t.Run("DuplicateSuppressionWithinSlotBoundary", func(t *testing.T) {
		ledger := filepath.Join(t.TempDir(), "dedup_suppress.jsonl")
		var stdout, stderr bytes.Buffer

		var cmdArgs []string
		if runtime.GOOS == "windows" {
			cmdArgs = []string{"cmd", "/c", "echo initial-run"}
		} else {
			cmdArgs = []string{"echo", "initial-run"}
		}

		// Initial run at 10:05 with 1h interval -> slot is 10:00:00Z
		argv1 := append([]string{
			"run",
			"--job", "job-dedup-boundary",
			"--ledger", ledger,
			"--interval", "1h",
			"--timeout", "5s",
			"--at", "2026-09-06T10:05:00Z",
			"--",
		}, cmdArgs...)

		if code := runCron(&stdout, &stderr, argv1); code != 0 {
			t.Fatalf("initial run failed: exit %d, stderr=%s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "status=ran") {
			t.Errorf("initial run stdout missing 'status=ran': %s", stdout.String())
		}

		stdout.Reset()
		stderr.Reset()

		// Second run at 10:45 with 1h interval -> maps to the SAME slot (10:00:00Z)
		var cmdArgs2 []string
		if runtime.GOOS == "windows" {
			cmdArgs2 = []string{"cmd", "/c", "echo duplicate-should-not-run"}
		} else {
			cmdArgs2 = []string{"echo", "duplicate-should-not-run"}
		}

		argv2 := append([]string{
			"run",
			"--job", "job-dedup-boundary",
			"--ledger", ledger,
			"--interval", "1h",
			"--timeout", "5s",
			"--at", "2026-09-06T10:45:00Z",
			"--",
		}, cmdArgs2...)

		code2 := runCron(&stdout, &stderr, argv2)
		if code2 != cronExitDeduped {
			t.Fatalf("expected dedup exit %d, got %d (stdout=%s stderr=%s)", cronExitDeduped, code2, stdout.String(), stderr.String())
		}
		if strings.Contains(stdout.String(), "duplicate-should-not-run") {
			t.Fatalf("suppressed command executed on duplicate!")
		}
		if !strings.Contains(stdout.String(), "skipped_duplicate") {
			t.Errorf("stdout missing 'skipped_duplicate': %s", stdout.String())
		}

		// Verify fires ledger recorded dedup
		fires, err := cronReadFires(ledger)
		if err != nil {
			t.Fatalf("read fires: %v", err)
		}
		dedupCount := 0
		for _, f := range fires {
			if f.Outcome == cronOutcomeDeduped {
				dedupCount++
			}
		}
		if dedupCount != 1 {
			t.Errorf("expected 1 dedup fire record in ledger, got %d", dedupCount)
		}

		// Verify executed runs count remains 1
		runs, err := cronReadRuns(ledger)
		if err != nil || len(runs) != 1 {
			t.Errorf("expected exactly 1 executed run in ledger, got %d", len(runs))
		}
	})

	// 3. Verifies non-zero exit handling
	t.Run("NonZeroExitHandling", func(t *testing.T) {
		ledger := filepath.Join(t.TempDir(), "nonzero.jsonl")
		var stdout, stderr bytes.Buffer

		var cmdArgs []string
		if runtime.GOOS == "windows" {
			cmdArgs = []string{"cmd", "/c", "exit 42"}
		} else {
			cmdArgs = []string{"sh", "-c", "exit 42"}
		}

		argv := append([]string{
			"run",
			"--job", "job-nonzero",
			"--ledger", ledger,
			"--interval", "1h",
			"--timeout", "5s",
			"--",
		}, cmdArgs...)

		code := runCron(&stdout, &stderr, argv)
		if code != 42 {
			t.Fatalf("expected exit code 42, got %d (stderr=%s)", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "status=failed") {
			t.Errorf("stdout missing 'status=failed': %s", stdout.String())
		}

		runs, err := cronReadRuns(ledger)
		if err != nil || len(runs) != 1 {
			t.Fatalf("expected 1 run in ledger, got %d (err: %v)", len(runs), err)
		}
		if runs[0].ExitCode != 42 || runs[0].Status != cronRunStatusFailed {
			t.Errorf("ledger record mismatch: %+v", runs[0])
		}
	})

	// 4. Verifies timeout handling
	t.Run("TimeoutHandling", func(t *testing.T) {
		ledger := filepath.Join(t.TempDir(), "timeout.jsonl")
		var stdout, stderr bytes.Buffer

		var cmdArgs []string
		if runtime.GOOS == "windows" {
			cmdArgs = []string{"powershell", "-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 3"}
		} else {
			cmdArgs = []string{"sleep", "3"}
		}

		killInvoked := false
		origKill := cronRunKillTree
		cronRunKillTree = func(pid int) (bool, string) {
			killInvoked = true
			return origKill(pid)
		}
		defer func() { cronRunKillTree = origKill }()

		argv := append([]string{
			"run",
			"--job", "job-timeout-test",
			"--ledger", ledger,
			"--interval", "1h",
			"--timeout", "100ms",
			"--",
		}, cmdArgs...)

		code := runCron(&stdout, &stderr, argv)
		if code != cronRunExitTimeout {
			t.Fatalf("expected exit code %d (timeout), got %d (stderr=%s)", cronRunExitTimeout, code, stderr.String())
		}
		if !killInvoked {
			t.Errorf("expected process tree kill to be invoked on timeout")
		}
		if !strings.Contains(stdout.String(), "status=timeout") {
			t.Errorf("stdout missing 'status=timeout': %s", stdout.String())
		}

		runs, err := cronReadRuns(ledger)
		if err != nil || len(runs) != 1 {
			t.Fatalf("expected 1 run in ledger, got %d (err: %v)", len(runs), err)
		}
		if runs[0].ExitCode != cronRunExitTimeout || runs[0].Status != cronRunStatusTimeout {
			t.Errorf("ledger record mismatch: %+v", runs[0])
		}
	})
}

func TestCronRunNormalExecution(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "cron_run.jsonl")
	var stdout, stderr bytes.Buffer

	// Choose a fast, reliable cross-platform command
	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo hello-run"}
	} else {
		cmdArgs = []string{"echo", "hello-run"}
	}

	argv := append([]string{
		"run",
		"--job", "test-job-normal",
		"--ledger", ledger,
		"--interval", "10m",
		"--timeout", "5s",
		"--at", "2026-09-06T12:00:00Z",
		"--",
	}, cmdArgs...)

	code := runCron(&stdout, &stderr, argv)
	if code != 0 {
		t.Fatalf("runCron failed with exit code %d: stderr=%s", code, stderr.String())
	}

	if !strings.Contains(stdout.String(), "hello-run") {
		t.Fatalf("stdout did not contain 'hello-run': %s", stdout.String())
	}

	// Verify fire record in ledger
	fires, err := cronReadFires(ledger)
	if err != nil {
		t.Fatalf("cronReadFires error: %v", err)
	}
	if len(fires) != 1 {
		t.Fatalf("expected 1 fire record, got %d", len(fires))
	}
	if fires[0].Job != "test-job-normal" || fires[0].Outcome != cronOutcomeFired {
		t.Fatalf("unexpected fire record: %+v", fires[0])
	}

	// Verify run record in ledger
	runs, err := cronReadRuns(ledger)
	if err != nil {
		t.Fatalf("cronReadRuns error: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected 1 run record, got %d", len(runs))
	}
	r := runs[0]
	if r.Job != "test-job-normal" {
		t.Errorf("expected job 'test-job-normal', got %q", r.Job)
	}
	if r.Outcome != cronRunOutcomeSucceeded {
		t.Errorf("expected outcome 'succeeded', got %q", r.Outcome)
	}
	if r.ExitCode != 0 {
		t.Errorf("expected exit_code 0, got %d", r.ExitCode)
	}
	if r.DurationMS < 0 {
		t.Errorf("duration_ms < 0: %d", r.DurationMS)
	}
	if r.Slot != fires[0].Slot {
		t.Errorf("slot mismatch: run slot %q vs fire slot %q", r.Slot, fires[0].Slot)
	}
}

func TestCronRunDuplicateSlotSuppression(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "cron_run_dedup.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo first-run"}
	} else {
		cmdArgs = []string{"echo", "first-run"}
	}

	argv1 := append([]string{
		"run",
		"--job", "test-job-dedup",
		"--ledger", ledger,
		"--interval", "1h",
		"--timeout", "5s",
		"--at", "2026-09-06T12:00:00Z",
		"--",
	}, cmdArgs...)

	code1 := runCron(&stdout, &stderr, argv1)
	if code1 != 0 {
		t.Fatalf("first run failed with exit code %d: stderr=%s", code1, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()

	// Run second time in the same interval slot (within 1h of 12:00:00Z)
	var cmdArgs2 []string
	if runtime.GOOS == "windows" {
		cmdArgs2 = []string{"cmd", "/c", "echo second-run"}
	} else {
		cmdArgs2 = []string{"echo", "second-run"}
	}

	argv2 := append([]string{
		"run",
		"--job", "test-job-dedup",
		"--ledger", ledger,
		"--interval", "1h",
		"--timeout", "5s",
		"--at", "2026-09-06T12:30:00Z",
		"--",
	}, cmdArgs2...)

	code2 := runCron(&stdout, &stderr, argv2)
	if code2 != cronExitDeduped {
		t.Fatalf("expected dedup exit code %d, got %d (stdout=%s stderr=%s)", cronExitDeduped, code2, stdout.String(), stderr.String())
	}

	// Should print dedup message
	if !strings.Contains(stdout.String(), "deduped test-job-dedup") {
		t.Fatalf("expected dedup message on stdout, got: %s", stdout.String())
	}
	if strings.Contains(stdout.String(), "second-run") {
		t.Fatalf("command should not have executed on dedup")
	}

	// Verify only 1 run record exists
	runs, err := cronReadRuns(ledger)
	if err != nil {
		t.Fatalf("cronReadRuns error: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected 1 run record, got %d", len(runs))
	}
}

func TestCronRunNonZeroExitPropagation(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "cron_run_fail.jsonl")
	var stdout, stderr bytes.Buffer

	// Command that exits with code 7
	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "exit 7"}
	} else {
		cmdArgs = []string{"sh", "-c", "exit 7"}
	}

	argv := append([]string{
		"run",
		"--job", "test-job-fail",
		"--ledger", ledger,
		"--interval", "1h",
		"--timeout", "5s",
		"--",
	}, cmdArgs...)

	code := runCron(&stdout, &stderr, argv)
	if code != 7 {
		t.Fatalf("expected exit code 7, got %d (stderr=%s)", code, stderr.String())
	}

	runs, err := cronReadRuns(ledger)
	if err != nil {
		t.Fatalf("cronReadRuns error: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected 1 run record, got %d", len(runs))
	}
	r := runs[0]
	if r.Outcome != cronRunOutcomeFailed {
		t.Errorf("expected outcome 'failed', got %q", r.Outcome)
	}
	if r.ExitCode != 7 {
		t.Errorf("expected exit_code 7, got %d", r.ExitCode)
	}
	if r.Error == "" {
		t.Errorf("expected non-empty error message")
	}
}

func TestCronRunTimeoutEnforcement(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "cron_run_timeout.jsonl")
	var stdout, stderr bytes.Buffer

	// Command that sleeps longer than timeout (50ms timeout vs 2s sleep)
	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"powershell", "-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 2"}
	} else {
		cmdArgs = []string{"sleep", "2"}
	}

	killCalled := false
	origKill := cronRunKillTree
	cronRunKillTree = func(pid int) (bool, string) {
		killCalled = true
		return origKill(pid)
	}
	defer func() {
		cronRunKillTree = origKill
	}()

	argv := append([]string{
		"run",
		"--job", "test-job-timeout",
		"--ledger", ledger,
		"--interval", "1h",
		"--timeout", "100ms",
		"--",
	}, cmdArgs...)

	code := runCron(&stdout, &stderr, argv)
	if code != cronRunExitTimeout {
		t.Fatalf("expected exit code %d (timeout), got %d (stderr=%s)", cronRunExitTimeout, code, stderr.String())
	}
	if !killCalled {
		t.Fatalf("expected cronRunKillTree to have been invoked")
	}

	runs, err := cronReadRuns(ledger)
	if err != nil {
		t.Fatalf("cronReadRuns error: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected 1 run record, got %d", len(runs))
	}
	r := runs[0]
	if r.Outcome != cronRunOutcomeTimeout {
		t.Errorf("expected outcome 'timeout', got %q", r.Outcome)
	}
	if r.ExitCode != cronRunExitTimeout {
		t.Errorf("expected exit_code %d, got %d", cronRunExitTimeout, r.ExitCode)
	}
}
