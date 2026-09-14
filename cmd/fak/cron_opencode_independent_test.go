package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestCronOpenCodeIndCrashHelper is this file's own deterministic stub child.
// It is gated by FAK_CRON_IND_HELPER so it is inert during a normal test run.
// Modes:
//
//	crash-stdout-then-ok - crash signature on STDOUT on attempt 1, then succeed
//	empty-exit9          - no output at all, exit 9
//	segfault-only        - "segmentation fault" alone (no bun marker), exit 3
//	buncrash-only        - "bun has crashed" alone (no segfault marker), exit 3
func TestCronOpenCodeIndCrashHelper(t *testing.T) {
	if os.Getenv("FAK_CRON_IND_HELPER") != "1" {
		return
	}
	mode := os.Getenv("FAK_CRON_IND_MODE")
	counter := os.Getenv("FAK_CRON_IND_COUNTER")
	n := 0
	if counter != "" {
		if b, err := os.ReadFile(counter); err == nil {
			n, _ = strconv.Atoi(strings.TrimSpace(string(b)))
		}
		n++
		_ = os.WriteFile(counter, []byte(strconv.Itoa(n)), 0o644)
	}
	switch mode {
	case "crash-stdout-then-ok":
		if n <= 1 {
			// Signature on STDOUT, nothing on stderr, then a non-zero exit.
			if _, err := os.Stdout.WriteString("panic(thread 11): Segmentation fault at address 0x0\noh no: Bun has crashed. This indicates a bug in Bun, not your code.\n"); err != nil {
				os.Exit(1)
			}
			os.Exit(3)
		}
		if _, err := os.Stdout.WriteString("{\"session_id\": \"ses_stdout_retry_ok\"}\n"); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	case "empty-exit9":
		os.Exit(9)
	case "segfault-only":
		if _, err := os.Stderr.WriteString("panic(thread 11): Segmentation fault at address 0x0\n"); err != nil {
			os.Exit(1)
		}
		os.Exit(3)
	case "buncrash-only":
		if _, err := os.Stderr.WriteString("oh no: Bun has crashed. This indicates a bug in Bun, not your code.\n"); err != nil {
			os.Exit(1)
		}
		os.Exit(3)
	}
	os.Exit(0)
}

func cronOpenCodeIndHelperCommand(t *testing.T, mode string) ([]string, []string) {
	dir := t.TempDir()
	counter := filepath.Join(dir, "attempts")
	cmdArgs := []string{os.Args[0], "-test.run=^TestCronOpenCodeIndCrashHelper$"}
	env := []string{
		"FAK_CRON_IND_HELPER=1",
		"FAK_CRON_IND_MODE=" + mode,
		"FAK_CRON_IND_COUNTER=" + counter,
	}
	t.Cleanup(func() { _ = os.Remove(counter) })
	return cmdArgs, env
}

func TestCronOpenCodeIndependentSignaturePartialMarkersNoFire(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want bool
	}{
		{"segfault marker alone", "panic(thread 1): Segmentation fault at address 0x0", false},
		{"bun has crashed marker alone", "oh no: Bun has crashed. This indicates a bug in Bun, not your code.", false},
		{"bun.report marker alone", "see https://bun.report/1.3.14 for a report", false},
		{"segfault plus unrelated bun word", "Segmentation fault\nrunning bun tests\nall good", false},
		{"both markers, segfault first", "Segmentation fault\nBun has crashed", true},
		{"both markers, crash first", "oh no: Bun has crashed.\npanic(thread 1): Segmentation fault at address 0x0", true},
		{"segfault plus bun.report", "Segmentation fault\nhttps://bun.report/1.3.14", true},
		{"case-insensitive", "SEGMENTATION FAULT\nBUN HAS CRASHED", true},
		{"empty", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := cronOpenCodeBunCrashSignature(tc.in); got != tc.want {
				t.Errorf("cronOpenCodeBunCrashSignature(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestCronOpenCodeIndependentNegativeCrashRetriesClampedToZero(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_negative_retries.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "crash-then-ok")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-negative-retries",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: -5,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v (stderr: %s)", err, stderr.String())
	}
	if receipt.Outcome != "failed" {
		t.Errorf("negative CrashRetries must clamp to 0 (no retry); got outcome %q", receipt.Outcome)
	}
	if receipt.Attempts != 1 {
		t.Errorf("negative CrashRetries must clamp to 0 attempts=1, got %d", receipt.Attempts)
	}
	if receipt.ExitCode != 3 {
		t.Errorf("expected crash exit_code 3, got %d", receipt.ExitCode)
	}
	if !receipt.CrashSignature {
		t.Errorf("expected crash_signature true (crash occurred), got false")
	}
	if receipt.CrashRecovered {
		t.Errorf("crash_recovered must be false when retry is disabled")
	}
}

func TestCronOpenCodeIndependentNoCrashSuccessFastPathMemo(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_fast_path.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo {\"session_id\": \"ses_fast_path_ok\"}"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo '{\"session_id\": \"ses_fast_path_ok\"}'"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:          "job-fast-path",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 3,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Attempts != 1 {
		t.Errorf("expected a single attempt when output carries no crash signature, got %d", receipt.Attempts)
	}
	if receipt.CrashSignature {
		t.Errorf("a clean run must NOT be stamped crash_signature=true")
	}
	if receipt.CrashRecovered {
		t.Errorf("a clean run must NOT be stamped crash_recovered=true")
	}
	if receipt.Outcome != "succeeded" {
		t.Errorf("expected succeeded, got %q", receipt.Outcome)
	}
	raw, _ := os.ReadFile(ledger)
	if strings.Contains(string(raw), `"crash_recovered":true`) {
		t.Errorf("ledger must not claim crash_recovered=true for a clean run: %s", string(raw))
	}
	if strings.Contains(string(raw), `"crash_signature":true`) {
		t.Errorf("ledger must not claim crash_signature=true for a clean run: %s", string(raw))
	}
}

func TestCronOpenCodeIndependentSignatureOnStdoutDetected(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_stdout_crash.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeIndHelperCommand(t, "crash-stdout-then-ok")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-stdout-crash",
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
	if !receipt.CrashSignature {
		t.Errorf("signature printed on STDOUT must be detected; got crash_signature=false")
	}
	if receipt.Attempts != 2 {
		t.Errorf("expected retry to fire on stdout-borne signature (attempts=2), got %d", receipt.Attempts)
	}
	if receipt.Outcome != "succeeded" || !receipt.CrashRecovered {
		t.Errorf("expected recovery on second attempt, got outcome=%q recovered=%v", receipt.Outcome, receipt.CrashRecovered)
	}
}

func TestCronOpenCodeIndependentEmptyOutputNonzeroExitNoRetry(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_empty_nocrash.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeIndHelperCommand(t, "empty-exit9")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-empty-nocrash",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 4,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.Attempts != 1 {
		t.Errorf("empty-output nonzero exit must NOT retry; got attempts=%d", receipt.Attempts)
	}
	if receipt.CrashSignature {
		t.Errorf("empty output must not be a crash signature")
	}
	if receipt.ExitCode != 9 {
		t.Errorf("expected exit_code 9, got %d", receipt.ExitCode)
	}
}

func TestCronOpenCodeIndependentSegfaultOnlyNotRetried(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_segfault_only.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeIndHelperCommand(t, "segfault-only")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-segfault-only",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 3,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.CrashSignature {
		t.Errorf("'segmentation fault' without a Bun marker must not be a crash signature")
	}
	if receipt.Attempts != 1 {
		t.Errorf("must not retry on a bare segfault marker; got attempts=%d", receipt.Attempts)
	}
}

func TestCronOpenCodeIndependentBunCrashOnlyNotRetried(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_buncrash_only.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeIndHelperCommand(t, "buncrash-only")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-buncrash-only",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Env:          env,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 3,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.CrashSignature {
		t.Errorf("'bun has crashed' without a segfault marker must not be a crash signature")
	}
	if receipt.Attempts != 1 {
		t.Errorf("must not retry on a bare bun-crash marker; got attempts=%d", receipt.Attempts)
	}
}

func TestCronOpenCodeIndependentCrashRetryAppendsFireLedgerOnce(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_crash_ledger.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "crash-then-ok")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-crash-ledger-once",
		Ledger:       ledger,
		Interval:     time.Hour,
		At:           "2026-09-07T10:05:00Z",
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
	if receipt.Attempts != 2 {
		t.Fatalf("precondition: expected 2 attempts, got %d", receipt.Attempts)
	}

	fires, err := cronReadFires(ledger)
	if err != nil {
		t.Fatalf("cronReadFires error: %v", err)
	}
	fired := 0
	for _, f := range fires {
		if f.Job == opts.Job && f.Outcome == cronOutcomeFired {
			fired++
		}
	}
	if fired != 1 {
		t.Errorf("retry must not re-run CAS fire step: expected exactly 1 fired record, got %d", fired)
	}

	receipts, err := cronReadOpenCodeReceipts(ledger)
	if err != nil {
		t.Fatalf("cronReadOpenCodeReceipts error: %v", err)
	}
	if len(receipts) != 1 {
		t.Errorf("retry must append exactly 1 run receipt, got %d", len(receipts))
	}
}

func TestCronOpenCodeIndependentBunVersionFromOutput(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_bun_version.jsonl")
	var stdout, stderr bytes.Buffer
	cmdArgs, env := cronOpenCodeCrashHelperCommand(t, "crash-then-ok")

	opts := ScheduledOpenCodeOptions{
		Job:          "job-bun-version",
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
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.BunVersion != "1.3.14" {
		t.Errorf("expected bun_version 1.3.14 extracted from crash banner, got %q", receipt.BunVersion)
	}
}

func TestCronOpenCodeIndependentNonopencodeReceiptZeroVersion(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "opencode_probe_skip.jsonl")
	var stdout, stderr bytes.Buffer

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo {\"session_id\": \"ses_probe_skip\"}"}
	} else {
		cmdArgs = []string{"sh", "-c", "echo '{\"session_id\": \"ses_probe_skip\"}'"}
	}

	opts := ScheduledOpenCodeOptions{
		Job:          "job-probe-skip",
		Ledger:       ledger,
		Timeout:      10 * time.Second,
		Command:      cmdArgs,
		Stdout:       &stdout,
		Stderr:       &stderr,
		EmitReceipt:  true,
		CrashRetries: 0,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if err != nil {
		t.Fatalf("unexpected harness error: %v", err)
	}
	if receipt.OpenCodeVersion != "" {
		t.Errorf("non-opencode command must skip the --version probe; got %q", receipt.OpenCodeVersion)
	}
}
