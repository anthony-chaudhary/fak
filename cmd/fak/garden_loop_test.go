package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestGardenLoopLockAcquireAndRelease(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()

	lock1, ok1, err := acquireGardenLoopLock(dir, now, time.Hour)
	if err != nil || !ok1 || lock1 == nil {
		t.Fatalf("first acquire failed: ok=%v, err=%v", ok1, err)
	}

	// Second acquire should be refused (overlap prevention).
	lock2, ok2, err := acquireGardenLoopLock(dir, now, time.Hour)
	if err != nil {
		t.Fatalf("second acquire error: %v", err)
	}
	if ok2 || lock2 != nil {
		t.Fatalf("second acquire succeeded unexpectedly: %+v", lock2)
	}

	lock1.release()

	// Third acquire after release should succeed.
	lock3, ok3, err := acquireGardenLoopLock(dir, now, time.Hour)
	if err != nil || !ok3 || lock3 == nil {
		t.Fatalf("third acquire failed: ok=%v, err=%v", ok3, err)
	}
	lock3.release()
}

func TestGardenLoopSingleIterationDryRun(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "loop-ledger.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenLoop(&stdout, &stderr, []string{
		"--repo", dir,
		"--iterations", "1",
		"--live=false",
		"--interval", "10ms",
		"--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("runGardenLoop failed with code %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "[cycle 1]") {
		t.Fatalf("expected cycle 1 in output, got: %s", out)
	}
	if !strings.Contains(out, "watchdog:") {
		t.Fatalf("expected watchdog sweep in output, got: %s", out)
	}
}

func TestGardenLoopSingleIterationJSON(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "loop-ledger.jsonl")

	var stdout, stderr bytes.Buffer
	code := runGardenLoop(&stdout, &stderr, []string{
		"--repo", dir,
		"--iterations", "1",
		"--live=false",
		"--interval", "10ms",
		"--ledger", ledger,
		"--json",
	})
	if code != 0 {
		t.Fatalf("runGardenLoop failed with code %d, stderr: %s", code, stderr.String())
	}

	var rep gardenLoopCycleReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to parse JSON output: %v, raw: %s", err, stdout.String())
	}
	if rep.Cycle != 1 {
		t.Fatalf("expected cycle 1, got %d", rep.Cycle)
	}
	if rep.Schema != gardenLoopSchema {
		t.Fatalf("expected schema %q, got %q", gardenLoopSchema, rep.Schema)
	}
	if rep.Live {
		t.Fatalf("expected live=false, got true")
	}
}

func TestGardenLoopGracefulContextCancellation(t *testing.T) {
	dir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())

	opts := gardenLoopOptions{
		Repo:            dir,
		Interval:        50 * time.Millisecond,
		Live:            false,
		CleanBins:       false,
		TreeDoctor:      false,
		PackRefs:        false,
		MaxAgeDays:      7,
		WatchdogTimeout: time.Second,
		TickBudget:      time.Second,
		Iterations:      0, // infinite
		AsJSON:          true,
	}

	done := make(chan int, 1)
	var stdout, stderr bytes.Buffer
	go func() {
		done <- runGardenLoopProcess(ctx, &stdout, &stderr, opts)
	}()

	// Allow first cycle to run, then cancel context.
	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("expected exit code 0 on cancel, got %d", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for garden loop to exit on cancellation")
	}
}
func TestGardenLoopRegisterDurableUnit(t *testing.T) {
	// Hermetic: stub the OS-registration step so --register never shells out and
	// never bakes this transient test binary into a real host scheduler task.
	// Pin the executable seam to a durable install path so the transient-artifact
	// refusal does not short-circuit the registration path under test.
	prevExe := gardenLoopExecutable
	gardenLoopExecutable = func() (string, error) {
		return filepath.Join(string(filepath.Separator)+"opt", "fak", "fak"), nil
	}
	t.Cleanup(func() { gardenLoopExecutable = prevExe })

	prevReg := gardenLoopRegisterOS
	var gotBin string
	gardenLoopRegisterOS = func(_ io.Writer, _ io.Writer, fakBin, _ string, _ time.Duration, _ bool) int {
		gotBin = fakBin
		return 0
	}
	t.Cleanup(func() { gardenLoopRegisterOS = prevReg })

	dir := t.TempDir()
	registry := filepath.Join(dir, "loop-registry.json")

	var stdout, stderr bytes.Buffer
	code := runGardenLoop(&stdout, &stderr, []string{
		"--repo", dir,
		"--register",
		"--registry", registry,
		"--interval", "2h",
	})
	// Registration may fail or succeed OS scheduler step depending on platform permissions,
	// but the durable loop in registry must always be written.
	reg, err := loopmgr.LoadRegistry(registry)
	if err != nil {
		t.Fatalf("failed to load registry: %v", err)
	}
	job, ok := reg.Get(gardenLoopID)
	if !ok {
		t.Fatalf("job %q not found in registry", gardenLoopID)
	}
	if !job.State.Armed() {
		t.Fatalf("job state = %q, want armed", job.State)
	}
	if job.Schedule.IntervalSeconds != 7200 {
		t.Fatalf("job interval = %d, want 7200", job.Schedule.IntervalSeconds)
	}
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if gotBin == "" {
		t.Fatalf("OS registration not reached; stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
	t.Logf("OS registration reached with resolved binary %q", gotBin)
}

// TestGardenLoopRegisterRefusesTransientTestBinary witnesses the fix for the
// live FleetStaleWorkGarden failure class: `--register` run from a transient
// test/build binary must NOT persist a host-scheduler unit pointing at that
// artifact (which breaks with ERROR_FILE_NOT_FOUND once the temp dir is
// cleaned). The registry half still succeeds; the OS step is skipped with a
// named reason and exit 0.
func TestGardenLoopRegisterRefusesTransientTestBinary(t *testing.T) {
	cases := map[string]string{
		"go-build work dir": filepath.Join("C:", "Users", "u", "AppData", "Local", "Temp", "go-build709495352", "b001", "fak.test.exe"),
		"tempdir root":      filepath.Join(os.TempDir(), "fak.test.exe"),
		"b001 link step":    filepath.Join("repo", "b001", "fak.exe"),
		"repo worktree":     filepath.Join("C:", "work", "fak", ".worktrees", "ticket-x", "fak.exe"),
		"repo gotmp":        filepath.Join("C:", "work", "fak", ".worktrees", ".gotmp-foo", "TestX", "002", "fak"),
	}
	for name, execPath := range cases {
		execPath := execPath
		t.Run(name, func(t *testing.T) {
			if !transientExecutable(execPath) {
				t.Fatalf("transientExecutable(%q) = false, want true", execPath)
			}
			prevExe := gardenLoopExecutable
			gardenLoopExecutable = func() (string, error) { return execPath, nil }
			t.Cleanup(func() { gardenLoopExecutable = prevExe })

			// If the refusal is bypassed, this stub would record it: a hit here
			// means the transient path reached real OS registration.
			prevReg := gardenLoopRegisterOS
			osReached := false
			gardenLoopRegisterOS = func(_ io.Writer, _ io.Writer, _, _ string, _ time.Duration, _ bool) int {
				osReached = true
				return 0
			}
			t.Cleanup(func() { gardenLoopRegisterOS = prevReg })

			dir := t.TempDir()
			registry := filepath.Join(dir, "loop-registry.json")
			var stdout, stderr bytes.Buffer
			code := runGardenLoop(&stdout, &stderr, []string{
				"--repo", dir,
				"--register",
				"--registry", registry,
				"--interval", "2h",
			})
			if code != 0 {
				t.Fatalf("exit code = %d, want 0 (a skipped OS step is a valid outcome)\nstdout=%s\nstderr=%s", code, stdout.String(), stderr.String())
			}
			if !strings.Contains(stdout.String(), "transient test/build artifact") {
				t.Fatalf("stdout missing transient-artifact skip reason:\n%s", stdout.String())
			}
			if osReached {
				t.Fatalf("transient path %q reached real OS registration — must be refused", execPath)
			}
			// The durable registry half must still be written.
			reg, err := loopmgr.LoadRegistry(registry)
			if err != nil {
				t.Fatalf("failed to load registry: %v", err)
			}
			if _, ok := reg.Get(gardenLoopID); !ok {
				t.Fatalf("job %q not found in registry", gardenLoopID)
			}
		})
	}

	// Negative control: durable installed paths are NOT transient.
	for _, durable := range []string{
		filepath.Join("C:", "Users", "u", "bin", "fak.exe"),
		filepath.Join("C:", "VulkanSDK", "1.4.350.0", "Bin", "fak.exe"),
		"/usr/local/bin/fak",
	} {
		if transientExecutable(durable) {
			t.Fatalf("transientExecutable flagged a durable install path %q", durable)
		}
	}
}
