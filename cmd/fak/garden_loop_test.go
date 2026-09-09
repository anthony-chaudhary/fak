package main

import (
	"bytes"
	"context"
	"encoding/json"
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
	_ = code
}
