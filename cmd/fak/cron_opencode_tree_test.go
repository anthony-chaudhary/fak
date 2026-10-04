package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// cronTreeHelperEnv selects a role for a re-exec of this test binary in
// TestCronOpenCodeHardTimeoutKillsGrandchild. It is inert in a normal run.
const cronTreeHelperEnv = "FAK_CRON_TREE_HELPER"

// cronTreeTiming returns the hard ceiling for the witness run and how long the
// grandchild lingers before writing its survivor marker. The linger must exceed
// the ceiling plus the time teardown takes; Windows gets more of both because
// re-exec start-up and process teardown are slower there on a loaded host.
func cronTreeTiming() (hard, linger time.Duration) {
	if runtime.GOOS == "windows" {
		return 8 * time.Second, 14 * time.Second
	}
	return 1500 * time.Millisecond, 4 * time.Second
}

// TestCronOpenCodeHardTimeoutKillsGrandchild witnesses that the hard ceiling
// takes the attempt's whole process tree. A scheduled run often wraps OpenCode
// in another command (for example one that injects credentials), so the
// OpenCode session is the runner's grandchild. When the hard ceiling fires,
// that grandchild must die with the attempt rather than keep editing after its
// supervisor has given up. The helper tree mirrors that shape: the middle
// process starts a grandchild that writes a survivor marker only if it
// outlives the ceiling.
func TestCronOpenCodeHardTimeoutKillsGrandchild(t *testing.T) {
	switch os.Getenv(cronTreeHelperEnv) {
	case "middle":
		cronTreeMiddle(t)
		return
	case "grandchild":
		_, linger := cronTreeTiming()
		time.Sleep(linger)
		_ = os.WriteFile(os.Getenv("FAK_CRON_TREE_SURVIVOR"), []byte("survived"), 0o600)
		return
	}
	if testing.Short() {
		t.Skip("spawns a helper process tree and waits out a hard ceiling")
	}
	t.Parallel()

	markers := t.TempDir()
	ready := filepath.Join(markers, "ready")
	survivor := filepath.Join(markers, "survivor")
	hard, linger := cronTreeTiming()
	var stdout, stderr bytes.Buffer
	started := time.Now()
	receipt, err := RunScheduledOpenCode(ScheduledOpenCodeOptions{
		Job:          "job-hard-timeout-tree",
		Ledger:       filepath.Join(markers, "ledger.jsonl"),
		Timeout:      hard,
		HardTimeout:  hard,
		CrashRetries: 0,
		Command:      []string{os.Args[0], "-test.run=^TestCronOpenCodeHardTimeoutKillsGrandchild$"},
		Env: []string{
			cronTreeHelperEnv + "=middle",
			"FAK_CRON_TREE_READY=" + ready,
			"FAK_CRON_TREE_SURVIVOR=" + survivor,
		},
		Stdout:      &stdout,
		Stderr:      &stderr,
		EmitReceipt: true,
	})
	if err != nil {
		t.Fatalf("RunScheduledOpenCode: %v (stderr %q)", err, stderr.String())
	}
	if receipt.Outcome != "timeout" {
		t.Fatalf("outcome = %q, want timeout (stderr %q)", receipt.Outcome, stderr.String())
	}
	data, err := os.ReadFile(ready)
	if err != nil {
		t.Fatalf("grandchild never started before the hard ceiling: %v", err)
	}
	if pid, convErr := strconv.Atoi(strings.TrimSpace(string(data))); convErr != nil || pid <= 0 {
		t.Fatalf("invalid grandchild pid %q: %v", data, convErr)
	}
	// The grandchild started after `started`, so by this instant it has either
	// been killed or written its marker.
	if wait := time.Until(started.Add(linger + time.Second)); wait > 0 {
		time.Sleep(wait)
	}
	if _, err := os.Stat(survivor); !os.IsNotExist(err) {
		t.Fatalf("grandchild outlived the hard ceiling: %v", err)
	}
}

// cronTreeMiddle stands in for the wrapper command: it starts the
// grandchild, records its PID once it exists, and then waits on it, so it is
// still alive with its child when the ceiling fires.
func cronTreeMiddle(t *testing.T) {
	child := exec.Command(os.Args[0], "-test.run=^TestCronOpenCodeHardTimeoutKillsGrandchild$")
	child.Env = append(os.Environ(), cronTreeHelperEnv+"=grandchild")
	if err := child.Start(); err != nil {
		t.Fatalf("start grandchild: %v", err)
	}
	if err := os.WriteFile(os.Getenv("FAK_CRON_TREE_READY"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
		t.Fatalf("write ready marker: %v", err)
	}
	_ = child.Wait()
}
