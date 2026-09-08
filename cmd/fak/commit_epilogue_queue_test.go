package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func TestCommitQueueOnBusy(t *testing.T) {
	repo := t.TempDir()

	// Mock commitLaneWaitFn to simulate a busy lane lock
	oldWait := commitLaneWaitFn
	defer func() { commitLaneWaitFn = oldWait }()
	commitLaneWaitFn = func(dir string, timeout time.Duration) (bool, safecommit.LockWaitReceipt) {
		return false, safecommit.LockWaitReceipt{
			HolderPID:   12345,
			HolderAlive: true,
			ElapsedNS:   1000000,
			DeadlineNS:  1000000,
		}
	}

	testFile := filepath.Join(repo, "foo.txt")
	if err := os.WriteFile(testFile, []byte("change\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Without --queue-on-busy, it must exit with safecommit.ExitLockBusy (exit 3)
	var stdout, stderr bytes.Buffer
	rc := runCommit(&stdout, &stderr, []string{
		"--dir", repo,
		"--path", "foo.txt",
		"-m", "fix(#12084): test busy lane (fak kernel)",
	})
	if rc != safecommit.ExitLockBusy {
		t.Fatalf("expected ExitLockBusy (%d), got %d; stdout=%s stderr=%s", safecommit.ExitLockBusy, rc, stdout.String(), stderr.String())
	}

	// 2. With --queue-on-busy, it must succeed (exit 0) and queue to .dispatch-runs/epilogues
	stdout.Reset()
	stderr.Reset()
	rc = runCommit(&stdout, &stderr, []string{
		"--dir", repo,
		"--path", "foo.txt",
		"-m", "fix(#12084): test busy lane (fak kernel)",
		"--queue-on-busy",
	})
	if rc != 0 {
		t.Fatalf("expected exit 0 with --queue-on-busy, got %d; stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "QUEUED: commit lane busy; submitted epilogue") {
		t.Fatalf("expected queued message in stdout, got %s", stdout.String())
	}

	// Verify epilogue on disk
	runsDir := filepath.Join(repo, ".dispatch-runs")
	recs, err := dispatchtick.ListEpilogues(runsDir, dispatchtick.EpilogueStatusPending)
	if err != nil {
		t.Fatalf("list epilogues: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("expected 1 queued epilogue, got %d", len(recs))
	}
	if recs[0].Issue != 12084 || recs[0].Lane != "kernel" || len(recs[0].Paths) != 1 || recs[0].Paths[0] != "foo.txt" {
		t.Fatalf("unexpected queued epilogue: %+v", recs[0])
	}

	// 3. With FAK_COMMIT_QUEUE=1, env var activates queue-on-busy
	t.Setenv("FAK_COMMIT_QUEUE", "1")
	stdout.Reset()
	stderr.Reset()
	rc = runCommit(&stdout, &stderr, []string{
		"--dir", repo,
		"--path", "foo.txt",
		"-m", "fix(#12084): test env queue (fak kernel)",
	})
	if rc != 0 {
		t.Fatalf("expected exit 0 with FAK_COMMIT_QUEUE=1, got %d; stderr=%s", rc, stderr.String())
	}

	recs, err = dispatchtick.ListEpilogues(runsDir, dispatchtick.EpilogueStatusPending)
	if err != nil {
		t.Fatalf("list epilogues: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("expected 2 queued epilogues, got %d", len(recs))
	}

	// 4. With --queue-on-busy and --json, it outputs JSON with status queued
	stdout.Reset()
	stderr.Reset()
	rc = runCommit(&stdout, &stderr, []string{
		"--dir", repo,
		"--path", "foo.txt",
		"-m", "fix(#12084): test json queue (fak kernel)",
		"--queue-on-busy",
		"--json",
	})
	if rc != 0 {
		t.Fatalf("expected exit 0 with --json, got %d; stderr=%s", rc, stderr.String())
	}
	var jsonResp map[string]interface{}
	if err := json.Unmarshal(stdout.Bytes(), &jsonResp); err != nil {
		t.Fatalf("unmarshal json: %v; raw=%s", err, stdout.String())
	}
	if jsonResp["status"] != "queued" {
		t.Fatalf("json status = %v, want queued", jsonResp["status"])
	}
}
