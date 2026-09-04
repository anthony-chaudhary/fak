package ops

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestWorkspaceManagerSweep(t *testing.T) {
	tmp := t.TempDir()
	gitDir := filepath.Join(tmp, ".git")
	_ = os.MkdirAll(gitDir, 0o755)

	// Create a dead-PID lock
	deadLock := filepath.Join(gitDir, "fak-commit.lock")
	_ = os.WriteFile(deadLock, []byte("99999999"), 0o644)

	// Create a live-PID lock
	liveLock := filepath.Join(gitDir, "index.lock")
	_ = os.WriteFile(liveLock, []byte("12345"), 0o644)

	wm := NewWorkspaceManager(tmp)
	wm.ProcessAlive = func(pid int) bool {
		return pid == 12345
	}

	// Dry run
	dryRes, err := wm.SweepLocksAndWorktrees(context.Background(), true)
	if err != nil {
		t.Fatalf("dry run error: %v", err)
	}
	if len(dryRes.LocksEvicted) != 1 {
		t.Errorf("expected 1 lock evicted in dry run, got %d: %v", len(dryRes.LocksEvicted), dryRes.LocksEvicted)
	}

	// Apply
	res, err := wm.SweepLocksAndWorktrees(context.Background(), false)
	if err != nil {
		t.Fatalf("apply error: %v", err)
	}
	if len(res.LocksEvicted) != 1 || res.LocksEvicted[0] != "fak-commit.lock" {
		t.Errorf("expected fak-commit.lock evicted, got %v", res.LocksEvicted)
	}

	// Verify dead lock is removed and live lock is kept
	if _, err := os.Stat(deadLock); !os.IsNotExist(err) {
		t.Errorf("expected dead lock to be removed, stat err: %v", err)
	}
	if _, err := os.Stat(liveLock); err != nil {
		t.Errorf("expected live lock to remain, stat err: %v", err)
	}
}
