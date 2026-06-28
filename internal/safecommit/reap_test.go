package safecommit

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

// deadPID returns a pid that is no longer running, by spawning a trivial child and
// waiting for it to exit. The kernel will not have recycled the number microseconds
// later when the test reads it.
func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=NoSuchTestZZZ")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn helper: %v", err)
	}
	pid := cmd.Process.Pid
	_ = cmd.Wait() // reap; the pid is now dead
	return pid
}

// TestReapStaleLockRemovesDeadHolder proves the field-bug fix: a lockfile recording a
// DEAD holder pid is removed by reapStaleLock, so the next committer is not wedged. This
// is the automatic form of the manual `rm .git/fak-commit.lock` that unblocked a
// 56-minute commit stall.
func TestReapStaleLockRemovesDeadHolder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fak-commit.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	reapStaleLock(path)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("stale lockfile not reaped: stat err = %v", err)
	}
}

// TestReapStaleLockKeepsLiveHolder proves the safety gate: a lockfile whose recorded pid
// is THIS (alive) process must NOT be removed — reaping a live committer's lock would let
// two writers race the shared trunk.
func TestReapStaleLockKeepsLiveHolder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fak-commit.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	reapStaleLock(path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("live holder's lockfile was wrongly reaped: %v", err)
	}
}

// TestReapStaleLockIgnoresGarbage proves the fail-safe paths: an absent file, an empty
// file, and a non-numeric body all leave reapStaleLock a no-op (it never deletes a lock it
// cannot attribute to a dead pid).
func TestReapStaleLockIgnoresGarbage(t *testing.T) {
	dir := t.TempDir()

	// Absent file: no panic, no error.
	reapStaleLock(filepath.Join(dir, "absent.lock"))

	for _, body := range []string{"", "   \n", "not-a-pid\n"} {
		path := filepath.Join(dir, "g.lock")
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("seed %q: %v", body, err)
		}
		reapStaleLock(path)
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("unattributable lockfile (%q) was wrongly reaped: %v", body, err)
		}
	}
}
