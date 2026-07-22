package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
)

// lockReapInitRepo initialises a REAL git temp repo (skipped when git is unavailable,
// e.g. the native-Windows path; runs under the WSL suite) and returns its dir. The
// orphan .lock reaper resolves <git-common-dir>/refs/fak/locks through a live
// `git rev-parse`, so these wiring tests need an actual repo, not a bare temp dir.
func lockReapInitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

// lockReapLocksDir resolves <git-common-dir>/refs/fak/locks for the repo at dir the
// same way internal/leaseref.Store.ReapLockFiles does, so a lock the test writes lands
// exactly where the reaper sweeps.
func lockReapLocksDir(t *testing.T, dir string) string {
	t.Helper()
	c := exec.Command("git", "rev-parse", "--path-format=absolute", "--git-common-dir")
	c.Dir = dir
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git rev-parse --git-common-dir: %v\n%s", err, out)
	}
	common := strings.TrimSpace(string(out))
	return filepath.Join(common, "refs", "fak", "locks")
}

// lockReapWriteLock writes a .lock file under the locks dir with the given age before
// now (a negative age dates it into the future, the clock-skew case), returning its
// absolute path. Age is applied via Chtimes so the sweep's mtime clock sees exactly the
// intended staleness.
func lockReapWriteLock(t *testing.T, locksDir, name string, now time.Time, age time.Duration) string {
	t.Helper()
	if err := os.MkdirAll(locksDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", locksDir, err)
	}
	path := filepath.Join(locksDir, name)
	if err := os.WriteFile(path, []byte("0000000000000000000000000000000000000000\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	stamp := now.Add(-age)
	if err := os.Chtimes(path, stamp, stamp); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
	return path
}

// TestGardenTickReapsOrphanLocks proves the once-per-tick, unconditional wiring in
// performGardenTick (#5348): with an EMPTY plan (no lease record is expired, so no
// ActReap decision fires), the tick still sweeps the orphan .lock namespace — a stale
// lock past the TTL is reaped, while a fresh (possibly-live CAS) lock and a future-mtime
// (clock-skew) lock are both KEPT. This is the "a ghost .lock can exist even when no
// lease record is expired" invariant.
func TestGardenTickReapsOrphanLocks(t *testing.T) {
	dir := lockReapInitRepo(t)
	locks := lockReapLocksDir(t, dir)
	now := time.Now()
	stale := lockReapWriteLock(t, locks, "session-dead.lock", now, 5*time.Hour)
	fresh := lockReapWriteLock(t, locks, "session-live.lock", now, 5*time.Millisecond)
	future := lockReapWriteLock(t, locks, "skewed.lock", now, -48*time.Hour)

	var stdout, stderr bytes.Buffer
	reaped, sessions, surfaced, lockFiles := performGardenTick(&stdout, &stderr, gardenbundle.TickPlan{}, dir, false)
	if stderr.Len() != 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}
	if reaped != 0 || sessions != 0 || surfaced != 0 {
		t.Fatalf("lease/session/surface counts = %d/%d/%d, want 0/0/0 (empty plan)", reaped, sessions, surfaced)
	}
	if lockFiles != 1 {
		t.Fatalf("lockFiles = %d, want 1 (only the stale lock is reaped)", lockFiles)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale lock still present after tick (err=%v)", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh lock wrongly removed: %v", err)
	}
	if _, err := os.Stat(future); err != nil {
		t.Fatalf("future-mtime lock wrongly removed (clock-skew fail-safe): %v", err)
	}
}

// TestGardenTickDryRunKeepsOrphanLocks proves the dry-run gate: a report-only tick
// performs NO delete, so a stale orphan .lock that a real tick would reap stays on disk
// and the lockFiles counter is 0. Mirrors how the lease/session reaps never Perform
// under dry-run.
func TestGardenTickDryRunKeepsOrphanLocks(t *testing.T) {
	dir := lockReapInitRepo(t)
	locks := lockReapLocksDir(t, dir)
	now := time.Now()
	stale := lockReapWriteLock(t, locks, "session-dead.lock", now, 5*time.Hour)

	var stdout, stderr bytes.Buffer
	_, _, _, lockFiles := performGardenTick(&stdout, &stderr, gardenbundle.TickPlan{}, dir, true)
	if stderr.Len() != 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}
	if lockFiles != 0 {
		t.Fatalf("dry-run lockFiles = %d, want 0 (no delete under dry-run)", lockFiles)
	}
	if _, err := os.Stat(stale); err != nil {
		t.Fatalf("dry-run wrongly removed the stale lock: %v", err)
	}
}

// TestLeaserefReapReportsLockSweep proves the fourth sweep wired into runLeaserefReap
// (#5348): a stale orphan .lock is reaped and the sweep is reported in the summary line
// alongside leases / sessions / intents.
func TestLeaserefReapReportsLockSweep(t *testing.T) {
	dir := lockReapInitRepo(t)
	locks := lockReapLocksDir(t, dir)
	now := time.Now()
	stale := lockReapWriteLock(t, locks, "session-dead.lock", now, 5*time.Hour)
	fresh := lockReapWriteLock(t, locks, "session-live.lock", now, 5*time.Millisecond)

	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"reap", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref reap exit=%d stderr=%q", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "1 orphan lock(s)") {
		t.Fatalf("reap summary missing the orphan-lock sweep: %q", got)
	}
	if !strings.Contains(got, "1 live lock(s) kept") {
		t.Fatalf("reap summary missing the kept (live) lock count: %q", got)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale lock still present after reap (err=%v)", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("reap wrongly removed the fresh live lock: %v", err)
	}
}
