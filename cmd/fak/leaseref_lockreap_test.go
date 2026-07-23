package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/witness"
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

	// Keep the once-per-tick growth collect (#5349) off the real Fleet tree/ledger:
	// census only the temp repo (which has no oversized files), never delete.
	t.Setenv("FAK_FLEET_DIR", "")
	t.Setenv("LOCALAPPDATA", "")
	t.Setenv("FAK_GARDEN_GROWTH_LEDGER", filepath.Join(t.TempDir(), "growth-reap.jsonl"))

	var stdout, stderr bytes.Buffer
	reaped, sessions, surfaced, lockFiles, _, intents, _ := performGardenTick(&stdout, &stderr, gardenbundle.TickPlan{}, dir, dir, false, false)
	if stderr.Len() != 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}
	if reaped != 0 || sessions != 0 || surfaced != 0 || intents != 0 {
		t.Fatalf("lease/session/surface/intent counts = %d/%d/%d/%d, want 0/0/0/0 (empty plan)", reaped, sessions, surfaced, intents)
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
	_, _, _, lockFiles, _, _, _ := performGardenTick(&stdout, &stderr, gardenbundle.TickPlan{}, dir, dir, true, false)
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

// TestGardenTickReapsExpiredIntents proves the reap-parity wiring in performGardenTick
// (#5345): the automatic tick now sweeps lapsed intent leases too, not just leases +
// sessions + orphan .locks. A refs/fak/locks/intent-<key> whose TTL has expired is
// reaped under an EMPTY plan (no ActReap decision fires), while nothing else is touched;
// a second tick reaps nothing (idempotent). This closes the same reaper asymmetry that
// bit sessions in #5344 — before this wiring only the manual `fak leaseref reap` swept
// intents, so the scheduled loop grew the intent namespace unbounded.
func TestGardenTickReapsExpiredIntents(t *testing.T) {
	dir := lockReapInitRepo(t)
	now := time.Now()

	// Seed one already-expired intent: claim it an hour in the past with a 60s TTL, so
	// its expiry sits ~59 min before the tick's real-clock ReapIntents(time.Now()) call.
	store := leaseref.NewInDir(dir)
	past := now.Add(-time.Hour)
	if _, v, err := store.ClaimIntent(context.Background(), leaseref.IntentRecord{Target: "#77", Holder: "gc-parity", TTLSeconds: 60}, past); err != nil || !v.OK {
		t.Fatalf("seed expired intent: ok=%v err=%v", v.OK, err)
	}

	// Keep the once-per-tick growth collect off the real Fleet tree/ledger.
	t.Setenv("FAK_FLEET_DIR", "")
	t.Setenv("LOCALAPPDATA", "")
	t.Setenv("FAK_GARDEN_GROWTH_LEDGER", filepath.Join(t.TempDir(), "growth-reap.jsonl"))

	var stdout, stderr bytes.Buffer
	reaped, sessions, surfaced, lockFiles, _, intents, _ := performGardenTick(&stdout, &stderr, gardenbundle.TickPlan{}, dir, dir, false, false)
	if stderr.Len() != 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}
	if intents != 1 {
		t.Fatalf("intents = %d, want 1 (the lapsed intent is reaped on the tick)", intents)
	}
	if reaped != 0 || sessions != 0 || surfaced != 0 || lockFiles != 0 {
		t.Fatalf("collateral reap lease/session/surface/lock = %d/%d/%d/%d, want 0/0/0/0", reaped, sessions, surfaced, lockFiles)
	}

	// The intent ref is gone: a second tick reaps nothing (idempotent, like the other sweeps).
	var so2, se2 bytes.Buffer
	_, _, _, _, _, intents2, _ := performGardenTick(&so2, &se2, gardenbundle.TickPlan{}, dir, dir, false, false)
	if intents2 != 0 {
		t.Fatalf("second tick intents = %d, want 0 (idempotent reap)", intents2)
	}
}

// TestGardenTickFoldsSentinelDecisionsNote proves the decisions-note fold wiring in
// performGardenTick (#5361, census row 12): the automatic tick bounds the empty-tree
// sentinel note — the ONE unbounded producer on refs/notes/fak/decisions that every
// pre-commit refusal appends to — folding it to a keep-last-N tail and reporting the
// dropped-line count. FAK_GARDEN_DECISIONS_KEEP tightens the bound so a small seed
// exercises the fold; a dry-run tick performs NO fold, and a second acting tick at the
// same bound folds nothing more (idempotent). The library fold itself is proven in
// internal/witness; this asserts performGardenTick actually invokes it and threads the
// count through the tick's return.
func TestGardenTickFoldsSentinelDecisionsNote(t *testing.T) {
	dir := lockReapInitRepo(t)

	// Keep the once-per-tick growth collect off the real Fleet tree/ledger.
	t.Setenv("FAK_FLEET_DIR", "")
	t.Setenv("LOCALAPPDATA", "")
	t.Setenv("FAK_GARDEN_GROWTH_LEDGER", filepath.Join(t.TempDir(), "growth-reap.jsonl"))
	// Tighten the keep bound so five seeded refusals exceed it and the fold has work.
	t.Setenv("FAK_GARDEN_SENTINEL_KEEP", "2")

	// Seed the empty-tree sentinel note with five pre-commit refusals (commitSHA=="").
	rec := witness.NewRecorderForDir(dir)
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		if err := rec.AppendDecision(ctx, "", witness.Decision{Op: "commit", Verdict: witness.VerdictRefuse, ReasonClass: "OFF_TRUNK"}); err != nil {
			t.Fatalf("seed refusal %d: %v", i, err)
		}
	}

	// A DRY-RUN tick performs no fold even with a fat sentinel note (mirrors the reaps).
	var dso, dse bytes.Buffer
	if _, _, _, _, _, _, dryFolded := performGardenTick(&dso, &dse, gardenbundle.TickPlan{}, dir, dir, true, false); dryFolded != 0 {
		t.Fatalf("dry-run folded = %d, want 0 (no side effect under dry-run)", dryFolded)
	}
	if pre, _ := rec.ReadDecisions(ctx, ""); len(pre) != 5 {
		t.Fatalf("dry-run must not touch the sentinel note, got %d lines, want 5", len(pre))
	}

	// An acting tick folds the note to the last 2 lines: 3 dropped.
	var stdout, stderr bytes.Buffer
	_, _, _, _, _, _, folded := performGardenTick(&stdout, &stderr, gardenbundle.TickPlan{}, dir, dir, false, false)
	if stderr.Len() != 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}
	if folded != 3 {
		t.Fatalf("folded = %d, want 3 (5 refusals folded to the last 2)", folded)
	}
	if post, _ := rec.ReadDecisions(ctx, ""); len(post) != 2 {
		t.Fatalf("sentinel note should hold 2 lines after the fold, got %d", len(post))
	}

	// Idempotent: a second acting tick at the same bound folds nothing more.
	var so2, se2 bytes.Buffer
	if _, _, _, _, _, _, folded2 := performGardenTick(&so2, &se2, gardenbundle.TickPlan{}, dir, dir, false, false); folded2 != 0 {
		t.Fatalf("second tick folded = %d, want 0 (idempotent)", folded2)
	}
}
