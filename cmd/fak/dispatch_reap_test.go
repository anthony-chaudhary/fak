package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// mkDispatchRuns makes a fresh .dispatch-runs/ dir under a temp root so ClassifyPath
// labels the files inside it ClassDispatchLog (disposable). The dispatch tree name is
// load-bearing: a sidecar suffix like .witness/.txt is only disposable BECAUSE it
// sits under .dispatch-runs/.
func mkDispatchRuns(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), ".dispatch-runs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	return dir
}

// mkAged writes name into dir and back-dates its mtime by age. Returns the full path.
func mkAged(t *testing.T, dir, name string, now time.Time, age time.Duration) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("sidecar\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mt := now.Add(-age)
	if err := os.Chtimes(p, mt, mt); err != nil {
		t.Fatal(err)
	}
	return p
}

func reapedSet(reap []coldSidecar) map[string]bool {
	m := map[string]bool{}
	for _, s := range reap {
		m[filepath.Base(s.Path)] = true
	}
	return m
}

// TestSurveyColdReapsColdPastFloor: a COLD sidecar past the grace floor is reaped,
// while a sidecar inside the grace window is kept.
func TestSurveyColdReapsColdPastFloor(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	mkAged(t, dir, "anchor.log", now, 1*time.Hour)    // in grace → kept, and the newest run
	mkAged(t, dir, "cold.witness", now, 48*time.Hour) // past the 24h floor → reaped
	reap, kept, err := surveyColdDispatchSidecars(dir, 24*time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	got := reapedSet(reap)
	if !got["cold.witness"] {
		t.Errorf("cold.witness past the floor must be reaped; reap=%v", got)
	}
	if got["anchor.log"] {
		t.Error("anchor.log inside the grace window must be kept, not reaped")
	}
	if kept < 1 {
		t.Errorf("kept = %d, want >= 1 (the grace-window anchor)", kept)
	}
}

// TestSurveyColdKeepsNewestRunWhenAllCold: even when EVERY sidecar is older than the
// grace floor, the newest run's cluster (the freshest file plus its burst-mates
// within dispatchRunSpread) is kept; only the strictly older run is reaped.
func TestSurveyColdKeepsNewestRunWhenAllCold(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	mkAged(t, dir, "newest-a.log", now, 40*time.Hour)                   // freshest — the newest run
	mkAged(t, dir, "newest-b.witness", now, 40*time.Hour+5*time.Minute) // same run burst → kept
	mkAged(t, dir, "older.txt", now, 80*time.Hour)                      // an older run → reaped
	reap, kept, err := surveyColdDispatchSidecars(dir, 24*time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	got := reapedSet(reap)
	if !got["older.txt"] {
		t.Errorf("the strictly older run must be reaped; reap=%v", got)
	}
	if got["newest-a.log"] || got["newest-b.witness"] {
		t.Errorf("the newest run's sidecars must be kept even when all are cold; reap=%v", got)
	}
	if kept != 2 {
		t.Errorf("kept = %d, want 2 (the two newest-run sidecars)", kept)
	}
}

// TestSurveyColdKeepsGraceWindow: a sidecar inside the grace window is kept even
// though an older sibling is reaped.
func TestSurveyColdKeepsGraceWindow(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	mkAged(t, dir, "recent.wave", now, 2*time.Hour) // inside the 24h grace → kept
	mkAged(t, dir, "stale.json", now, 50*time.Hour) // past the floor → reaped
	reap, _, err := surveyColdDispatchSidecars(dir, 24*time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	got := reapedSet(reap)
	if got["recent.wave"] {
		t.Error("recent.wave inside the grace window must be kept")
	}
	if !got["stale.json"] {
		t.Errorf("stale.json past the floor must be reaped; reap=%v", got)
	}
}

// TestSurveyColdFloorZeroFallsBackToDefault: a mis-wired floor of 0 must NOT sweep a
// live file — it falls back to the 24h default, so a 1h-old sidecar is kept while a
// 48h-old one is still reaped.
func TestSurveyColdFloorZeroFallsBackToDefault(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	mkAged(t, dir, "fresh.log", now, 1*time.Hour)        // must survive the zero-floor fallback
	mkAged(t, dir, "ancient.witness", now, 48*time.Hour) // still reaped under the 24h fallback
	reap, _, err := surveyColdDispatchSidecars(dir, 0, now)
	if err != nil {
		t.Fatal(err)
	}
	got := reapedSet(reap)
	if got["fresh.log"] {
		t.Error("floor=0 must fall back to the 24h default and KEEP a 1h-old live file")
	}
	if !got["ancient.witness"] {
		t.Errorf("floor=0 fallback should still reap a 48h-old sidecar; reap=%v", got)
	}
}

// TestSurveyColdExcludesNonDisposable: a non-disposable class (a lane-journal WAL)
// that happens to sit in the swept dir is NEVER reaped, even when cold, while a
// disposable log beside it is.
func TestSurveyColdExcludesNonDisposable(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	mkAged(t, dir, "anchor.witness", now, 30*time.Hour)     // newest run → kept
	mkAged(t, dir, "lane-journal.jsonl", now, 80*time.Hour) // non-disposable WAL → never reaped
	mkAged(t, dir, "doomed.log", now, 80*time.Hour)         // disposable + cold → reaped
	reap, _, err := surveyColdDispatchSidecars(dir, 24*time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	got := reapedSet(reap)
	if got["lane-journal.jsonl"] {
		t.Error("a non-disposable WAL must never be reaped, even when cold")
	}
	if !got["doomed.log"] {
		t.Errorf("a cold disposable log must be reaped; reap=%v", got)
	}
}

// TestRunDispatchReapDryRunDeletesNothing: the default (no --apply) lists the
// would-reap set but deletes nothing on disk.
func TestRunDispatchReapDryRunDeletesNothing(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	anchor := mkAged(t, dir, "anchor.log", now, 0)            // fresh → newest run/grace, kept
	cold := mkAged(t, dir, "cold.witness", now, 48*time.Hour) // reap-eligible

	var out, errb bytes.Buffer
	if rc := runDispatchReap(&out, &errb, []string{"--dir", dir}); rc != 0 {
		t.Fatalf("dry-run rc = %d, want 0 (stderr=%s)", rc, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "would reap") || !strings.Contains(s, "0 deleted") || !strings.Contains(s, "DRY-RUN") {
		t.Errorf("dry-run output missing the would-reap/0-deleted banner:\n%s", s)
	}
	for _, p := range []string{anchor, cold} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("dry-run must not delete %s: %v", p, err)
		}
	}
}

// TestRunDispatchReapApplyDeletes: --apply deletes the cold sidecar, keeps the newest
// run, and the ledger records the reaped decision.
func TestRunDispatchReapApplyDeletes(t *testing.T) {
	dir := mkDispatchRuns(t)
	now := time.Now()
	anchor := mkAged(t, dir, "anchor.log", now, 0)            // fresh → kept
	cold := mkAged(t, dir, "cold.witness", now, 48*time.Hour) // reaped
	ledger := filepath.Join(t.TempDir(), "reap.jsonl")

	var out, errb bytes.Buffer
	rc := runDispatchReap(&out, &errb, []string{"--dir", dir, "--apply", "--ledger", ledger})
	if rc != 0 {
		t.Fatalf("apply rc = %d, want 0 (stderr=%s)", rc, errb.String())
	}
	if _, err := os.Stat(cold); !os.IsNotExist(err) {
		t.Errorf("apply should delete the cold sidecar, but it remains (%v)", err)
	}
	if _, err := os.Stat(anchor); err != nil {
		t.Errorf("apply must NOT delete the newest run's sidecar: %v", err)
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("ledger not written: %v", err)
	}
	if !strings.Contains(string(data), "\"action\":\"reaped\"") {
		t.Errorf("ledger missing a reaped decision:\n%s", data)
	}
}

// TestRunDispatchReapMissingDirIsClean: a nonexistent dispatch dir reaps nothing and
// exits 0 (nothing to collect is success, not an error).
func TestRunDispatchReapMissingDirIsClean(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "no-such-dispatch-runs")
	var out, errb bytes.Buffer
	if rc := runDispatchReap(&out, &errb, []string{"--dir", missing}); rc != 0 {
		t.Fatalf("missing dir rc = %d, want 0 (stderr=%s)", rc, errb.String())
	}
	if !strings.Contains(out.String(), "nothing reapable") {
		t.Errorf("missing dir should report nothing reapable:\n%s", out.String())
	}
}
