package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeSizedGrowthFile creates a file of a given logical size (via Truncate, so a
// large size costs no write) and stamps its mtime to age seconds/hours in the
// past, so the growth census reads it as HOT (fresh) or COLD (aged).
func writeSizedGrowthFile(t *testing.T, path string, size int64, age time.Duration) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	if err := f.Truncate(size); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	mt := time.Now().Add(-age)
	if err := os.Chtimes(path, mt, mt); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
}

// dosMetricsFail (16 MB) is the ClassDosMetrics FAIL budget; a disposable telemetry
// sink over it and COLD is reapable. laneJournalFail is the same 16 MB budget for
// the non-disposable lease WAL, which is protected even when over budget and cold.
const overBudget = 17 << 20 // 17 MB > the 16 MB ClassDosMetrics/ClassLaneJournal FAIL budget

// TestCollectGrowthLogsDryRunLedgersButDeletesNothing proves the DELETE-SAFE
// default: with apply off, the collect writes the would-reap set to the reap
// ledger (the soak evidence) but removes nothing from disk.
func TestCollectGrowthLogsDryRunLedgersButDeletesNothing(t *testing.T) {
	root := t.TempDir()
	cold := filepath.Join(root, "a", "metrics", "observations.jsonl")
	writeSizedGrowthFile(t, cold, overBudget, time.Hour) // COLD, disposable, over budget
	ledger := filepath.Join(t.TempDir(), "growth-reap.jsonl")

	var stderr bytes.Buffer
	reaped := collectGrowthLogs(&stderr, []string{root}, false, ledger)

	if reaped != 0 {
		t.Fatalf("apply-off must delete nothing, reaped=%d", reaped)
	}
	if _, err := os.Stat(cold); err != nil {
		t.Fatalf("apply-off must keep the file on disk: %v", err)
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if !strings.Contains(string(data), "would-reap") || !strings.Contains(string(data), "observations.jsonl") {
		t.Fatalf("ledger must record the would-reap set, got: %s", data)
	}
}

// TestCollectGrowthLogsKeepsHotDisposable proves the HOT protection: an oversized
// disposable file modified within HotAgeSec is never reaped, even with apply on
// (a live writer may still depend on it).
func TestCollectGrowthLogsKeepsHotDisposable(t *testing.T) {
	root := t.TempDir()
	hot := filepath.Join(root, "b", "metrics", "observations.jsonl")
	writeSizedGrowthFile(t, hot, overBudget, 0) // mtime ~now => HOT
	ledger := filepath.Join(t.TempDir(), "growth-reap.jsonl")

	var stderr bytes.Buffer
	reaped := collectGrowthLogs(&stderr, []string{root}, true, ledger) // apply ON

	if reaped != 0 {
		t.Fatalf("a HOT file must never be reaped even with apply, reaped=%d", reaped)
	}
	if _, err := os.Stat(hot); err != nil {
		t.Fatalf("HOT protection must keep the file: %v", err)
	}
}

// TestCollectGrowthLogsSparesNonDisposableLedger proves a non-disposable WAL /
// chained ledger is never in the reap set, even when COLD and over budget: it must
// be bounded at its write site, not hard-deleted.
func TestCollectGrowthLogsSparesNonDisposableLedger(t *testing.T) {
	root := t.TempDir()
	wal := filepath.Join(root, "c", "lane-journal.jsonl")
	writeSizedGrowthFile(t, wal, overBudget, time.Hour) // COLD + over budget, non-disposable
	ledger := filepath.Join(t.TempDir(), "growth-reap.jsonl")

	var stderr bytes.Buffer
	reaped := collectGrowthLogs(&stderr, []string{root}, true, ledger) // apply ON

	if reaped != 0 {
		t.Fatalf("a non-disposable WAL must never be reaped, reaped=%d", reaped)
	}
	if _, err := os.Stat(wal); err != nil {
		t.Fatalf("non-disposable ledger must be kept: %v", err)
	}
}

// TestCollectGrowthLogsApplyOptInDeletesColdDisposable proves the apply opt-in via
// FAK_GARDEN_GROWTH_COLLECT=apply deletes a COLD oversized disposable file while
// sparing a non-disposable ledger sitting right beside it.
func TestCollectGrowthLogsApplyOptInDeletesColdDisposable(t *testing.T) {
	t.Setenv("FAK_GARDEN_GROWTH_COLLECT", "apply")
	root := t.TempDir()
	cold := filepath.Join(root, "a", "metrics", "observations.jsonl")
	writeSizedGrowthFile(t, cold, overBudget, time.Hour)
	wal := filepath.Join(root, "c", "lane-journal.jsonl")
	writeSizedGrowthFile(t, wal, overBudget, time.Hour)
	ledger := filepath.Join(t.TempDir(), "growth-reap.jsonl")

	var stderr bytes.Buffer
	reaped := collectGrowthLogs(&stderr, []string{root}, growthApplyEnabled(false), ledger)

	if reaped != 1 {
		t.Fatalf("apply must reap exactly the cold disposable file, reaped=%d", reaped)
	}
	if _, err := os.Stat(cold); !os.IsNotExist(err) {
		t.Fatalf("apply must delete the cold oversized disposable file, stat err=%v", err)
	}
	if _, err := os.Stat(wal); err != nil {
		t.Fatalf("apply must spare the non-disposable ledger: %v", err)
	}
	data, _ := os.ReadFile(ledger)
	if !strings.Contains(string(data), "reaped") {
		t.Fatalf("ledger must record the reap, got: %s", data)
	}
}

// TestGrowthApplyEnabledDefaultsOff proves the safety default: with no env and no
// flag the collect is ledger-only (apply off); the env opt-in or the flag flips it.
func TestGrowthApplyEnabledDefaultsOff(t *testing.T) {
	t.Setenv("FAK_GARDEN_GROWTH_COLLECT", "")
	if growthApplyEnabled(false) {
		t.Fatalf("unset env + no flag must stay ledger-only (default-off)")
	}
	if !growthApplyEnabled(true) {
		t.Fatalf("--growth-apply flag must enable apply")
	}
	t.Setenv("FAK_GARDEN_GROWTH_COLLECT", "apply")
	if !growthApplyEnabled(false) {
		t.Fatalf("FAK_GARDEN_GROWTH_COLLECT=apply must enable apply")
	}
}

// TestGrowthCensusRootsAddsFleetTreeWhenPresent proves the Fleet roots scope: the
// repo root always, the Fleet tree only when it resolves to a real directory, and
// a clean skip when neither the env nor LOCALAPPDATA point at one.
func TestGrowthCensusRootsAddsFleetTreeWhenPresent(t *testing.T) {
	repo := t.TempDir()
	t.Setenv("FAK_FLEET_DIR", "")
	t.Setenv("LOCALAPPDATA", "")
	if got := growthCensusRoots(repo); len(got) != 1 || got[0] != repo {
		t.Fatalf("no fleet env: want [repo], got %v", got)
	}
	fleet := t.TempDir()
	t.Setenv("FAK_FLEET_DIR", fleet)
	if got := growthCensusRoots(repo); len(got) != 2 || got[1] != fleet {
		t.Fatalf("with fleet dir: want [repo, fleet], got %v", got)
	}
	t.Setenv("FAK_FLEET_DIR", filepath.Join(fleet, "does-not-exist"))
	if got := growthCensusRoots(repo); len(got) != 1 {
		t.Fatalf("a missing fleet dir must be skipped cleanly, got %v", got)
	}
}
