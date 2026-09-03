package codexlifecycle

// #10673 reconciliation tests. The committed fixture
// (testdata/reconcile/issue-10673) reproduces the audited day-directory
// signature — task_started exceeding observed terminals by a nonzero raw
// delta — plus one junk file, and conformance.json carries the exact expected
// counts derived from the fixture timeline, independent of the
// implementation under test.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

var reconcileFixtureDir = filepath.Join("testdata", "reconcile", "issue-10673")

// confReconcileCounts mirrors the subset of ReconcileCounts JSON keys that
// conformance.json pins.
type confReconcileCounts struct {
	Rollouts          int     `json:"rollouts"`
	TaskStarted       int     `json:"task_started"`
	TaskComplete      int     `json:"task_complete"`
	TurnAborted       int     `json:"turn_aborted"`
	RawUnaccounted    int     `json:"raw_unaccounted"`
	RawUnaccountedPct float64 `json:"raw_unaccounted_pct"`
	Complete          int     `json:"complete"`
	Aborted           int     `json:"aborted"`
	Superseded        int     `json:"superseded"`
	ProcessDeath      int     `json:"process_death"`
	Live              int     `json:"live"`
	ClosedTotal       int     `json:"closed_total"`
	Residual          int     `json:"residual_unaccounted"`
	ResidualPct       float64 `json:"residual_pct"`
	UnclassifiedAfter int     `json:"unclassified_after"`
	Orphans           int     `json:"orphans"`
	Reused            int     `json:"reused"`
	MultiplyTerm      int     `json:"multiply_terminated"`
}

type confReconcile struct {
	Schema string `json:"schema"`
	Expect struct {
		Scanned    int                            `json:"scanned"`
		Unreadable int                            `json:"unreadable"`
		Totals     confReconcileCounts            `json:"totals"`
		ByProvider map[string]confReconcileCounts `json:"by_provider"`
	} `json:"expect"`
}

func loadReconcileFixture(t *testing.T) (ReconcileReport, confReconcile) {
	t.Helper()
	rep, err := ScanReconcileCorpus(reconcileFixtureDir, ScanOptions{FreshWithin: 0})
	if err != nil {
		t.Fatalf("ScanReconcileCorpus: %v", err)
	}
	blob, err := os.ReadFile(filepath.Join(reconcileFixtureDir, "conformance.json"))
	if err != nil {
		t.Fatalf("read conformance: %v", err)
	}
	var conf confReconcile
	if err := json.Unmarshal(blob, &conf); err != nil {
		t.Fatalf("decode conformance: %v", err)
	}
	return rep, conf
}

func assertReconcileCounts(t *testing.T, where string, got ReconcileCounts, want confReconcileCounts) {
	t.Helper()
	if got.Rollouts != want.Rollouts ||
		got.TaskStarted != want.TaskStarted ||
		got.TaskComplete != want.TaskComplete ||
		got.TurnAborted != want.TurnAborted ||
		got.RawUnaccounted != want.RawUnaccounted ||
		got.Complete != want.Complete ||
		got.Aborted != want.Aborted ||
		got.Superseded != want.Superseded ||
		got.ProcessDeath != want.ProcessDeath ||
		got.Live != want.Live ||
		got.ClosedTotal != want.ClosedTotal ||
		got.ResidualUnaccounted != want.Residual ||
		got.UnclassifiedAfter != want.UnclassifiedAfter ||
		got.Orphans != want.Orphans ||
		got.Reused != want.Reused ||
		got.MultiplyTerminated != want.MultiplyTerm {
		t.Errorf("%s = %+v, want %+v", where, got, want)
	}
	if !eqF(got.RawUnaccountedPct, want.RawUnaccountedPct) || !eqF(got.ResidualPct, want.ResidualPct) {
		t.Errorf("%s pcts = %g/%g, want %g/%g", where, got.RawUnaccountedPct, got.ResidualPct, want.RawUnaccountedPct, want.ResidualPct)
	}
}

// The fixture reproduces the audited #10673 signature end to end and matches
// the conformance values exactly: a nonzero RAW unaccounted delta that the
// fold's synthesized terminals fully explain (residual 0), with observed and
// synthesized outcomes kept in separate columns.
func TestReconcileFixture_Conformance(t *testing.T) {
	rep, conf := loadReconcileFixture(t)
	if rep.Schema != conf.Schema {
		t.Errorf("schema = %q, want %q", rep.Schema, conf.Schema)
	}
	if rep.Scanned != conf.Expect.Scanned || rep.Unreadable != conf.Expect.Unreadable {
		t.Errorf("scanned/unreadable = %d/%d, want %d/%d", rep.Scanned, rep.Unreadable, conf.Expect.Scanned, conf.Expect.Unreadable)
	}
	assertReconcileCounts(t, "totals", rep.Totals, conf.Expect.Totals)
	if len(conf.Expect.ByProvider) == 0 {
		t.Fatal("conformance.json has no by_provider expectations")
	}
	if len(rep.ByProvider) != len(conf.Expect.ByProvider) {
		t.Errorf("by_provider rows = %d (%v), want %d", len(rep.ByProvider), rep.ProviderVersions(), len(conf.Expect.ByProvider))
	}
	for key, want := range conf.Expect.ByProvider {
		got, ok := rep.ByProvider[key]
		if !ok {
			t.Errorf("missing by_provider row %q", key)
			continue
		}
		assertReconcileCounts(t, "by_provider["+key+"]", *got, want)
	}
	if !rep.AllStartsTyped {
		t.Error("all_starts_typed = false, want true (the fold types every start)")
	}
}

// A closed corpus — every start reaches an observed terminal — reports a zero
// raw delta AND zero residual. THE GATE: this is the healthy-run shape.
func TestScanReconcileCorpus_ClosedCorpusReportsZeroDelta(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeRollout(t, dir, "a.jsonl", now,
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-09-01T08:00:01.000Z", "A"), complete("2026-09-01T08:00:30.000Z", "A"))
	writeRollout(t, dir, "b.jsonl", now,
		meta("s2", "fak", "0.144.4", `C:\work\fak`),
		started("2026-09-01T08:01:01.000Z", "B"), aborted("2026-09-01T08:02:00.000Z", "B"))
	rep, err := ScanReconcileCorpus(dir, ScanOptions{FreshWithin: 0})
	if err != nil {
		t.Fatalf("ScanReconcileCorpus: %v", err)
	}
	if rep.Scanned != 2 {
		t.Fatalf("scanned = %d, want 2", rep.Scanned)
	}
	if rep.Totals.TaskStarted != 2 || rep.Totals.TaskComplete != 1 || rep.Totals.TurnAborted != 1 {
		t.Errorf("raw = %d/%d/%d, want 2/1/1", rep.Totals.TaskStarted, rep.Totals.TaskComplete, rep.Totals.TurnAborted)
	}
	if rep.Totals.RawUnaccounted != 0 || !eqF(rep.Totals.RawUnaccountedPct, 0) {
		t.Errorf("raw_unaccounted = %d (%g%%), want 0 (0%%)", rep.Totals.RawUnaccounted, rep.Totals.RawUnaccountedPct)
	}
	if rep.Totals.ResidualUnaccounted != 0 || rep.Totals.ClosedTotal != 2 {
		t.Errorf("closed/residual = %d/%d, want 2/0", rep.Totals.ClosedTotal, rep.Totals.ResidualUnaccounted)
	}
}

// Synthesis is counted SEPARATELY from observation: the raw delta stays the
// honest 498-vs-288-style number while the fold's Superseded/ProcessDeath
// columns explain it, and observed Complete/Aborted never absorb them.
func TestScanReconcileCorpus_SynthesisCountedSeparately(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	stale := now.Add(-72 * time.Hour)
	writeRollout(t, dir, "gap.jsonl", stale,
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-09-01T08:00:01.000Z", "A"), // abandoned mid-session -> superseded
		started("2026-09-01T08:01:00.000Z", "B"),
		complete("2026-09-01T08:01:30.000Z", "B"),
		started("2026-09-01T08:02:00.000Z", "C")) // open at end, stale -> process_death
	rep, err := ScanReconcileCorpus(dir, ScanOptions{FreshWithin: time.Hour, Now: now})
	if err != nil {
		t.Fatalf("ScanReconcileCorpus: %v", err)
	}
	if rep.Totals.TaskStarted != 3 || rep.Totals.TaskComplete != 1 {
		t.Fatalf("raw = %d started / %d complete, want 3/1", rep.Totals.TaskStarted, rep.Totals.TaskComplete)
	}
	if rep.Totals.RawUnaccounted != 2 || !eqF(rep.Totals.RawUnaccountedPct, 200*float64(1)/float64(3)) {
		t.Errorf("raw_unaccounted = %d (%g%%), want 2 (66.7%%)", rep.Totals.RawUnaccounted, rep.Totals.RawUnaccountedPct)
	}
	if rep.Totals.Complete != 1 || rep.Totals.Superseded != 1 || rep.Totals.ProcessDeath != 1 {
		t.Errorf("outcomes = complete %d / superseded %d / death %d, want 1/1/1",
			rep.Totals.Complete, rep.Totals.Superseded, rep.Totals.ProcessDeath)
	}
	if rep.Totals.ClosedTotal != 3 || rep.Totals.ResidualUnaccounted != 0 {
		t.Errorf("closed/residual = %d/%d, want 3/0 (synthesis explains the whole delta)", rep.Totals.ClosedTotal, rep.Totals.ResidualUnaccounted)
	}
}

// A fresh rollout's open final start is LIVE, not closed: the residual keeps
// it unaccounted instead of letting synthesis claim a terminal that did not
// happen. (The Live-vs-ProcessDeath discriminator itself is witnessed in
// corpus_test.go.)
func TestScanReconcileCorpus_LiveStaysOpenInResidual(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeRollout(t, dir, "live.jsonl", now.Add(-time.Minute),
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-09-01T08:00:01.000Z", "A"))
	rep, err := ScanReconcileCorpus(dir, ScanOptions{FreshWithin: time.Hour, Now: now})
	if err != nil {
		t.Fatalf("ScanReconcileCorpus: %v", err)
	}
	if rep.Totals.Live != 1 {
		t.Fatalf("live = %d, want 1", rep.Totals.Live)
	}
	if rep.Totals.ClosedTotal != 0 || rep.Totals.ResidualUnaccounted != 1 {
		t.Errorf("closed/residual = %d/%d, want 0/1 (a live turn is open, not synthesized-closed)",
			rep.Totals.ClosedTotal, rep.Totals.ResidualUnaccounted)
	}
	if rep.Totals.RawUnaccounted != 1 {
		t.Errorf("raw_unaccounted = %d, want 1", rep.Totals.RawUnaccounted)
	}
}

// An orphan terminal (a truncated head, or a terminal for a turn started in an
// earlier file) makes raw terminals EXCEED starts: the delta goes negative and
// must stay honest, not clamped to zero.
func TestScanReconcileCorpus_OrphanTerminalKeepsDeltaHonest(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeRollout(t, dir, "head.jsonl", now,
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		complete("2026-09-01T08:00:30.000Z", "GHOST"))
	rep, err := ScanReconcileCorpus(dir, ScanOptions{FreshWithin: 0})
	if err != nil {
		t.Fatalf("ScanReconcileCorpus: %v", err)
	}
	if rep.Scanned != 1 {
		t.Fatalf("scanned = %d, want 1 (an orphan-bearing rollout is still evidence)", rep.Scanned)
	}
	if rep.Totals.Orphans != 1 {
		t.Errorf("orphans = %d, want 1", rep.Totals.Orphans)
	}
	if rep.Totals.RawUnaccounted != -1 {
		t.Errorf("raw_unaccounted = %d, want -1 (honest negative, never clamped)", rep.Totals.RawUnaccounted)
	}
}
