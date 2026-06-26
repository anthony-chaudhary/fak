package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeFixture writes the given JSONL lines to a temp journal and returns its path.
func writeFixture(t *testing.T, lines []string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "verdict-journal.jsonl")
	body := ""
	for _, l := range lines {
		body += l + "\n"
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// TestFoldRates is the core hermetic check: a fixed fixture journal folds to the exact
// closure_rate and regression_rate, ESCALATE is excluded from regression, and a
// cause=regressed revert is counted.
func TestFoldRates(t *testing.T) {
	path := writeFixture(t, []string{
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"verify","verdict":"NOT_SHIPPED"}`,
		`{"syscall":"memory_recall","verdict":"RECALL_FRESH"}`,
		`{"syscall":"improve","verdict":"KEEP"}`,
		`{"syscall":"improve","verdict":"KEEP"}`,
		`{"syscall":"improve","verdict":"REVERT","detail":{"revert_cause":"regressed"}}`,
		`{"syscall":"improve","verdict":"ESCALATE"}`,
	})
	h, err := computeHealth(path, 0)
	if err != nil {
		t.Fatalf("computeHealth: %v", err)
	}
	// closure = 3 SHIPPED / 4 verify decisions = 0.75
	if !approx(h.ClosureRate, 0.75) {
		t.Errorf("closure_rate = %v, want 0.75", h.ClosureRate)
	}
	// regression = 1 REVERT / (2 KEEP + 1 REVERT) = 0.333; ESCALATE excluded
	if !approx(h.RegressionRate, 0.333) {
		t.Errorf("regression_rate = %v, want 0.333", h.RegressionRate)
	}
	if h.Counts.RegressedRevert != 1 {
		t.Errorf("regressed_revert = %d, want 1", h.Counts.RegressedRevert)
	}
	if h.Counts.Escalate != 1 {
		t.Errorf("escalate = %d, want 1", h.Counts.Escalate)
	}
	if h.Window.ImproveCycles != 4 {
		t.Errorf("improve_cycles = %d, want 4 (2 KEEP + 1 REVERT + 1 ESCALATE)", h.Window.ImproveCycles)
	}
	if h.Window.VerifyChecks != 4 {
		t.Errorf("verify_checks = %d, want 4", h.Window.VerifyChecks)
	}
	if h.Window.RowsConsidered != 9 {
		t.Errorf("rows_considered = %d, want 9 (all rows, incl. the memory_recall row)", h.Window.RowsConsidered)
	}
}

// TestBaselineReproducible proves the recorded Baseline anchor is the honest fold of a
// journal matching the real .dos/verdict-journal.jsonl at #382 introduction (9/11 verify
// SHIPPED, 2/4 improve KEEP-decisions reverted) — verify-don't-trust the constant.
func TestBaselineReproducible(t *testing.T) {
	var lines []string
	for i := 0; i < 9; i++ {
		lines = append(lines, `{"syscall":"verify","verdict":"SHIPPED"}`)
	}
	for i := 0; i < 2; i++ {
		lines = append(lines, `{"syscall":"verify","verdict":"NOT_SHIPPED"}`)
	}
	for i := 0; i < 2; i++ {
		lines = append(lines, `{"syscall":"improve","verdict":"KEEP"}`)
	}
	for i := 0; i < 2; i++ {
		lines = append(lines, `{"syscall":"improve","verdict":"REVERT","detail":{"revert_cause":"regressed"}}`)
	}
	path := writeFixture(t, lines)
	h, err := computeHealth(path, 0)
	if err != nil {
		t.Fatalf("computeHealth: %v", err)
	}
	if !approx(h.ClosureRate, Baseline.ClosureRate) {
		t.Errorf("folded closure_rate %v != recorded baseline %v", h.ClosureRate, Baseline.ClosureRate)
	}
	if !approx(h.RegressionRate, Baseline.RegressionRate) {
		t.Errorf("folded regression_rate %v != recorded baseline %v", h.RegressionRate, Baseline.RegressionRate)
	}
	// Folding the baseline journal yields exactly the baseline → healthy, zero delta.
	if !h.Healthy {
		t.Errorf("folding the baseline journal should read healthy, got %s", h.Note)
	}
	if !approx(h.Delta.ClosureRate, 0) || !approx(h.Delta.RegressionRate, 0) {
		t.Errorf("baseline-vs-baseline delta = %+v, want zero", h.Delta)
	}
}

// TestWindowCap proves --window N considers only the most-recent N rows.
func TestWindowCap(t *testing.T) {
	path := writeFixture(t, []string{
		`{"syscall":"improve","verdict":"REVERT"}`, // old — excluded by a window of 3
		`{"syscall":"improve","verdict":"REVERT"}`, // old — excluded
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"improve","verdict":"KEEP"}`,
	})
	h, err := computeHealth(path, 3) // last 3 rows only
	if err != nil {
		t.Fatalf("computeHealth: %v", err)
	}
	if h.Window.RowsConsidered != 3 {
		t.Fatalf("rows_considered = %d, want 3", h.Window.RowsConsidered)
	}
	// Last 3 rows: 2 verify SHIPPED + 1 KEEP → closure 1.0, regression 0.0 (no revert).
	if !approx(h.ClosureRate, 1.0) {
		t.Errorf("windowed closure_rate = %v, want 1.0", h.ClosureRate)
	}
	if !approx(h.RegressionRate, 0.0) {
		t.Errorf("windowed regression_rate = %v, want 0.0 (the two REVERTs fell outside the window)", h.RegressionRate)
	}
}

// TestEmptyAndMissingAreHonestZero proves a missing or empty journal folds to an all-zero
// result with no error (read-only, fail-safe) — the metric exists even before the loop
// has emitted a cycle.
func TestEmptyAndMissingAreHonestZero(t *testing.T) {
	// Missing file.
	h, err := computeHealth(filepath.Join(t.TempDir(), "nope.jsonl"), 0)
	if err != nil {
		t.Fatalf("missing journal should not error, got %v", err)
	}
	if h.ClosureRate != 0 || h.RegressionRate != 0 || h.Window.RowsConsidered != 0 {
		t.Errorf("missing journal should fold to zero, got closure=%v regression=%v rows=%d",
			h.ClosureRate, h.RegressionRate, h.Window.RowsConsidered)
	}
	// Empty file.
	empty := writeFixture(t, nil)
	h2, err := computeHealth(empty, 0)
	if err != nil {
		t.Fatalf("empty journal should not error, got %v", err)
	}
	if h2.Window.ImproveCycles != 0 || h2.Window.VerifyChecks != 0 {
		t.Errorf("empty journal should have no cycles/checks, got %+v", h2.Window)
	}
}

// TestMalformedLineTolerated proves a junk line is skipped, not fatal — the checking
// layer must survive an evolving append log.
func TestMalformedLineTolerated(t *testing.T) {
	path := writeFixture(t, []string{
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`this is not json`,
		`{"syscall":"verify","verdict":"NOT_SHIPPED"}`,
	})
	h, err := computeHealth(path, 0)
	if err != nil {
		t.Fatalf("computeHealth: %v", err)
	}
	if h.Window.VerifyChecks != 2 {
		t.Errorf("verify_checks = %d, want 2 (junk line skipped)", h.Window.VerifyChecks)
	}
	if !approx(h.ClosureRate, 0.5) {
		t.Errorf("closure_rate = %v, want 0.5", h.ClosureRate)
	}
}
