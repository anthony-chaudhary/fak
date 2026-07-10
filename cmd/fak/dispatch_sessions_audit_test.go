package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

// wasteRow builds a finished (or live) session row for the waste-lens tests.
func wasteRow(backend, lane, outcome string, live bool, tokens uint64, cacheShare float64) dispatchSessionRow {
	return dispatchSessionRow{
		Backend: backend, Lane: lane, Outcome: outcome, Live: live,
		Tokens: tokens, CacheReadShare: cacheShare,
	}
}

func snapOf(rows ...dispatchSessionRow) dispatchSessionsSnapshot {
	return dispatchSessionsSnapshot{Sessions: rows, SessionCount: len(rows)}
}

// TestFoldDispatchSessionsWasteRate: a backend/lane group where most finished
// sessions wasted work trips the storm-rate finding; a clean group trips nothing.
func TestFoldDispatchSessionsWasteRate(t *testing.T) {
	snap := snapOf(
		// opencode/cmd: 3 of 4 finished sessions wasted → 75% >= 50% → finding.
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRetryStorm), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeQuotaWalled), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeWastedSpawn), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeShipped), false, 0, 0),
		// claude/docs: all shipped → clean.
		wasteRow("claude", "docs", string(dispatchaudit.OutcomeShipped), false, 0, 0),
		wasteRow("claude", "docs", string(dispatchaudit.OutcomeShipped), false, 0, 0),
		wasteRow("claude", "docs", string(dispatchaudit.OutcomeShipped), false, 0, 0),
	)
	findings := foldDispatchSessionsWaste(snap, defaultDispatchSessionsWasteThresholds())
	if len(findings) != 1 {
		t.Fatalf("want exactly 1 finding, got %d: %+v", len(findings), findings)
	}
	f := findings[0]
	if !strings.Contains(f.CodeSite, "waste-rate/opencode/cmd") {
		t.Errorf("code-site = %q, want the opencode/cmd waste-rate site", f.CodeSite)
	}
	if f.Fingerprint == "" || len(f.Fingerprint) != 16 {
		t.Errorf("fingerprint = %q, want a 16-hex hash", f.Fingerprint)
	}
	if !strings.Contains(f.Title, "opencode") || !strings.Contains(f.Title, "lane cmd") {
		t.Errorf("title = %q, want backend+lane", f.Title)
	}
}

// TestFoldDispatchSessionsWasteCacheAndNoop trips the cache-read-collapse and
// token-heavy-no-op detectors from the token-accounting fields.
func TestFoldDispatchSessionsWasteCacheAndNoop(t *testing.T) {
	snap := snapOf(
		// Two substantial sessions reading almost no cache → collapse.
		wasteRow("claude", "core", string(dispatchaudit.OutcomeShipped), false, 30000, 0.05),
		wasteRow("claude", "core", string(dispatchaudit.OutcomeShipped), false, 40000, 0.02),
		// Two token-heavy no-ops → the no-op cluster.
		wasteRow("claude", "core", string(dispatchaudit.OutcomeNoOp), false, 25000, 0.5),
		wasteRow("claude", "core", string(dispatchaudit.OutcomeWastedSpawn), false, 22000, 0.5),
	)
	findings := foldDispatchSessionsWaste(snap, defaultDispatchSessionsWasteThresholds())
	sites := map[string]bool{}
	for _, f := range findings {
		sites[f.CodeSite] = true
	}
	if !sites["cache-read-collapse/claude/core"] {
		t.Errorf("missing cache-read-collapse finding; got %v", sites)
	}
	if !sites["token-heavy-noop/claude/core"] {
		t.Errorf("missing token-heavy-noop finding; got %v", sites)
	}
}

// TestFoldDispatchSessionsWasteLiveExcluded: a still-running worker is not yet waste,
// so a group made only of live sessions produces nothing.
func TestFoldDispatchSessionsWasteLiveExcluded(t *testing.T) {
	snap := snapOf(
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRunning), true, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRunning), true, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRunning), true, 0, 0),
	)
	if f := foldDispatchSessionsWaste(snap, defaultDispatchSessionsWasteThresholds()); len(f) != 0 {
		t.Fatalf("live-only group produced findings: %+v", f)
	}
}

// TestFoldDispatchSessionsWasteDeterministic: same snapshot → identical findings.
func TestFoldDispatchSessionsWasteDeterministic(t *testing.T) {
	snap := snapOf(
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRetryStorm), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeQuotaWalled), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeWastedSpawn), false, 0, 0),
		wasteRow("zbackend", "z", string(dispatchaudit.OutcomeNoOp), false, 30000, 0.5),
		wasteRow("zbackend", "z", string(dispatchaudit.OutcomeNoOp), false, 30000, 0.5),
	)
	th := defaultDispatchSessionsWasteThresholds()
	a := foldDispatchSessionsWaste(snap, th)
	b := foldDispatchSessionsWaste(snap, th)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("fold not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

// TestRunDispatchSessionsAuditDryRun: the default (--file-issues without --confirm)
// prints fingerprinted candidates and writes NO markers and calls no gh — a sweep is
// side-effect free until an operator opts into --confirm.
func TestRunDispatchSessionsAuditDryRun(t *testing.T) {
	runsDir := t.TempDir()
	snap := snapOf(
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRetryStorm), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeQuotaWalled), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeWastedSpawn), false, 0, 0),
	)
	findings := foldDispatchSessionsWaste(snap, defaultDispatchSessionsWasteThresholds())
	if len(findings) != 1 {
		t.Fatalf("fixture should yield 1 finding, got %d", len(findings))
	}
	fp := findings[0].Fingerprint

	var stdout, stderr strings.Builder
	code := runDispatchSessionsAudit(&stdout, &stderr, runsDir, snap, false /* dry-run */, 0)
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "dry-run") {
		t.Errorf("dry-run output missing the dry-run notice:\n%s", out)
	}
	if !strings.Contains(out, fp) {
		t.Errorf("dry-run output missing the candidate fingerprint %s:\n%s", fp, out)
	}
	// The load-bearing invariant: dry-run left no marker on disk.
	if dispatchaudit.AlreadyFiled(runsDir, fp) {
		t.Errorf("dry-run wrote a filed-marker for %s; it must not", fp)
	}
}

// TestRunDispatchSessionsAuditDedup: a fingerprint already marked as filed is deduped
// out of the dry-run candidate list.
func TestRunDispatchSessionsAuditDedup(t *testing.T) {
	runsDir := t.TempDir()
	snap := snapOf(
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeRetryStorm), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeQuotaWalled), false, 0, 0),
		wasteRow("opencode", "cmd", string(dispatchaudit.OutcomeWastedSpawn), false, 0, 0),
	)
	fp := foldDispatchSessionsWaste(snap, defaultDispatchSessionsWasteThresholds())[0].Fingerprint
	if err := dispatchaudit.MarkFiled(runsDir, fp); err != nil {
		t.Fatalf("mark filed: %v", err)
	}

	var stdout, stderr strings.Builder
	if code := runDispatchSessionsAudit(&stdout, &stderr, runsDir, snap, false, 0); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "already filed") {
		t.Errorf("want a deduped/already-filed notice for the marked fingerprint:\n%s", out)
	}
	if strings.Contains(out, "candidate improvement tickets") {
		t.Errorf("marked fingerprint should not appear as a fresh candidate:\n%s", out)
	}
}

// TestRunDispatchSessionsAuditClean: no waste → a clean, zero-exit report.
func TestRunDispatchSessionsAuditClean(t *testing.T) {
	snap := snapOf(
		wasteRow("claude", "docs", string(dispatchaudit.OutcomeShipped), false, 0, 0),
		wasteRow("claude", "docs", string(dispatchaudit.OutcomeShipped), false, 0, 0),
	)
	var stdout, stderr strings.Builder
	if code := runDispatchSessionsAudit(&stdout, &stderr, t.TempDir(), snap, false, 0); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "no systemic waste") {
		t.Errorf("clean snapshot should report no systemic waste:\n%s", stdout.String())
	}
}
