package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// writeBudgetLedger writes a gateway-usage ledger with the given rows to a temp
// file and returns its path — the real on-disk records `fak budget` reads.
func writeBudgetLedger(t *testing.T, rows ...gatewayusageledger.Row) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gateway-usage.jsonl")
	for _, r := range rows {
		if err := gatewayusageledger.Append(path, r); err != nil {
			t.Fatalf("append ledger row: %v", err)
		}
	}
	return path
}

// TestRunBudgetReportsSpendVsTargetFromRealRecords is the #2091 done-condition
// witness end to end: `fak budget` reads REAL gateway-usage records, picks the
// current task, and reports spend vs a soft target with the per-category
// breakdown — the "you've spent N, budget M, here is where it went" signal.
func TestRunBudgetReportsSpendVsTargetFromRealRecords(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	older := gatewayusageledger.NewRow("exit", "guard", "stdio", "sess-old", 0, nil,
		gatewayusageledger.Counters{InputTokens: 1, OutputTokens: 1, ObservedTurns: 1, Total: 1}, now.Add(-time.Hour))
	current := gatewayusageledger.NewRow("exit", "guard", "stdio", "sess-now", 0, nil,
		gatewayusageledger.Counters{
			InputTokens:        30000,
			OutputTokens:       10000,
			CachedPromptTokens: 25000,
			ObservedTurns:      18,
			Total:              54,
		}, now)
	path := writeBudgetLedger(t, older, current)

	var out, errOut bytes.Buffer
	rc := runBudget(&out, &errOut, []string{"--json", "--ledger", path, "--target-tokens", "100000", "--target-turns", "40"})
	if rc != 0 {
		t.Fatalf("runBudget rc = %d, stderr:\n%s", rc, errOut.String())
	}

	var r metrics.BudgetReadout
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("decode readout: %v\n%s", err, out.String())
	}
	// No --session: the most recent row (sess-now) is the current task.
	if r.Session != "sess-now" {
		t.Errorf("session = %q, want sess-now (the most recent task)", r.Session)
	}
	if r.Tokens.Spent != 40000 || r.Tokens.Remaining != 60000 || r.Tokens.PercentUsed != 40 {
		t.Errorf("tokens = %+v, want spent 40000 / remaining 60000 / 40%%", r.Tokens)
	}
	if r.Turns.Spent != 18 || r.Turns.Remaining != 22 {
		t.Errorf("turns = %+v, want spent 18 / remaining 22", r.Turns)
	}
	if r.CachedTokens != 25000 {
		t.Errorf("cached tokens = %d, want 25000", r.CachedTokens)
	}
	if len(r.Categories) != 5 {
		t.Fatalf("categories = %d, want 5", len(r.Categories))
	}
	if defects := metrics.GateBudgetLabeled(r); len(defects) > 0 {
		t.Errorf("readout failed the labeling gate: %v", defects)
	}
}

// TestRunBudgetSelectsNamedSession pins --session selection against the ledger.
func TestRunBudgetSelectsNamedSession(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	a := gatewayusageledger.NewRow("exit", "guard", "", "sess-a", 0, nil,
		gatewayusageledger.Counters{InputTokens: 100, OutputTokens: 100}, now.Add(-time.Minute))
	b := gatewayusageledger.NewRow("exit", "guard", "", "sess-b", 0, nil,
		gatewayusageledger.Counters{InputTokens: 999, OutputTokens: 1}, now)
	path := writeBudgetLedger(t, a, b)

	var out, errOut bytes.Buffer
	if rc := runBudget(&out, &errOut, []string{"--json", "--ledger", path, "--session", "sess-a"}); rc != 0 {
		t.Fatalf("runBudget rc = %d, stderr:\n%s", rc, errOut.String())
	}
	var r metrics.BudgetReadout
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if r.Session != "sess-a" || r.Tokens.Spent != 200 {
		t.Errorf("named-session readout = session %q spent %d, want sess-a / 200", r.Session, r.Tokens.Spent)
	}
	// No target passed: the axis reports raw spend without inventing a percentage.
	if r.Tokens.HasTarget {
		t.Errorf("no --target-tokens should leave the token axis target-less: %+v", r.Tokens)
	}
}

// TestRunBudgetEmptyLedger is the clean first-run path: no records yet is a
// non-zero exit with a clear message, not a crash or a fabricated readout.
func TestRunBudgetEmptyLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.jsonl")
	var out, errOut bytes.Buffer
	if rc := runBudget(&out, &errOut, []string{"--ledger", path}); rc != 1 {
		t.Fatalf("runBudget on empty ledger rc = %d, want 1; out=%q err=%q", rc, out.String(), errOut.String())
	}
}

// TestBudgetToolCalls pins the tool-call source preference: adjudication Total
// first, then Submits, then Admitted (the guard-proxy path has Submits == 0).
func TestBudgetToolCalls(t *testing.T) {
	if got := budgetToolCalls(gatewayusageledger.Counters{Total: 7, Submits: 3, Admitted: 2}); got != 7 {
		t.Errorf("Total-present = %d, want 7", got)
	}
	if got := budgetToolCalls(gatewayusageledger.Counters{Submits: 3, Admitted: 2}); got != 3 {
		t.Errorf("Submits fallback = %d, want 3", got)
	}
	if got := budgetToolCalls(gatewayusageledger.Counters{Admitted: 2}); got != 2 {
		t.Errorf("Admitted fallback = %d, want 2", got)
	}
}
