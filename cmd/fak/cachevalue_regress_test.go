package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cvregress"
)

// regressLedgerRow renders one Track-1 WITNESSED ledger line. It marshals the REAL
// cachevalueledger.Row rather than hand-writing JSON so the fixture cannot drift from the
// schema it claims to seed (#3695 is explicitly not allowed to redefine that schema).
func regressLedgerRow(t *testing.T, date string, turns, prompt, reused uint64) string {
	t.Helper()
	b, err := json.Marshal(cachevalueledger.Row{
		Schema:       cachevalueledger.Schema,
		Date:         date,
		SessionType:  "nightrun",
		Provider:     "anthropic",
		Context:      "agent",
		Turns:        turns,
		PromptTokens: prompt,
		ReusedTokens: reused,
	})
	if err != nil {
		t.Fatalf("marshal ledger row: %v", err)
	}
	return string(b)
}

// seedRegressLedger writes the seeded corpus to a temp ledger and returns its path.
func seedRegressLedger(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cache-value.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("seed ledger: %v", err)
	}
	return path
}

// runRegress drives the wired `fak cachevalue regress` consumer against a hermetic ledger.
func runRegress(t *testing.T, argv ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runCachevalueRegress(&stdout, &stderr, argv)
	return code, stdout.String(), stderr.String()
}

// TestCachevalueRegressFiresAndGates is the #3695 witness that the fence FIRES: a seeded
// session far below the pinned floor is flagged, named in the render, and -- only under
// --gate -- turns into a non-zero exit. Without --gate the same regression stays exit 0, so
// the default render remains a read-only report.
func TestCachevalueRegressFiresAndGates(t *testing.T) {
	// hit-rate 10% (pinned floor 40%), write-amp 9.00 (pinned ceiling 1.50).
	ledger := seedRegressLedger(t,
		regressLedgerRow(t, "2026-07-01", 6, 100000, 70000), // healthy: 70% hit, 0.43 amp
		regressLedgerRow(t, "2026-07-02", 8, 200000, 20000), // REGRESSED
	)

	code, stdout, stderr := runRegress(t, "--ledger", ledger)
	if code != 0 {
		t.Fatalf("without --gate: exit = %d, want 0 (stderr=%q)", code, stderr)
	}
	if !strings.Contains(stdout, "REGRESSED") {
		t.Fatalf("without --gate: stdout missing REGRESSED verdict:\n%s", stdout)
	}
	// The sample flagged session the issue's witness asks for must be legible by date.
	if !strings.Contains(stdout, "2026-07-02") {
		t.Fatalf("flagged session 2026-07-02 not named in render:\n%s", stdout)
	}
	// The healthy session must NOT be flagged: it may appear nowhere in the flagged block.
	if strings.Contains(stdout, "2026-07-01") {
		t.Fatalf("healthy session 2026-07-01 was flagged:\n%s", stdout)
	}

	gated, _, _ := runRegress(t, "--ledger", ledger, "--gate")
	if gated != 1 {
		t.Fatalf("with --gate: exit = %d, want 1 on REGRESSED", gated)
	}
}

// TestCachevalueRegressJSONNamesTheFlaggedSession asserts the machine payload carries the
// per-session hit%/write-amp evidence and the reason, not just a bare verdict.
func TestCachevalueRegressJSONNamesTheFlaggedSession(t *testing.T) {
	ledger := seedRegressLedger(t, regressLedgerRow(t, "2026-07-02", 8, 200000, 20000))

	code, stdout, stderr := runRegress(t, "--ledger", ledger, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr=%q)", code, stderr)
	}
	var rep cvregress.Report
	if err := json.Unmarshal([]byte(stdout), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, stdout)
	}
	if rep.Verdict != "REGRESSED" || rep.OK {
		t.Fatalf("verdict = %q ok = %v, want REGRESSED/false", rep.Verdict, rep.OK)
	}
	if len(rep.Regressions) != 1 {
		t.Fatalf("regressions = %d, want 1", len(rep.Regressions))
	}
	got := rep.Regressions[0]
	if got.Date != "2026-07-02" || !got.Regressed {
		t.Fatalf("flagged session = %+v, want 2026-07-02 regressed", got)
	}
	if got.HitRatePct < 9.9 || got.HitRatePct > 10.1 {
		t.Fatalf("hit_rate_pct = %.3f, want ~10.0", got.HitRatePct)
	}
	if got.WriteAmp < 8.9 || got.WriteAmp > 9.1 {
		t.Fatalf("write_amp = %.3f, want ~9.0", got.WriteAmp)
	}
	if !strings.Contains(got.Reason, "below pinned floor") {
		t.Fatalf("reason = %q, want the pinned-floor reason", got.Reason)
	}
}

// TestCachevalueRegressClearsOnHealthyCorpus is the other half of the witness: the fence
// CLEARS. A corpus that holds the pin is OK and stays exit 0 even under --gate.
func TestCachevalueRegressClearsOnHealthyCorpus(t *testing.T) {
	ledger := seedRegressLedger(t,
		regressLedgerRow(t, "2026-07-01", 6, 100000, 70000),
		regressLedgerRow(t, "2026-07-02", 4, 80000, 60000),
	)

	code, stdout, stderr := runRegress(t, "--ledger", ledger, "--gate")
	if code != 0 {
		t.Fatalf("healthy corpus under --gate: exit = %d, want 0 (stderr=%q)", code, stderr)
	}
	if !strings.Contains(stdout, "OK") {
		t.Fatalf("stdout missing OK verdict:\n%s", stdout)
	}
	if strings.Contains(stdout, "flagged sessions") {
		t.Fatalf("healthy corpus rendered a flagged block:\n%s", stdout)
	}
}

// TestCachevalueRegressInsufficientFallsOpen pins the fall-open posture: a corpus of only
// single-turn and below-min-size rows is missing evidence, not a regression, so --gate stays
// green. A fence that reds on a thin corpus would be retired by CI within a week.
func TestCachevalueRegressInsufficientFallsOpen(t *testing.T) {
	ledger := seedRegressLedger(t,
		regressLedgerRow(t, "2026-07-01", 1, 100000, 0), // single-turn: dropped
		regressLedgerRow(t, "2026-07-02", 5, 120, 0),    // below min-prompt-tokens: dropped
	)

	code, stdout, stderr := runRegress(t, "--ledger", ledger, "--gate")
	if code != 0 {
		t.Fatalf("thin corpus under --gate: exit = %d, want 0 (stderr=%q)", code, stderr)
	}
	if !strings.Contains(stdout, "INSUFFICIENT") {
		t.Fatalf("stdout missing INSUFFICIENT verdict:\n%s", stdout)
	}
}

// TestCachevalueRegressPinIsOperatorRatchetable proves the pin is a knob, not a constant: the
// same healthy corpus reds once the operator ratchets the floor above what it realizes.
func TestCachevalueRegressPinIsOperatorRatchetable(t *testing.T) {
	ledger := seedRegressLedger(t, regressLedgerRow(t, "2026-07-01", 6, 100000, 70000))

	if code, _, _ := runRegress(t, "--ledger", ledger, "--gate"); code != 0 {
		t.Fatalf("at the default pin: exit = %d, want 0", code)
	}
	code, stdout, _ := runRegress(t, "--ledger", ledger, "--gate", "--hit-rate-floor", "85")
	if code != 1 {
		t.Fatalf("at a ratcheted 85%% floor: exit = %d, want 1\n%s", code, stdout)
	}
}

// TestCachevalueRegressRejectsBadInput keeps the usage contract: a malformed --since and a
// stray positional are usage errors (exit 2), not a silently-empty green fold.
func TestCachevalueRegressRejectsBadInput(t *testing.T) {
	if code, _, _ := runRegress(t, "--since", "07-2026"); code != 2 {
		t.Fatalf("malformed --since: exit = %d, want 2", code)
	}
	if code, _, _ := runRegress(t, "unexpected"); code != 2 {
		t.Fatalf("stray positional: exit = %d, want 2", code)
	}
}
