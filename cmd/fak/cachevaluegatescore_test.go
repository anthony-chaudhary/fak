package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// writeSavingsLedger renders the given Track-2 rows through the same encoder the durable
// appender uses (schema-stamped, dimension-normalized) so the fixture round-trips through
// ReadSavingsLedgerFile exactly as a real ledger would.
func writeSavingsLedger(t *testing.T, dir string, rows ...cachevaluereport.SavingsRow) string {
	t.Helper()
	path := filepath.Join(dir, "cache-savings.jsonl")
	var b strings.Builder
	for _, row := range rows {
		line, err := cachevaluereport.AppendSavingsLine(row)
		if err != nil {
			t.Fatalf("encode savings row: %v", err)
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write savings ledger: %v", err)
	}
	return path
}

// runGate drives the wired `fak score cachevalue-gate` consumer against a hermetic savings
// ledger (Track-1 and gateway-usage point at nonexistent temp paths -> empty), returning its
// exit code plus captured stdout/stderr.
func runGate(t *testing.T, savingsPath string, extraArgv ...string) (int, string, string) {
	t.Helper()
	dir := filepath.Dir(savingsPath)
	argv := append([]string{
		"-ledger", filepath.Join(dir, "absent-track1.jsonl"),
		"-savings-ledger", savingsPath,
		"-usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
	}, extraArgv...)
	var stdout, stderr bytes.Buffer
	code := runCacheValueGateScore(&stdout, &stderr, argv)
	return code, stdout.String(), stderr.String()
}

// coldShedRows produce a candidate with gross == net (a fully-cold shed books at 1.0x, so the
// honest blend already equals gross): fak 1000 / (provider 360 + fak 1000) = 73.53%, divergence 0.
func coldShedRows() []cachevaluereport.SavingsRow {
	return []cachevaluereport.SavingsRow{
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 400, SavedTokenEquiv: 360, NetSavedTokenEquiv: 360},
		{Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 1000, CompactionCacheReadTokens: 0,
			SavedTokenEquiv: 1000, NetSavedTokenEquiv: 1000},
	}
}

// TestCacheValueGatePassesHealthyCandidate: a healthy candidate (gross==net, no net regression)
// against a zero baseline is a clean PASS (exit 0), proving the installed gate does not
// false-positive on an honest fold.
func TestCacheValueGatePassesHealthyCandidate(t *testing.T) {
	dir := t.TempDir()
	savings := writeSavingsLedger(t, dir, coldShedRows()...)

	code, stdout, stderr := runGate(t, savings)
	if code != 0 {
		t.Fatalf("healthy candidate: exit = %d, want 0 (PASS)\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	if strings.Contains(stderr, "BLOCK") {
		t.Fatalf("healthy candidate should not print a BLOCK banner; stderr:\n%s", stderr)
	}
}

// TestCacheValueGateBlocksNetRegression: pinning a net floor ABOVE the candidate's net share
// makes the wired consumer BLOCK (exit 1) with the net_nonregression fence -- the headline
// fak_share_net regressed below the last-accepted-honest floor.
func TestCacheValueGateBlocksNetRegression(t *testing.T) {
	dir := t.TempDir()
	savings := writeSavingsLedger(t, dir, coldShedRows()...)
	// Candidate net is ~0.7353; pin the floor at 0.80 so net regressed below it.
	baseline := filepath.Join(dir, "baseline.json")
	if err := os.WriteFile(baseline, []byte(`{"corpus":{"candidate_fak_share_net":0.80,"candidate_fak_share_gross":0.80}}`), 0o644); err != nil {
		t.Fatalf("write baseline: %v", err)
	}

	code, stdout, stderr := runGate(t, savings, "-baseline", baseline)
	if code != 1 {
		t.Fatalf("net regression: exit = %d, want 1 (BLOCK)\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	if !strings.Contains(stderr, "BLOCK") || !strings.Contains(stderr, "net_nonregression") {
		t.Fatalf("net regression should name net_nonregression in the BLOCK banner; stderr:\n%s", stderr)
	}
}

// TestCacheValueGateBlocksFreshGrossUp: a blended (warm+cold) shed opens a gross-net gap past the
// reward-hack alarm; the divergence-ceiling fence is baseline-independent, so the consumer BLOCKs
// (exit 1) even with NO baseline pinned -- a fresh 1.0x-on-warm gross-up is caught at a zero pin.
func TestCacheValueGateBlocksFreshGrossUp(t *testing.T) {
	dir := t.TempDir()
	// Blended shed: shed 1000 with a warm witness of 400 -> net blend 640, gross 1000.
	// net = 640/1000 = 0.64, gross = 1000/1360 = 0.7353, divergence 0.0953 > 0.05.
	rows := []cachevaluereport.SavingsRow{
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 400, SavedTokenEquiv: 360, NetSavedTokenEquiv: 360},
		{Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 1000, CompactionCacheReadTokens: 400,
			SavedTokenEquiv: 640, NetSavedTokenEquiv: 640},
	}
	savings := writeSavingsLedger(t, dir, rows...)

	code, stdout, stderr := runGate(t, savings)
	if code != 1 {
		t.Fatalf("fresh gross-up: exit = %d, want 1 (BLOCK)\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	if !strings.Contains(stderr, "divergence_ceiling") {
		t.Fatalf("fresh gross-up should name divergence_ceiling in the BLOCK banner; stderr:\n%s", stderr)
	}
}

// TestCacheValueGateJSONExitReflectsBlock: --json emits the machine payload with a clean stderr
// (no operator banner) and still exits non-zero on a block, so a CI consumer reads exit code AND
// payload.ok together.
func TestCacheValueGateJSONExitReflectsBlock(t *testing.T) {
	dir := t.TempDir()
	savings := writeSavingsLedger(t, dir, coldShedRows()...)
	baseline := filepath.Join(dir, "baseline.json")
	if err := os.WriteFile(baseline, []byte(`{"corpus":{"candidate_fak_share_net":0.90,"candidate_fak_share_gross":0.90}}`), 0o644); err != nil {
		t.Fatalf("write baseline: %v", err)
	}

	code, stdout, stderr := runGate(t, savings, "-baseline", baseline, "-json")
	if code != 1 {
		t.Fatalf("json block: exit = %d, want 1\nstdout:\n%s", code, stdout)
	}
	if strings.Contains(stderr, "BLOCK") {
		t.Fatalf("--json should keep stderr clean of the operator banner; stderr:\n%s", stderr)
	}
	if !strings.Contains(stdout, "\"ok\": false") || !strings.Contains(stdout, scorecard.CacheValueGateSchema) {
		t.Fatalf("--json payload should carry ok:false and the gate schema; stdout:\n%s", stdout)
	}
}

// TestCacheValueGateFactsFromReportMapping pins the pure percent->fraction mapping, including the
// nil (empty/upside-down corpus) -> 0 floor.
func TestCacheValueGateFactsFromReportMapping(t *testing.T) {
	gross, net := 73.5, 64.0
	facts := cacheValueGateFactsFromReport(cachevaluereport.FleetBenefitReport{
		FakShareGrossPct: &gross, FakSharePct: &net,
	})
	if facts.FakShareGross != 0.735 || facts.FakShareNet != 0.64 {
		t.Fatalf("mapping = gross %v net %v, want 0.735 / 0.64", facts.FakShareGross, facts.FakShareNet)
	}
	empty := cacheValueGateFactsFromReport(cachevaluereport.FleetBenefitReport{})
	if empty.FakShareGross != 0 || empty.FakShareNet != 0 {
		t.Fatalf("nil shares should fold to 0/0, got gross %v net %v", empty.FakShareGross, empty.FakShareNet)
	}
}

// TestLoadCacheValueGateBaseline covers the two pins: an empty path is a zero floor; a prior
// gate --json snapshot pins the floor to its candidate shares.
func TestLoadCacheValueGateBaseline(t *testing.T) {
	var stderr bytes.Buffer
	zero, ok := loadCacheValueGateBaseline(&stderr, "test", "")
	if !ok || zero.FakShareNet != 0 || zero.FakShareGross != 0 {
		t.Fatalf("empty path should be a zero floor; got %+v ok=%v", zero, ok)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(`{"corpus":{"candidate_fak_share_gross":0.5,"candidate_fak_share_net":0.42}}`), 0o644); err != nil {
		t.Fatalf("write snapshot: %v", err)
	}
	pinned, ok := loadCacheValueGateBaseline(&stderr, "test", path)
	if !ok || pinned.FakShareGross != 0.5 || pinned.FakShareNet != 0.42 {
		t.Fatalf("pinned baseline = %+v ok=%v, want gross 0.5 net 0.42", pinned, ok)
	}
}
