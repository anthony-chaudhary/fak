package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// THE CAPTURED LOOP-TURN ARTIFACT (#2523).
//
// This is the witness the issue asks for: a dispatch turn that swept its finished workers
// and fanned out the leaf one of them shipped, with nobody asking it to. The sweep's
// records and the commit-path reader are stubbed — the seam under test is the automatic
// invocation, not git — and everything else is real: the real planner, the real taxonomy,
// the real durable ledger row.
//
// The fixture deliberately mixes the five shapes a real sweep returns, so the artifact
// shows the honesty counters carrying weight and not just a happy path.
//
// Regenerate after an intentional taxonomy/payload change with UPDATE_GOLDEN=1.
func TestDispatchIssueFanoutCapturesTheLoopTurnArtifact(t *testing.T) {
	old := dispatchWitnessCommitPaths
	t.Cleanup(func() { dispatchWitnessCommitPaths = old })
	dispatchWitnessCommitPaths = func(_, sha string) ([]string, bool) {
		switch sha {
		case "abc1234": // shipped inside a leaf, twice over — one fan-out, not two
			return []string{"internal/issuefanout/turn.go", "internal/issuefanout/ledger.go", "cmd/fak/dispatch_fanout.go"}, true
		case "def5678": // shipped, but outside every internal/<leaf>/
			return []string{"docs/spine-first-defaults.md"}, true
		}
		return nil, false // the commit is gone from this clone
	}

	runsDir := t.TempDir()
	records := []dispatchtick.WitnessRecord{
		{Issue: 2523, SHA: "abc1234", Claim: dispatchtick.ClaimWitnessed},
		{Issue: 2524, SHA: "def5678", Claim: dispatchtick.ClaimWitnessed},
		{Issue: 2525, SHA: "ghi9012", Claim: dispatchtick.ClaimWitnessed},
		{Issue: 2526, SHA: "jkl3456", Claim: dispatchtick.ClaimUnwitnessed},
		{Issue: 2527, Claim: dispatchtick.ClaimNoCommit},
	}
	at := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)

	got := dispatchIssueFanout(".", runsDir, records, at)
	if got == nil {
		t.Fatal("a turn that witnessed a shipped spine must produce an artifact")
	}
	blob, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatalf("encode artifact: %v", err)
	}
	blob = append(blob, '\n')

	goldenPath := filepath.Join("testdata", "dispatch_issue_fanout_turn.golden.json")
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(goldenPath, blob, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if string(blob) != string(want) {
		t.Fatalf("loop-turn artifact mismatch:\n--- got ---\n%s\n--- want (%s) ---\n%s", blob, goldenPath, want)
	}

	// The durable half of the done condition: the ledger row exists on disk, written by
	// the turn itself, and it names the lane that shipped.
	ledger, err := os.ReadFile(filepath.Join(runsDir, dispatchFanoutLedgerName))
	if err != nil {
		t.Fatalf("read invocation ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(ledger)), "\n")
	if len(lines) != 1 {
		t.Fatalf("want exactly one durable invocation row, got %d:\n%s", len(lines), ledger)
	}
	for _, want := range []string{`"leaf":"issuefanout"`, `"outcome":"success"`, `"at":"2026-08-03T12:00:00Z"`} {
		if !strings.Contains(lines[0], want) {
			t.Fatalf("ledger row %q missing %q", lines[0], want)
		}
	}
	if strings.Contains(lines[0], "abc1234") || strings.Contains(lines[0], runsDir) {
		t.Fatalf("ledger row leaks a boundary the row contract excludes: %q", lines[0])
	}
	t.Logf("captured loop-turn artifact:\n%s\nledger row:\n%s", blob, lines[0])
}

// A sweep with nothing shipped adds no payload key at all, so an idle live tick stays
// byte-identical to the one before this seam existed.
func TestDispatchIssueFanoutIsSilentWhenNothingShipped(t *testing.T) {
	old := dispatchWitnessCommitPaths
	t.Cleanup(func() { dispatchWitnessCommitPaths = old })
	dispatchWitnessCommitPaths = func(string, string) ([]string, bool) {
		t.Fatal("no witnessed commit in this sweep — the paths must never be read")
		return nil, false
	}
	runsDir := t.TempDir()
	for _, records := range [][]dispatchtick.WitnessRecord{
		nil,
		{{Issue: 1, SHA: "aaa", Claim: dispatchtick.ClaimUnwitnessed}},
		{{Issue: 2, Claim: dispatchtick.ClaimNoCommit}},
		{{Issue: 3, SHA: "  ", Claim: dispatchtick.ClaimWitnessed}},
	} {
		if got := dispatchIssueFanout(".", runsDir, records, time.Now()); got != nil {
			t.Fatalf("records %+v produced %v, want no payload key", records, got)
		}
	}
	if _, err := os.Stat(filepath.Join(runsDir, dispatchFanoutLedgerName)); !os.IsNotExist(err) {
		t.Fatal("a turn that planned nothing must not create the ledger file")
	}
}

// The issue's own confusion risk, pinned: "Wire the default at ONE seam; two call sites
// double-fire on every turn." A second call site would fan out the same leaf twice per
// tick and, once filed, double-file every fanout-<leaf>-<slug> key.
func TestFanoutDefaultIsWiredAtExactlyOneSeam(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	calls := map[string]int{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		src := string(b)
		if n := strings.Count(src, "dispatchIssueFanout(") - strings.Count(src, "func dispatchIssueFanout("); n > 0 {
			calls[name] = n
		}
	}
	if len(calls) != 1 || calls["dispatch_tick_evaluate.go"] != 1 {
		t.Fatalf("the fan-out default must have exactly one call site (dispatch_tick_evaluate.go), got %v", calls)
	}
}
