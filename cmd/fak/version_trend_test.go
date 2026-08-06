package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

const trendLedgerFixture = `{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":10,"last_commit":"aaa","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":25,"last_commit":"ccc","last_date":"2026-07-03T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g11","last_date":"2026-07-01T00:00:00Z","score":3}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"internal/gateway","kind":"internal","rev":8,"last_commit":"g22","last_date":"2026-07-03T00:00:00Z","score":7.5}
`

// TestRunVersionTrendHuman drives the CLI shell end-to-end against a temp ledger
// and checks the human table: window header, rev delta, stamp count, and the
// score delta line.
func TestRunVersionTrendHuman(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ledger.jsonl")
	if err := os.WriteFile(path, []byte(trendLedgerFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut strings.Builder
	code := runVersionTrend(&out, &errOut, []string{"--ledger", path})
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	got := out.String()
	for _, want := range []string{
		"4 rows over 2026-07-01..2026-07-03  2 modules",
		"Δr+15  r10→r25  2 stamps  cmd/fak",
		"internal/gateway",
		"Δscore +4.5",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("trend output missing %q in:\n%s", want, got)
		}
	}
}

// trendCapLedgerFixture carries three movers — cmd/fak fastest, then
// internal/gateway, then internal/modver — plus two modules stamped exactly once.
// A single stamp cannot witness growth, so those two land in the dormant list.
// That is the shape --top needs: enough rows in BOTH sections that a per-section
// cap is distinguishable from one combined truncation.
const trendCapLedgerFixture = `{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":10,"last_commit":"aaa","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":25,"last_commit":"ccc","last_date":"2026-07-03T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g11","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"internal/gateway","kind":"internal","rev":8,"last_commit":"g22","last_date":"2026-07-03T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/modver","kind":"internal","rev":1,"last_commit":"m11","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"internal/modver","kind":"internal","rev":2,"last_commit":"m22","last_date":"2026-07-03T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/pathutil","kind":"internal","rev":4,"last_commit":"p11","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-02T00:00:00Z","module":"internal/sessionctl","kind":"internal","rev":6,"last_commit":"s11","last_date":"2026-07-02T00:00:00Z"}
`

// TestRunVersionTrendJSONTopCapsEachSection pins the claim runVersionTrend makes
// when it caps AFTER Select rather than through it: --top is N PER SECTION, and
// --json emits the same capped set the human table renders. One combined
// truncation under the default rev-delta sort would drop every dormant module
// before the reader ever saw the dormant list, so the load-bearing assertion is
// that a dormant module SURVIVES --top 1 alongside the top mover.
func TestRunVersionTrendJSONTopCapsEachSection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ledger.jsonl")
	if err := os.WriteFile(path, []byte(trendCapLedgerFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut strings.Builder
	code := runVersionTrend(&out, &errOut, []string{"--ledger", path, "--json", "--top", "1"})
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	var rep modver.TrendReport
	if err := json.Unmarshal([]byte(out.String()), &rep); err != nil {
		t.Fatalf("decode trend JSON: %v\nraw:\n%s", err, out.String())
	}
	// Rows counts the whole ledger even under a cap: the header reports the ledger
	// that was asked, not just the slice that answered.
	if rep.Rows != 8 {
		t.Errorf("rows = %d, want 8 (the cap must not shrink the ledger count)", rep.Rows)
	}
	if got, want := rep.Window, [2]string{"2026-07-01T00:00:00Z", "2026-07-03T00:00:00Z"}; got != want {
		t.Errorf("window = %v, want %v", got, want)
	}
	// --top 1 keeps exactly one module per section, movers first.
	if len(rep.Modules) != 2 {
		t.Fatalf("modules = %d, want 2 (1 mover + 1 dormant); got %+v", len(rep.Modules), rep.Modules)
	}
	mover, dormant := rep.Modules[0], rep.Modules[1]
	if mover.Module != "cmd/fak" {
		t.Errorf("top mover = %q, want cmd/fak (fastest at Δr+15 over the window)", mover.Module)
	}
	if mover.Dormant {
		t.Errorf("top mover %q marked dormant", mover.Module)
	}
	if mover.RevDelta != 15 {
		t.Errorf("top mover rev delta = %d, want 15", mover.RevDelta)
	}
	if dormant.Module != "internal/pathutil" {
		t.Errorf("dormant = %q, want internal/pathutil (stalest single stamp)", dormant.Module)
	}
	if !dormant.Dormant {
		t.Errorf("dormant entry %q not marked dormant", dormant.Module)
	}
	// The whole point of the per-section cap: the mid-ranked movers are gone, but
	// the dormant section still reported someone.
	for _, gone := range []string{"internal/gateway", "internal/modver", "internal/sessionctl"} {
		for _, m := range rep.Modules {
			if m.Module == gone {
				t.Errorf("module %q survived --top 1", gone)
			}
		}
	}
	// And the human table renders that SAME capped set — the two surfaces may not
	// disagree about what --top selected.
	var humanOut, humanErr strings.Builder
	if code := runVersionTrend(&humanOut, &humanErr, []string{"--ledger", path, "--top", "1"}); code != 0 {
		t.Fatalf("human exit %d: %s", code, humanErr.String())
	}
	human := humanOut.String()
	for _, want := range []string{"movers: 1", "dormant: 1", "cmd/fak", "internal/pathutil"} {
		if !strings.Contains(human, want) {
			t.Errorf("human table missing %q in:\n%s", want, human)
		}
	}
	for _, gone := range []string{"internal/gateway", "internal/modver", "internal/sessionctl"} {
		if strings.Contains(human, gone) {
			t.Errorf("human table kept %q under --top 1:\n%s", gone, human)
		}
	}
}

// TestRunVersionTrendMissingLedger witnesses the guided error when no ledger
// exists yet (exit 1, not a panic or a silent empty report).
func TestRunVersionTrendMissingLedger(t *testing.T) {
	var out, errOut strings.Builder
	code := runVersionTrend(&out, &errOut, []string{"--ledger", filepath.Join(t.TempDir(), "nope.jsonl")})
	if code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	if !strings.Contains(errOut.String(), "no ledger at") {
		t.Errorf("want guided 'no ledger' error, got: %s", errOut.String())
	}
}
