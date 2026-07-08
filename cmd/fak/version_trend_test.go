package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
