package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunProjectSelfcheck proves the `fak project selfcheck` subcommand runs the
// deterministic fold + ledger/trend proof and reports OK, no gh, no key.
func TestRunProjectSelfcheck(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runProjectSelfcheck(&out, &errb, nil); code != 0 {
		t.Fatalf("selfcheck exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "SELFCHECK OK") {
		t.Fatalf("selfcheck stdout missing OK banner:\n%s", out.String())
	}
}

// TestRunProjectReportAppendHistory proves `fak project report --from-items … --append-history
// --ledger …` folds hermetically (no gh), trends against the durable ledger, and durably
// appends the tick — the accrual the weekly cadence step rides.
func TestRunProjectReportAppendHistory(t *testing.T) {
	dir := t.TempDir()
	items := filepath.Join(dir, "items.json")
	if err := os.WriteFile(items, []byte(
		`[{"issue":1,"status":"Todo","generation":"now","priority":"P1"},`+
			`{"issue":2,"status":"Backlog","generation":"next","priority":"P2"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	ledger := filepath.Join(dir, "history.jsonl")

	// Pre-seed a prior tick dated well in the past (distinct generated_at) so the trend
	// is computed against it deterministically — a fully-classified board here sheds the
	// seed's 3 unclassified items, so the direction is "improved".
	seed := `{"schema":"fak.project-ledger/1","date":"2020-01-01","commit":"seed",` +
		`"generated_at":"2020-01-01T00:00:00Z","verdict":"ACTION","measured":true,"total":5,"unclassified":3}` + "\n"
	if err := os.WriteFile(ledger, []byte(seed), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runProjectReport(&out, &errb, []string{"--from-items", items, "--append-history", "--ledger", ledger, "--json"})
	if code != 0 {
		t.Fatalf("report exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, `"trend"`) || strings.Contains(got, `"direction": "new"`) {
		t.Fatalf("expected a trend computed against the seeded row (not 'new'):\n%s", got)
	}
	if !strings.Contains(got, `"direction": "improved"`) {
		t.Fatalf("classifying every item should read as improved vs the seed:\n%s", got)
	}

	// The tick durably appended a second row.
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("ledger unreadable: %v", err)
	}
	rows := 0
	for _, ln := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if strings.TrimSpace(ln) != "" {
			rows++
		}
	}
	if rows != 2 {
		t.Fatalf("ledger has %d rows after append, want 2 (seed + tick)\n%s", rows, data)
	}
}
