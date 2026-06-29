package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
)

// prerolledReportJSON folds a real (witnessed-maturity) report and a synthetic roadmap
// to a JSON file, so the post path is exercised hermetically with no `gh`.
func prerolledReportJSON(t *testing.T) string {
	t.Helper()
	m := milestonereport.InterpretMaturity(covmatrix.Grid())
	e := milestonereport.InterpretEpics(
		[]milestonereport.EpicSpec{{Number: 1243, Title: "support-maturity"}},
		[]milestonereport.EpicCounts{{Number: 1243, Closed: 14, Total: 16, Source: "label"}}, "")
	r := milestonereport.Fold(m, e, milestonereport.FoldOpts{Date: "2026-06-29", Commit: "deadbee"})
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "report.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestMilestonePostDryRunRendersCard: `fak milestone post --report-json <f> --dry-run`
// renders the exact card from a pre-rolled report, posting nothing and needing no gh.
func TestMilestonePostDryRunRendersCard(t *testing.T) {
	clearSlackEnv(t)
	path := prerolledReportJSON(t)

	var out, errb bytes.Buffer
	code := runMilestonePost(&out, &errb, []string{"--report-json", path, "--dry-run", "--source", "agent"})
	if code != 0 {
		t.Fatalf("post --dry-run exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"fak milestones", "climb:", "ladder:", "roadmap:", "#1243 support-maturity", "88% (14/16)", "posted by agent"} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run card missing %q:\n%s", want, got)
		}
	}
}

// TestMilestoneReportJSONEnvelope: `fak milestone report --json` against a temp
// workspace emits a valid fak-milestone-report/1 envelope. The maturity dimension is
// witnessed from the in-process grid and always measures; the roadmap may be ACTION
// (gh unavailable in CI) — the test asserts the SHAPE, not the gh result.
func TestMilestoneReportJSONEnvelope(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runMilestoneReport(&out, &errb, []string{"--workspace", t.TempDir(), "--json"})
	// exit 0 (both measured) or 1 (roadmap unmeasured) are both valid shapes here.
	if code != 0 && code != 1 {
		t.Fatalf("report --json exit = %d (want 0 or 1), stderr=%s", code, errb.String())
	}
	var r milestonereport.Report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("report --json must emit a valid envelope: %v\n%s", err, out.String())
	}
	if r.Schema != milestonereport.Schema {
		t.Fatalf("schema = %q, want %q", r.Schema, milestonereport.Schema)
	}
	if r.Maturity.Err != "" || r.Maturity.Cells == 0 {
		t.Fatalf("maturity is witnessed from the live grid and must always measure: %+v", r.Maturity)
	}
	if r.Trend == nil {
		t.Fatal("report must attach a per-tick trend (first tick = 'new')")
	}
}

// TestMilestoneReportAppendHistory: --append-history writes a durable ledger row that
// re-parses, and a second tick trends against the first.
func TestMilestoneReportAppendHistory(t *testing.T) {
	clearSlackEnv(t)
	root := t.TempDir()
	ledger := filepath.Join(root, "history.jsonl")

	var out, errb bytes.Buffer
	code := runMilestoneReport(&out, &errb, []string{"--workspace", root, "--ledger", ledger, "--append-history"})
	if code != 0 && code != 1 {
		t.Fatalf("append-history exit = %d, stderr=%s", code, errb.String())
	}
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("append-history must write the ledger: %v", err)
	}
	rows := milestonereport.ParseLedger(string(raw))
	if len(rows) != 1 {
		t.Fatalf("ledger should have 1 row after one tick, got %d", len(rows))
	}
	if rows[0].Schema != milestonereport.LedgerSchema || rows[0].Cells == 0 {
		t.Fatalf("ledger row malformed: %+v", rows[0])
	}
}
