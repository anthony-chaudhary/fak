package cadencereport

import (
	"encoding/json"
	"strings"
	"testing"
)

// jsonMap unmarshals a JSON literal into the map[string]any shape the live
// runner hands the interpreters, so tests exercise the exact float64/string
// types json.Unmarshal produces (not hand-built Go types).
func jsonMap(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("bad test JSON: %v", err)
	}
	return m
}

func TestInterpretScores(t *testing.T) {
	good := InterpretScores(jsonMap(t, `{
		"total_debt": 40, "measured": 13, "errored": 0,
		"trend": {"direction": "improved", "summary": "improved -4 vs @abc (was 44, now 40)"}
	}`), "")
	if good.Debt != 40 || good.Measured != 13 {
		t.Fatalf("debt/measured = %d/%d, want 40/13", good.Debt, good.Measured)
	}
	if good.TrendDirection != "improved" || !strings.Contains(good.TrendSummary, "was 44") {
		t.Fatalf("trend = %q / %q", good.TrendDirection, good.TrendSummary)
	}
	if !good.OK || good.Err != "" {
		t.Fatalf("good scores should be OK with no err, got ok=%v err=%q", good.OK, good.Err)
	}

	regressed := InterpretScores(jsonMap(t, `{"total_debt": 50, "measured": 13, "errored": 0, "trend": {"direction": "regressed"}}`), "")
	if regressed.OK {
		t.Fatal("a regressed score must not be OK")
	}

	unmeasured := InterpretScores(jsonMap(t, `{"total_debt": 40, "measured": 12, "errored": 1, "trend": {"direction": "flat"}}`), "")
	if unmeasured.OK || unmeasured.Err == "" {
		t.Fatalf("errored scorecard must set Err and not be OK, got ok=%v err=%q", unmeasured.OK, unmeasured.Err)
	}

	failed := InterpretScores(nil, "timed out after 300s")
	if failed.Err != "timed out after 300s" || failed.TrendDirection != "unknown" {
		t.Fatalf("failed run = %+v", failed)
	}
}

func TestInterpretReleases(t *testing.T) {
	good := InterpretReleases(jsonMap(t, `{
		"ok": true, "verdict": "OK",
		"rolling": {"last_tag": "v1.2.3"},
		"next_action": {"kind": "wait", "detail": "nothing release-worthy pending"}
	}`), "")
	if good.Version != "v1.2.3" || good.ActionKind != "wait" || !good.OK {
		t.Fatalf("good releases = %+v", good)
	}
	if !strings.Contains(good.ActionDetail, "nothing release-worthy") {
		t.Fatalf("detail = %q", good.ActionDetail)
	}

	noTag := InterpretReleases(jsonMap(t, `{"ok": false, "verdict": "ACTION", "rolling": {"last_tag": null}, "next_action": {"kind": "confirm_ci", "detail": "x"}}`), "")
	if noTag.Version != "(none)" {
		t.Fatalf("missing tag should render (none), got %q", noTag.Version)
	}

	failed := InterpretReleases(nil, "non-JSON output: boom")
	if failed.Err == "" || failed.Verdict != "ERROR" {
		t.Fatalf("failed run = %+v", failed)
	}
}

func okScores() Scores {
	return Scores{Debt: 40, Measured: 13, TrendDirection: "improved", TrendSummary: "improved -4", OK: true}
}
func okWork() Work { return Work{WindowDays: 7, Commits: 23, Ships: 18} }
func okReleases() Releases {
	return Releases{Version: "v1.2.3", ActionKind: "wait", ActionDetail: "pending", Verdict: "OK", OK: true}
}
func foldOpts() FoldOpts {
	return FoldOpts{Workspace: "/repo", Commit: "abc1234", GeneratedAt: "2026-06-26T00:00:00Z", Date: "2026-06-26"}
}

func TestFoldRecorded(t *testing.T) {
	r := Fold(okScores(), okWork(), okReleases(), foldOpts())
	if !r.OK || r.Verdict != "OK" || r.Finding != "cadence_recorded" {
		t.Fatalf("clean fold = ok=%v verdict=%q finding=%q", r.OK, r.Verdict, r.Finding)
	}
	if r.Schema != Schema {
		t.Fatalf("schema = %q", r.Schema)
	}
	for _, want := range []string{"debt 40", "23 commit", "18 ship", "v1.2.3", "wait"} {
		if !strings.Contains(r.Reason, want) {
			t.Fatalf("reason %q missing %q", r.Reason, want)
		}
	}
}

func TestFoldScoreRegressionIsAdvisoryNotGate(t *testing.T) {
	s := okScores()
	s.TrendDirection = "regressed"
	s.OK = false
	r := Fold(s, okWork(), okReleases(), foldOpts())
	// The scorecard ratchet owns debt regressions; the cadence report must not
	// double-gate them -- it stays OK and surfaces the regression advisory.
	if !r.OK || r.Finding != "cadence_advisory" {
		t.Fatalf("score regression should be advisory-OK, got ok=%v finding=%q", r.OK, r.Finding)
	}
	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("advisory regression must not fail --check, got exit %d", code)
	}
}

func TestFoldUnmeasuredGates(t *testing.T) {
	w := okWork()
	w.Err = "git rev-list failed: not a repo"
	r := Fold(okScores(), w, okReleases(), foldOpts())
	if r.OK || r.Verdict != "ACTION" || r.Finding != "cadence_unmeasured" {
		t.Fatalf("unmeasured dimension must gate, got ok=%v verdict=%q finding=%q", r.OK, r.Verdict, r.Finding)
	}
	code, msg := CheckGate(r)
	if code != 1 || !strings.Contains(msg, "INCOMPLETE") {
		t.Fatalf("CheckGate over unmeasured = %d %q", code, msg)
	}
}

func TestParseLedgerTolerant(t *testing.T) {
	content := strings.Join([]string{
		`{"schema":"fak-cadence-ledger/1","date":"2026-06-20","commit":"a","scores_debt":44,"work_commits":20,"work_ships":10}`,
		``,
		`not json at all`,
		`{"date":"","commit":"skipme"}`,
		`{"schema":"fak-cadence-ledger/1","date":"2026-06-26","commit":"b","scores_debt":40,"work_commits":25,"work_ships":18}`,
	}, "\n")
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("want 2 valid rows, got %d: %+v", len(rows), rows)
	}
	if rows[0].Date != "2026-06-20" || rows[1].ScoresDebt != 40 {
		t.Fatalf("rows mis-parsed: %+v", rows)
	}
}

func TestTrendVsLast(t *testing.T) {
	prior := []LedgerRow{
		{Date: "2026-06-20", Commit: "a", ScoresDebt: 44, WorkCommits: 20, WorkShips: 15, GeneratedAt: "2026-06-20T00:00:00Z"},
		{Date: "2026-06-23", Commit: "b", ScoresDebt: 42, WorkCommits: 22, WorkShips: 16, GeneratedAt: "2026-06-23T00:00:00Z"},
	}
	row := LedgerRow{Date: "2026-06-26", Commit: "c", ScoresDebt: 40, WorkCommits: 25, WorkShips: 18, WorkWindowDays: 7, GeneratedAt: "2026-06-26T00:00:00Z"}
	tr := TrendVsLast(row, prior)
	if tr.Direction != "improved" || tr.DebtDelta != -2 || tr.DebtFrom != 42 || tr.DebtTo != 40 {
		t.Fatalf("improved trend = %+v", tr)
	}
	if tr.PrevDate != "2026-06-23" {
		t.Fatalf("trend should compare vs the latest prior row, got prev %q", tr.PrevDate)
	}
	if tr.WorkCommitsDelta != 3 || tr.WorkShipsDelta != 2 {
		t.Fatalf("work deltas wrong: commits %+d, ships %+d", tr.WorkCommitsDelta, tr.WorkShipsDelta)
	}

	first := TrendVsLast(row, nil)
	if first.Direction != "new" || !strings.Contains(first.Summary, "first cadence tick") {
		t.Fatalf("first tick = %+v", first)
	}

	worse := TrendVsLast(LedgerRow{Date: "2026-06-27", ScoresDebt: 50, WorkCommits: 18, WorkShips: 12, GeneratedAt: "2026-06-27T00:00:00Z"}, prior)
	if worse.Direction != "regressed" || worse.DebtDelta != 8 {
		t.Fatalf("regressed trend = %+v", worse)
	}

	flat := TrendVsLast(LedgerRow{Date: "2026-06-27", ScoresDebt: 42, WorkCommits: 22, WorkShips: 16, GeneratedAt: "2026-06-27T00:00:00Z"}, prior)
	if flat.Direction != "flat" {
		t.Fatalf("flat trend = %+v", flat)
	}
}

func TestTrendExcludesSameGeneratedAt(t *testing.T) {
	// An idempotent re-append (same generated_at) must not trend against itself.
	prior := []LedgerRow{
		{Date: "2026-06-20", ScoresDebt: 44, WorkCommits: 20, WorkShips: 15, GeneratedAt: "2026-06-20T00:00:00Z"},
		{Date: "2026-06-26", ScoresDebt: 40, WorkCommits: 25, WorkShips: 18, GeneratedAt: "2026-06-26T12:00:00Z"},
	}
	row := LedgerRow{Date: "2026-06-26", ScoresDebt: 40, WorkCommits: 25, WorkShips: 18, GeneratedAt: "2026-06-26T12:00:00Z"}
	tr := TrendVsLast(row, prior)
	if tr.PrevDate != "2026-06-20" {
		t.Fatalf("same generated_at row should be excluded, got prev %q", tr.PrevDate)
	}
}

func TestRowFromReportRoundTrip(t *testing.T) {
	r := Fold(okScores(), okWork(), okReleases(), foldOpts())
	row := RowFromReport(r)
	if row.Schema != LedgerSchema || row.Date != "2026-06-26" || row.ScoresDebt != 40 ||
		row.WorkCommits != 23 || row.WorkShips != 18 || row.ReleaseVersion != "v1.2.3" || row.ReleaseAction != "wait" {
		t.Fatalf("row projection = %+v", row)
	}
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatal(err)
	}
	var back LedgerRow
	if err := json.Unmarshal([]byte(line), &back); err != nil {
		t.Fatalf("ledger line not valid JSON: %v", err)
	}
	if back.ScoresDebt != 40 || back.ScoresTrend != "improved" {
		t.Fatalf("round-trip lost fields: %+v", back)
	}
}

func TestWithGateJSON(t *testing.T) {
	r := Fold(okScores(), okWork(), okReleases(), foldOpts())
	code, msg := CheckGate(r)
	g := r.WithGate(code, msg)
	if g.GateExit == nil || *g.GateExit != 0 || g.GateMessage == "" {
		t.Fatalf("gate fields not set: %+v", g)
	}
	b, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"gate_exit"`) {
		t.Fatalf("gate_exit not emitted in JSON: %s", b)
	}
}

func TestRenderSmoke(t *testing.T) {
	r := Fold(okScores(), okWork(), okReleases(), foldOpts())
	tr := TrendVsLast(RowFromReport(r), []LedgerRow{{Date: "2026-06-20", ScoresDebt: 44, GeneratedAt: "2026-06-20T00:00:00Z"}})
	r.Trend = &tr
	out := Render(r)
	for _, want := range []string{"cadence report", "scores", "work", "releases", "trend:", "->"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}
