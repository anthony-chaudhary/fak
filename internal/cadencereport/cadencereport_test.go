package cadencereport

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	maturityscore "github.com/anthony-chaudhary/fak/internal/maturity"
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
	if good.GradeDebt != 40 {
		t.Fatalf("grade debt fallback = %d, want 40 for legacy payload", good.GradeDebt)
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

	withSeverity := InterpretScores(jsonMap(t, `{"total_debt": 500, "grade_debt": 7, "measured": 20, "errored": 0, "trend": {"direction": "flat"}}`), "")
	if withSeverity.GradeDebt != 7 {
		t.Fatalf("grade_debt should come from payload when present, got %d", withSeverity.GradeDebt)
	}
}

func TestInterpretScoresFromFile(t *testing.T) {
	payload := `{"total_debt": 40, "measured": 13, "errored": 0, "trend": {"direction": "improved", "summary": "improved -4 vs @abc"}}`

	// A good payload via --scores-from must fold to the SAME SCORES as the live
	// interpret path (so the captured pane drives an identical result).
	dir := t.TempDir()
	good := filepath.Join(dir, "scores.json")
	if err := os.WriteFile(good, []byte(payload), 0o644); err != nil {
		t.Fatal(err)
	}
	fromFile := InterpretScoresFromFile(good, nil)
	live := InterpretScores(jsonMap(t, payload), "")
	if fromFile != live {
		t.Fatalf("--scores-from = %+v, want it to equal the live interpret %+v", fromFile, live)
	}

	// Stdin path ("-") reads the same payload from the injected reader.
	fromStdin := InterpretScoresFromFile("-", strings.NewReader(payload))
	if fromStdin != live {
		t.Fatalf("--scores-from - = %+v, want %+v", fromStdin, live)
	}

	// A garbled file degrades to an ERRORED SCORES dimension (Err set, not OK,
	// trend unknown) — never a silent zero.
	garbled := filepath.Join(dir, "garbled.json")
	if err := os.WriteFile(garbled, []byte("not json at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := InterpretScoresFromFile(garbled, nil)
	if g.OK || g.Err == "" || g.TrendDirection != "unknown" {
		t.Fatalf("garbled file should degrade to errored, got %+v", g)
	}

	// A missing file likewise degrades to errored.
	m := InterpretScoresFromFile(filepath.Join(dir, "nope.json"), nil)
	if m.OK || m.Err == "" {
		t.Fatalf("missing file should degrade to errored, got %+v", m)
	}

	// And an errored SCORES dimension drives Fold -> cadence_unmeasured -> gate exit 1.
	report := Fold(g, okWork(), okReleases(), foldOpts())
	if report.Finding != "cadence_unmeasured" {
		t.Fatalf("errored scores should fold to cadence_unmeasured, got %q", report.Finding)
	}
	if code, _ := CheckGate(report); code != 1 {
		t.Fatalf("cadence_unmeasured should gate exit 1, got %d", code)
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

func TestWithPublishStalenessIsInformationalAndSurfaced(t *testing.T) {
	// The projection layers the lag on without flipping OK (informational, never gating).
	base := Releases{Version: "v0.34.0", ActionKind: "cut_release", Verdict: "ACTION", OK: false}
	r := WithPublishStaleness(base, 1602, 3.4, "very_stale")
	if r.CommitsBehind != 1602 || r.DaysBehind != 3.4 || r.PublishVerdict != "very_stale" {
		t.Fatalf("staleness not layered on: %+v", r)
	}
	if r.OK != base.OK {
		t.Fatalf("WithPublishStaleness must not change OK")
	}

	// A fresh @latest (0 behind) is rendered without a lag suffix and trends as 0.
	fresh := WithPublishStaleness(Releases{Version: "v9.9.9", ActionKind: "wait", OK: true}, 0, 0, "fresh")
	if publishLagSuffix(fresh) != "" {
		t.Fatalf("fresh @latest must render no lag suffix, got %q", publishLagSuffix(fresh))
	}

	// The lag appears in the fold summary (reason) and the human render, and trends in the row.
	rep := Fold(okScores(), okWork(), r, foldOpts())
	if !strings.Contains(rep.Reason, "1602 behind") {
		t.Fatalf("fold reason should surface the lag, got %q", rep.Reason)
	}
	if !strings.Contains(Render(rep), "1602 behind") {
		t.Fatalf("render should surface the lag")
	}
	if RowFromReport(rep).ReleaseCommitsBehind != 1602 {
		t.Fatalf("ledger row should carry the lag for trending")
	}
}

func TestMaturityFromScorecard(t *testing.T) {
	got := MaturityFromScorecard(maturityscore.ScorecardPayload{
		OK: true,
		Corpus: map[string]any{
			"maturity_debt": 0,
			"score":         81,
			"grade":         "B",
			"capabilities":  12,
			"ladder_skips":  0,
			"backlog":       3,
			"distribution": map[string]int{
				"tested":    5,
				"dogfooded": 7,
			},
		},
		Backlog: []maturityscore.NextWork{
			{
				Lane:     "dgxbridge",
				FromRung: maturityscore.RungProposed,
				Gap:      maturityscore.RungPrototyped,
				Title:    "prototype dgxbridge: land a v1 in internal/dgxbridge",
				Witness:  "a non-test .go file exists under internal/dgxbridge",
			},
			{
				Lane:     "alpha",
				FromRung: maturityscore.RungPrototyped,
				Gap:      maturityscore.RungTested,
				Title:    "test alpha: add unit tests covering internal/alpha",
				Witness:  "a *_test.go in internal/alpha",
			},
		},
	})
	if got.Score != 81 || got.Grade != "B" || got.Backlog != 3 || got.NextLane != "dgxbridge" ||
		got.Distribution["dogfooded"] != 7 || !got.OK {
		t.Fatalf("maturity projection = %+v", got)
	}
	if got.NextLane != "dgxbridge" || got.RouteLane != "alpha" ||
		got.RouteKey != "maturity/alpha/tested" || got.RouteSkippedPrivate != 1 {
		t.Fatalf("maturity route preview = %+v", got)
	}
}

func okScores() Scores {
	return Scores{Debt: 40, Measured: 13, TrendDirection: "improved", TrendSummary: "improved -4", OK: true}
}
func okMaturity() Maturity {
	return Maturity{
		Debt:         0,
		Score:        78,
		Grade:        "C",
		Capabilities: 111,
		Backlog:      88,
		Distribution: map[string]int{
			"proposed":   1,
			"prototyped": 0,
			"tested":     18,
			"dogfooded":  56,
			"default":    36,
		},
		NextLane:            "dgxbridge",
		NextItem:            "prototype dgxbridge: land a v1 in internal/dgxbridge",
		RouteKey:            "maturity/advmodel/dogfooded",
		RouteLane:           "advmodel",
		RouteItem:           "maturity(advmodel): dogfood the capability in fak",
		RouteSkippedPrivate: 1,
		OK:                  true,
	}
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

func TestFoldMaturityDebtIsAdvisoryNotGate(t *testing.T) {
	m := okMaturity()
	m.Debt = 2
	m.LadderSkips = 2
	m.OK = false
	r := FoldWithMaturity(okScores(), m, okWork(), okReleases(), foldOpts())
	if !r.OK || r.Finding != "cadence_advisory" {
		t.Fatalf("maturity debt should be advisory-OK, got ok=%v finding=%q", r.OK, r.Finding)
	}
	if !strings.Contains(r.Reason, "maturity ladder-skip debt") {
		t.Fatalf("reason should surface maturity advisory, got %q", r.Reason)
	}
	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("maturity advisory must not fail --check, got exit %d", code)
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

func TestFoldUnmeasuredMaturityGates(t *testing.T) {
	m := okMaturity()
	m.Err = "maturity read failed"
	r := FoldWithMaturity(okScores(), m, okWork(), okReleases(), foldOpts())
	if r.OK || r.Verdict != "ACTION" || r.Finding != "cadence_unmeasured" {
		t.Fatalf("unmeasured maturity must gate, got ok=%v verdict=%q finding=%q", r.OK, r.Verdict, r.Finding)
	}
	if !strings.Contains(r.Reason, "maturity") {
		t.Fatalf("reason should name maturity, got %q", r.Reason)
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

func TestStandingNormalizesDifficultyChange(t *testing.T) {
	prior := ProjectStanding(LedgerRow{
		Date:                 "2026-06-20",
		Commit:               "a",
		GeneratedAt:          "2026-06-20T00:00:00Z",
		ScoresDebt:           10,
		ScoresGradeDebt:      10,
		ScoresMeasured:       10,
		MaturityScore:        80,
		MaturityCapabilities: 10,
		WorkCommits:          20,
		WorkShips:            15,
	}, nil)
	row := ProjectStanding(LedgerRow{
		Date:                 "2026-06-26",
		Commit:               "b",
		GeneratedAt:          "2026-06-26T00:00:00Z",
		ScoresDebt:           20,
		ScoresGradeDebt:      20,
		ScoresMeasured:       20,
		MaturityScore:        80,
		MaturityCapabilities: 10,
		WorkCommits:          25,
		WorkShips:            18,
	}, []LedgerRow{prior})

	tr := TrendVsLast(row, []LedgerRow{prior})
	if tr.DebtDelta != 10 {
		t.Fatalf("raw debt delta = %d, want +10 to prove difficulty changed", tr.DebtDelta)
	}
	if tr.Direction != "flat" || tr.StandingDelta != 0 || row.StandingScore != prior.StandingScore {
		t.Fatalf("standing should stay flat under equal normalized health: row=%+v trend=%+v prior=%+v", row, tr, prior)
	}
	if tr.StandingDifficultyDelta <= 0 {
		t.Fatalf("difficulty should record the harder tick, got %+d", tr.StandingDifficultyDelta)
	}
}

func TestStandingCanClimbAndFall(t *testing.T) {
	base := ProjectStanding(LedgerRow{
		Date:                 "2026-06-20",
		GeneratedAt:          "2026-06-20T00:00:00Z",
		ScoresDebt:           20,
		ScoresGradeDebt:      20,
		ScoresMeasured:       10,
		MaturityScore:        75,
		MaturityCapabilities: 10,
	}, nil)
	better := ProjectStanding(LedgerRow{
		Date:                 "2026-06-21",
		GeneratedAt:          "2026-06-21T00:00:00Z",
		ScoresDebt:           12,
		ScoresGradeDebt:      12,
		ScoresMeasured:       10,
		MaturityScore:        85,
		MaturityCapabilities: 10,
	}, []LedgerRow{base})
	if better.StandingScore <= base.StandingScore || better.StandingDelta <= 0 {
		t.Fatalf("standing did not climb: base=%+v better=%+v", base, better)
	}
	up := TrendVsLast(better, []LedgerRow{base})
	if up.Direction != "improved" {
		t.Fatalf("up trend = %+v, want improved", up)
	}

	worse := ProjectStanding(LedgerRow{
		Date:                 "2026-06-22",
		GeneratedAt:          "2026-06-22T00:00:00Z",
		ScoresDebt:           28,
		ScoresGradeDebt:      28,
		ScoresMeasured:       10,
		MaturityScore:        65,
		MaturityCapabilities: 10,
	}, []LedgerRow{better})
	if worse.StandingScore >= better.StandingScore || worse.StandingDelta >= 0 {
		t.Fatalf("standing did not fall: better=%+v worse=%+v", better, worse)
	}
	down := TrendVsLast(worse, []LedgerRow{better})
	if down.Direction != "regressed" {
		t.Fatalf("down trend = %+v, want regressed", down)
	}
}

func TestTrendCarriesMaturityDeltas(t *testing.T) {
	prior := []LedgerRow{{Date: "2026-06-25", ScoresDebt: 40, MaturityScore: 70, MaturityDebt: 1, MaturityBacklog: 10, GeneratedAt: "2026-06-25T00:00:00Z"}}
	row := LedgerRow{Date: "2026-06-26", ScoresDebt: 40, MaturityScore: 78, MaturityDebt: 0, MaturityBacklog: 8, GeneratedAt: "2026-06-26T00:00:00Z"}
	tr := TrendVsLast(row, prior)
	if tr.MaturityScoreDelta != 8 || tr.MaturityDebtDelta != -1 || tr.MaturityBacklogDelta != -2 {
		t.Fatalf("maturity deltas wrong: %+v", tr)
	}
	if !strings.Contains(tr.Summary, "maturity score +8") {
		t.Fatalf("trend summary should surface maturity delta, got %q", tr.Summary)
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
	r := FoldWithMaturity(okScores(), okMaturity(), okWork(), okReleases(), foldOpts())
	row := RowFromReport(r)
	if row.Schema != LedgerSchema || row.Date != "2026-06-26" || row.ScoresDebt != 40 ||
		row.MaturityScore != 78 || row.MaturityBacklog != 88 || row.MaturityTested != 18 ||
		row.MaturityRouteKey != "maturity/advmodel/dogfooded" || row.MaturityRouteLane != "advmodel" ||
		row.MaturityRouteSkipped != 1 ||
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
	if back.ScoresDebt != 40 || back.ScoresTrend != "improved" || back.MaturityDefault != 36 ||
		back.MaturityRouteLane != "advmodel" {
		t.Fatalf("round-trip lost fields: %+v", back)
	}
}

func TestRenderSurfacesMaturityRoutePreview(t *testing.T) {
	r := FoldWithMaturity(okScores(), okMaturity(), okWork(), okReleases(), foldOpts())
	out := Render(r)
	for _, want := range []string{
		"next dgxbridge",
		"route advmodel",
		"1 private skipped",
		"fak maturity route --fetch-existing --limit 3",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
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

func TestShipsBySubjects(t *testing.T) {
	// Pin the ship-stamp grammar boundary the WORK-DONE count must honor. Each
	// subject is decided through hooks.StampOf, the same grammar the pre-commit
	// lint binds to, so a body-only / merge / release subject is NOT a ship.
	subjects := []string{
		"Merge branch 'feat' into main",                                 // merge -> none, not a ship
		"chore: see (fak gateway) for context then do other work",       // mid-subject mention, not a trailing stamp -> none
		"fix(gateway): treat same-tick ready as positive (fak gateway)", // trailer -> ship, gateway
		"fak/blob: add the spool reader",                                // direct -> ship, blob
		"v1.2.3: release the cut",                                       // release bundle -> not a per-leaf ship
		"wip: scratch nothing landed yet",                               // bare WIP -> none, not a ship
		"docs(typo): tidy a heading (fak gatway)",                       // off-lane typo -> still a ship (grammar, not taxonomy)
	}
	ships, byLane := shipsBySubjects(subjects)
	// gateway trailer + blob direct + the gatway typo = 3 grammar-valid ships.
	if ships != 3 {
		t.Fatalf("ships = %d, want 3", ships)
	}
	want := map[string]int{"gateway": 1, "blob": 1, "gatway": 1}
	if len(byLane) != len(want) {
		t.Fatalf("byLane = %v, want %v", byLane, want)
	}
	for leaf, n := range want {
		if byLane[leaf] != n {
			t.Fatalf("byLane[%q] = %d, want %d (full: %v)", leaf, byLane[leaf], n, byLane)
		}
	}

	// No ships -> a nil ByLane map (so the JSON envelope omits by_lane and Render
	// skips the breakdown line), never an empty non-nil map.
	none, nilLane := shipsBySubjects([]string{"Merge x", "wip: y", "v1.0.0: z"})
	if none != 0 || nilLane != nil {
		t.Fatalf("no-ship case = (%d, %v), want (0, nil)", none, nilLane)
	}
}

func TestRenderSmoke(t *testing.T) {
	r := FoldWithMaturity(okScores(), okMaturity(), okWork(), okReleases(), foldOpts())
	tr := TrendVsLast(RowFromReport(r), []LedgerRow{{Date: "2026-06-20", ScoresDebt: 44, GeneratedAt: "2026-06-20T00:00:00Z"}})
	r.Trend = &tr
	out := Render(r)
	for _, want := range []string{"cadence report", "scores", "maturity", "work", "releases", "trend:", "->"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}
