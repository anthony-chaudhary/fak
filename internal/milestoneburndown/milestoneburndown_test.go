package milestoneburndown

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

// fixedNow is the injected clock every pure test folds against, so verdicts are
// reproducible without a wall-clock read.
var fixedNow = time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC)

func TestInterpretClassifiesEveryStatus(t *testing.T) {
	ms := []Milestone{
		{Number: 1, Title: "Overdue one", DueOn: "2026-07-01T00:00:00Z", Open: 5, Closed: 5, ClosedInWindow: 3, WindowDays: 28},
		{Number: 2, Title: "At risk pace", DueOn: "2026-07-20T00:00:00Z", Open: 20, Closed: 0, ClosedInWindow: 1, WindowDays: 28},
		{Number: 3, Title: "On track", DueOn: "2026-08-30T00:00:00Z", Open: 2, Closed: 8, ClosedInWindow: 14, WindowDays: 28},
		{Number: 4, Title: "No due", DueOn: "", Open: 3, Closed: 1, ClosedInWindow: 0, WindowDays: 28},
		{Number: 5, Title: "Done", DueOn: "2026-07-05T00:00:00Z", Open: 0, Closed: 9, ClosedInWindow: 2, WindowDays: 28},
		{Number: 6, Title: "At risk no closures", DueOn: "2026-07-15T00:00:00Z", Open: 4, Closed: 0, ClosedInWindow: 0, WindowDays: 28},
	}
	p := Interpret(ms, 28, fixedNow, "")

	if !p.OK {
		t.Fatalf("expected measured portfolio, got Err=%q", p.Err)
	}
	want := map[int]string{
		1: StatusOverdue,
		2: StatusAtRisk,
		3: StatusOnTrack,
		4: StatusNoDueDate,
		5: StatusDone,
		6: StatusAtRisk,
	}
	byNum := map[int]Row{}
	for _, r := range p.Rows {
		byNum[r.Number] = r
	}
	for num, status := range want {
		if got := byNum[num].Status; got != status {
			t.Errorf("milestone #%d: status = %q, want %q (note: %s)", num, got, status, byNum[num].Note)
		}
	}

	if p.Total != 6 || p.Overdue != 1 || p.AtRisk != 2 || p.NoDueDate != 1 || p.OnTrack != 1 || p.Done != 1 {
		t.Errorf("counts wrong: total=%d overdue=%d atrisk=%d nodue=%d ontrack=%d done=%d",
			p.Total, p.Overdue, p.AtRisk, p.NoDueDate, p.OnTrack, p.Done)
	}
	// 3*overdue + 2*at-risk + 1*no-due-date = 3 + 4 + 1 = 8.
	if p.AtRiskDebt != 8 {
		t.Errorf("at-risk debt = %d, want 8", p.AtRiskDebt)
	}
	if p.OpenTotal != 34 {
		t.Errorf("open total = %d, want 34", p.OpenTotal)
	}
}

func TestInterpretSeverityOrderingAndProjection(t *testing.T) {
	ms := []Milestone{
		{Number: 3, Title: "On track", DueOn: "2026-08-30T00:00:00Z", Open: 2, Closed: 8, ClosedInWindow: 14, WindowDays: 28},
		{Number: 2, Title: "At risk later due", DueOn: "2026-07-20T00:00:00Z", Open: 20, Closed: 0, ClosedInWindow: 1, WindowDays: 28},
		{Number: 6, Title: "At risk sooner due", DueOn: "2026-07-15T00:00:00Z", Open: 4, Closed: 0, ClosedInWindow: 0, WindowDays: 28},
		{Number: 1, Title: "Overdue", DueOn: "2026-07-01T00:00:00Z", Open: 5, Closed: 5, ClosedInWindow: 3, WindowDays: 28},
	}
	p := Interpret(ms, 28, fixedNow, "")
	gotOrder := make([]int, 0, len(p.Rows))
	for _, r := range p.Rows {
		gotOrder = append(gotOrder, r.Number)
	}
	// overdue(1) first, then at-risk soonest-due first (6 due 07-15 before 2 due
	// 07-20), then on-track(3).
	wantOrder := []int{1, 6, 2, 3}
	if fmt.Sprint(gotOrder) != fmt.Sprint(wantOrder) {
		t.Fatalf("severity order = %v, want %v", gotOrder, wantOrder)
	}

	// On-track row projects a drain date before its due date.
	var onTrack Row
	for _, r := range p.Rows {
		if r.Number == 3 {
			onTrack = r
		}
	}
	if onTrack.ProjDate == "" || onTrack.ProjDays < 0 {
		t.Errorf("on-track row should project a drain date, got projDays=%d projDate=%q", onTrack.ProjDays, onTrack.ProjDate)
	}
	if onTrack.RequiredRate <= 0 {
		t.Errorf("on-track row should carry a required rate, got %v", onTrack.RequiredRate)
	}
}

func TestInterpretDegradedNoVelocity(t *testing.T) {
	// ClosedInWindow = -1 means velocity is unmeasured; the fold falls back to
	// urgency only: due within soonDays with open work => AT_RISK, else ON_TRACK.
	ms := []Milestone{
		{Number: 10, Title: "soon no velocity", DueOn: "2026-07-13T00:00:00Z", Open: 3, Closed: 1, ClosedInWindow: -1, WindowDays: 28},
		{Number: 11, Title: "far no velocity", DueOn: "2026-09-01T00:00:00Z", Open: 3, Closed: 1, ClosedInWindow: -1, WindowDays: 28},
	}
	p := Interpret(ms, 28, fixedNow, "")
	byNum := map[int]Row{}
	for _, r := range p.Rows {
		byNum[r.Number] = r
	}
	if byNum[10].Status != StatusAtRisk || byNum[10].HasVelocity {
		t.Errorf("#10 = %q hasVel=%v, want AT_RISK w/o velocity", byNum[10].Status, byNum[10].HasVelocity)
	}
	if byNum[11].Status != StatusOnTrack || byNum[11].HasVelocity {
		t.Errorf("#11 = %q hasVel=%v, want ON_TRACK w/o velocity", byNum[11].Status, byNum[11].HasVelocity)
	}
	if !strings.Contains(byNum[10].Note, "velocity unmeasured") {
		t.Errorf("#10 note should flag unmeasured velocity, got %q", byNum[10].Note)
	}
}

func TestFoldVerdictAndNextAction(t *testing.T) {
	ms := []Milestone{
		{Number: 1, Title: "Overdue one", DueOn: "2026-07-01T00:00:00Z", Open: 5, Closed: 5, ClosedInWindow: 3, WindowDays: 28},
		{Number: 3, Title: "On track", DueOn: "2026-08-30T00:00:00Z", Open: 2, Closed: 8, ClosedInWindow: 14, WindowDays: 28},
	}
	rep := Fold(Interpret(ms, 28, fixedNow, ""), FoldOpts{Date: "2026-07-10", Commit: "abc1234"})
	if rep.Verdict != "OK" || rep.Finding != findingRecorded {
		t.Fatalf("verdict/finding = %q/%q, want OK/%s", rep.Verdict, rep.Finding, findingRecorded)
	}
	if !strings.Contains(rep.NextAction, "OVERDUE") || !strings.Contains(rep.NextAction, "#1") {
		t.Errorf("next action should name the overdue milestone #1, got %q", rep.NextAction)
	}
	// Advisory gate passes on a measured report (OVERDUE is a fact, not incompleteness).
	if g := CheckGate(rep); g.Exit != 0 {
		t.Errorf("gate should pass on measured report, got exit %d (%s)", g.Exit, g.Message)
	}
}

func TestFoldUnmeasuredRedsGate(t *testing.T) {
	rep := Fold(Interpret(nil, 28, fixedNow, "list milestones: gh: not authenticated"), FoldOpts{})
	if rep.OK || rep.Verdict != "ACTION" || rep.Finding != findingUnmeasured {
		t.Fatalf("unmeasured report should be ACTION/%s, got ok=%v verdict=%q finding=%q",
			findingUnmeasured, rep.OK, rep.Verdict, rep.Finding)
	}
	g := CheckGate(rep)
	if g.Exit != 1 || !strings.Contains(g.Message, "INCOMPLETE") {
		t.Errorf("gate should red on unmeasured, got exit %d msg %q", g.Exit, g.Message)
	}
	withGate := rep.WithGate(g)
	if withGate.GateExit == nil || *withGate.GateExit != 1 {
		t.Errorf("WithGate should stamp gate_exit=1, got %v", withGate.GateExit)
	}
}

func TestLedgerRowAndTrend(t *testing.T) {
	cur := LedgerRow{AtRiskDebt: 8, Overdue: 1, AtRisk: 2, Date: "2026-07-10"}
	prior := LedgerRow{AtRiskDebt: 5, Overdue: 0, AtRisk: 1, Date: "2026-07-03"}

	// Falling at-risk debt vs the latest prior row is an improvement.
	improved := TrendVsLast(LedgerRow{AtRiskDebt: 3, Date: "2026-07-10"}, []LedgerRow{prior})
	if improved.Direction != "improved" || improved.DebtDelta != -2 {
		t.Errorf("falling debt should be improved, got %q delta=%d", improved.Direction, improved.DebtDelta)
	}
	// Rising debt is a regression; the overdue delta is carried onto the trend.
	regressed := TrendVsLast(cur, []LedgerRow{prior})
	if regressed.Direction != "regressed" || regressed.DebtDelta != 3 || regressed.OverdueDelta != 1 {
		t.Errorf("rising debt should be regressed, got %q delta=%d overdueDelta=%d",
			regressed.Direction, regressed.DebtDelta, regressed.OverdueDelta)
	}
	// Equal debt is flat.
	flat := TrendVsLast(LedgerRow{AtRiskDebt: 5, Date: "2026-07-10"}, []LedgerRow{prior})
	if flat.Direction != "flat" {
		t.Errorf("equal debt should be flat, got %q", flat.Direction)
	}
	// No prior row is a fresh series.
	fresh := TrendVsLast(cur, nil)
	if fresh.Direction != "new" {
		t.Errorf("no prior should be new, got %q", fresh.Direction)
	}
	// With several priors, the trend is measured against the most recent one.
	multi := TrendVsLast(cur, []LedgerRow{
		{AtRiskDebt: 99, Date: "2026-06-01"},
		prior, // 2026-07-03 is the latest before cur
	})
	if multi.PrevDate != "2026-07-03" || multi.DebtDelta != 3 {
		t.Errorf("trend should use the latest prior, got prev=%q delta=%d", multi.PrevDate, multi.DebtDelta)
	}
}

// TestLedgerRoundTrip proves a rendered ledger line parses back to the same row,
// and that ParseLedger tolerates blank and malformed lines.
func TestLedgerRoundTrip(t *testing.T) {
	row := RowFromReport(Fold(Interpret(
		[]Milestone{{Number: 1, Title: "x", DueOn: "2026-07-01T00:00:00Z", Open: 2, Closed: 1, ClosedInWindow: 0, WindowDays: 28}},
		28, fixedNow, ""), FoldOpts{Commit: "abc123", GeneratedAt: "2026-07-10T00:00:00Z", Date: "2026-07-10"}))
	line, err := trendreport.AppendLedgerLine(row)
	if err != nil {
		t.Fatalf("AppendLedgerLine: %v", err)
	}
	content := "\n" + line + "\n{ not json }\n"
	got := ParseLedger(content)
	if len(got) != 1 {
		t.Fatalf("ParseLedger should keep exactly the one valid row, got %d", len(got))
	}
	if got[0].Date != row.Date || got[0].AtRiskDebt != row.AtRiskDebt || got[0].Schema != LedgerSchema {
		t.Errorf("round-trip mismatch: got %+v want %+v", got[0], row)
	}
}

func TestWithTrendRegressionDowngradesFinding(t *testing.T) {
	ms := []Milestone{{Number: 1, Title: "x", DueOn: "2026-07-01T00:00:00Z", Open: 1, Closed: 0, ClosedInWindow: 0, WindowDays: 28}}
	rep := Fold(Interpret(ms, 28, fixedNow, ""), FoldOpts{})
	if rep.Finding != findingRecorded {
		t.Fatalf("precondition: finding=%q", rep.Finding)
	}
	rep = rep.WithTrend(Trend{Direction: "regressed", DebtDelta: 2, PrevDate: "2026-07-03"})
	if rep.Finding != findingAdvisory {
		t.Errorf("regression should downgrade to advisory, got %q", rep.Finding)
	}
	// Advisory is still NOT a gate red — only unmeasured reds.
	if g := CheckGate(rep); g.Exit != 0 {
		t.Errorf("advisory regression must not red the gate, got exit %d", g.Exit)
	}
}

// ---- collector (injected runner, no gh / no network) ----------------------

func TestDecodeMilestones(t *testing.T) {
	js := `[
      {"number":2,"title":"At risk","html_url":"u2","due_on":"2026-07-20T08:00:00Z","state":"open","open_issues":20,"closed_issues":0},
      {"number":3,"title":"On track","html_url":"u3","due_on":null,"state":"open","open_issues":2,"closed_issues":8}
    ]`
	got, err := decodeMilestones([]byte(js))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 2 || got[0].Number != 2 || got[0].Open != 20 || got[1].DueOn != "" {
		t.Fatalf("unexpected decode: %+v", got)
	}
	// Empty / whitespace payload decodes to nothing, not an error.
	if r, err := decodeMilestones([]byte("  \n ")); err != nil || len(r) != 0 {
		t.Errorf("empty payload: got %v err %v", r, err)
	}
}

func TestCountClosedSince(t *testing.T) {
	since := time.Date(2026, 6, 12, 0, 0, 0, 0, time.UTC)
	js := `[
      {"closed_at":"2026-07-01T00:00:00Z"},
      {"closed_at":"2026-06-20T00:00:00Z"},
      {"closed_at":"2026-05-01T00:00:00Z"},
      {"closed_at":null},
      {"closed_at":""}
    ]`
	n, err := countClosedSince([]byte(js), since)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 2 {
		t.Fatalf("closed-in-window = %d, want 2 (two after %s)", n, since.Format("2006-01-02"))
	}
}

func TestCollectWithInjectedRunner(t *testing.T) {
	milestones := `[
      {"number":2,"title":"At risk","html_url":"u2","due_on":"2026-07-20T08:00:00Z","state":"open","open_issues":20,"closed_issues":0},
      {"number":3,"title":"On track","html_url":"u3","due_on":"2026-08-30T08:00:00Z","state":"open","open_issues":2,"closed_issues":8}
    ]`
	runner := func(args []string) (string, string, bool) {
		if len(args) < 2 || args[0] != "api" {
			return "", "unexpected args", false
		}
		path := args[1]
		switch {
		case strings.Contains(path, "/milestones?"):
			return milestones, "", true
		case strings.Contains(path, "milestone=2"):
			// one issue closed inside the window
			return `[{"closed_at":"2026-07-01T00:00:00Z"}]`, "", true
		case strings.Contains(path, "milestone=3"):
			// fourteen closed inside the window -> healthy pace
			var b strings.Builder
			b.WriteString("[")
			for i := 0; i < 14; i++ {
				if i > 0 {
					b.WriteString(",")
				}
				b.WriteString(`{"closed_at":"2026-07-01T00:00:00Z"}`)
			}
			b.WriteString("]")
			return b.String(), "", true
		default:
			return "[]", "", true
		}
	}

	p := Collect("owner/name", runner, 28, fixedNow)
	if !p.OK || p.Total != 2 {
		t.Fatalf("expected 2 measured milestones, got ok=%v total=%d err=%q", p.OK, p.Total, p.Err)
	}
	byNum := map[int]Row{}
	for _, r := range p.Rows {
		byNum[r.Number] = r
	}
	if byNum[2].Status != StatusAtRisk {
		t.Errorf("#2 with 1 closure over 28d and 20 open due in 10d should be AT_RISK, got %q (%s)", byNum[2].Status, byNum[2].Note)
	}
	if !byNum[2].HasVelocity {
		t.Errorf("#2 should carry a measured velocity")
	}
	if byNum[3].Status != StatusOnTrack {
		t.Errorf("#3 with 14 closures over 28d and 2 open should be ON_TRACK, got %q (%s)", byNum[3].Status, byNum[3].Note)
	}
	// At-risk sorts before on-track.
	if p.Rows[0].Number != 2 {
		t.Errorf("at-risk milestone should sort first, got #%d", p.Rows[0].Number)
	}
}

func TestCollectListReadFailureIsUnmeasured(t *testing.T) {
	runner := func(args []string) (string, string, bool) {
		return "", "gh: could not resolve to a Repository", false
	}
	p := Collect("owner/name", runner, 28, fixedNow)
	if p.OK || p.Err == "" {
		t.Fatalf("list read failure should be unmeasured, got ok=%v err=%q", p.OK, p.Err)
	}
	if !strings.Contains(p.Err, "list milestones") {
		t.Errorf("err should name the failing step, got %q", p.Err)
	}
}

// Render should never panic and should surface the most-urgent action line.
func TestRenderSmoke(t *testing.T) {
	ms := []Milestone{
		{Number: 1, Title: "Overdue one that has a very long title indeed exceeding the column width", DueOn: "2026-07-01T00:00:00Z", Open: 5, Closed: 5, ClosedInWindow: 3, WindowDays: 28},
		{Number: 4, Title: "No due", DueOn: "", Open: 3, Closed: 1, ClosedInWindow: 0, WindowDays: 28},
	}
	out := Render(Fold(Interpret(ms, 28, fixedNow, ""), FoldOpts{Date: "2026-07-10"}))
	if !strings.Contains(out, "MILESTONE BURNDOWN") || !strings.Contains(out, "OVERDUE") {
		t.Errorf("render missing expected content:\n%s", out)
	}
}

func TestCohortTrackingDecouplesDiscoveryFromDrain(t *testing.T) {
	base := BurndownBaseline{
		Schema:      BaselineSchema,
		Date:        "2026-07-13",
		Commit:      "31c6538b",
		OpenTotal:   100,
		ClosedTotal: 50,
		Total:       150,
	}
	// 40 issues from the baseline cohort were closed (closed 50 -> 90),
	// but 40 new issues were filed (open 100 - 40 + 40 = 100).
	// Gross open stays 100 (net change 0), but 40% of original cohort actually drained.
	cur := Portfolio{
		OpenTotal:   100,
		ClosedTotal: 90,
	}
	cohort := ComputeCohortProgress(base, cur)

	if cohort.BaselineOpen != 100 {
		t.Errorf("baseline open = %d, want 100", cohort.BaselineOpen)
	}
	if cohort.CurrentOpen != 100 {
		t.Errorf("current open = %d, want 100", cohort.CurrentOpen)
	}
	if cohort.CohortDrained != 40 {
		t.Errorf("cohort drained = %d, want 40", cohort.CohortDrained)
	}
	if cohort.CohortRemaining != 60 {
		t.Errorf("cohort remaining = %d, want 60", cohort.CohortRemaining)
	}
	if cohort.DrainPct != 40.0 {
		t.Errorf("drain pct = %v, want 40.0", cohort.DrainPct)
	}
	if cohort.SurvivalPct != 60.0 {
		t.Errorf("survival pct = %v, want 60.0", cohort.SurvivalPct)
	}
	if cohort.Discovered != 40 {
		t.Errorf("discovered = %d, want 40", cohort.Discovered)
	}
	if cohort.ExpansionRatio != 1.40 {
		t.Errorf("expansion ratio = %v, want 1.40", cohort.ExpansionRatio)
	}
	if cohort.DiscoveryMult != 1.00 {
		t.Errorf("discovery multiplier = %v, want 1.00", cohort.DiscoveryMult)
	}

	p := cur.WithCohort(cohort)
	if p.Cohort == nil || p.Cohort.CohortDrained != 40 {
		t.Fatalf("WithCohort failed to attach cohort to Portfolio")
	}
}

func TestLedgerCohortRoundTripAndLegacyCompatibility(t *testing.T) {
	// 1. Round-trip row with full cohort fields
	row := LedgerRow{
		Schema:         LedgerSchema,
		Date:           "2026-09-07",
		Commit:         "ce4a342d9",
		GeneratedAt:    "2026-09-07T14:20:12Z",
		Verdict:        "OK",
		Total:          20,
		OpenTotal:      580,
		ClosedTotal:    120,
		AtRiskDebt:     38,
		BaselineOpen:   386,
		CohortDrained:  120,
		CohortSurvival: 68.9,
		CohortDrain:    31.1,
		Discovered:     314,
		ExpansionRatio: 1.81,
	}
	line, err := trendreport.AppendLedgerLine(row)
	if err != nil {
		t.Fatalf("AppendLedgerLine: %v", err)
	}
	rows := ParseLedger(line)
	if len(rows) != 1 {
		t.Fatalf("expected 1 parsed row, got %d", len(rows))
	}
	got := rows[0]
	if got.BaselineOpen != 386 || got.CohortDrained != 120 || got.CohortDrain != 31.1 ||
		got.CohortSurvival != 68.9 || got.Discovered != 314 || got.ExpansionRatio != 1.81 ||
		got.ClosedTotal != 120 {
		t.Errorf("round-trip mismatch: got %+v want %+v", got, row)
	}

	// 2. Legacy compatibility: line without closed_total or cohort fields
	legacyLine := `{"schema":"fak-milestone-burndown-ledger/1","date":"2026-07-13","commit":"31c6538b","generated_at":"2026-07-13T11:32:31Z","verdict":"OK","total":17,"overdue":1,"at_risk":2,"no_due_date":6,"on_track":8,"done":0,"open_total":386,"at_risk_debt":13}`
	parsed := ParseLedger(legacyLine)
	if len(parsed) != 1 {
		t.Fatalf("expected 1 legacy row, got %d", len(parsed))
	}
	leg := parsed[0]
	if leg.OpenTotal != 386 || leg.Date != "2026-07-13" || leg.BaselineOpen != 0 || leg.CohortDrained != 0 {
		t.Errorf("legacy parse mismatch: %+v", leg)
	}
}

func TestTrendVsLastCohortProgression(t *testing.T) {
	prior := LedgerRow{
		Date:           "2026-08-24",
		Commit:         "307ae3213",
		AtRiskDebt:     37,
		BaselineOpen:   386,
		CohortDrain:    25.0,
		CohortSurvival: 75.0,
		Discovered:     150,
		ExpansionRatio: 1.39,
	}
	cur := LedgerRow{
		Date:           "2026-08-31",
		Commit:         "ecdae2ce0",
		AtRiskDebt:     39,
		BaselineOpen:   386,
		CohortDrain:    31.5,
		CohortSurvival: 68.5,
		Discovered:     190,
		ExpansionRatio: 1.49,
	}

	tr := TrendVsLast(cur, []LedgerRow{prior})
	if tr.CohortDrainFrom != 25.0 || tr.CohortDrainTo != 31.5 {
		t.Errorf("cohort drain from/to = %v/%v, want 25.0/31.5", tr.CohortDrainFrom, tr.CohortDrainTo)
	}
	if tr.CohortDrainDelta != 6.5 {
		t.Errorf("cohort drain delta = %v, want 6.5", tr.CohortDrainDelta)
	}
	if tr.CohortSurvivalDelta != -6.5 {
		t.Errorf("cohort survival delta = %v, want -6.5", tr.CohortSurvivalDelta)
	}
	if tr.DiscoveredDelta != 40 {
		t.Errorf("discovered delta = %d, want 40", tr.DiscoveredDelta)
	}
	if tr.ExpansionDelta != 0.10 {
		t.Errorf("expansion delta = %v, want 0.10", tr.ExpansionDelta)
	}
	if !strings.Contains(tr.Summary, "cohort drained +6.5% vs 2026-08-24") {
		t.Errorf("summary missing cohort drained delta clause: %q", tr.Summary)
	}

	// Flat cohort drain delta does not append cohort clause
	flatCur := cur
	flatCur.CohortDrain = 25.0
	flatTr := TrendVsLast(flatCur, []LedgerRow{prior})
	if strings.Contains(flatTr.Summary, "cohort drained") {
		t.Errorf("flat cohort drain should not append cohort drained clause, got %q", flatTr.Summary)
	}
}

func TestRenderCohortProgress(t *testing.T) {
	ms := []Milestone{
		{Number: 1, Title: "Milestone One", DueOn: "2026-07-20T00:00:00Z", Open: 2, Closed: 8, ClosedInWindow: 5, WindowDays: 28},
	}
	p := Interpret(ms, 28, fixedNow, "")
	cohort := CohortProgress{
		BaselineDate:    "2026-07-13",
		BaselineOpen:    100,
		CurrentOpen:     90,
		CohortDrained:   30,
		CohortRemaining: 70,
		DrainPct:        30.0,
		SurvivalPct:     70.0,
		Discovered:      20,
		ExpansionRatio:  1.20,
		DiscoveryMult:   0.67,
	}
	p = p.WithCohort(cohort)
	rep := Fold(p, FoldOpts{Date: "2026-07-20", Commit: "1234abc"})
	out := Render(rep)

	wantSubstrings := []string{
		"cohort progress (2026-07-13, 100 baseline open):",
		"original cohort: 30.0% drained (30/100 closed, 70 surviving | 70.0% survival)",
		"gross discovery: 20 new issues filed (expansion ratio 1.20x, discovery:drain 0.67:1)",
		"net-open change: -10 (gross open conflates 30 closures with 20 discoveries)",
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(out, want) {
			t.Errorf("Render missing %q in output:\n%s", want, out)
		}
	}
}

func TestBaselineFromLedger(t *testing.T) {
	rows := []LedgerRow{
		{Date: "2026-07-13", Commit: "31c6538b", OpenTotal: 386, ClosedTotal: 50, Total: 17},
		{Date: "2026-07-20", Commit: "abcdef12", OpenTotal: 350, ClosedTotal: 90, Total: 17},
	}

	// 1. Default (date == "") returns earliest row
	b1, ok := BaselineFromLedger(rows, "")
	if !ok || b1.Date != "2026-07-13" || b1.OpenTotal != 386 || b1.Schema != BaselineSchema {
		t.Errorf("BaselineFromLedger default: got %+v, ok=%v", b1, ok)
	}

	// 2. Exact date
	b2, ok := BaselineFromLedger(rows, "2026-07-20")
	if !ok || b2.Date != "2026-07-20" || b2.OpenTotal != 350 {
		t.Errorf("BaselineFromLedger exact date: got %+v, ok=%v", b2, ok)
	}

	// 3. Unknown date
	_, ok = BaselineFromLedger(rows, "2025-01-01")
	if ok {
		t.Errorf("BaselineFromLedger unknown date should return false")
	}

	// 4. Empty rows
	_, ok = BaselineFromLedger(nil, "")
	if ok {
		t.Errorf("BaselineFromLedger empty rows should return false")
	}

	// 5. ParseBaseline valid JSON
	raw := `{"schema":"fak-milestone-burndown-baseline/1","date":"2026-07-13","commit":"31c6538b","open_total":386,"closed_total":50,"total":17}`
	parsed, err := ParseBaseline([]byte(raw))
	if err != nil || parsed.OpenTotal != 386 || parsed.Date != "2026-07-13" {
		t.Fatalf("ParseBaseline failed: %v, parsed=%+v", err, parsed)
	}

	// 6. ParseBaseline schema mismatch
	badSchema := `{"schema":"bad-schema","date":"2026-07-13"}`
	if _, err := ParseBaseline([]byte(badSchema)); err == nil {
		t.Errorf("ParseBaseline should reject bad schema")
	}
}

func TestCohortBoundaryConditionsAndInvariants(t *testing.T) {
	// 1. BaselineOpen == 0
	baseZero := BurndownBaseline{OpenTotal: 0}
	cpZero := ComputeCohortProgress(baseZero, Portfolio{OpenTotal: 50, ClosedTotal: 20})
	if cpZero.BaselineOpen != 0 || cpZero.DrainPct != 0 || cpZero.SurvivalPct != 0 {
		t.Errorf("zero baseline should yield 0 pcts: %+v", cpZero)
	}

	// 2. CohortDrained == 0: DiscoveryMult must not divide by zero or yield NaN/Inf
	base := BurndownBaseline{OpenTotal: 100, ClosedTotal: 10, Total: 110}
	cpNoDrain := ComputeCohortProgress(base, Portfolio{OpenTotal: 120, ClosedTotal: 10})
	if cpNoDrain.CohortDrained != 0 {
		t.Errorf("drained = %d, want 0", cpNoDrain.CohortDrained)
	}
	if cpNoDrain.DiscoveryMult != 0.0 {
		t.Errorf("discovery multiplier should be 0.0 when 0 drained, got %v", cpNoDrain.DiscoveryMult)
	}
	if math.IsNaN(cpNoDrain.DiscoveryMult) || math.IsInf(cpNoDrain.DiscoveryMult, 0) {
		t.Errorf("discovery multiplier must be finite")
	}

	// 3. Closures exceed BaselineOpen: CohortDrained clamped to BaselineOpen, remaining 0
	cpOver := ComputeCohortProgress(base, Portfolio{OpenTotal: 20, ClosedTotal: 200})
	if cpOver.CohortDrained != 100 {
		t.Errorf("drained should clamp to baseline open 100, got %d", cpOver.CohortDrained)
	}
	if cpOver.CohortRemaining != 0 {
		t.Errorf("remaining should be 0, got %d", cpOver.CohortRemaining)
	}
	if cpOver.DrainPct != 100.0 || cpOver.SurvivalPct != 0.0 {
		t.Errorf("drain/survival pct = %v/%v, want 100.0/0.0", cpOver.DrainPct, cpOver.SurvivalPct)
	}

	// 4. Invariant: CohortDrained + CohortRemaining == BaselineOpen
	base100 := BurndownBaseline{OpenTotal: 100, ClosedTotal: 0, Total: 100}
	for closed := 0; closed <= 100; closed += 17 {
		cp := ComputeCohortProgress(base100, Portfolio{OpenTotal: 100 - closed, ClosedTotal: closed})
		if cp.CohortDrained+cp.CohortRemaining != 100 {
			t.Errorf("invariant broken for closed=%d: %d + %d != 100",
				closed, cp.CohortDrained, cp.CohortRemaining)
		}
		if math.Abs((cp.DrainPct+cp.SurvivalPct)-100.0) > 0.1 {
			t.Errorf("pct sum invariant broken for closed=%d: %v + %v != 100.0",
				closed, cp.DrainPct, cp.SurvivalPct)
		}
	}
}
