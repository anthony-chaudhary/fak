package milestonereport

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
)

// --- the CLIMB dimension ----------------------------------------------------

func TestInterpretMaturityLowersTheGrid(t *testing.T) {
	cells := []covmatrix.Cell{
		{Family: "a", Backend: "cpu", Support: covmatrix.Supported},       // M4
		{Family: "a", Backend: "metal", Support: covmatrix.ProofPathOnly}, // M3
		{Family: "b", Backend: "cpu", Support: covmatrix.Fenced},          // M1
		{Family: "b", Backend: "metal", Support: covmatrix.Undefined},     // M0
	}
	m := InterpretMaturity(cells)
	if m.Err != "" || !m.OK {
		t.Fatalf("a populated grid must measure cleanly, got err=%q ok=%v", m.Err, m.OK)
	}
	if m.Cells != 4 {
		t.Fatalf("cells = %d, want 4", m.Cells)
	}
	// Distribution seeds every rung; the four cells land at M0/M1/M3/M4.
	for _, r := range supportmaturity.Rungs {
		if _, ok := m.Dist[r.String()]; !ok {
			t.Fatalf("dist missing seeded rung %s", r)
		}
	}
	if m.Dist["M4"] != 1 || m.Dist["M3"] != 1 || m.Dist["M1"] != 1 || m.Dist["M0"] != 1 {
		t.Fatalf("dist = %v, want one each at M0/M1/M3/M4", m.Dist)
	}
	if got := m.Dist["M0"] + m.Dist["M1"] + m.Dist["M2"] + m.Dist["M3"] + m.Dist["M4"] + m.Dist["M5"] + m.Dist["M6"] + m.Dist["M7"]; got != m.Cells {
		t.Fatalf("dist sums to %d, want cells %d", got, m.Cells)
	}
	if m.Matured != 1 { // only the SUPPORTED (M4) cell is at/above the matured floor
		t.Fatalf("matured = %d, want 1 (only the M4 cell)", m.Matured)
	}
	if m.Highest != "M4" || m.HighestRank != int(supportmaturity.M4Correct) {
		t.Fatalf("highest = %q/%d, want M4", m.Highest, m.HighestRank)
	}
	if m.ProgressPct <= 0 || m.ProgressPct >= 100 {
		t.Fatalf("progress = %.1f, want strictly between 0 and 100", m.ProgressPct)
	}
	if len(m.Worst) == 0 || !strings.Contains(m.Worst[0], "M0") {
		t.Fatalf("worst should lead with the M0 cell, got %v", m.Worst)
	}
}

func TestInterpretMaturityProgressIsMonotone(t *testing.T) {
	allLow := []covmatrix.Cell{{Support: covmatrix.Undefined}, {Support: covmatrix.Undefined}}
	allHigh := []covmatrix.Cell{{Support: covmatrix.Supported}, {Support: covmatrix.Supported}}
	low := InterpretMaturity(allLow)
	high := InterpretMaturity(allHigh)
	if !(low.ProgressPct < high.ProgressPct) {
		t.Fatalf("an all-UNDEFINED grid (%.1f) must score below an all-SUPPORTED grid (%.1f)", low.ProgressPct, high.ProgressPct)
	}
	if low.ProgressPct != 0 {
		t.Fatalf("an all-M0 grid must have 0%% progress, got %.1f", low.ProgressPct)
	}
}

func TestInterpretMaturityEmptyGridErrors(t *testing.T) {
	m := InterpretMaturity(nil)
	if m.Err == "" || m.OK {
		t.Fatalf("an empty grid must error, got err=%q ok=%v", m.Err, m.OK)
	}
	// Even errored, the distribution is fully seeded (no nil-map panic downstream).
	if len(m.Dist) != len(supportmaturity.Rungs) {
		t.Fatalf("dist must be seeded even when errored, got %v", m.Dist)
	}
}

func TestInterpretMaturityOverLiveGrid(t *testing.T) {
	// The real grid must fold without error and stay internally consistent — this is
	// the witnessed path the report ships on.
	m := InterpretMaturity(covmatrix.Grid())
	if m.Err != "" || !m.OK || m.Cells == 0 {
		t.Fatalf("live grid must measure cleanly, got %+v", m)
	}
	sum := 0
	for _, n := range m.Dist {
		sum += n
	}
	if sum != m.Cells {
		t.Fatalf("live dist sums to %d, want cells %d", sum, m.Cells)
	}
}

// --- the ROADMAP dimension --------------------------------------------------

var sampleSpecs = []EpicSpec{
	{Number: 1, Title: "alpha"},
	{Number: 2, Title: "beta"},
}

func TestInterpretEpicsAllGood(t *testing.T) {
	counts := []EpicCounts{
		{Number: 1, Closed: 3, Total: 4, Source: "label"},
		{Number: 2, Closed: 1, Total: 2, Source: "checklist"},
	}
	e := InterpretEpics(sampleSpecs, counts, "")
	if e.Err != "" || !e.OK {
		t.Fatalf("all-good epics must measure cleanly, got err=%q ok=%v", e.Err, e.OK)
	}
	if e.Measured != 2 || e.Tracked != 2 {
		t.Fatalf("measured/tracked = %d/%d, want 2/2", e.Measured, e.Tracked)
	}
	if e.Closed != 4 || e.Total != 6 {
		t.Fatalf("closed/total = %d/%d, want 4/6", e.Closed, e.Total)
	}
	if e.OverallPct < 66 || e.OverallPct > 67 { // 4/6 = 66.7%
		t.Fatalf("overall pct = %.1f, want ~66.7", e.OverallPct)
	}
	if e.Rows[0].Source != "label" || e.Rows[1].Source != "checklist" {
		t.Fatalf("source provenance lost: %+v", e.Rows)
	}
}

func TestInterpretEpicsWholeCommandFailureGates(t *testing.T) {
	// A `gh` binary failure errors EVERY row and the dimension — the unmeasured gate.
	e := InterpretEpics(sampleSpecs, nil, "gh: command not found")
	if e.Err == "" || e.OK {
		t.Fatalf("a whole-command failure must error the dimension, got err=%q ok=%v", e.Err, e.OK)
	}
	if e.Measured != 0 {
		t.Fatalf("measured = %d, want 0", e.Measured)
	}
	for _, row := range e.Rows {
		if row.Err == "" {
			t.Fatalf("every row must carry the run error, got %+v", row)
		}
	}
}

func TestInterpretEpicsPartialFailureIsAdvisoryNotGated(t *testing.T) {
	// One epic reads, one fails to read → the dimension is MEASURED (no gate), with a
	// non-gating partial note; the failed row is excluded from the overall pct.
	counts := []EpicCounts{
		{Number: 1, Closed: 2, Total: 4, Source: "label"},
		{Number: 2, Err: "no child signal"},
	}
	e := InterpretEpics(sampleSpecs, counts, "")
	if e.Err != "" || !e.OK {
		t.Fatalf("a partial failure must NOT error the dimension, got err=%q ok=%v", e.Err, e.OK)
	}
	if e.PartialNote == "" {
		t.Fatalf("a partial failure must record a partial note")
	}
	if e.Measured != 1 || e.Closed != 2 || e.Total != 4 {
		t.Fatalf("only the readable epic counts toward the pct, got measured=%d closed=%d total=%d", e.Measured, e.Closed, e.Total)
	}
}

func TestInterpretEpicsNeverFabricatesZero(t *testing.T) {
	// An epic with no readable child signal must surface as an errored row, never a
	// fabricated 0/0 "0% done" — the load-bearing honesty seam.
	counts := []EpicCounts{
		{Number: 1, Err: "no child signal (no track label, no checklist)"},
		{Number: 2, Err: "no child signal (no track label, no checklist)"},
	}
	e := InterpretEpics(sampleSpecs, counts, "")
	if e.Measured != 0 || e.Err == "" {
		t.Fatalf("all-unreadable epics must error the dimension, not report 0%%, got %+v", e)
	}
	for _, row := range e.Rows {
		if row.Pct != 0 || row.Total != 0 {
			continue
		}
		if row.Err == "" {
			t.Fatalf("an unreadable epic must carry Err, not a silent 0/0: %+v", row)
		}
	}
}

// --- the work-class split ---------------------------------------------------

// TestInterpretEpicsSplitsProgramsFromDiscrete is the load-bearing test for this
// change: an ONGOING-program epic (#1010 kernel-opt) is measured and surfaced but
// its child tally is EXCLUDED from the roadmap completion %, while a DISCRETE epic
// (#1315 native harness) folds into the %. This is what stops a never-"done"
// optimization program from being read as a stalled deliverable.
func TestInterpretEpicsSplitsProgramsFromDiscrete(t *testing.T) {
	specs := []EpicSpec{
		{Number: 1010, Title: "GLM kernel"},     // kernel-optimization (ongoing)
		{Number: 1315, Title: "native harness"}, // discrete
		{Number: 1301, Title: "cache P&L"},      // cache-optimization (ongoing)
	}
	counts := []EpicCounts{
		{Number: 1010, Closed: 7, Total: 10, Source: "label"},
		{Number: 1315, Closed: 3, Total: 4, Source: "label"},
		{Number: 1301, Closed: 2, Total: 5, Source: "checklist"},
	}
	e := InterpretEpics(specs, counts, "")
	if e.Programs != 2 || e.Discrete != 1 {
		t.Fatalf("split = %d program / %d discrete, want 2/1", e.Programs, e.Discrete)
	}
	// OverallPct folds ONLY the discrete epic (3/4), NOT the two programs' children.
	if e.Closed != 3 || e.Total != 4 {
		t.Fatalf("roadmap completion must fold discrete epics only, got closed=%d total=%d (want 3/4)", e.Closed, e.Total)
	}
	if e.OverallPct < 74 || e.OverallPct > 76 { // 3/4 = 75%
		t.Fatalf("overall pct = %.1f, want ~75 (discrete only)", e.OverallPct)
	}
	// Every measured row still carries its closed/total (programs are surfaced, just
	// not folded into the %).
	byNum := map[int]EpicRow{}
	for _, r := range e.Rows {
		byNum[r.Number] = r
	}
	if r := byNum[1010]; !r.Ongoing() || r.Closed != 7 || r.Total != 10 {
		t.Fatalf("the kernel program row must be ongoing and carry its tally, got %+v", r)
	}
	if r := byNum[1315]; r.Ongoing() {
		t.Fatalf("#1315 must classify as discrete, got ongoing")
	}
}

// TestRenderSplitsSections proves the human render carries two labeled sections and
// that an ongoing program is rendered WITHOUT a misleading completion %.
func TestRenderSplitsSections(t *testing.T) {
	specs := []EpicSpec{
		{Number: 1010, Title: "GLM kernel", Generation: "next"},    // ongoing
		{Number: 1315, Title: "native harness", Generation: "now"}, // discrete
	}
	counts := []EpicCounts{
		{Number: 1010, Closed: 7, Total: 10, Source: "label"},
		{Number: 1315, Closed: 3, Total: 4, Source: "label"},
	}
	r := Fold(goodMaturity(), InterpretEpics(specs, counts, ""), FoldOpts{Date: "2026-06-29", Commit: "abc"})
	out := Render(r)
	for _, want := range []string{"discrete epics (-> done):", "ongoing programs", "kernel-optimization", "shipped", "in-flight"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
	// The kernel program must NOT be rendered as "70%" — an ongoing program has no %.
	if strings.Contains(out, "GLM kernel — 70%") || strings.Contains(out, "GLM kernel [kernel-optimization] — 70%") {
		t.Fatalf("an ongoing program must never render a completion %%\n%s", out)
	}
}

func TestInterpretEpicsSummarizesGenerationLanes(t *testing.T) {
	specs := []EpicSpec{
		{Number: 1315, Title: "native harness", Generation: "gen/now"}, // discrete
		{Number: 1010, Title: "GLM kernel", Generation: "next"},        // ongoing
		{Number: 7, Title: "future option", Generation: "future"},      // unreadable
	}
	counts := []EpicCounts{
		{Number: 1315, Closed: 2, Total: 4, Source: "label"},
		{Number: 1010, Closed: 7, Total: 10, Source: "label"},
	}
	e := InterpretEpics(specs, counts, "")
	byGen := map[string]GenerationRow{}
	for _, row := range e.Generations {
		byGen[row.Generation] = row
	}
	if row := byGen["now"]; row.Tracked != 1 || row.Discrete != 1 || row.Closed != 2 || row.Total != 4 || row.OverallPct != 50 {
		t.Fatalf("now generation row = %+v, want one 2/4 discrete epic", row)
	}
	if row := byGen["now"]; row.DebtScore != 2 || row.StaleIssues != 2 {
		t.Fatalf("now generation debt = %+v, want score 2 from two stale-risk issues", row)
	}
	if row := byGen["next"]; row.Tracked != 1 || row.Programs != 1 || row.Discrete != 0 || row.Total != 0 {
		t.Fatalf("next generation row = %+v, want one ongoing program with no completion pct", row)
	}
	if row := byGen["next"]; row.DebtScore != 2 || row.UnpromotedBets != 1 {
		t.Fatalf("next generation debt = %+v, want score 2 from one unpromoted bet", row)
	}
	if row := byGen["second-next"]; row.Tracked != 0 {
		t.Fatalf("second-next generation row = %+v, want seeded zero row", row)
	}
	if row := byGen["future"]; row.Tracked != 1 || row.Errored != 1 || row.Measured != 0 {
		t.Fatalf("future generation row = %+v, want one unreadable row", row)
	}
	if row := byGen["future"]; row.DebtScore != 3 || row.MissingWitnesses != 1 {
		t.Fatalf("future generation debt = %+v, want score 3 from one missing witness", row)
	}
}

func TestGenerationDebtFlagsUnclassifiedLabelShipMismatch(t *testing.T) {
	specs := []EpicSpec{{Number: 10, Title: "unclassified shipped row", Generation: ""}}
	counts := []EpicCounts{{Number: 10, Closed: 1, Total: 1, Source: "label"}}
	e := InterpretEpics(specs, counts, "")
	var row GenerationRow
	for _, r := range e.Generations {
		if r.Generation == "unclassified" {
			row = r
			break
		}
	}
	if row.Tracked != 1 || row.LabelShipMismatches != 1 || row.DebtScore != 2 {
		t.Fatalf("unclassified generation debt = %+v, want one label/ship mismatch with score 2", row)
	}
}

// --- the fold ---------------------------------------------------------------

func goodMaturity() Maturity {
	return InterpretMaturity([]covmatrix.Cell{{Family: "a", Backend: "cpu", Support: covmatrix.Supported}})
}

func goodEpics() Epics {
	return InterpretEpics([]EpicSpec{{Number: 1, Title: "x"}}, []EpicCounts{{Number: 1, Closed: 1, Total: 2, Source: "label"}}, "")
}

func TestFoldRecordedWhenBothMeasured(t *testing.T) {
	r := Fold(goodMaturity(), goodEpics(), FoldOpts{Date: "2026-06-29", Commit: "abc"})
	if !r.OK || r.Verdict != "OK" || r.Finding != "milestone_recorded" {
		t.Fatalf("both measured must record OK, got %+v", r)
	}
	if r.Schema != Schema {
		t.Fatalf("schema = %q, want %q", r.Schema, Schema)
	}
	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("a recorded report must gate 0, got %d", code)
	}
}

func TestFoldUnmeasuredRoadmapGates(t *testing.T) {
	// The roadmap dimension errored (gh down) → ACTION/milestone_unmeasured → gate 1.
	badEpics := InterpretEpics([]EpicSpec{{Number: 1, Title: "x"}}, nil, "gh: not found")
	r := Fold(goodMaturity(), badEpics, FoldOpts{Date: "2026-06-29"})
	if r.OK || r.Finding != "milestone_unmeasured" {
		t.Fatalf("an unmeasured roadmap must be ACTION/unmeasured, got %+v", r)
	}
	code, msg := CheckGate(r)
	if code != 1 || !strings.Contains(msg, "INCOMPLETE") {
		t.Fatalf("an unmeasured report must gate 1, got %d %q", code, msg)
	}
}

func TestWithTrendAdvisoryOnRegressionDoesNotGate(t *testing.T) {
	// A regressed trend rewrites a recorded report to advisory — but it must STILL gate
	// 0 (a regression is a measured fact, not an incomplete report).
	r := Fold(goodMaturity(), goodEpics(), FoldOpts{Date: "2026-06-29"})
	r = r.WithTrend(Trend{Direction: "regressed", Summary: "climb regressed -5.0%"})
	if r.Finding != "milestone_advisory" {
		t.Fatalf("a regressed trend must mark advisory, got %q", r.Finding)
	}
	if !strings.Contains(r.Reason, "advisory") {
		t.Fatalf("the advisory reason must carry the trend, got %q", r.Reason)
	}
	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("an advisory report must gate 0 (not a second quality gate), got %d", code)
	}
}

// --- the ledger + trend -----------------------------------------------------

func TestLedgerRoundTrip(t *testing.T) {
	r := Fold(goodMaturity(), goodEpics(), FoldOpts{Date: "2026-06-29", Commit: "abc", GeneratedAt: "2026-06-29T00:00:00Z"})
	row := RowFromReport(r)
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatalf("append line: %v", err)
	}
	// Tolerant of blank + garbled lines mixed in.
	rows := ParseLedger("\n" + line + "\nnot-json\n{}\n")
	if len(rows) != 1 {
		t.Fatalf("parse recovered %d rows, want 1 (blank/garbled/dateless skipped)", len(rows))
	}
	if rows[0].Date != "2026-06-29" || rows[0].Schema != LedgerSchema {
		t.Fatalf("round-trip lost fields: %+v", rows[0])
	}
}

func TestTrendVsLastDirections(t *testing.T) {
	base := LedgerRow{Date: "2026-06-28", MaturityProgress: 40, EpicOverallPct: 50, Matured: 2, GeneratedAt: "t0"}

	first := TrendVsLast(base, nil)
	if first.Direction != "new" {
		t.Fatalf("first tick must be new, got %q", first.Direction)
	}

	up := LedgerRow{Date: "2026-06-29", MaturityProgress: 45, EpicOverallPct: 50, Matured: 3, GeneratedAt: "t1"}
	if d := TrendVsLast(up, []LedgerRow{base}).Direction; d != "improved" {
		t.Fatalf("a higher climb must be improved, got %q", d)
	}

	down := LedgerRow{Date: "2026-06-29", MaturityProgress: 35, EpicOverallPct: 50, GeneratedAt: "t1"}
	if d := TrendVsLast(down, []LedgerRow{base}).Direction; d != "regressed" {
		t.Fatalf("a lower climb must be regressed, got %q", d)
	}

	flat := LedgerRow{Date: "2026-06-29", MaturityProgress: 40, EpicOverallPct: 50, GeneratedAt: "t1"}
	if d := TrendVsLast(flat, []LedgerRow{base}).Direction; d != "flat" {
		t.Fatalf("equal climb + roadmap must be flat, got %q", d)
	}

	// Roadmap advancing while climb holds counts as improved.
	roadUp := LedgerRow{Date: "2026-06-29", MaturityProgress: 40, EpicOverallPct: 60, GeneratedAt: "t1"}
	if d := TrendVsLast(roadUp, []LedgerRow{base}).Direction; d != "improved" {
		t.Fatalf("a roadmap gain with flat climb must be improved, got %q", d)
	}
}

// --- render -----------------------------------------------------------------

func TestRenderCarriesBothDimensions(t *testing.T) {
	r := Fold(goodMaturity(), goodEpics(), FoldOpts{Date: "2026-06-29", Commit: "abc"})
	r = r.WithTrend(TrendVsLast(RowFromReport(r), nil))
	out := Render(r)
	for _, want := range []string{"milestone report", "climb", "ladder:", "M0:", "roadmap", "generation lanes:", "now:", "debt", "second-next: 0 tracked", "#1 x", "->", "trend:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
}

func TestRenderShowsGhFailureHonestly(t *testing.T) {
	badEpics := InterpretEpics([]EpicSpec{{Number: 9, Title: "z"}}, nil, "gh: not found")
	r := Fold(goodMaturity(), badEpics, FoldOpts{Date: "2026-06-29"})
	out := Render(r)
	if !strings.Contains(out, "gh read failed") {
		t.Fatalf("an unreadable epic must render 'gh read failed', not 0%%\n%s", out)
	}
	if strings.Contains(out, "#9 z — 0%") {
		t.Fatalf("must never fabricate a 0%% for an unreadable epic\n%s", out)
	}
}

func TestCheckGateExitCodes(t *testing.T) {
	rec := Fold(goodMaturity(), goodEpics(), FoldOpts{})
	if code, _ := CheckGate(rec); code != 0 {
		t.Fatalf("recorded -> 0, got %d", code)
	}
	bad := Fold(goodMaturity(), InterpretEpics([]EpicSpec{{Number: 1}}, nil, "boom"), FoldOpts{})
	if code, _ := CheckGate(bad); code != 1 {
		t.Fatalf("unmeasured -> 1, got %d", code)
	}
}
