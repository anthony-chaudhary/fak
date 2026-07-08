package cachevaluereport

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// seriesFor finds the (length, outcome) series in a shape-trend report, or nil if absent.
func seriesFor(r ShapeTrendReport, l LengthBand, o OutcomeBand) *ShapeSeries {
	for i := range r.Series {
		if r.Series[i].Length == l && r.Series[i].Outcome == o {
			return &r.Series[i]
		}
	}
	return nil
}

func TestFoldShapeTrend_EmptyIsInsufficientButOK(t *testing.T) {
	r := FoldShapeTrend(nil, fixedNow)
	if !r.OK {
		t.Fatalf("empty shape-trend should be OK (a report, not a gate); got OK=false")
	}
	if r.Verdict != "INSUFFICIENT" {
		t.Fatalf("empty shape-trend verdict = %q, want INSUFFICIENT", r.Verdict)
	}
	if len(r.Series) != 0 || len(r.Weeks) != 0 || r.TotalSessions != 0 {
		t.Fatalf("empty trend should have no series/weeks/sessions; got %d series, %d weeks, %d sessions",
			len(r.Series), len(r.Weeks), r.TotalSessions)
	}
	if !r.VsNaiveMultipleExcluded || r.PublishableValueFamily == "" {
		t.Fatalf("#1066 fence self-labels missing: excluded=%v family=%q", r.VsNaiveMultipleExcluded, r.PublishableValueFamily)
	}
	if r.Schema != ShapeTrendSchema {
		t.Fatalf("schema = %q, want %q", r.Schema, ShapeTrendSchema)
	}
}

func TestFoldShapeTrend_SingleTurnOnlyInsufficient(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ReuseRatio: 0},
		{Date: weekB, SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ReuseRatio: 0},
	}
	r := FoldShapeTrend(rows, fixedNow)
	if r.Verdict != "INSUFFICIENT" || !r.OK {
		t.Fatalf("single-turn-only verdict/ok = %q/%v, want INSUFFICIENT/true", r.Verdict, r.OK)
	}
	if r.MultiTurnSessions != 0 || r.TotalSessions != 2 {
		t.Fatalf("multi/total = %d/%d, want 0/2", r.MultiTurnSessions, r.TotalSessions)
	}
}

// A single-week corpus reports `new` for the shape (no prior week to compare).
func TestFoldShapeTrend_SingleWeekIsNew(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 900, ReuseRatio: 0.90},
	}
	r := FoldShapeTrend(rows, fixedNow)
	if r.Verdict != "MEASURED" {
		t.Fatalf("verdict = %q, want MEASURED", r.Verdict)
	}
	if len(r.Weeks) != 1 {
		t.Fatalf("weeks = %d, want 1", len(r.Weeks))
	}
	s := seriesFor(r, LengthLong, OutcomeWarm)
	if s == nil {
		t.Fatalf("missing long×warm series")
	}
	if len(s.Weeks) != 1 {
		t.Fatalf("long×warm points = %d, want 1", len(s.Weeks))
	}
	if s.Weeks[0].Trend != TrendNew {
		t.Fatalf("single-week trend = %q, want new", s.Weeks[0].Trend)
	}
	// Only reuse-bearing shape this week → within-week share of reused tokens is 1.0.
	if got := s.Weeks[0].ShareOfReusedTokens; got < 0.999 || got > 1.001 {
		t.Fatalf("long×warm share_of_reused_tokens = %.4f, want ~1.0", got)
	}
	// A brand-new shape gained share by definition; the headline reflects it as gained.
	if len(r.LatestLost) != 0 {
		t.Fatalf("single-week corpus should lose no shape; got %v", r.LatestLost)
	}
}

// A corpus where the warm×long share of reused tokens FALLS across two weeks reports
// `regressed` for that shape — the core drift witness from the issue.
func TestFoldShapeTrend_WarmLongShareFallsIsRegressed(t *testing.T) {
	rows := []cachevalueledger.Row{
		// Week A: long×warm is the ONLY reuse-bearing shape → within-week reuse share 1.0.
		{Date: weekAEarly, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 900, ReuseRatio: 0.90},

		// Week B: long×warm still present but a short×warm shape now carries most of the
		// week's reuse → long×warm's within-week reuse share collapses to 600/3000 = 0.20.
		{Date: weekB, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 600, ReuseRatio: 0.60},
		{Date: weekB, SessionType: "serve", Turns: 3, PromptTokens: 3000, ReusedTokens: 2400, ReuseRatio: 0.80},
	}
	r := FoldShapeTrend(rows, fixedNow)
	if r.Verdict != "MEASURED" {
		t.Fatalf("verdict = %q, want MEASURED", r.Verdict)
	}
	if len(r.Weeks) != 2 {
		t.Fatalf("weeks = %d, want 2 (%v)", len(r.Weeks), r.Weeks)
	}

	s := seriesFor(r, LengthLong, OutcomeWarm)
	if s == nil || len(s.Weeks) != 2 {
		t.Fatalf("long×warm series = %+v, want 2 weekly points", s)
	}
	if s.Weeks[0].Trend != TrendNew {
		t.Fatalf("long×warm week[0] trend = %q, want new", s.Weeks[0].Trend)
	}
	if s.Weeks[1].Trend != TrendRegressed {
		t.Fatalf("long×warm week[1] trend = %q, want regressed (share fell 1.0 → 0.20)", s.Weeks[1].Trend)
	}
	if s.LatestTrend != TrendRegressed {
		t.Fatalf("long×warm LatestTrend = %q, want regressed", s.LatestTrend)
	}
	if s.Weeks[1].DeltaShareOfReusedTokens >= 0 {
		t.Fatalf("long×warm week[1] delta = %.4f, want negative", s.Weeks[1].DeltaShareOfReusedTokens)
	}

	// The most-recent week's headline names long×warm as a lost shape and short×warm,
	// which appeared new-and-dominant, is not double-counted as lost.
	if r.LatestWeek != r.Weeks[1] {
		t.Fatalf("LatestWeek = %q, want %q", r.LatestWeek, r.Weeks[1])
	}
	foundLost := false
	for _, lbl := range r.LatestLost {
		if lbl == shapeLabel(LengthLong, OutcomeWarm) {
			foundLost = true
		}
	}
	if !foundLost {
		t.Fatalf("LatestLost = %v, want it to include %q", r.LatestLost, shapeLabel(LengthLong, OutcomeWarm))
	}
}

// A shape whose within-week reuse share is essentially unchanged reads `flat` (the
// reuseEpsilon dead-band), not a manufactured improved/regressed.
func TestFoldShapeTrend_UnchangedShareIsFlat(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 800, ReuseRatio: 0.80},
		{Date: weekB, SessionType: "guard", Turns: 8, PromptTokens: 1000, ReusedTokens: 800, ReuseRatio: 0.80},
	}
	r := FoldShapeTrend(rows, fixedNow)
	s := seriesFor(r, LengthLong, OutcomeWarm)
	if s == nil || len(s.Weeks) != 2 {
		t.Fatalf("long×warm series = %+v, want 2 weekly points", s)
	}
	// Both weeks: long×warm is the only reuse-bearing shape → share 1.0 both weeks → flat.
	if s.Weeks[1].Trend != TrendFlat {
		t.Fatalf("long×warm week[1] trend = %q, want flat (share held at 1.0)", s.Weeks[1].Trend)
	}
}

func TestFoldShapeTrend_ZeroTurnRowsSkipped(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 0, PromptTokens: 999},
		{Date: weekAEarly, SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600, ReuseRatio: 0.60},
	}
	r := FoldShapeTrend(rows, fixedNow)
	if r.TotalSessions != 1 {
		t.Fatalf("TotalSessions = %d, want 1 (zero-turn row skipped)", r.TotalSessions)
	}
	if s := seriesFor(r, LengthLong, OutcomeWarm); s == nil || len(s.Weeks) != 1 {
		t.Fatalf("want a long×warm series with 1 weekly point; got %+v", s)
	}
}
