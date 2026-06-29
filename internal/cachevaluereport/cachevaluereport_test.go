package cachevaluereport

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

var fixedNow = time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

// 2026-06-15 (Mon) and 2026-06-24 (Wed) fall in two consecutive ISO weeks, so rows
// dated in each land in distinct buckets.
const (
	weekAEarly = "2026-06-15"
	weekALate  = "2026-06-18"
	weekB      = "2026-06-24"
)

func TestFold_EmptyIsInsufficientButOK(t *testing.T) {
	r := Fold(nil, fixedNow)
	if !r.OK {
		t.Fatalf("empty roll-up should be OK (a report, not a gate); got OK=false")
	}
	if r.Verdict != "INSUFFICIENT" {
		t.Fatalf("empty roll-up verdict = %q, want INSUFFICIENT", r.Verdict)
	}
	if len(r.Buckets) != 0 || r.TotalSessions != 0 {
		t.Fatalf("empty roll-up should have no buckets/sessions; got %d buckets, %d sessions", len(r.Buckets), r.TotalSessions)
	}
	if !r.VsNaiveMultipleExcluded || r.PublishableValueFamily == "" {
		t.Fatalf("#1066 fence self-labels missing: excluded=%v family=%q", r.VsNaiveMultipleExcluded, r.PublishableValueFamily)
	}
}

func TestFold_SingleTurnOnlyIsInsufficient(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ColdTurns: 1},
		{Date: weekALate, SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0, ColdTurns: 1},
	}
	r := Fold(rows, fixedNow)
	if r.Verdict != "INSUFFICIENT" {
		t.Fatalf("single-turn-only verdict = %q, want INSUFFICIENT (no multi-turn reuse to trend)", r.Verdict)
	}
	if r.TotalSessions != 2 {
		t.Fatalf("TotalSessions = %d, want 2", r.TotalSessions)
	}
	if len(r.Buckets) != 1 {
		t.Fatalf("both rows are in week A; want 1 bucket, got %d", len(r.Buckets))
	}
	if r.Buckets[0].RealizedReuseRatio != 0 {
		t.Fatalf("single-turn bucket realized reuse = %v, want 0", r.Buckets[0].RealizedReuseRatio)
	}
}

func TestFold_ZeroTurnRowsSkipped(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 0, PromptTokens: 999},
		{Date: weekAEarly, SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600},
	}
	r := Fold(rows, fixedNow)
	if r.TotalRows != 1 {
		t.Fatalf("TotalRows = %d, want 1 (zero-turn row skipped)", r.TotalRows)
	}
}

func TestFold_TwoWeekTrendImproved(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekALate, SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600, FrozenTurns: 6, PartialTurns: 2, ColdTurns: 2},
		{Date: weekB, SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 750, FrozenTurns: 8, PartialTurns: 1, ColdTurns: 1},
	}
	r := Fold(rows, fixedNow)
	if r.Verdict != "MEASURED" {
		t.Fatalf("verdict = %q, want MEASURED", r.Verdict)
	}
	if len(r.Buckets) != 2 {
		t.Fatalf("want 2 weekly buckets, got %d", len(r.Buckets))
	}
	first, second := r.Buckets[0], r.Buckets[1]
	if !approx(first.RealizedReuseRatio, 0.60) {
		t.Fatalf("week-A realized reuse = %v, want ~0.60", first.RealizedReuseRatio)
	}
	if !approx(second.RealizedReuseRatio, 0.75) {
		t.Fatalf("week-B realized reuse = %v, want ~0.75", second.RealizedReuseRatio)
	}
	if first.Trend != TrendNew {
		t.Fatalf("first bucket trend = %q, want new", first.Trend)
	}
	if second.Trend != TrendImproved {
		t.Fatalf("second bucket trend = %q, want improved", second.Trend)
	}
	if !approx(r.LatestReuseRatio, 0.75) || r.LatestTrend != TrendImproved {
		t.Fatalf("latest = %v/%q, want 0.75/improved", r.LatestReuseRatio, r.LatestTrend)
	}
	if first.Thin || second.Thin {
		t.Fatalf("10 multi-turn turns each is above MinBucketTurns=%d; neither should be thin", MinBucketTurns)
	}
}

func TestFold_RegressedAndThin(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekALate, SessionType: "serve", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
		// week B: only 3 multi-turn turns -> below MinBucketTurns=8 -> thin; reuse drops.
		{Date: weekB, SessionType: "serve", Turns: 3, PromptTokens: 1000, ReusedTokens: 400},
	}
	r := Fold(rows, fixedNow)
	if len(r.Buckets) != 2 {
		t.Fatalf("want 2 buckets, got %d", len(r.Buckets))
	}
	second := r.Buckets[1]
	if second.Trend != TrendRegressed {
		t.Fatalf("week-B trend = %q, want regressed", second.Trend)
	}
	if !second.Thin {
		t.Fatalf("week-B has 3 multi-turn turns (< %d); want Thin=true", MinBucketTurns)
	}
}

func TestFold_SessionTypeBreakdown(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekAEarly, SessionType: "guard", Turns: 5, PromptTokens: 100, ReusedTokens: 50},
		{Date: weekALate, SessionType: "serve", Turns: 5, PromptTokens: 100, ReusedTokens: 50},
		{Date: weekALate, SessionType: "guard", Turns: 5, PromptTokens: 100, ReusedTokens: 50},
	}
	r := Fold(rows, fixedNow)
	if len(r.Buckets) != 1 {
		t.Fatalf("all rows in week A; want 1 bucket, got %d", len(r.Buckets))
	}
	bt := r.Buckets[0].BySessionType
	if bt["guard"] != 2 || bt["serve"] != 1 {
		t.Fatalf("BySessionType = %v, want guard:2 serve:1", bt)
	}
	if r.Buckets[0].Start != weekAEarly {
		t.Fatalf("bucket Start = %q, want earliest date %q", r.Buckets[0].Start, weekAEarly)
	}
}

func TestRender_NonEmptyAndFenced(t *testing.T) {
	rows := []cachevalueledger.Row{
		{Date: weekALate, SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600},
	}
	out := Render(Fold(rows, fixedNow))
	if !strings.Contains(out, "WITNESSED") {
		t.Fatalf("Render output should label the track WITNESSED:\n%s", out)
	}
	if !strings.Contains(out, "marginal-over-tuned-warm-KV") {
		t.Fatalf("Render output should carry the #1066 fence family:\n%s", out)
	}
	// The fence is STRUCTURAL: Fold never computes the vs-naive 1/(1-reuse) multiple,
	// so no Bucket field can carry it. (The family string deliberately NAMES the
	// excluded multiple to document the fence, so a substring check on the rendered
	// text would false-positive on that self-description.)
	if !Fold(rows, fixedNow).VsNaiveMultipleExcluded {
		t.Fatalf("report must self-label the vs-naive multiple as excluded (#1066)")
	}
}

func approx(got, want float64) bool {
	d := got - want
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}
