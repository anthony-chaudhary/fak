package metrics

import (
	"strings"
	"testing"
)

// TestGenerationRoadmapRowsClosed binds the closed row vocabulary the issue names:
// frontier, trend, promotion candidates, stale assumptions, velocity — in that
// order, and each with a human label. A drift here is a spec change, not a silent
// row addition.
func TestGenerationRoadmapRowsClosed(t *testing.T) {
	want := []RoadmapRow{
		RowFrontier,
		RowTrend,
		RowPromotionCandidates,
		RowStaleAssumptions,
		RowVelocity,
	}
	if len(RoadmapRows) != len(want) {
		t.Fatalf("RoadmapRows = %v, want %v", RoadmapRows, want)
	}
	for i, r := range want {
		if RoadmapRows[i] != r {
			t.Fatalf("RoadmapRows[%d] = %q, want %q", i, RoadmapRows[i], r)
		}
		// every declared row has a real human label distinct from its raw key
		if label := RoadmapRows[i].Label(); label == "" || label == string(r) {
			t.Fatalf("row %q has no human label (got %q)", r, label)
		}
	}
}

// TestGenerationRoadmapHorizonOrder pins the now->future horizon ordering — the
// closed generation vocabulary, sorted by horizon and never by priority.
func TestGenerationRoadmapHorizonOrder(t *testing.T) {
	want := []string{"now", "next", "second-next", "future"}
	if len(RoadmapGenerations) != len(want) {
		t.Fatalf("RoadmapGenerations = %v, want %v", RoadmapGenerations, want)
	}
	for i, s := range want {
		if RoadmapGenerations[i] != s {
			t.Fatalf("RoadmapGenerations[%d] = %q, want %q", i, RoadmapGenerations[i], s)
		}
	}
}

// TestGenerationRoadmapRenderShowsAllRowsAndHorizons proves the dashboard renders
// the orthogonality header plus every row for every horizon — including a horizon
// missing from the snapshot, which must appear as an empty column (the full closed
// set is always shown, so a stalled horizon can't hide by being absent).
func TestGenerationRoadmapRenderShowsAllRowsAndHorizons(t *testing.T) {
	rm := GenerationRoadmap{
		Period: "7d",
		Columns: []GenerationColumn{
			{Stream: "now", Label: "gen/now", Frontier: "native harness", Trend: TrendAdvancing, PromotionCandidates: 2, StaleAssumptions: 0, Velocity: 5.5},
			{Stream: "second-next", Label: "gen/second-next", Frontier: "portfolio dashboard", Trend: TrendHolding, PromotionCandidates: 1, StaleAssumptions: 3, Velocity: 0.5},
			// "next" and "future" intentionally omitted — Render must synthesize them.
		},
	}
	out := rm.Render()

	if !strings.Contains(out, OrthogonalityNote) {
		t.Fatalf("render missing orthogonality note:\n%s", out)
	}
	for _, kw := range []string{"priority", "shared trunk", "feature gate"} {
		if !strings.Contains(strings.ToLower(out), kw) {
			t.Fatalf("orthogonality note does not name %q:\n%s", kw, out)
		}
	}
	for _, row := range RoadmapRows {
		if !strings.Contains(out, row.Label()) {
			t.Fatalf("render missing row %q:\n%s", row.Label(), out)
		}
	}
	for _, stream := range RoadmapGenerations {
		if !strings.Contains(out, "gen/"+stream) {
			t.Fatalf("render missing horizon column gen/%s:\n%s", stream, out)
		}
	}
	if !strings.Contains(out, "native harness") || !strings.Contains(out, "portfolio dashboard") {
		t.Fatalf("render dropped a frontier cell:\n%s", out)
	}
	// A missing horizon shows unknown trend, not a dropped column.
	if !strings.Contains(out, string(TrendUnknown)) {
		t.Fatalf("synthesized horizon should render trend %q:\n%s", TrendUnknown, out)
	}
	// Deterministic: same snapshot renders identically.
	if rm.Render() != out {
		t.Fatal("Render is not deterministic")
	}
}

// TestGenerationRoadmapCellFormatting checks the numeric/empty cell forms so a
// zero-velocity horizon reads as "0", not blank, and an empty frontier reads "-".
func TestGenerationRoadmapCellFormatting(t *testing.T) {
	c := GenerationColumn{Stream: "future", PromotionCandidates: 0, StaleAssumptions: 4, Velocity: 0}
	if got := c.cell(RowFrontier); got != "-" {
		t.Fatalf("empty frontier cell = %q, want %q", got, "-")
	}
	if got := c.cell(RowTrend); got != string(TrendUnknown) {
		t.Fatalf("empty trend cell = %q, want %q", got, TrendUnknown)
	}
	if got := c.cell(RowStaleAssumptions); got != "4" {
		t.Fatalf("stale-assumptions cell = %q, want %q", got, "4")
	}
	if got := c.cell(RowVelocity); got != "0" {
		t.Fatalf("zero velocity cell = %q, want %q", got, "0")
	}
}
