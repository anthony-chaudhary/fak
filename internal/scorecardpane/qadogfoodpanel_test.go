package scorecardpane

import (
	"strings"
	"testing"
	"time"
)

// TestQADogfoodPanelFoldsCounts proves the fold reports each of the four figures the
// issue's done condition names — open, stale, closure-witness, and the root-point
// percent — over a mixed set, worst-and-best cases both present.
func TestQADogfoodPanelFoldsCounts(t *testing.T) {
	issues := []QADogfoodIssue{
		// Healthy open origin control: open, fresh, witnessed, full root-point fields.
		{Number: 1, Open: true, Stale: false, ClosureWitness: "go test ./x -run Y",
			RootPointChange: "Add the origin control", DoneCondition: "The panel reports X"},
		// Stale open issue, still witnessed + root-pointed.
		{Number: 2, Open: true, Stale: true, ClosureWitness: "go test ./z",
			RootPointChange: "Move the check earlier", DoneCondition: "The check refuses at origin"},
		// Open but bare: no witness, no root-point fields (after-the-fact cleanup debt).
		{Number: 3, Open: true},
		// Closed, witnessed, root-pointed — counts toward closure-witness + root-point
		// but not open/stale.
		{Number: 4, Open: false, ClosureWitness: "go test ./w",
			RootPointChange: "Root control", DoneCondition: "Done when W"},
	}

	p := FoldQADogfoodPanel(issues)

	if p.Schema != QADogfoodPanelSchema {
		t.Errorf("schema = %q, want %q", p.Schema, QADogfoodPanelSchema)
	}
	if p.Total != 4 {
		t.Errorf("Total = %d, want 4", p.Total)
	}
	if p.OpenCount != 3 {
		t.Errorf("OpenCount = %d, want 3", p.OpenCount)
	}
	if p.StaleCount != 1 {
		t.Errorf("StaleCount = %d, want 1", p.StaleCount)
	}
	if p.ClosureWitnessCount != 3 {
		t.Errorf("ClosureWitnessCount = %d, want 3", p.ClosureWitnessCount)
	}
	if p.RootPointCount != 3 {
		t.Errorf("RootPointCount = %d, want 3", p.RootPointCount)
	}
	if p.RootPointPercent != 75 {
		t.Errorf("RootPointPercent = %v, want 75", p.RootPointPercent)
	}
}

// TestQADogfoodPanelEmptyIsWellFormed proves the empty set never divides by zero and
// renders a well-formed zero line.
func TestQADogfoodPanelEmptyIsWellFormed(t *testing.T) {
	p := FoldQADogfoodPanel(nil)
	if p.Total != 0 || p.RootPointPercent != 0 {
		t.Errorf("empty panel = %+v, want zero counts and 0%% root-point", p)
	}
	line := RenderQADogfoodPanel(p)
	if !strings.Contains(line, "0 tracked") || !strings.Contains(line, "0% of tracked") {
		t.Errorf("empty render = %q, want a well-formed zero line", line)
	}
}

// TestQADogfoodPanelRootPointRequiresBothFields proves an issue missing either the
// root-point change or the done condition does NOT count toward root-point coverage.
func TestQADogfoodPanelRootPointRequiresBothFields(t *testing.T) {
	onlyChange := QADogfoodIssue{Open: true, RootPointChange: "Move it earlier"}
	onlyDone := QADogfoodIssue{Open: true, DoneCondition: "Done when Y"}
	both := QADogfoodIssue{Open: true, RootPointChange: "Move it earlier", DoneCondition: "Done when Y"}
	if onlyChange.HasRootPointFields() || onlyDone.HasRootPointFields() {
		t.Error("a single root-point field should not satisfy HasRootPointFields")
	}
	if !both.HasRootPointFields() {
		t.Error("both root-point fields should satisfy HasRootPointFields")
	}
	p := FoldQADogfoodPanel([]QADogfoodIssue{onlyChange, onlyDone, both})
	if p.RootPointCount != 1 {
		t.Errorf("RootPointCount = %d, want 1 (only the fully-fielded issue)", p.RootPointCount)
	}
}

// TestQADogfoodPanelStaleHorizon proves QADogfoodStale is open-gated, horizon-gated,
// and never invents staleness for a closed issue or an unknown last-touch time.
func TestQADogfoodPanelStaleHorizon(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	horizon := 14 * 24 * time.Hour
	old := now.Add(-30 * 24 * time.Hour)
	recent := now.Add(-1 * 24 * time.Hour)

	if !QADogfoodStale(true, old, now, horizon) {
		t.Error("open + untouched 30d should be stale")
	}
	if QADogfoodStale(true, recent, now, horizon) {
		t.Error("open + touched 1d ago should not be stale")
	}
	if QADogfoodStale(false, old, now, horizon) {
		t.Error("a closed issue is never stale")
	}
	if QADogfoodStale(true, time.Time{}, now, horizon) {
		t.Error("an unknown last-touch time is never stale (undercount, don't invent)")
	}
	// A non-positive horizon falls back to the default (14d), so 30d old is stale.
	if !QADogfoodStale(true, old, now, 0) {
		t.Error("non-positive horizon should fall back to the default and count 30d as stale")
	}
}
