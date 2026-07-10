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

// TestQADogfoodPanelWitnessProven proves the #3839 upgrade: a closure witness only
// counts as clean when it was actually RUN and PASSED. Over the three fixture states
// the issue names — present+pass, present+fail, absent — plus a present-but-unrun
// case, a failing or unrun witness is NOT a proven closure and, when the issue is
// closed, becomes surfaced closure debt.
func TestQADogfoodPanelWitnessProven(t *testing.T) {
	issues := []QADogfoodIssue{
		// present + pass: declared, ran, passed → proven, a clean closure witness.
		{Number: 1, Open: false, ClosureWitness: "go test ./a", WitnessRun: true, WitnessPassed: true},
		// present + fail: declared, ran, FAILED → not proven; closed-but-unproven debt.
		{Number: 2, Open: false, ClosureWitness: "go test ./b", WitnessRun: true, WitnessPassed: false},
		// present + unrun: declared but never executed → not proven; closed-but-unproven.
		{Number: 3, Open: false, ClosureWitness: "go test ./c"},
		// absent: no witness → not run, not proven, and not an unproven-closure (a
		// missing witness is a different gap than a declared-but-unproven one).
		{Number: 4, Open: false},
	}

	p := FoldQADogfoodPanel(issues)

	if p.ClosureWitnessCount != 3 {
		t.Errorf("ClosureWitnessCount = %d, want 3 (declared: #1,#2,#3)", p.ClosureWitnessCount)
	}
	if p.WitnessRunCount != 2 {
		t.Errorf("WitnessRunCount = %d, want 2 (ran: #1,#2)", p.WitnessRunCount)
	}
	if p.WitnessPassCount != 1 {
		t.Errorf("WitnessPassCount = %d, want 1 (proven: only #1)", p.WitnessPassCount)
	}
	if p.UnprovenClosureCount != 2 {
		t.Errorf("UnprovenClosureCount = %d, want 2 (closed+declared+unproven: #2 failed, #3 unrun)", p.UnprovenClosureCount)
	}

	// The load-bearing distinction: a failing witness is NOT a clean closure witness,
	// even though it declares one.
	failing := issues[1]
	if !failing.HasClosureWitness() {
		t.Fatal("#2 declares a witness")
	}
	if failing.WitnessProven() {
		t.Error("#2's witness FAILED — it must not count as a proven/clean closure witness")
	}
	if !issues[0].WitnessProven() {
		t.Error("#1's witness ran and passed — it must count as proven")
	}
	if issues[2].WitnessProven() {
		t.Error("#3's witness was never run — it must not count as proven")
	}

	// The render must make the distinction visible, not fold proven and unproven into
	// one number.
	line := RenderQADogfoodPanel(p)
	if !strings.Contains(line, "1 proven") || !strings.Contains(line, "2 unproven-closed") {
		t.Errorf("render must distinguish proven from unproven: %q", line)
	}
}

// TestQADogfoodPanelOpenWitnessNotClosureDebt proves an OPEN issue with a declared-
// but-unrun witness is not yet closure debt — the witness is only owed at closure.
func TestQADogfoodPanelOpenWitnessNotClosureDebt(t *testing.T) {
	p := FoldQADogfoodPanel([]QADogfoodIssue{
		{Number: 1, Open: true, ClosureWitness: "go test ./a"}, // open, unrun
	})
	if p.UnprovenClosureCount != 0 {
		t.Errorf("UnprovenClosureCount = %d, want 0 (an open issue is not closure debt)", p.UnprovenClosureCount)
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
