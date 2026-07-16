package operatorbrief

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// residualUnit builds a RESIDUAL overlay unit with a caller-chosen title/leaf, so
// a test can steer whether choicetriage reads it as a genuine authority decision.
func residualUnit(leaf, title string, resolves ...string) steerpr.Unit {
	return steerpr.Unit{
		Leaf:     leaf,
		Title:    title,
		Band:     steerpr.BandResidual,
		Resolves: resolves,
		Commits:  []steerpr.Commit{{SHA: "aaa111", Subject: title, Leaf: leaf, Band: steerpr.BandResidual}},
	}
}

// TestOSPStateReflectsMeasurement pins the source stamp: absent -> missing,
// unreadable -> unmeasured (never a clean zero), readable -> ok.
func TestOSPStateReflectsMeasurement(t *testing.T) {
	if got := ospState(nil); got.Status != "missing" {
		t.Errorf("ospState(nil).Status = %q, want missing", got.Status)
	}
	if got := ospState(&OSP{Unreadable: true}); got.Status != "unmeasured" || got.Finding != "osp_unmeasured" {
		t.Errorf("ospState(unreadable) = {%q,%q}, want {unmeasured, osp_unmeasured}", got.Status, got.Finding)
	}
	readable := ospState(&OSP{Units: []steerpr.Unit{residualUnit("gateway", "fix(gateway): tighten the overlay fold")}})
	if readable.Status != "ok" {
		t.Errorf("ospState(readable).Status = %q, want ok", readable.Status)
	}
	if !strings.HasPrefix(readable.Finding, "osp_residual_") {
		t.Errorf("ospState(readable).Finding = %q, want an osp_residual_* count", readable.Finding)
	}
}

// TestAddOSPResidualNonAuthorityGoesToWatch is the witness the issue names: a
// RESIDUAL unit that choicetriage judges NOT an authority decision must land in
// watch, not human — the overlay cannot spam the pager.
func TestAddOSPResidualNonAuthorityGoesToWatch(t *testing.T) {
	var r Report
	addOSP(&r, OSP{Units: []steerpr.Unit{
		residualUnit("gateway", "fix(gateway): tighten the overlay fold"),
	}})
	if len(r.Human) != 0 {
		t.Fatalf("non-authority RESIDUAL unit paged: Human = %d, want 0", len(r.Human))
	}
	if len(r.Watch) != 1 {
		t.Fatalf("non-authority RESIDUAL unit not watched: Watch = %d, want 1", len(r.Watch))
	}
	if r.Watch[0].Source != "osp" {
		t.Errorf("watch item Source = %q, want osp", r.Watch[0].Source)
	}
}

// TestAddOSPResidualAuthorityPages is the complement: a RESIDUAL unit that DOES
// name authority a person holds (a release/publish decision) reaches the human
// bucket.
func TestAddOSPResidualAuthorityPages(t *testing.T) {
	var r Report
	addOSP(&r, OSP{Units: []steerpr.Unit{
		residualUnit("release", "approve the pending release publish"),
	}})
	if len(r.Human) != 1 {
		t.Fatalf("authority RESIDUAL unit did not page: Human = %d, want 1", len(r.Human))
	}
	if r.Human[0].Source != "osp" || r.Human[0].Severity != "decision" {
		t.Errorf("human item = {%q,%q}, want {osp, decision}", r.Human[0].Source, r.Human[0].Severity)
	}
	if len(r.Watch) != 0 {
		t.Errorf("authority RESIDUAL unit also watched: Watch = %d, want 0", len(r.Watch))
	}
}

// TestAddOSPBandsMapToBuckets pins the non-residual mapping: UNVERIFIABLE -> watch,
// CLEARED -> background.
func TestAddOSPBandsMapToBuckets(t *testing.T) {
	var r Report
	addOSP(&r, OSP{Units: []steerpr.Unit{
		{Leaf: "a", Title: "docs(a): note", Band: steerpr.BandUnverifiable},
		{Leaf: "b", Title: "feat(b): thing", Band: steerpr.BandCleared},
	}})
	if len(r.Watch) != 1 || r.Watch[0].Source != "osp" {
		t.Errorf("UNVERIFIABLE unit not watched: Watch = %v", r.Watch)
	}
	if len(r.Background) != 1 || r.Background[0].Source != "osp" {
		t.Errorf("CLEARED unit not backgrounded: Background = %v", r.Background)
	}
	if len(r.Human) != 0 {
		t.Errorf("non-residual bands paged: Human = %d, want 0", len(r.Human))
	}
}

// TestAddOSPUnreadableWatchesNeverPages proves an unreadable overlay is surfaced
// (never a clean zero) but never pages a human.
func TestAddOSPUnreadableWatchesNeverPages(t *testing.T) {
	var r Report
	addOSP(&r, OSP{Unreadable: true, Note: "payload deleted"})
	if len(r.Human) != 0 {
		t.Fatalf("unreadable overlay paged: Human = %d, want 0", len(r.Human))
	}
	if len(r.Watch) != 1 || !strings.Contains(r.Watch[0].Title, "unmeasured") {
		t.Fatalf("unreadable overlay not surfaced as an unmeasured watch: Watch = %v", r.Watch)
	}
}

// TestFoldOSPUnreadableLeavesBriefValid proves the acceptance gate: deleting the
// OSP payload marks the source unmeasured and does NOT red the brief on its own.
// The brief is folded with every required source measured, so the only variable
// is the OSP overlay: with it unreadable, the brief must still gate clean (no
// human bucket) and carry an "unmeasured" osp source stamp.
func TestFoldOSPUnreadableLeavesBriefValid(t *testing.T) {
	base := measuredInputs()
	base.OSP = &OSP{Unreadable: true}
	r := Fold(base)

	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("unreadable OSP redded the brief: gate exit %d, want 0", code)
	}
	if s := sourceByName(r.Sources, "osp"); s == nil || s.Status != "unmeasured" {
		t.Fatalf("osp source stamp = %+v, want status unmeasured", s)
	}
}

// TestFoldOSPResidualPileLengthensTimebox proves OSP items feed the existing
// attention weighting: more residual-authority units => a longer human timebox,
// honestly rather than silently.
func TestFoldOSPResidualPileLengthensTimebox(t *testing.T) {
	one := measuredInputs()
	one.OSP = &OSP{Units: []steerpr.Unit{
		residualUnit("release", "approve release publish #1"),
	}}
	many := measuredInputs()
	many.OSP = &OSP{Units: []steerpr.Unit{
		residualUnit("release", "approve release publish #1"),
		residualUnit("auth", "grant the auth credential #2"),
		residualUnit("policy", "make the policy priority call #3"),
	}}

	small := Fold(one).Attention.BudgetMinutes
	big := Fold(many).Attention.BudgetMinutes
	if big <= small {
		t.Fatalf("residual pile did not lengthen the timebox: 1-unit=%dm, 3-unit=%dm (want 3-unit > 1-unit)", small, big)
	}
}

// TestOSPBriefCaptureBucketsAcrossBands is the captured-brief witness: it folds a
// four-band overlay through the exact Fold path the `fak operator brief --json`
// CLI calls, asserts each band lands in its bucket, and emits the folded brief so
// the capture is inspectable. RESIDUAL splits by authority: the release/publish
// unit pages (human), the plain gateway unit is a watch. UNVERIFIABLE is a watch,
// CLEARED is background — and the osp source stamp reads measured (ok).
func TestOSPBriefCaptureBucketsAcrossBands(t *testing.T) {
	in := measuredInputs()
	in.OSP = &OSP{
		Schema: steerpr.Schema,
		Units: []steerpr.Unit{
			residualUnit("release", "approve the pending release publish", "#5024"),
			residualUnit("gateway", "fix(gateway): tighten the overlay fold"),
			{Leaf: "cache", Title: "docs(cache): explain the ladder", Band: steerpr.BandUnverifiable},
			{Leaf: "kv", Title: "feat(kv): witnessed reuse win", Band: steerpr.BandCleared},
		},
	}
	r := Fold(in)

	want := map[string]string{
		"human":      "approve the pending release publish",
		"watch":      "fix(gateway): tighten the overlay fold",
		"background": "feat(kv): witnessed reuse win (cleared)",
	}
	if !hasOSPItem(r.Human, want["human"]) {
		t.Errorf("release/publish RESIDUAL unit not in human bucket: %+v", r.Human)
	}
	if !hasOSPItem(r.Watch, want["watch"]) {
		t.Errorf("non-authority RESIDUAL unit not in watch bucket: %+v", r.Watch)
	}
	if !hasOSPItem(r.Watch, "docs(cache): explain the ladder (unverifiable)") {
		t.Errorf("UNVERIFIABLE unit not in watch bucket: %+v", r.Watch)
	}
	if !hasOSPItem(r.Background, want["background"]) {
		t.Errorf("CLEARED unit not in background bucket: %+v", r.Background)
	}
	if s := sourceByName(r.Sources, "osp"); s == nil || s.Status != "ok" {
		t.Errorf("osp source stamp = %+v, want status ok (measured)", s)
	}

	blob, err := json.MarshalIndent(struct {
		Sources []SourceState `json:"sources"`
		Counts  Counts        `json:"counts"`
		Human   []Item        `json:"human"`
		Watch   []Item        `json:"watch"`
		Backgd  []Item        `json:"background"`
	}{r.Sources, r.Counts, r.Human, r.Watch, r.Background}, "", "  ")
	if err != nil {
		t.Fatalf("marshal captured brief: %v", err)
	}
	t.Logf("captured operator brief (OSP-sourced entries bucketed):\n%s", blob)
}

// hasOSPItem reports whether any osp-sourced item in items carries the title.
func hasOSPItem(items []Item, title string) bool {
	for _, it := range items {
		if it.Source == "osp" && it.Title == title {
			return true
		}
	}
	return false
}

// measuredInputs returns a brief whose required sources are all measured and
// clean (no human bucket), so a test can isolate the OSP overlay as the only
// variable that could change the brief's verdict.
func measuredInputs() Inputs {
	c := cleanCadence()
	p := cleanProgram()
	m := cleanMilestone()
	return Inputs{Cadence: &c, Program: &p, Milestone: &m}
}

// sourceByName returns the source stamp with the given name, or nil.
func sourceByName(srcs []SourceState, name string) *SourceState {
	for i := range srcs {
		if srcs[i].Name == name {
			return &srcs[i]
		}
	}
	return nil
}
