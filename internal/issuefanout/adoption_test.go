package issuefanout

import (
	"reflect"
	"testing"
)

// A shipped leaf clears the floor only at MinFanout filed follow-ons; below it,
// the leaf is a gap — a spine that shipped without its 3..50+ backlog — and the
// report's OK bit goes false so a pipeline can gate on it.
func TestAdoptionGapAndFloor(t *testing.T) {
	var markers []string
	for _, slug := range []string{"qa-a", "qa-b", "qa-c"} { // MinFanout == 3
		markers = append(markers, "fanout-spinez-"+slug)
	}
	markers = append(markers, "fanout-gappy-qa-a", "fanout-gappy-qa-b") // MinFanout-1

	rep := Adoption([]string{"spinez", "gappy", "barez"}, markers)

	if rep.ShippedLeaves != 3 {
		t.Fatalf("ShippedLeaves: got %d want 3", rep.ShippedLeaves)
	}
	if rep.ClearedLeaves != 1 || rep.GapLeaves != 2 || rep.OK {
		t.Fatalf("standing: got cleared=%d gap=%d ok=%v want 1/2/false", rep.ClearedLeaves, rep.GapLeaves, rep.OK)
	}
	if !reflect.DeepEqual(rep.Gaps, []string{"barez", "gappy"}) {
		t.Fatalf("Gaps: got %v want [barez gappy] (leaf-sorted)", rep.Gaps)
	}
	byLeaf := map[string]LeafAdoption{}
	for _, l := range rep.Leaves {
		byLeaf[l.Leaf] = l
	}
	if got := byLeaf["gappy"]; got.FanoutFiled != MinFanout-1 || got.ClearsFloor || got.Gap != 1 {
		t.Fatalf("gappy at floor-1: got %+v want filed=%d clears=false gap=1", got, MinFanout-1)
	}
	if got := byLeaf["spinez"]; got.FanoutFiled != MinFanout || !got.ClearsFloor || got.Gap != 0 {
		t.Fatalf("spinez at floor: got %+v want filed=%d clears=true gap=0", got, MinFanout)
	}
	if got := byLeaf["barez"]; got.FanoutFiled != 0 || got.Gap != MinFanout {
		t.Fatalf("barez with no markers: got %+v want filed=0 gap=%d", got, MinFanout)
	}
}

// A fanout key for a leaf outside the shipped set is an orphan (the dual gap),
// a non-fanout key is ignored, and a leaf whose name prefixes another's does not
// steal its count (longest-prefix-match wins).
func TestAdoptionOrphansAndPrefixCollision(t *testing.T) {
	markers := []string{
		"fanout-resume-qa-a", "fanout-resume-qa-b", "fanout-resume-qa-c",
		"fanout-resumewatch-qa-a", "fanout-resumewatch-qa-b", "fanout-resumewatch-qa-c",
		"fanout-ghost-qa-a", // ghost is not shipped -> orphan
		"not-a-fanout-key",  // not a fan-out marker -> ignored, not an orphan
	}
	rep := Adoption([]string{"resume", "resumewatch"}, markers)

	byLeaf := map[string]int{}
	for _, l := range rep.Leaves {
		byLeaf[l.Leaf] = l.FanoutFiled
	}
	if byLeaf["resume"] != 3 {
		t.Fatalf("resume count: got %d want 3 (must not absorb resumewatch keys)", byLeaf["resume"])
	}
	if byLeaf["resumewatch"] != 3 {
		t.Fatalf("resumewatch count: got %d want 3", byLeaf["resumewatch"])
	}
	if !reflect.DeepEqual(rep.OrphanMarkers, []string{"fanout-ghost-qa-a"}) {
		t.Fatalf("orphans: got %v want [fanout-ghost-qa-a] (the non-fanout key is not an orphan)", rep.OrphanMarkers)
	}
}

// Pure: same inputs -> byte-identical report; duplicate shipped leaves and
// duplicate marker keys both collapse.
func TestAdoptionDeterministicAndDedup(t *testing.T) {
	leaves := []string{"beta", "alpha", "beta"} // unsorted + duplicate
	markers := []string{
		"fanout-alpha-x-1", "fanout-alpha-y-2", "fanout-alpha-z-3",
		"fanout-alpha-x-1", // duplicate key -> counted once
	}
	a := Adoption(leaves, markers)
	b := Adoption(leaves, markers)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("two Adoption calls over the same input differ")
	}
	if a.ShippedLeaves != 2 {
		t.Fatalf("leaf dedup: got ShippedLeaves %d want 2", a.ShippedLeaves)
	}
	if len(a.Leaves) != 2 || a.Leaves[0].Leaf != "alpha" || a.Leaves[1].Leaf != "beta" {
		t.Fatalf("leaves not sorted+deduped: %+v", a.Leaves)
	}
	if a.Leaves[0].FanoutFiled != 3 {
		t.Fatalf("key dedup: alpha counted %d want 3 (duplicate key must not double-count)", a.Leaves[0].FanoutFiled)
	}
	if a.Schema != AdoptionSchema || a.MinFanout != MinFanout {
		t.Fatalf("report header: got schema=%q min=%d want %q/%d", a.Schema, a.MinFanout, AdoptionSchema, MinFanout)
	}
}

// Empty input is a valid, vacuously-OK zero report, not a panic.
func TestAdoptionEmpty(t *testing.T) {
	rep := Adoption(nil, nil)
	if rep.ShippedLeaves != 0 || rep.GapLeaves != 0 || !rep.OK || len(rep.Leaves) != 0 || len(rep.OrphanMarkers) != 0 {
		t.Fatalf("empty adoption not a clean zero: %+v", rep)
	}
	if rep.Schema != AdoptionSchema {
		t.Fatalf("schema missing on empty report: %q", rep.Schema)
	}
}
