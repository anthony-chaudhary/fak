package metrics

import (
	"strings"
	"testing"
)

// TestEdgeKindsClosed binds the closed edge vocabulary the issue names: forward,
// inverted, intra-generation, unknown-horizon — in that order, each with a human
// label. A drift here is a spec change, not a silent kind addition.
func TestEdgeKindsClosed(t *testing.T) {
	want := []EdgeKind{EdgeForward, EdgeInverted, EdgeIntra, EdgeUnknown}
	if len(EdgeKinds) != len(want) {
		t.Fatalf("EdgeKinds = %v, want %v", EdgeKinds, want)
	}
	for i, k := range want {
		if EdgeKinds[i] != k {
			t.Fatalf("EdgeKinds[%d] = %q, want %q", i, EdgeKinds[i], k)
		}
		if label := EdgeKinds[i].Label(); label == "" || label == string(k) {
			t.Fatalf("kind %q has no human label (got %q)", k, label)
		}
	}
}

// TestEdgeKindClassification pins the rank-direction rule: a farther option depending
// on a nearer one is forward, the reverse is inverted, same horizon is intra, and an
// off-vocabulary horizon fails closed to unknown. Direction is decided by horizon rank
// alone — the orthogonality guarantee, so priority never appears here.
func TestEdgeKindClassification(t *testing.T) {
	cases := []struct {
		name string
		edge DependencyEdge
		want EdgeKind
	}{
		{"future-depends-on-now", DependencyEdge{From: "opt-f", FromStream: "future", To: "seam-n", ToStream: "now"}, EdgeForward},
		{"next-depends-on-now", DependencyEdge{From: "opt-x", FromStream: "next", To: "seam-n", ToStream: "now"}, EdgeForward},
		{"now-depends-on-future", DependencyEdge{From: "opt-n", FromStream: "now", To: "opt-f", ToStream: "future"}, EdgeInverted},
		{"second-next-depends-on-future", DependencyEdge{From: "opt-s", FromStream: "second-next", To: "opt-f", ToStream: "future"}, EdgeInverted},
		{"same-horizon", DependencyEdge{From: "a", FromStream: "now", To: "b", ToStream: "now"}, EdgeIntra},
		{"bad-from", DependencyEdge{From: "a", FromStream: "someday", To: "b", ToStream: "now"}, EdgeUnknown},
		{"bad-to", DependencyEdge{From: "a", FromStream: "now", To: "b", ToStream: "eventually"}, EdgeUnknown},
	}
	for _, tc := range cases {
		if got := tc.edge.Kind(); got != tc.want {
			t.Fatalf("%s: Kind() = %q, want %q", tc.name, got, tc.want)
		}
	}

	// CrossesGeneration is true only for forward/inverted, false for intra/unknown.
	if !(DependencyEdge{FromStream: "future", ToStream: "now"}).CrossesGeneration() {
		t.Fatal("forward edge should cross a generation boundary")
	}
	if (DependencyEdge{FromStream: "now", ToStream: "now"}).CrossesGeneration() {
		t.Fatal("intra-generation edge must not count as crossing")
	}
	if (DependencyEdge{FromStream: "now", ToStream: "someday"}).CrossesGeneration() {
		t.Fatal("unknown-horizon edge must not count as crossing")
	}
}

// TestDemotionCriterionOnlyForInversions proves the demotion criterion is named for an
// inverted edge (naming both promotion and demotion evidence) and empty for every other
// kind — the proof bar for gen work is a dependency edge WITH demotion criteria.
func TestDemotionCriterionOnlyForInversions(t *testing.T) {
	inv := DependencyEdge{From: "near", FromStream: "now", To: "far", ToStream: "future"}
	crit := inv.DemotionCriterion()
	if crit == "" {
		t.Fatal("inverted edge must carry a demotion criterion")
	}
	for _, kw := range []string{"promote", "demote", "gen/now", "gen/future"} {
		if !strings.Contains(crit, kw) {
			t.Fatalf("demotion criterion missing %q: %q", kw, crit)
		}
	}
	for _, e := range []DependencyEdge{
		{FromStream: "future", ToStream: "now"}, // forward
		{FromStream: "now", ToStream: "now"},    // intra
		{FromStream: "now", ToStream: "bad"},    // unknown
	} {
		if got := e.DemotionCriterion(); got != "" {
			t.Fatalf("non-inverted edge %+v should have no criterion, got %q", e, got)
		}
	}
}

// TestClassifyDeterministicAndGrouped checks Classify tags every edge, groups by kind
// (EdgeKinds order), and is stable — the same edges always render identically.
func TestClassifyDeterministicAndGrouped(t *testing.T) {
	edges := []DependencyEdge{
		{From: "n1", FromStream: "now", To: "f1", ToStream: "future"},              // inverted
		{From: "f2", FromStream: "future", To: "n2", ToStream: "now"},              // forward
		{From: "s1", FromStream: "second-next", To: "s2", ToStream: "second-next"}, // intra
		{From: "b1", FromStream: "now", To: "b2", ToStream: "nope"},                // unknown
	}
	rep := Classify(edges)
	if len(rep.Edges) != len(edges) {
		t.Fatalf("Classify dropped edges: got %d, want %d", len(rep.Edges), len(edges))
	}
	// Grouped by kind in EdgeKinds order: forward first, then inverted, intra, unknown.
	wantKinds := []EdgeKind{EdgeForward, EdgeInverted, EdgeIntra, EdgeUnknown}
	for i, k := range wantKinds {
		if rep.Edges[i].Kind != k {
			t.Fatalf("Edges[%d].Kind = %q, want %q (grouping drifted)", i, rep.Edges[i].Kind, k)
		}
	}

	counts := rep.CountByKind()
	for _, k := range EdgeKinds {
		if counts[k] != 1 {
			t.Fatalf("CountByKind[%q] = %d, want 1", k, counts[k])
		}
	}

	if got := rep.Inversions(); len(got) != 1 || got[0].From != "n1" {
		t.Fatalf("Inversions() = %+v, want the single n1->f1 edge", got)
	}
}

// TestDependencyReportRender proves the report renders the orthogonality header, a
// count line, every edge, and a demotion criterion for each inversion — and is
// deterministic.
func TestDependencyReportRender(t *testing.T) {
	rep := Classify([]DependencyEdge{
		{From: "near-opt", FromStream: "now", To: "far-opt", ToStream: "future", Note: "blocked on unmatured seam"},
		{From: "future-bet", FromStream: "future", To: "now-seam", ToStream: "now"},
	})
	out := rep.Render()

	if !strings.Contains(out, OrthogonalityNote) {
		t.Fatalf("render missing orthogonality note:\n%s", out)
	}
	for _, kw := range []string{"priority", "shared trunk", "feature gate"} {
		if !strings.Contains(strings.ToLower(out), kw) {
			t.Fatalf("orthogonality note does not name %q:\n%s", kw, out)
		}
	}
	// Every edge kind label present in the count line.
	for _, k := range EdgeKinds {
		if !strings.Contains(out, string(k)) {
			t.Fatalf("render missing kind %q in counts:\n%s", k, out)
		}
	}
	// Endpoints rendered with their horizon labels.
	for _, kw := range []string{"gen/now:near-opt", "gen/future:far-opt", "blocked on unmatured seam"} {
		if !strings.Contains(out, kw) {
			t.Fatalf("render missing %q:\n%s", kw, out)
		}
	}
	// The inversion surfaces its demotion criterion.
	if !strings.Contains(out, "inversions (promotion blockers)") {
		t.Fatalf("render missing inversions section:\n%s", out)
	}
	if !strings.Contains(out, "demote near-opt to gen/future") {
		t.Fatalf("render missing demotion criterion:\n%s", out)
	}
	if rep.Render() != out {
		t.Fatal("Render is not deterministic")
	}
}

// TestEmptyReportRender checks the empty case reads honestly: zero counts and a
// "(none)" edge line, never a blank or a panic.
func TestEmptyReportRender(t *testing.T) {
	out := Classify(nil).Render()
	if !strings.Contains(out, "(none)") {
		t.Fatalf("empty report should render (none):\n%s", out)
	}
	if !strings.Contains(out, "forward=0") {
		t.Fatalf("empty report should render zero counts:\n%s", out)
	}
}
