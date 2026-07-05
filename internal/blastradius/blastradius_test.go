package blastradius

import (
	"reflect"
	"testing"
)

// graph models a small intra-module import chain, keyed by repo-relative package dir
// (the namespace the shell's go-list fold produces): c imports b imports a; the d/x
// pair is a disjoint island. edges[p] is what p imports, so Select walks the reverse
// (importers) direction to find a's dependents.
func graph() map[string][]string {
	return map[string][]string{
		"internal/b": {"internal/a"},
		"internal/c": {"internal/b"},
		"internal/d": {"internal/x"},
	}
}

func TestEstimateHoldsAffectedLeasesExcludesDisjoint(t *testing.T) {
	leases := []Lease{
		{Lane: "lane-b", TreeGlobs: []string{"internal/b/**"}},           // imports a (dependent) -> held
		{Lane: "lane-a-file", TreeGlobs: []string{"internal/a/impl.go"}}, // the broken tree itself -> held
		{Lane: "lane-d", TreeGlobs: []string{"internal/d/**"}},           // disjoint island -> run
		{Lane: "lane-x", TreeGlobs: []string{"internal/x/**"}},           // a imports nothing; x is unrelated -> run
	}
	issues := []Issue{
		{ID: "1001", Paths: []string{"internal/c/new.go"}}, // c transitively imports a -> held
		{ID: "1002", Paths: []string{"internal/z/new.go"}}, // untouched tree -> run
	}

	got := Estimate(graph(), "internal/a", leases, issues)

	// The dependency blast radius of a is {a, b, c}, sorted.
	wantRadius := []string{"internal/a", "internal/b", "internal/c"}
	if !reflect.DeepEqual(got.Radius, wantRadius) {
		t.Fatalf("radius = %v, want %v", got.Radius, wantRadius)
	}

	// Exactly the intersecting leases are held, in input order, with the matched trees.
	wantLeases := []AffectedLease{
		{Lane: "lane-b", TreeGlobs: []string{"internal/b/**"}, Matched: []string{"internal/b"}},
		{Lane: "lane-a-file", TreeGlobs: []string{"internal/a/impl.go"}, Matched: []string{"internal/a"}},
	}
	if !reflect.DeepEqual(got.Leases, wantLeases) {
		t.Fatalf("held leases = %#v, want %#v", got.Leases, wantLeases)
	}
	if want := []string{"lane-d", "lane-x"}; !reflect.DeepEqual(got.ExcludedLeases, want) {
		t.Fatalf("excluded leases = %v, want %v", got.ExcludedLeases, want)
	}

	// The queued-issue intersection behaves the same.
	wantIssues := []AffectedIssue{
		{ID: "1001", Paths: []string{"internal/c/new.go"}, Matched: []string{"internal/c"}},
	}
	if !reflect.DeepEqual(got.Issues, wantIssues) {
		t.Fatalf("held issues = %#v, want %#v", got.Issues, wantIssues)
	}
	if want := []string{"1002"}; !reflect.DeepEqual(got.ExcludedIssues, want) {
		t.Fatalf("excluded issues = %v, want %v", got.ExcludedIssues, want)
	}
}

// A lease covering a broad ancestor dir (e.g. all of internal/**) intersects every
// radius tree, so its Matched lists them all — the operator can see why the hold is wide.
func TestEstimateMatchedListsEveryIntersectedRadiusTree(t *testing.T) {
	got := Estimate(graph(), "internal/a", []Lease{{Lane: "wide", TreeGlobs: []string{"internal/**"}}}, nil)
	if len(got.Leases) != 1 {
		t.Fatalf("expected the wide lease held, got %#v", got.Leases)
	}
	want := []string{"internal/a", "internal/b", "internal/c"}
	if !reflect.DeepEqual(got.Leases[0].Matched, want) {
		t.Fatalf("matched = %v, want the full radius %v", got.Leases[0].Matched, want)
	}
}

// A broken key absent from the graph selects just itself: only a lease that directly
// touches that tree is held (the conservative floor), never a spurious dependent.
func TestEstimateBrokenNotInGraphSelectsItselfOnly(t *testing.T) {
	leases := []Lease{
		{Lane: "on-broken", TreeGlobs: []string{"internal/orphan/**"}},
		{Lane: "elsewhere", TreeGlobs: []string{"internal/b/**"}},
	}
	got := Estimate(graph(), "internal/orphan", leases, nil)
	if want := []string{"internal/orphan"}; !reflect.DeepEqual(got.Radius, want) {
		t.Fatalf("radius = %v, want %v", got.Radius, want)
	}
	if len(got.Leases) != 1 || got.Leases[0].Lane != "on-broken" {
		t.Fatalf("held = %#v, want only lane on-broken", got.Leases)
	}
	if want := []string{"elsewhere"}; !reflect.DeepEqual(got.ExcludedLeases, want) {
		t.Fatalf("excluded = %v, want %v", got.ExcludedLeases, want)
	}
}

// Empty inputs yield an empty-but-non-nil affected set: the JSON renders [] arrays,
// not null, and nothing is spuriously held.
func TestEstimateEmptyInputsRenderEmptyArrays(t *testing.T) {
	got := Estimate(graph(), "internal/a", nil, nil)
	for _, tc := range []struct {
		name string
		v    any
	}{
		{"leases", got.Leases},
		{"issues", got.Issues},
		{"excluded_leases", got.ExcludedLeases},
		{"excluded_issues", got.ExcludedIssues},
	} {
		if reflect.ValueOf(tc.v).IsNil() {
			t.Errorf("%s is nil; want a non-nil empty slice for stable JSON", tc.name)
		}
		if reflect.ValueOf(tc.v).Len() != 0 {
			t.Errorf("%s = %v, want empty", tc.name, tc.v)
		}
	}
}

// A lease whose declared globs are all invalid (empty / bare star / escaping) never
// matches — an unparseable tree must not silently hold everything.
func TestEstimateInvalidLeaseTreeNeverHeld(t *testing.T) {
	leases := []Lease{
		{Lane: "empty", TreeGlobs: []string{"", "   "}},
		{Lane: "star", TreeGlobs: []string{"**"}},
		{Lane: "escape", TreeGlobs: []string{"../outside"}},
	}
	got := Estimate(graph(), "internal/a", leases, nil)
	if len(got.Leases) != 0 {
		t.Fatalf("invalid-tree leases held: %#v", got.Leases)
	}
	if want := []string{"empty", "star", "escape"}; !reflect.DeepEqual(got.ExcludedLeases, want) {
		t.Fatalf("excluded = %v, want %v", got.ExcludedLeases, want)
	}
}

// Pure: the same inputs always produce an identical AffectedSet.
func TestEstimateDeterministic(t *testing.T) {
	leases := []Lease{
		{Lane: "b", TreeGlobs: []string{"internal/b/**"}},
		{Lane: "d", TreeGlobs: []string{"internal/d/**"}},
	}
	issues := []Issue{{ID: "1", Paths: []string{"internal/a/x.go"}}}
	a := Estimate(graph(), "internal/a", leases, issues)
	b := Estimate(graph(), "internal/a", leases, issues)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Estimate not deterministic:\n%#v\n%#v", a, b)
	}
}
