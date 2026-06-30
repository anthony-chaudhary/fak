package superloop

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// TestRegistryScorecardRefsReal is the no-drift witness: every scorecard member in
// the registry must reference a REAL control-pane card key, so a super loop can never
// send an operator at a scorecard that does not exist. It re-derives the valid keys
// from scorecardpane.Cards (the same source the control pane folds) and fails if a
// member drifts away from it.
func TestRegistryScorecardRefsReal(t *testing.T) {
	valid := map[string]bool{}
	for _, c := range scorecardpane.Cards {
		valid[c.Key] = true
	}
	for _, ref := range ScorecardRefs() {
		if !valid[ref] {
			t.Errorf("scorecard member %q is not a real control-pane card key (drifted from scorecardpane.Cards)", ref)
		}
	}
}

// TestRegistryWellFormed checks the structural invariants the shell relies on: every
// super loop has a name, a title, at least one member, and every member has a kind +
// ref.
func TestRegistryWellFormed(t *testing.T) {
	for _, s := range Registry() {
		if s.Name == "" || s.Title == "" {
			t.Errorf("super loop %+v missing name/title", s)
		}
		if len(s.Members) == 0 {
			t.Errorf("super loop %q has no members", s.Name)
		}
		for _, m := range s.Members {
			if m.Kind == "" || m.Ref == "" {
				t.Errorf("super loop %q has a malformed member %+v", s.Name, m)
			}
		}
		if got, ok := Lookup(s.Name); !ok || got.Name != s.Name {
			t.Errorf("Lookup(%q) did not round-trip", s.Name)
		}
	}
}

func TestManageBenchmarksBridgesToBenchLoop(t *testing.T) {
	s, ok := Lookup("manage-benchmarks")
	if !ok {
		t.Fatal("manage-benchmarks not registered")
	}
	refs := map[MemberKind]map[string]bool{}
	for _, m := range s.Members {
		if refs[m.Kind] == nil {
			refs[m.Kind] = map[string]bool{}
		}
		refs[m.Kind][m.Ref] = true
	}
	if !refs[KindScorecard]["bench_dx"] {
		t.Fatal("manage-benchmarks must include the benchmark-DX scorecard")
	}
	if !refs[KindLoop]["nightrun"] {
		t.Fatal("manage-benchmarks must include the nightrun collection loop")
	}
	if !refs[KindSurface]["fak bench-loop status"] {
		t.Fatal("manage-benchmarks must descend to the concrete bench-loop status surface")
	}

	rep := Walk(s, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "bench_dx"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindLoop, Ref: "nightrun"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindSurface, Ref: "fak bench-loop status"}, Container: true, Detail: "domain fold"},
	})
	if len(rep.Worklist) != 1 {
		t.Fatalf("surface descend pointer should remain in the worklist, got %d items", len(rep.Worklist))
	}
	if got := rep.Worklist[0].Action; got != "enter `fak bench-loop status`" {
		t.Fatalf("surface action = %q", got)
	}
}

// TestClassifySuperVsNormal is the differentiation witness: a registered super loop
// satisfies all five properties; a normal leaf loop satisfies none of the structural
// ones and is classified NOT super, with the reason naming the first failing rung.
func TestClassifySuperVsNormal(t *testing.T) {
	s, ok := Lookup("improve-quality")
	if !ok {
		t.Fatal("improve-quality not registered")
	}
	v := Classify(FactsFor(s))
	if !v.IsSuper {
		t.Fatalf("registered super loop classified as not-super: %s", v.Reason)
	}
	if len(v.Properties) != 5 {
		t.Fatalf("want 5 properties, got %d", len(v.Properties))
	}
	for _, p := range v.Properties {
		if !p.Holds {
			t.Errorf("super loop property %q does not hold: got=%v want=%v", p.Name, p.Got, p.Want)
		}
	}

	leaf := Classify(LeafFacts("dispatch-tick"))
	if leaf.IsSuper {
		t.Error("a normal leaf loop must not classify as a super loop")
	}
	// The first failing rung for a bare leaf is has_members.
	if want := "has_members"; !containsProp(leaf, want, false) {
		t.Errorf("leaf should fail %q", want)
	}
}

// TestClassifyPartialNotSuper proves the AND-gate: a loop that walks members and reads
// them first but does NOT select worst-first is still not a super loop — every rung
// must hold.
func TestClassifyPartialNotSuper(t *testing.T) {
	partial := LoopFacts{
		Name:              "half-super",
		MemberCount:       3,
		WalksFirst:        true,
		SelectsWorstFirst: false, // the missing rung
		ExitsOnAggregate:  true,
		ActsAtOwnAltitude: false,
	}
	v := Classify(partial)
	if v.IsSuper {
		t.Fatal("a loop missing worst-first selection must not classify as super")
	}
	if !containsProp(v, "selects_worst_first", false) {
		t.Errorf("reason should name selects_worst_first; got %q", v.Reason)
	}
}

// TestWalkWorstFirst checks the SELECT step: dark/unmeasured leaves rank first, then
// debt descending, containers in the descend band, and a clean measured leaf is
// dropped from the worklist. Aggregate debt sums only measured leaves (not the
// container).
func TestWalkWorstFirst(t *testing.T) {
	s := Super{
		Name:  "t",
		Title: "test",
		Floor: 0,
		Members: []Member{
			{Kind: KindScorecard, Ref: "a"},
			{Kind: KindScorecard, Ref: "b"},
			{Kind: KindScorecard, Ref: "clean"},
			{Kind: KindLoop, Ref: "darkloop"},
			{Kind: KindGarden, Ref: "garden"},
		},
	}
	statuses := []MemberStatus{
		{Member: s.Members[0], Debt: 10, Measured: true},         // debt 10
		{Member: s.Members[1], Debt: 600, Measured: true},        // debt 600 (heaviest)
		{Member: s.Members[2], Debt: 0, Measured: true},          // clean -> dropped
		{Member: s.Members[3], Dark: true, Measured: true},       // dark -> most urgent
		{Member: s.Members[4], Container: true, Measured: false}, // descend pointer
	}
	rep := Walk(s, statuses)

	if rep.TotalDebt != 610 {
		t.Errorf("total debt: want 610 (10+600, container excluded), got %d", rep.TotalDebt)
	}
	if rep.Dark != 1 {
		t.Errorf("dark count: want 1, got %d", rep.Dark)
	}
	if rep.Unmeasured != 0 {
		t.Errorf("unmeasured: want 0 (container is not counted), got %d", rep.Unmeasured)
	}
	if rep.Satisfied {
		t.Error("must not be satisfied with debt and a dark loop")
	}
	// Worklist excludes the clean member: 5 members - 1 clean = 4.
	if len(rep.Worklist) != 4 {
		t.Fatalf("worklist len: want 4, got %d", len(rep.Worklist))
	}
	// Order: dark leaf, then debt 600, then debt 10, then container.
	wantOrder := []string{"darkloop", "b", "a", "garden"}
	for i, ref := range wantOrder {
		if rep.Worklist[i].Member.Ref != ref {
			t.Errorf("worklist[%d]: want %q, got %q", i, ref, rep.Worklist[i].Member.Ref)
		}
		if rep.Worklist[i].Rank != i+1 {
			t.Errorf("worklist[%d] rank: want %d, got %d", i, i+1, rep.Worklist[i].Rank)
		}
	}
	if rep.Finding != "superloop_dark" {
		t.Errorf("finding: want superloop_dark, got %q", rep.Finding)
	}
}

// TestWalkSatisfied: all leaves measured-clean and live, no container in the way ->
// satisfied, verdict OK.
func TestWalkSatisfied(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "a"}, {Kind: KindScorecard, Ref: "b"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Debt: 0, Measured: true},
		{Member: s.Members[1], Debt: 0, Measured: true},
	})
	if !rep.Satisfied {
		t.Errorf("want satisfied; reason=%q", rep.Reason)
	}
	if rep.Verdict != "OK" || rep.Finding != "superloop_satisfied" {
		t.Errorf("want OK/superloop_satisfied, got %s/%s", rep.Verdict, rep.Finding)
	}
	if len(rep.Worklist) != 0 {
		t.Errorf("clean walk should have an empty worklist, got %d", len(rep.Worklist))
	}
}

// TestWalkUnmeasuredBlocks: an unreadable leaf can never read as clean — it blocks
// Satisfied and raises the unmeasured finding even at zero measured debt.
func TestWalkUnmeasuredBlocks(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "a"},
	}}
	rep := Walk(s, []MemberStatus{{Member: s.Members[0], Measured: false}})
	if rep.Satisfied {
		t.Error("an unmeasured member must block satisfied")
	}
	if rep.Finding != "superloop_unmeasured" {
		t.Errorf("want superloop_unmeasured, got %q", rep.Finding)
	}
	if rep.Unmeasured != 1 {
		t.Errorf("want 1 unmeasured, got %d", rep.Unmeasured)
	}
}

// TestWalkSatisfiedWithContainer pins the load-bearing container rule: a clean,
// all-measured walk that ALSO carries a container (a descend pointer) is still
// SATISFIED — the container is excluded from the unmeasured tally, so it cannot flip
// a clean intent to permanently-unsatisfied, yet it is still surfaced for descent. A
// regression that counted the container as unmeasured would red every container-
// bearing intent (improve-quality, improve-loops, manage-benchmarks all carry one)
// while the rest of the suite stayed green.
func TestWalkSatisfiedWithContainer(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "a"},
		{Kind: KindGarden, Ref: "garden"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Debt: 0, Measured: true},
		{Member: s.Members[1], Container: true, Measured: false},
	})
	if !rep.Satisfied {
		t.Errorf("a clean walk carrying a container must be satisfied; reason=%q unmeasured=%d", rep.Reason, rep.Unmeasured)
	}
	if rep.Unmeasured != 0 {
		t.Errorf("a container must not count as unmeasured, got %d", rep.Unmeasured)
	}
	if rep.Verdict != "OK" {
		t.Errorf("want OK, got %s", rep.Verdict)
	}
	// The container is still surfaced as a descend pointer even on a satisfied walk.
	if len(rep.Worklist) != 1 || rep.Worklist[0].Member.Ref != "garden" {
		t.Errorf("container should remain a descend pointer in the worklist, got %+v", rep.Worklist)
	}
}

// TestWalkUnmeasuredBeatsDark pins walkVerdict's precedence: when a walk has BOTH an
// unmeasured leaf and a dark leaf, the unmeasured finding wins (a status we could not
// even read is more conservative than a known-dark loop). If the dark branch were
// reordered above unmeasured, this would silently flip and the unmeasured-only /
// dark-only tests would both still pass.
func TestWalkUnmeasuredBeatsDark(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "unread"},
		{Kind: KindLoop, Ref: "darkloop"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: false},
		{Member: s.Members[1], Dark: true, Measured: true},
	})
	if rep.Finding != "superloop_unmeasured" {
		t.Errorf("unmeasured must take precedence over dark; got finding %q", rep.Finding)
	}
	if rep.Verdict != "ACTION" {
		t.Errorf("want ACTION, got %s", rep.Verdict)
	}
	if rep.Unmeasured != 1 || rep.Dark != 1 {
		t.Errorf("want unmeasured=1 dark=1, got unmeasured=%d dark=%d", rep.Unmeasured, rep.Dark)
	}
}

func containsProp(v Verdict, name string, holds bool) bool {
	for _, p := range v.Properties {
		if p.Name == name {
			return p.Holds == holds
		}
	}
	return false
}
