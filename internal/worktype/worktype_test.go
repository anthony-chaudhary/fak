package worktype

import "testing"

// TestOngoingPredicate pins the one branch the reports depend on: the two
// optimization programs are ongoing, a discrete epic and an unknown class are not.
func TestOngoingPredicate(t *testing.T) {
	cases := []struct {
		c    Class
		want bool
	}{
		{KernelOptimization, true},
		{CacheOptimization, true},
		{DiscreteEpic, false},
		{Class("not-a-real-class"), false},
	}
	for _, tc := range cases {
		if got := tc.c.Ongoing(); got != tc.want {
			t.Errorf("Class(%q).Ongoing() = %v, want %v", tc.c, got, tc.want)
		}
	}
}

// TestClassifyEpicDeclaredVsDefault proves the declared epics route to their program
// and an undeclared epic falls through to the discrete default — the conservative
// behavior that keeps a newly-tracked epic on a completion % until consciously promoted.
func TestClassifyEpicDeclaredVsDefault(t *testing.T) {
	if got := ClassifyEpic(1010); got != KernelOptimization {
		t.Errorf("ClassifyEpic(1010) = %q, want kernel-optimization", got)
	}
	if got := ClassifyEpic(1301); got != CacheOptimization {
		t.Errorf("ClassifyEpic(1301) = %q, want cache-optimization", got)
	}
	if got := ClassifyEpic(1351); got != CacheOptimization {
		t.Errorf("ClassifyEpic(1351) = %q, want cache-optimization", got)
	}
	// An epic never declared is a discrete deliverable, not a silent program.
	if got := ClassifyEpic(999999); got != DiscreteEpic {
		t.Errorf("ClassifyEpic(undeclared) = %q, want discrete-epic", got)
	}
}

// TestDeclaredEpicsAreAllOngoing is the load-bearing invariant: every epic in the
// declared map MUST classify as an ongoing program. A discrete epic in the map would
// be a contradiction — the map's only purpose is to name the programs.
func TestDeclaredEpicsAreAllOngoing(t *testing.T) {
	declared := DeclaredEpics()
	if len(declared) == 0 {
		t.Fatal("no epics declared as ongoing programs — the split has nothing to act on")
	}
	for _, n := range declared {
		if c := ClassifyEpic(n); !c.Ongoing() {
			t.Errorf("declared epic #%d classifies as %q, which is not ongoing — the map must only name programs", n, c)
		}
	}
}

// TestProgramsHaveMetadata ensures every ongoing-program class in Programs resolves
// to a registry entry with a track label and an operating doc, so the report can
// always point an operator at the program's spine.
func TestProgramsHaveMetadata(t *testing.T) {
	if len(Programs) == 0 {
		t.Fatal("no ongoing programs declared")
	}
	for _, c := range Programs {
		if !c.Ongoing() {
			t.Errorf("Programs contains %q, which is not an ongoing class", c)
		}
		p, ok := ProgramFor(c)
		if !ok {
			t.Errorf("ProgramFor(%q) not found — Programs and the registry are out of sync", c)
			continue
		}
		if p.OperatingDoc == "" {
			t.Errorf("program %q has no operating doc", c)
		}
		if p.Class != c {
			t.Errorf("program %q registry entry has Class %q", c, p.Class)
		}
	}
	// A class that is NOT an ongoing program has no program metadata.
	if _, ok := ProgramFor(DiscreteEpic); ok {
		t.Error("ProgramFor(DiscreteEpic) returned ok=true — a discrete epic is not a program")
	}
}

// TestDefinitionsNonEmpty guards that every closed-vocabulary class carries the
// written definition the disambiguation discipline requires.
func TestDefinitionsNonEmpty(t *testing.T) {
	for _, c := range []Class{KernelOptimization, CacheOptimization, DiscreteEpic} {
		if c.Definition() == "" || c.Definition() == "unknown work class" {
			t.Errorf("class %q has no definition", c)
		}
		if c.Label() == "" {
			t.Errorf("class %q has no label", c)
		}
	}
}
