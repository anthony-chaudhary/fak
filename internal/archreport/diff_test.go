package archreport

import (
	"reflect"
	"testing"
)

func TestDiffNamesEveryArchitectureChangeDeterministically(t *testing.T) {
	before := Report{Leaves: []Leaf{
		{Name: "alpha", DeclaredTier: 1, DeclaredTierName: "primitive", Dependencies: []string{"abi", "old"}, Violations: []string{"alpha -> old"}},
		{Name: "old", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
		{Name: "removed", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
	}}
	after := Report{Leaves: []Leaf{
		{Name: "new", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
		{Name: "alpha", DeclaredTier: 3, DeclaredTierName: "mechanism", Dependencies: []string{"abi", "new"}, Violations: []string{"alpha -> new"}},
		{Name: "old", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
	}}
	got := Diff(before, after)
	if !reflect.DeepEqual(got.AddedLeaves, []string{"new"}) || !reflect.DeepEqual(got.RemovedLeaves, []string{"removed"}) {
		t.Fatalf("leaves=%+v", got)
	}
	wantTier := []TierChange{{Leaf: "alpha", Before: 1, BeforeName: "primitive", After: 3, AfterName: "mechanism"}}
	if !reflect.DeepEqual(got.TierChanges, wantTier) {
		t.Fatalf("tiers=%+v", got.TierChanges)
	}
	if !reflect.DeepEqual(got.AddedEdges, []EdgeChange{{From: "alpha", To: "new"}}) || !reflect.DeepEqual(got.RemovedEdges, []EdgeChange{{From: "alpha", To: "old"}}) {
		t.Fatalf("edges=%+v", got)
	}
	if !reflect.DeepEqual(got.IntroducedViolations, []string{"alpha -> new"}) || !reflect.DeepEqual(got.ResolvedViolations, []string{"alpha -> old"}) {
		t.Fatalf("violations=%+v", got)
	}
	if got.Verdict != "regression" {
		t.Fatalf("verdict=%q", got.Verdict)
	}
	if got.Changes() != 7 {
		t.Fatalf("changes=%d", got.Changes())
	}
	first, _ := got.JSON()
	second, _ := Diff(before, after).JSON()
	if string(first) != string(second) {
		t.Fatalf("non-deterministic JSON\n%s\n%s", first, second)
	}
}

func TestDiffEmpty(t *testing.T) {
	r := Report{Leaves: []Leaf{{Name: "abi", DeclaredTier: 0}}}
	got := Diff(r, r)
	if got.Schema != DiffSchema || got.Verdict != "clean" || got.Changes() != 0 {
		t.Fatalf("%+v", got)
	}
}

func TestDiffReportsTypedDiagnosticDeltasByStableIdentity(t *testing.T) {
	before := Report{Diagnostics: []Diagnostic{
		{Kind: DiagnosticStaleTierDeclaration, Leaf: "baseline", Message: "C:/before/internal/baseline does not exist", Recovery: "remove baseline row"},
		{Kind: "parse-error", Leaf: "fixed", Message: "before parse error", Recovery: "repair syntax"},
	}}
	after := Report{Diagnostics: []Diagnostic{
		{Kind: DiagnosticStaleTierDeclaration, Leaf: "baseline", Message: "D:/after/internal/baseline does not exist", Recovery: "remove baseline row"},
		{Kind: DiagnosticStaleTierDeclaration, Leaf: "zeta", Message: "zeta missing", Recovery: "remove zeta row"},
		{Kind: "parse-error", Leaf: "alpha", Message: "alpha broken", Recovery: "repair alpha"},
	}}
	got := Diff(before, after)
	wantIntroduced := []Diagnostic{
		{Kind: "parse-error", Leaf: "alpha", Message: "alpha broken", Recovery: "repair alpha"},
		{Kind: DiagnosticStaleTierDeclaration, Leaf: "zeta", Message: "zeta missing", Recovery: "remove zeta row"},
	}
	wantResolved := []Diagnostic{{Kind: "parse-error", Leaf: "fixed", Message: "before parse error", Recovery: "repair syntax"}}
	if !reflect.DeepEqual(got.IntroducedDiagnostics, wantIntroduced) || !reflect.DeepEqual(got.ResolvedDiagnostics, wantResolved) {
		t.Fatalf("diagnostic diff=%+v", got)
	}
	if got.Verdict != "regression" || got.Changes() != 3 {
		t.Fatalf("verdict=%q changes=%d", got.Verdict, got.Changes())
	}
}

func TestDiffResolvedDiagnosticsRemainClean(t *testing.T) {
	before := Report{Diagnostics: []Diagnostic{{Kind: DiagnosticStaleTierDeclaration, Leaf: "gone"}}}
	got := Diff(before, Report{})
	if got.Verdict != "clean" || len(got.ResolvedDiagnostics) != 1 || got.Changes() != 1 {
		t.Fatalf("diff=%+v", got)
	}
}

func TestDiffDerivesDeterministicFanInChanges(t *testing.T) {
	before := Report{Leaves: []Leaf{
		{Name: "grow", Dependents: []string{"one"}},
		{Name: "grow-tie", Dependents: []string{"one"}},
		{Name: "shrink", Dependents: []string{"one", "two", "three"}},
		{Name: "removed", Dependents: []string{"one", "two"}},
	}}
	after := Report{Leaves: []Leaf{
		{Name: "grow", Dependents: []string{"one", "two", "three"}},
		{Name: "grow-tie", Dependents: []string{"one", "two", "three"}},
		{Name: "shrink", Dependents: []string{"one"}},
		{Name: "added", Dependents: []string{"one"}},
	}}
	got := Diff(before, after)
	want := []FanInChange{
		{Leaf: "grow", Before: 1, After: 3, Delta: 2},
		{Leaf: "grow-tie", Before: 1, After: 3, Delta: 2},
		{Leaf: "added", Before: 0, After: 1, Delta: 1},
		{Leaf: "removed", Before: 2, After: 0, Delta: -2},
		{Leaf: "shrink", Before: 3, After: 1, Delta: -2},
	}
	if !reflect.DeepEqual(got.FanInChanges, want) {
		t.Fatalf("fan-in changes=%+v want=%+v", got.FanInChanges, want)
	}
	if got.Changes() != 2 { // Added/removed leaves only; fan-in changes are a derived view.
		t.Fatalf("changes=%d", got.Changes())
	}
}

func TestDiffFanInChangesMatchDirectEdgeDelta(t *testing.T) {
	before := Report{Leaves: []Leaf{
		{Name: "a", Dependencies: []string{"seam"}},
		{Name: "seam", Dependents: []string{"a"}},
	}}
	after := Report{Leaves: []Leaf{
		{Name: "a", Dependencies: []string{"seam"}},
		{Name: "b", Dependencies: []string{"seam"}},
		{Name: "seam", Dependents: []string{"a", "b"}},
	}}
	got := Diff(before, after)
	if !reflect.DeepEqual(got.FanInChanges, []FanInChange{{Leaf: "seam", Before: 1, After: 2, Delta: 1}}) || !reflect.DeepEqual(got.AddedEdges, []EdgeChange{{From: "b", To: "seam"}}) {
		t.Fatalf("diff=%+v", got)
	}
}

func TestDiffDerivesDeterministicTierGapChanges(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "worse", DeclaredTier: 4, ImportFloor: 3, TierGap: 1}, {Name: "better", DeclaredTier: 4, ImportFloor: 1, TierGap: 3}, {Name: "declared", DeclaredTier: 3, ImportFloor: 2, TierGap: 1}, {Name: "removed", DeclaredTier: 4, ImportFloor: 1, TierGap: 3}}}
	after := Report{Leaves: []Leaf{{Name: "worse", DeclaredTier: 4, ImportFloor: 1, TierGap: 3}, {Name: "better", DeclaredTier: 4, ImportFloor: 3, TierGap: 1}, {Name: "declared", DeclaredTier: 4, ImportFloor: 2, TierGap: 2}, {Name: "added", DeclaredTier: 5, ImportFloor: 1, TierGap: 4}}}
	got := Diff(before, after)
	want := []TierGapChange{{Leaf: "worse", DeclaredTier: 4, BeforeFloor: 3, AfterFloor: 1, BeforeGap: 1, AfterGap: 3, Delta: 2}, {Leaf: "declared", DeclaredTier: 4, BeforeFloor: 2, AfterFloor: 2, BeforeGap: 1, AfterGap: 2, Delta: 1}, {Leaf: "better", DeclaredTier: 4, BeforeFloor: 1, AfterFloor: 3, BeforeGap: 3, AfterGap: 1, Delta: -2}}
	if !reflect.DeepEqual(got.TierGapChanges, want) {
		t.Fatalf("tier-gap changes=%+v want=%+v", got.TierGapChanges, want)
	}
	if got.Changes() != 3 {
		t.Fatalf("changes=%d", got.Changes())
	} // added, removed, and declared-tier change only.
}

func TestDiffTierGapChangeMatchesReportedFloor(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "leaf", DeclaredTier: 4, ImportFloor: 3, TierGap: 1}}}
	after := Report{Leaves: []Leaf{{Name: "leaf", DeclaredTier: 4, ImportFloor: 2, TierGap: 2}}}
	got := Diff(before, after)
	if !reflect.DeepEqual(got.TierGapChanges, []TierGapChange{{Leaf: "leaf", DeclaredTier: 4, BeforeFloor: 3, AfterFloor: 2, BeforeGap: 1, AfterGap: 2, Delta: 1}}) {
		t.Fatalf("diff=%+v", got)
	}
	if got.Changes() != 0 {
		t.Fatalf("derived tier-gap view was double-counted: %d", got.Changes())
	}
}

func TestDiffTierGapIncreaseIsRegression(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "leaf", DeclaredTier: 4, ImportFloor: 3, TierGap: 1}}}
	after := Report{Leaves: []Leaf{{Name: "leaf", DeclaredTier: 4, ImportFloor: 2, TierGap: 2}}}
	if got := Diff(before, after); got.Verdict != "regression" {
		t.Fatalf("verdict=%q diff=%+v", got.Verdict, got)
	}
}

func TestDiffTierGapImprovementRemainsClean(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "leaf", DeclaredTier: 4, ImportFloor: 2, TierGap: 2}}}
	after := Report{Leaves: []Leaf{{Name: "leaf", DeclaredTier: 4, ImportFloor: 3, TierGap: 1}}}
	if got := Diff(before, after); got.Verdict != "clean" {
		t.Fatalf("verdict=%q diff=%+v", got.Verdict, got)
	}
}
