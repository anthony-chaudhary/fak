package archreport

import (
	"bytes"
	"reflect"
	"testing"
)

func TestDiffLateralArticulationPointChangesPreserveIdentityAndImpact(t *testing.T) {
	introduced := LateralArticulationPoint{Tier: 2, TierName: "foundation-composite", Name: "new", Fragments: [][]string{{"a"}, {"b", "c"}}, FragmentCount: 2, CouplingPairs: 2}
	resolved := LateralArticulationPoint{Tier: 2, TierName: "foundation-composite", Name: "gone", FragmentCount: 2, CouplingPairs: 1}
	beforeStable := LateralArticulationPoint{Tier: 3, TierName: "mechanism", Name: "seam", Fragments: [][]string{{"x"}, {"y"}}, FragmentCount: 2, CouplingPairs: 1}
	afterStable := LateralArticulationPoint{Tier: 3, TierName: "mechanism", Name: "seam", Fragments: [][]string{{"x", "z"}, {"y", "q"}}, FragmentCount: 2, CouplingPairs: 4}
	diff := Diff(Report{LateralArticulationPoints: []LateralArticulationPoint{resolved, beforeStable}}, Report{LateralArticulationPoints: []LateralArticulationPoint{introduced, afterStable}})
	if !reflect.DeepEqual(diff.IntroducedLateralArticulationPoints, []LateralArticulationPoint{introduced}) || !reflect.DeepEqual(diff.ResolvedLateralArticulationPoints, []LateralArticulationPoint{resolved}) {
		t.Fatalf("diff=%+v", diff)
	}
	want := LateralArticulationPointChange{Tier: 3, TierName: "mechanism", Name: "seam", BeforeFragments: beforeStable.Fragments, AfterFragments: afterStable.Fragments, BeforeFragmentCount: 2, AfterFragmentCount: 2, BeforeCouplingPairs: 1, AfterCouplingPairs: 4, Delta: 3}
	if !reflect.DeepEqual(diff.LateralArticulationPointChanges, []LateralArticulationPointChange{want}) || diff.Verdict != "regression" || diff.Changes() != 0 {
		t.Fatalf("diff=%+v", diff)
	}
}
func TestDiffResolvedAndDecreasedLateralArticulationPointsRemainClean(t *testing.T) {
	before := Report{LateralArticulationPoints: []LateralArticulationPoint{{Tier: 2, Name: "seam", CouplingPairs: 4, FragmentCount: 2, Fragments: [][]string{{"a", "b"}, {"c", "d"}}}, {Tier: 2, Name: "gone", CouplingPairs: 1}}}
	after := Report{LateralArticulationPoints: []LateralArticulationPoint{{Tier: 2, Name: "seam", CouplingPairs: 1, FragmentCount: 2, Fragments: [][]string{{"a"}, {"c"}}}}}
	diff := Diff(before, after)
	if diff.Verdict != "clean" || len(diff.ResolvedLateralArticulationPoints) != 1 || len(diff.LateralArticulationPointChanges) != 1 || diff.LateralArticulationPointChanges[0].Delta != -3 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffLateralBridgeChangesPreserveIdentityAndImpact(t *testing.T) {
	introduced := LateralBridge{Tier: 2, TierName: "foundation-composite", Left: "a", Right: "b", CouplingPairs: 6, LeftSide: []string{"a", "x"}, RightSide: []string{"b", "y", "z"}}
	resolved := LateralBridge{Tier: 2, TierName: "foundation-composite", Left: "c", Right: "d", CouplingPairs: 2}
	beforeStable := LateralBridge{Tier: 3, TierName: "mechanism", Left: "m", Right: "n", CouplingPairs: 2, LeftSide: []string{"m"}, RightSide: []string{"n", "o"}}
	afterStable := LateralBridge{Tier: 3, TierName: "mechanism", Left: "m", Right: "n", CouplingPairs: 6, LeftSide: []string{"m", "p"}, RightSide: []string{"n", "o", "q"}}
	diff := Diff(Report{LateralBridges: []LateralBridge{resolved, beforeStable}}, Report{LateralBridges: []LateralBridge{introduced, afterStable}})
	if !reflect.DeepEqual(diff.IntroducedLateralBridges, []LateralBridge{introduced}) || !reflect.DeepEqual(diff.ResolvedLateralBridges, []LateralBridge{resolved}) {
		t.Fatalf("diff=%+v", diff)
	}
	wantChange := LateralBridgeChange{Tier: 3, TierName: "mechanism", Left: "m", Right: "n", BeforeCouplingPairs: 2, AfterCouplingPairs: 6, Delta: 4, BeforeLeftSide: beforeStable.LeftSide, BeforeRightSide: beforeStable.RightSide, AfterLeftSide: afterStable.LeftSide, AfterRightSide: afterStable.RightSide}
	if !reflect.DeepEqual(diff.LateralBridgeChanges, []LateralBridgeChange{wantChange}) || diff.Verdict != "regression" || diff.Changes() != 0 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffResolvedAndDecreasedLateralBridgesRemainClean(t *testing.T) {
	before := Report{LateralBridges: []LateralBridge{{Tier: 2, Left: "a", Right: "b", CouplingPairs: 6}, {Tier: 2, Left: "c", Right: "d", CouplingPairs: 2}}}
	after := Report{LateralBridges: []LateralBridge{{Tier: 2, Left: "a", Right: "b", CouplingPairs: 2}}}
	diff := Diff(before, after)
	if diff.Verdict != "clean" || len(diff.ResolvedLateralBridges) != 1 || len(diff.LateralBridgeChanges) != 1 || diff.LateralBridgeChanges[0].Delta != -4 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffLateralCouplingPairsRepresentMergeAndSplit(t *testing.T) {
	before := Report{LateralComponents: []LateralComponent{
		{Tier: 2, TierName: "foundation-composite", Members: []string{"a", "b"}},
		{Tier: 2, TierName: "foundation-composite", Members: []string{"c", "d"}},
		{Tier: 3, TierName: "mechanism", Members: []string{"x", "y", "z"}},
	}}
	after := Report{LateralComponents: []LateralComponent{
		{Tier: 2, TierName: "foundation-composite", Members: []string{"a", "b", "c", "d"}},
		{Tier: 3, TierName: "mechanism", Members: []string{"x", "y"}},
	}}
	diff := Diff(before, after)
	wantIntroduced := []LateralCoupling{
		{Tier: 2, TierName: "foundation-composite", Left: "a", Right: "c"},
		{Tier: 2, TierName: "foundation-composite", Left: "a", Right: "d"},
		{Tier: 2, TierName: "foundation-composite", Left: "b", Right: "c"},
		{Tier: 2, TierName: "foundation-composite", Left: "b", Right: "d"},
	}
	wantResolved := []LateralCoupling{{Tier: 3, TierName: "mechanism", Left: "x", Right: "z"}, {Tier: 3, TierName: "mechanism", Left: "y", Right: "z"}}
	if !reflect.DeepEqual(diff.IntroducedLateralCouplings, wantIntroduced) || !reflect.DeepEqual(diff.ResolvedLateralCouplings, wantResolved) {
		t.Fatalf("introduced=%+v resolved=%+v", diff.IntroducedLateralCouplings, diff.ResolvedLateralCouplings)
	}
	if diff.Verdict != "regression" || diff.Changes() != 0 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffResolvedLateralCouplingsRemainClean(t *testing.T) {
	before := Report{LateralComponents: []LateralComponent{{Tier: 2, TierName: "foundation-composite", Members: []string{"a", "b", "c"}}}}
	after := Report{LateralComponents: []LateralComponent{{Tier: 2, TierName: "foundation-composite", Members: []string{"a", "b"}}}}
	diff := Diff(before, after)
	if diff.Verdict != "clean" || len(diff.IntroducedLateralCouplings) != 0 || len(diff.ResolvedLateralCouplings) != 2 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestTypedEdgeDiffKeepsCompatibilityProjection(t *testing.T) {
	rootward := ArchitectureEdge{From: "high", FromTier: 3, FromTierName: "mechanism", To: "low", ToTier: 1, ToTierName: "primitive", TierDelta: -2, Direction: "rootward"}
	lateral := ArchitectureEdge{From: "peer", FromTier: 2, FromTierName: "foundation-composite", To: "side", ToTier: 2, ToTierName: "foundation-composite", TierDelta: 0, Direction: "lateral"}
	upward := ArchitectureEdge{From: "low", FromTier: 1, FromTierName: "primitive", To: "high", ToTier: 3, ToTierName: "mechanism", TierDelta: 2, Direction: "upward"}
	resolved := ArchitectureEdge{From: "old", FromTier: 2, FromTierName: "foundation-composite", To: "gone", ToTier: 2, ToTierName: "foundation-composite", Direction: "lateral"}
	before := Report{Leaves: []Leaf{{Name: "old", Dependencies: []string{"gone"}}}, Edges: []ArchitectureEdge{resolved}}
	after := Report{Leaves: []Leaf{{Name: "high", Dependencies: []string{"low"}}, {Name: "low", Dependencies: []string{"high"}}, {Name: "peer", Dependencies: []string{"side"}}}, Edges: []ArchitectureEdge{upward, lateral, rootward}}
	diff := Diff(before, after)
	wantIntroduced := []ArchitectureEdge{rootward, upward, lateral}
	if !reflect.DeepEqual(diff.IntroducedTypedEdges, wantIntroduced) || !reflect.DeepEqual(diff.ResolvedTypedEdges, []ArchitectureEdge{resolved}) {
		t.Fatalf("introduced=%+v resolved=%+v", diff.IntroducedTypedEdges, diff.ResolvedTypedEdges)
	}
	wantAdded := []EdgeChange{{From: "high", To: "low"}, {From: "low", To: "high"}, {From: "peer", To: "side"}}
	if !reflect.DeepEqual(diff.AddedEdges, wantAdded) || !reflect.DeepEqual(diff.RemovedEdges, []EdgeChange{{From: "old", To: "gone"}}) {
		t.Fatalf("compat added=%+v removed=%+v", diff.AddedEdges, diff.RemovedEdges)
	}
	if diff.Changes() != 8 { // four leaf membership changes plus four edge changes; typed views add zero.
		t.Fatalf("changes=%d diff=%+v", diff.Changes(), diff)
	}
}

func TestDiffBlastPathChangesExposeStableImpactReroutes(t *testing.T) {
	before := Report{Leaves: []Leaf{
		{Name: "deep", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"deep", "target"}}}},
		{Name: "same", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"same", "alpha", "target"}}}},
		{Name: "shorter", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"shorter", "one", "two", "target"}}}},
	}}
	after := Report{Leaves: []Leaf{
		{Name: "deep", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"deep", "one", "two", "target"}}}},
		{Name: "same", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"same", "beta", "target"}}}},
		{Name: "shorter", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"shorter", "target"}}}},
	}}
	diff := Diff(before, after)
	want := []BlastPathChange{
		{Source: "deep", Dependent: "target", BeforePath: []string{"deep", "target"}, AfterPath: []string{"deep", "one", "two", "target"}, BeforeHops: 1, AfterHops: 3, Delta: 2},
		{Source: "shorter", Dependent: "target", BeforePath: []string{"shorter", "one", "two", "target"}, AfterPath: []string{"shorter", "target"}, BeforeHops: 3, AfterHops: 1, Delta: -2},
		{Source: "same", Dependent: "target", BeforePath: []string{"same", "alpha", "target"}, AfterPath: []string{"same", "beta", "target"}, BeforeHops: 2, AfterHops: 2, Delta: 0},
	}
	if !reflect.DeepEqual(diff.BlastPathChanges, want) {
		t.Fatalf("blast path changes=%+v want=%+v", diff.BlastPathChanges, want)
	}
	if diff.Verdict != "regression" || diff.Changes() != 0 || len(diff.IntroducedBlastImpacts) != 0 || len(diff.ResolvedBlastImpacts) != 0 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffEqualAndShorterBlastPathChangesRemainClean(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "same", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"same", "alpha", "target"}}}}, {Name: "shorter", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"shorter", "one", "target"}}}}}}
	after := Report{Leaves: []Leaf{{Name: "same", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"same", "beta", "target"}}}}, {Name: "shorter", BlastPaths: []BlastPath{{Dependent: "target", Path: []string{"shorter", "target"}}}}}}
	diff := Diff(before, after)
	if diff.Verdict != "clean" || len(diff.BlastPathChanges) != 2 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffBlastImpactIdentitySurvivesEqualCountReplacement(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "source", BlastRadius: 1, BlastPaths: []BlastPath{{Dependent: "old", Path: []string{"source", "old"}}}}}}
	after := Report{Leaves: []Leaf{{Name: "source", BlastRadius: 1, BlastPaths: []BlastPath{{Dependent: "new", Path: []string{"source", "middle", "new"}}}}}}
	diff := Diff(before, after)
	wantIntroduced := []BlastImpact{{Source: "source", Dependent: "new", Path: []string{"source", "middle", "new"}}}
	wantResolved := []BlastImpact{{Source: "source", Dependent: "old", Path: []string{"source", "old"}}}
	if !reflect.DeepEqual(diff.IntroducedBlastImpacts, wantIntroduced) || !reflect.DeepEqual(diff.ResolvedBlastImpacts, wantResolved) {
		t.Fatalf("introduced=%+v resolved=%+v", diff.IntroducedBlastImpacts, diff.ResolvedBlastImpacts)
	}
	if len(diff.BlastRadiusChanges) != 0 || diff.Verdict != "regression" || diff.Changes() != 0 {
		t.Fatalf("diff=%+v; typed impact views must not depend on count or double-count edges", diff)
	}
}

func TestDiffResolvedBlastImpactRemainsClean(t *testing.T) {
	diff := Diff(
		Report{Leaves: []Leaf{{Name: "source", BlastRadius: 1, BlastPaths: []BlastPath{{Dependent: "gone", Path: []string{"source", "gone"}}}}}},
		Report{Leaves: []Leaf{{Name: "source", BlastPaths: []BlastPath{}}}},
	)
	if diff.Verdict != "clean" || len(diff.IntroducedBlastImpacts) != 0 || len(diff.ResolvedBlastImpacts) != 1 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestCompareBlastRadiusChangesRankRegressionsWithoutDoubleCounting(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "deep", BlastRadius: 1}, {Name: "small", BlastRadius: 2}, {Name: "improved", BlastRadius: 5}}}
	after := Report{Leaves: []Leaf{{Name: "deep", BlastRadius: 5}, {Name: "small", BlastRadius: 3}, {Name: "improved", BlastRadius: 1}}}
	diff := Diff(before, after)
	want := []BlastRadiusChange{{Leaf: "deep", Before: 1, After: 5, Delta: 4}, {Leaf: "small", Before: 2, After: 3, Delta: 1}, {Leaf: "improved", Before: 5, After: 1, Delta: -4}}
	if !reflect.DeepEqual(diff.BlastRadiusChanges, want) {
		t.Fatalf("blast radius changes=%+v want=%+v", diff.BlastRadiusChanges, want)
	}
	if diff.Verdict != "regression" {
		t.Fatalf("verdict=%q want regression", diff.Verdict)
	}
	if diff.Changes() != 0 {
		t.Fatalf("changes=%d; derived blast-radius view must not double-count edges", diff.Changes())
	}
}

func TestCompareBlastRadiusContractionRemainsClean(t *testing.T) {
	diff := Diff(Report{Leaves: []Leaf{{Name: "leaf", BlastRadius: 3}}}, Report{Leaves: []Leaf{{Name: "leaf", BlastRadius: 1}}})
	if diff.Verdict != "clean" || len(diff.BlastRadiusChanges) != 1 || diff.BlastRadiusChanges[0].Delta != -2 {
		t.Fatalf("diff=%+v", diff)
	}
}

func TestDiffNamesEveryArchitectureChangeDeterministically(t *testing.T) {
	before := Report{Leaves: []Leaf{
		{Name: "alpha", DeclaredTier: 1, DeclaredTierName: "primitive", Dependencies: []string{"abi", "old"}, ViolationEdges: []ViolationEdge{{From: "alpha", FromTier: 1, FromTierName: "primitive", To: "old", ToTier: 2, ToTierName: "foundation-composite", TierDistance: 1}}, Violations: []string{"alpha -> old"}},
		{Name: "old", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
		{Name: "removed", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
	}}
	after := Report{Leaves: []Leaf{
		{Name: "new", DeclaredTier: 2, DeclaredTierName: "foundation-composite"},
		{Name: "alpha", DeclaredTier: 3, DeclaredTierName: "mechanism", Dependencies: []string{"abi", "new"}, ViolationEdges: []ViolationEdge{{From: "alpha", FromTier: 3, FromTierName: "mechanism", To: "new", ToTier: 4, ToTierName: "composer", TierDistance: 1}}, Violations: []string{"alpha -> new"}},
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
	wantIntroduced := []ViolationEdge{{From: "alpha", FromTier: 3, FromTierName: "mechanism", To: "new", ToTier: 4, ToTierName: "composer", TierDistance: 1}}
	wantResolved := []ViolationEdge{{From: "alpha", FromTier: 1, FromTierName: "primitive", To: "old", ToTier: 2, ToTierName: "foundation-composite", TierDistance: 1}}
	if !reflect.DeepEqual(got.IntroducedViolationEdges, wantIntroduced) || !reflect.DeepEqual(got.ResolvedViolationEdges, wantResolved) || !reflect.DeepEqual(got.IntroducedViolations, []string{"alpha -> new"}) || !reflect.DeepEqual(got.ResolvedViolations, []string{"alpha -> old"}) {
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

func TestDiffViolationIdentityIgnoresTierDisplayChanges(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{{From: "low", FromTier: 1, FromTierName: "old-low", To: "high", ToTier: 3, ToTierName: "old-high"}}}}}
	after := Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{{From: "low", FromTier: 1, FromTierName: "new-low", To: "high", ToTier: 3, ToTierName: "new-high"}}}}}
	got := Diff(before, after)
	if len(got.IntroducedViolationEdges) != 0 || len(got.ResolvedViolationEdges) != 0 || got.Verdict != "clean" {
		t.Fatalf("format-only tier labels fabricated delta: %+v", got)
	}
}

func TestDiffTypedViolationJSONKeepsCompatibilityProjection(t *testing.T) {
	edge := ViolationEdge{From: "low", FromTier: 1, FromTierName: "primitive", To: "high", ToTier: 3, ToTierName: "mechanism", TierDistance: 2}
	got := Diff(Report{}, Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{edge}}}})
	raw, err := got.JSON()
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"introduced_violation_edges"`, `"introduced_violations"`, `"from_tier"`} {
		if !bytes.Contains(raw, []byte(key)) {
			t.Fatalf("JSON missing %s: %s", key, raw)
		}
	}
}

func TestDiffRanksTypedViolationsByTierDistance(t *testing.T) {
	near := ViolationEdge{From: "alpha", FromTier: 1, To: "near", ToTier: 2, TierDistance: 1}
	far := ViolationEdge{From: "alpha", FromTier: 1, To: "far", ToTier: 4, TierDistance: 3}
	got := Diff(Report{}, Report{Leaves: []Leaf{{Name: "alpha", ViolationEdges: []ViolationEdge{near, far}}}})
	if !reflect.DeepEqual(got.IntroducedViolationEdges, []ViolationEdge{far, near}) || !reflect.DeepEqual(got.IntroducedViolations, []string{"alpha -> far", "alpha -> near"}) {
		t.Fatalf("diff=%+v", got)
	}
}

func TestDiffReportsViolationDistanceChangesDeterministically(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{{From: "low", To: "far", TierDistance: 1}, {From: "low", To: "near", TierDistance: 3}, {From: "stable", To: "same", TierDistance: 2}}}}}
	after := Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{{From: "low", To: "far", TierDistance: 3}, {From: "low", To: "near", TierDistance: 1}, {From: "stable", To: "same", TierDistance: 2}}}}}
	got := Diff(before, after)
	want := []ViolationDistanceChange{{From: "low", To: "far", BeforeDistance: 1, AfterDistance: 3, Delta: 2}, {From: "low", To: "near", BeforeDistance: 3, AfterDistance: 1, Delta: -2}}
	if !reflect.DeepEqual(got.ViolationDistanceChanges, want) || got.Verdict != "regression" || got.Changes() != 0 || len(got.IntroducedViolationEdges) != 0 || len(got.ResolvedViolationEdges) != 0 {
		t.Fatalf("diff=%+v", got)
	}
}

func TestDiffViolationDistanceImprovementRemainsClean(t *testing.T) {
	before := Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{{From: "low", To: "high", TierDistance: 2}}}}}
	after := Report{Leaves: []Leaf{{Name: "low", ViolationEdges: []ViolationEdge{{From: "low", To: "high", TierDistance: 1}}}}}
	got := Diff(before, after)
	if got.Verdict != "clean" || len(got.ViolationDistanceChanges) != 1 {
		t.Fatalf("diff=%+v", got)
	}
}
