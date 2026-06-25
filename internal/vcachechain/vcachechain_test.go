package vcachechain

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// vcachechain_test.go pins each M4 acceptance criterion (issue #719) to the decimal.
// The headline is the §11.0 cost gate: a 30k-token prefix replayed at r=0.1 to recall
// one 10-token unit is a 3000→10, i.e. 300× LOSS, so the gate REFUSES it and allows
// rebuild only for amortized fan-out (≥301 siblings).

// chain3 is the §11.0 DAG: anchor(20000) → mid(10000) → unit(10). The unit's
// replayed prefix is 30000 tokens; its fresh prefill is 10. Block counts feed the
// 20-block lookback (anchor+mid = 20 blocks → exactly one breakpoint boundary).
func chain3() PrefixDAG {
	return PrefixDAG{Nodes: []ChainNode{
		{ID: "anchor", ParentID: "", Tokens: 20000, Blocks: 10},
		{ID: "mid", ParentID: "anchor", Tokens: 10000, Blocks: 10},
		{ID: "unit", ParentID: "mid", Tokens: 10, Blocks: 1},
	}}
}

// --- Acceptance 1: prefix DAG (Validate, ChainTo, PrefixTokens) ---

func TestPrefixDAGValidate(t *testing.T) {
	if err := chain3().Validate(); err != nil {
		t.Fatalf("valid chain rejected: %v", err)
	}
	for name, dag := range map[string]PrefixDAG{
		"empty":       {},
		"multi-root":  {Nodes: []ChainNode{{ID: "a"}, {ID: "b"}}},
		"missing-par": {Nodes: []ChainNode{{ID: "a", ParentID: "ghost"}}},
		"dup-id":      {Nodes: []ChainNode{{ID: "a"}, {ID: "a", ParentID: "a"}}},
	} {
		if err := dag.Validate(); err == nil {
			t.Errorf("%s: expected a validation error, got nil", name)
		}
	}
	// Cycle: a→b→a.
	cycle := PrefixDAG{Nodes: []ChainNode{
		{ID: "root", ParentID: ""},
		{ID: "a", ParentID: "b"},
		{ID: "b", ParentID: "a"},
	}}
	if err := cycle.Validate(); err != ErrCycle {
		t.Errorf("cycle: got %v, want ErrCycle", err)
	}
}

func TestChainToOrdersRootToNode(t *testing.T) {
	chain, err := chain3().ChainTo("unit")
	if err != nil {
		t.Fatalf("ChainTo: %v", err)
	}
	got := []string{chain[0].ID, chain[1].ID, chain[2].ID}
	want := []string{"anchor", "mid", "unit"}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("chain order = %v, want %v", got, want)
		}
	}
}

func TestPrefixTokensIsAncestorsOnly(t *testing.T) {
	p, err := chain3().PrefixTokens("unit")
	if err != nil {
		t.Fatal(err)
	}
	if p != 30000 { // anchor(20000) + mid(10000); the unit's own 10 is NOT a prefix
		t.Errorf("PrefixTokens(unit) = %d, want 30000", p)
	}
	if p2, _ := chain3().PrefixTokens("mid"); p2 != 20000 {
		t.Errorf("PrefixTokens(mid) = %d, want 20000", p2)
	}
	if p3, _ := chain3().PrefixTokens("anchor"); p3 != 0 {
		t.Errorf("PrefixTokens(anchor) = %d, want 0 (root has no ancestors)", p3)
	}
}

// --- Acceptance 2: topological replay + send-one-then-fan ---

func TestTopologicalReplayLinearChainOnePerLevel(t *testing.T) {
	// A single-target linear chain yields one node per fan level; Fan is empty at
	// every level (no siblings).
	plan, err := chain3().TopologicalReplay([]string{"unit"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Levels) != 3 {
		t.Fatalf("linear chain levels = %d, want 3 (anchor,mid,unit)", len(plan.Levels))
	}
	for i, lvl := range plan.Levels {
		if len(lvl.Fan) != 0 {
			t.Errorf("level %d: Fan = %v, want empty (no siblings on a linear chain)", i, lvl.Fan)
		}
	}
	want := []string{"anchor", "mid", "unit"}
	for i := range want {
		if plan.Levels[i].Lead.ID != want[i] {
			t.Errorf("level %d lead = %s, want %s", i, plan.Levels[i].Lead.ID, want[i])
		}
	}
}

func TestTopologicalReplayFanOutGroupsSiblings(t *testing.T) {
	// A star anchor with two sibling units: level 0 = {anchor (lead)}, level 1 =
	// {one unit (lead), the other (Fan)}. The Fan is released only after the lead's
	// first streamed token (send-one-then-fan, §8 + Rule C2).
	star := PrefixDAG{Nodes: []ChainNode{
		{ID: "anchor", ParentID: "", Tokens: 4096, Blocks: 5},
		{ID: "u1", ParentID: "anchor", Tokens: 10, Blocks: 1},
		{ID: "u2", ParentID: "anchor", Tokens: 10, Blocks: 1},
	}}
	plan, err := star.TopologicalReplay([]string{"u1", "u2"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Levels) != 2 {
		t.Fatalf("fan-out levels = %d, want 2", len(plan.Levels))
	}
	if plan.Levels[0].Lead.ID != "anchor" || len(plan.Levels[0].Fan) != 0 {
		t.Errorf("level 0 = %+v, want lead=anchor no fan", plan.Levels[0])
	}
	if len(plan.Levels[1].Fan) != 1 {
		t.Errorf("level 1 fan size = %d, want 1 (one sibling released after the lead)", len(plan.Levels[1].Fan))
	}
}

func TestTopologicalReplaySkipsWarmPrefix(t *testing.T) {
	// WarmDepth=1 means the anchor is already cached; replay starts at "mid".
	plan, err := chain3().TopologicalReplay([]string{"unit"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Levels) != 2 {
		t.Fatalf("warm-depth-1 levels = %d, want 2 (mid, unit only)", len(plan.Levels))
	}
	if plan.Levels[0].Lead.ID != "mid" {
		t.Errorf("first cold node = %s, want mid (anchor is warm)", plan.Levels[0].Lead.ID)
	}
}

// --- Acceptance 3: 20-block lookback breakpoints (Rule C3) ---

func TestPlaceBreakpoints(t *testing.T) {
	cases := map[int][]int{
		10:  nil,              // fits one look-back
		15:  nil,              // exactly one block: no intermediate needed
		16:  {15},             // first breakpoint at 15
		50:  {15, 30, 45},     // 60 would exceed the span
		100: {15, 30, 45, 60}, // capped at 4 — the aggregation signal
		200: {15, 30, 45, 60}, // still capped at 4 even though 75/90/... < 200
	}
	for blocks, want := range cases {
		got := PlaceBreakpoints(blocks)
		if !eqInts(got, want) {
			t.Errorf("PlaceBreakpoints(%d) = %v, want %v", blocks, got, want)
		}
	}
}

func TestMergeBreakpointsDedupes(t *testing.T) {
	got := MergeBreakpoints([]int{15, 45}, []int{30, 45, 60})
	want := []int{15, 30, 45, 60}
	if !eqInts(got, want) {
		t.Errorf("MergeBreakpoints = %v, want %v", got, want)
	}
}

// --- Acceptance 4: the §11.0 cost gate (the headline) ---

func TestReplayCostIsPrefixTimesReadMult(t *testing.T) {
	// §11.0: P=30000, r=0.1 → replay cost 3000 token-equiv.
	if got := ReplayCost(30000, 0.1); got != 3000 {
		t.Errorf("ReplayCost(30000,0.1) = %g, want 3000", got)
	}
}

func TestRebuildNetPositiveSingleUnitRefused(t *testing.T) {
	// The §11.0 headline: a 30k-token prefix at r=0.1 to recall one 10-token unit
	// is a 300× LOSS (3000 read vs 10 saved) → REFUSED.
	if RebuildNetPositive(30000, 10, 1, 0.1) {
		t.Error("single-unit rebuild should be REFUSED (P·r=3000 >= U=10)")
	}
}

func TestBreakEvenSiblingsIs301(t *testing.T) {
	// §11.0 crossover: rebuild wins once S > P·r/U = 3000/10 = 300, i.e. S=301.
	got := BreakEvenSiblings(30000, 10, 0.1)
	if got != 301 {
		t.Errorf("BreakEvenSiblings = %g, want 301", got)
	}
}

func TestRebuildNetPositiveAmortizedFanOutAllowed(t *testing.T) {
	// The amortized exception: 301 sibling units sharing the hot prefix → rebuild.
	if !RebuildNetPositive(30000, 10, 301, 0.1) {
		t.Error("amortized rebuild (301 siblings) should be ALLOWED (3000 < 301·10)")
	}
	// 300 siblings still net-negative: 3000 is not < 3000.
	if RebuildNetPositive(30000, 10, 300, 0.1) {
		t.Error("300 siblings should still be REFUSED (3000 is not < 3000)")
	}
}

// --- Acceptance 5: Law D4 — never rebuild a secret/regulated chain ---

func TestPlanRecallSecretChainIsNoCache(t *testing.T) {
	secret := chain3()
	secret.Nodes[0].Secret = vcachegov.Secret // a secret ancestor poisons the chain
	plan, err := PlanRecall(secret, RecallRequest{TargetNodeID: "unit", SiblingsRecalled: 1000, ReadMult: 0.1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Decision != DecisionNoCache {
		t.Errorf("secret chain decision = %s, want no_cache (Law D4)", plan.Decision)
	}
}

// --- Acceptance 6: gated OFF by default ---

func TestPlanRecallGatedOffByDefault(t *testing.T) {
	// DefaultEnabled is false: even an amortized fan-out that WOULD clear the gate
	// returns DecisionGatedOff when enabled is false. This is the issue title.
	if DefaultEnabled {
		t.Error("DefaultEnabled must be false (M4 is gated OFF by default)")
	}
	plan, err := PlanRecall(chain3(), RecallRequest{TargetNodeID: "unit", SiblingsRecalled: 1000, ReadMult: 0.1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Decision != DecisionGatedOff {
		t.Errorf("disabled decision = %s, want gated_off", plan.Decision)
	}
}

// --- The full PlanRecall decision tree, end to end ---

func TestPlanRecallDecisionTree(t *testing.T) {
	dag := chain3()
	// Single unit, enabled: §11.0 refuses it.
	p, err := PlanRecall(dag, RecallRequest{TargetNodeID: "unit", SiblingsRecalled: 1, ReadMult: 0.1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if p.Decision != DecisionColdPrefill {
		t.Errorf("single-unit decision = %s, want cold_prefill", p.Decision)
	}
	if p.ReplayCost != 3000 || p.FreshPrefillCost != 10 || p.BreakEvenSiblings != 301 {
		t.Errorf("economics: cost=%g fresh=%g be=%d, want 3000/10/301", p.ReplayCost, p.FreshPrefillCost, p.BreakEvenSiblings)
	}
	// Amortized fan-out (401 siblings), enabled, warm anchor (WarmDepth=1): rebuild.
	p, err = PlanRecall(dag, RecallRequest{TargetNodeID: "unit", SiblingsRecalled: 401, ReadMult: 0.1, WarmDepth: 1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if p.Decision != DecisionRebuild {
		t.Errorf("amortized decision = %s, want rebuild", p.Decision)
	}
	if !p.Decision.IsRebuild() {
		t.Error("IsRebuild() should be true for DecisionRebuild")
	}
	// The replay schedule must skip the warm anchor (start at "mid").
	if len(p.Replay.Levels) == 0 || p.Replay.Levels[0].Lead.ID != "mid" {
		t.Errorf("rebuild replay start = %v, want lead=mid (anchor warm)", p.Replay.Levels)
	}
	// Unknown target errors.
	if _, err := PlanRecall(dag, RecallRequest{TargetNodeID: "ghost"}, true); err != ErrMissingNode {
		t.Errorf("unknown target err = %v, want ErrMissingNode", err)
	}
}

func TestBreakEvenSiblingsInfWhenNothingToSave(t *testing.T) {
	if got := BreakEvenSiblings(30000, 0, 0.1); !math.IsInf(got, 1) {
		t.Errorf("BreakEvenSiblings(…,U=0) = %g, want +Inf (nothing to save)", got)
	}
}

// --- The CLI proof surface (ProveRecall): §11.0 headline, to the decimal ---

func TestProveRecallHeadlineRefusedThenAmortizedProven(t *testing.T) {
	// Default §11.0: 30k prefix, r=0.1, one 10-token unit → REFUSED (300× loss).
	p := ProveRecall(ProveRecallInput{PrefixTokens: 30000, UnitTokens: 10, ReadMult: 0.1, Siblings: 1})
	if p.Status != ProofRefuted || p.Decision != DecisionColdPrefill {
		t.Errorf("single-unit proof = %s/%s, want refuted/cold_prefill", p.Status, p.Decision)
	}
	if p.ReplayCost != 3000 || p.FreshPrefillCost != 10 || p.LossRatio != 300 {
		t.Errorf("economics: cost=%g fresh=%g loss=%g, want 3000/10/300", p.ReplayCost, p.FreshPrefillCost, p.LossRatio)
	}
	if p.BreakEvenSiblings != 301 {
		t.Errorf("break-even = %d, want 301", p.BreakEvenSiblings)
	}
	if p.CorrectnessDependsOn {
		t.Error("CorrectnessDependsOn must be false (Law A2)")
	}
	// Amortized fan-out at the break-even: 301 siblings → PROVEN.
	p = ProveRecall(ProveRecallInput{PrefixTokens: 30000, UnitTokens: 10, ReadMult: 0.1, Siblings: 301})
	if p.Status != ProofProven || p.Decision != DecisionRebuild {
		t.Errorf("amortized proof (301) = %s/%s, want proven/rebuild", p.Status, p.Decision)
	}
	if p.AmortizedSavings != 3010 {
		t.Errorf("amortized savings = %g, want 3010", p.AmortizedSavings)
	}
	// Just below: 300 siblings still REFUSED.
	p = ProveRecall(ProveRecallInput{PrefixTokens: 30000, UnitTokens: 10, ReadMult: 0.1, Siblings: 300})
	if p.Status != ProofRefuted {
		t.Errorf("300-sibling proof = %s, want refuted (3000 is not < 3000)", p.Status)
	}
}

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
