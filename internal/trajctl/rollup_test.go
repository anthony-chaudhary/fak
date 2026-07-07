package trajctl

import (
	"reflect"
	"testing"
)

// treeState builds the fixture tree the #2549 done-condition names: a PAUSED parent epic
// with three children — a healthy leaf, a met leaf, and a detour leaf that overran its
// turn budget while the parent is paused (so curve.go raises DETOUR_OVERRUN on it).
//
//	epic (paused)
//	├── child-a  active, healthy, progress 0.80, weight 1
//	├── child-b  MET,    own progress 0.50 but LOCKED at 1.00, weight 1
//	└── child-c  active, DETOUR_OVERRUN, progress 0.30, budget 2 turns / 3 scored, weight 2
//
// Documented aggregate: agg(epic) = (1*0.80 + 1*1.00 + 2*0.30) / (1+1+2) = 2.40/4 = 0.60,
// and the detour propagates so signal(epic) = DETOUR_OVERRUN.
func treeState() State {
	pt := func(id string, v float64, ms int64) ScoreRow {
		return ScoreRow{ObjectiveID: id, Method: CommitScorerMethod, Version: "1", Value: v, UnixMillis: ms}
	}
	return State{
		Objectives: map[string]Objective{
			"epic":    {ID: "epic", Statement: "ship the release", Status: StatusPaused},
			"child-a": {ID: "child-a", ParentID: "epic", Statement: "feature a", Status: StatusActive},
			"child-b": {ID: "child-b", ParentID: "epic", Statement: "feature b", Status: StatusMet},
			"child-c": {ID: "child-c", ParentID: "epic", Statement: "detour c", Status: StatusActive, Budget: Budget{Turns: 2}},
		},
		Scores: []ScoreRow{
			pt("child-a", 0.80, 1),
			pt("child-b", 0.50, 1),
			// child-c: 3 scored turns > its 2-turn budget while the parent is paused -> DETOUR_OVERRUN.
			pt("child-c", 0.30, 1),
			pt("child-c", 0.30, 2),
			pt("child-c", 0.30, 3),
		},
	}
}

// TestRollup_TreeFold is the #2549 witness: the fixture tree folds to the documented
// aggregate — budget-weighted progress, met child locked at complete, and the detour
// overrun propagated to the paused parent.
func TestRollup_TreeFold(t *testing.T) {
	rep := treeState().Rollup()
	if rep.Schema != RollupSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, RollupSchema)
	}
	byID := map[string]ObjectiveRollup{}
	for _, n := range rep.Nodes {
		byID[n.ObjectiveID] = n
	}
	if len(rep.Nodes) != 4 {
		t.Fatalf("nodes = %d, want 4 (epic + 3 children)", len(rep.Nodes))
	}

	epic := byID["epic"]
	if epic.AggProgress != 0.6 {
		t.Errorf("epic.AggProgress = %v, want 0.60 (budget-weighted: (0.8 + 1.0(locked) + 2*0.3)/4)", epic.AggProgress)
	}
	if epic.Signal != SignalDetourOverrun {
		t.Errorf("epic.Signal = %q, want %q (detour propagates to the paused parent)", epic.Signal, SignalDetourOverrun)
	}
	if epic.OwnProgress != 0 || epic.OwnSignal != SignalHealthy {
		t.Errorf("epic flat curve = progress %v / signal %q, want 0 / HEALTHY (the parent has no own scores)", epic.OwnProgress, epic.OwnSignal)
	}
	if epic.Descendants != 3 {
		t.Errorf("epic.Descendants = %d, want 3", epic.Descendants)
	}
	if !reflect.DeepEqual(epic.Children, []string{"child-a", "child-b", "child-c"}) {
		t.Errorf("epic.Children = %v, want [child-a child-b child-c] (lexical)", epic.Children)
	}

	// The met child locks its contribution at complete even though its own curve reads 0.5.
	b := byID["child-b"]
	if b.AggProgress != 1 {
		t.Errorf("met child-b.AggProgress = %v, want 1.0 (met locks at complete)", b.AggProgress)
	}
	if b.OwnProgress != 0.5 {
		t.Errorf("child-b.OwnProgress = %v, want 0.5 (its flat curve, unlocked)", b.OwnProgress)
	}

	// The detour child carries the overrun that propagated upstream.
	c := byID["child-c"]
	if c.Signal != SignalDetourOverrun {
		t.Errorf("child-c.Signal = %q, want DETOUR_OVERRUN (overran its 2-turn budget while parent paused)", c.Signal)
	}
	if c.AggProgress != 0.3 {
		t.Errorf("child-c.AggProgress = %v, want 0.3 (a leaf folds to its own progress)", c.AggProgress)
	}

	// Only the epic is a root, and it lists worst-first.
	if !reflect.DeepEqual(rep.Roots, []string{"epic"}) {
		t.Errorf("roots = %v, want [epic] (children have a live parent)", rep.Roots)
	}
}

// TestRollup_AbandonedChildDropped pins that dead work is neither progress nor a drag: an
// abandoned child is removed from the fold entirely, so the parent reflects only its live
// children rather than being averaged down by an abandoned zero.
func TestRollup_AbandonedChildDropped(t *testing.T) {
	st := State{
		Objectives: map[string]Objective{
			"p":    {ID: "p", Status: StatusPaused},
			"live": {ID: "live", ParentID: "p", Status: StatusActive},
			"dead": {ID: "dead", ParentID: "p", Status: StatusAbandoned},
		},
		Scores: []ScoreRow{
			{ObjectiveID: "live", Method: CommitScorerMethod, Version: "1", Value: 0.4, UnixMillis: 1},
			{ObjectiveID: "dead", Method: CommitScorerMethod, Version: "1", Value: 0.0, UnixMillis: 1},
		},
	}
	p, ok := st.RollupFor("p")
	if !ok {
		t.Fatal("RollupFor(p) not found")
	}
	if p.AggProgress != 0.4 {
		t.Errorf("p.AggProgress = %v, want 0.4 (abandoned child dropped, not averaged in as a 0)", p.AggProgress)
	}
	if p.Descendants != 2 {
		t.Errorf("p.Descendants = %d, want 2 (both children counted structurally, only the live one weighed)", p.Descendants)
	}
}

// TestRollup_LeafIsOwnProgress pins the base case: a lone objective (no children) folds to
// its own witnessed progress, and RollupFor reports unknown ids as not found.
func TestRollup_LeafIsOwnProgress(t *testing.T) {
	st := State{
		Objectives: map[string]Objective{"solo": {ID: "solo", Status: StatusActive}},
		Scores:     []ScoreRow{{ObjectiveID: "solo", Method: CommitScorerMethod, Version: "1", Value: 0.7, UnixMillis: 1}},
	}
	solo, ok := st.RollupFor("solo")
	if !ok {
		t.Fatal("RollupFor(solo) not found")
	}
	if solo.AggProgress != 0.7 || solo.Descendants != 0 {
		t.Errorf("solo rollup = progress %v / descendants %d, want 0.7 / 0", solo.AggProgress, solo.Descendants)
	}
	if _, ok := st.RollupFor("nope"); ok {
		t.Error("RollupFor(nope) = ok true, want false for an undeclared objective")
	}
}
