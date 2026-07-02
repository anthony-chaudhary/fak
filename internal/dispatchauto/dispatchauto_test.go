package dispatchauto

import (
	"reflect"
	"testing"
)

// TestPlanAutoBindsTheScarcestCeiling proves the headline contract: the wave
// auto-sizes to the MINIMUM of every set ceiling, names its binding, and never
// needs an operator-typed count.
func TestPlanAutoBindsTheScarcestCeiling(t *testing.T) {
	cases := []struct {
		name    string
		in      Input
		target  int
		binding string
	}{
		{
			name:    "ready work binds on a roomy box",
			in:      Input{EffectiveCap: 8, DistinctPools: 6, ReadyWork: 3, LiveWorkers: 0},
			target:  3,
			binding: CeilingReadyWork,
		},
		{
			name:    "distinct pools bind so the wave never serializes on one usage bucket",
			in:      Input{EffectiveCap: 8, DistinctPools: 2, ReadyWork: 5},
			target:  2,
			binding: CeilingDistinctPools,
		},
		{
			name:    "preflight effective cap binds",
			in:      Input{EffectiveCap: 1, DistinctPools: 4, ReadyWork: 4},
			target:  1,
			binding: CeilingEffectiveCap,
		},
		{
			name:    "throughput target binds when set",
			in:      Input{EffectiveCap: 8, DistinctPools: 8, ReadyWork: 9, RequiredWorkers: 5},
			target:  5,
			binding: CeilingRequiredWorkers,
		},
		{
			name:    "zero pools means no wave",
			in:      Input{EffectiveCap: 8, DistinctPools: 0, ReadyWork: 9},
			target:  0,
			binding: CeilingDistinctPools,
		},
		{
			name:    "zero ready work means no wave even with free pools",
			in:      Input{EffectiveCap: 8, DistinctPools: 3, ReadyWork: 0},
			target:  0,
			binding: CeilingReadyWork,
		},
		{
			name: "node headroom binds when a roster is supplied",
			in: Input{
				EffectiveCap: 8, DistinctPools: 8, ReadyWork: 8,
				Nodes: []Node{
					{Name: "a", SeatCap: 2, Live: 1, Healthy: true},
					{Name: "b", SeatCap: 2, Live: 0, Healthy: true},
				},
			},
			target:  3,
			binding: CeilingNodeHeadroom,
		},
		{
			name: "an unhealthy node contributes no headroom",
			in: Input{
				EffectiveCap: 8, DistinctPools: 8, ReadyWork: 8,
				Nodes: []Node{
					{Name: "a", SeatCap: 4, Live: 0, Healthy: true},
					{Name: "b", SeatCap: 4, Live: 0, Healthy: false},
				},
			},
			target:  4,
			binding: CeilingNodeHeadroom,
		},
		{
			name: "an uncapped healthy node lifts the roster ceiling entirely",
			in: Input{
				EffectiveCap: 6, DistinctPools: 7, ReadyWork: 8,
				Nodes: []Node{
					{Name: "a", SeatCap: 1, Live: 1, Healthy: true},
					{Name: "b", SeatCap: 0, Live: 0, Healthy: true},
				},
			},
			target:  6,
			binding: CeilingEffectiveCap,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := PlanAuto(tc.in)
			if got.Target != tc.target || got.Binding != tc.binding {
				t.Fatalf("PlanAuto(%+v) = target %d binding %q, want target %d binding %q\nplan: %s",
					tc.in, got.Target, got.Binding, tc.target, tc.binding, got)
			}
		})
	}
}

// TestPlanAutoRefillConverges proves the steady-state semantics the one-shot
// wave verbs lack: refill tops the population up to Target and never
// over-launches or goes negative.
func TestPlanAutoRefillConverges(t *testing.T) {
	base := Input{EffectiveCap: 4, DistinctPools: 4, ReadyWork: 9}

	in := base
	in.LiveWorkers = 1
	if got := PlanAuto(in); got.Refill != 3 {
		t.Fatalf("live=1 target=4: refill = %d, want 3", got.Refill)
	}

	in.LiveWorkers = 4
	got := PlanAuto(in)
	if got.Refill != 0 {
		t.Fatalf("live=4 target=4: refill = %d, want 0 (converged)", got.Refill)
	}
	if len(got.Assignments) != 0 {
		t.Fatalf("converged plan still placed workers: %+v", got.Assignments)
	}

	in.LiveWorkers = 7 // more live than target: never a negative refill
	if got := PlanAuto(in); got.Refill != 0 {
		t.Fatalf("live=7 target=4: refill = %d, want 0", got.Refill)
	}
}

// TestPlanAutoSlicesSharedContext proves each launched worker gets an explicit
// even slice of the fleet context budget.
func TestPlanAutoSlicesSharedContext(t *testing.T) {
	in := Input{EffectiveCap: 3, DistinctPools: 3, ReadyWork: 3, SharedContextTokens: 90_000}
	got := PlanAuto(in)
	if got.PerWorkerContextTokens != 30_000 {
		t.Fatalf("per-worker context = %d, want 30000", got.PerWorkerContextTokens)
	}
	for _, a := range got.Assignments {
		if a.ContextTokens != 30_000 {
			t.Fatalf("assignment %d carries context %d, want 30000", a.Seq, a.ContextTokens)
		}
	}
	// Unset budget ⇒ no per-worker slice, not a division by zero.
	in.SharedContextTokens = 0
	if got := PlanAuto(in); got.PerWorkerContextTokens != 0 {
		t.Fatalf("unset budget: per-worker context = %d, want 0", got.PerWorkerContextTokens)
	}
	// Zero target ⇒ no slice.
	if got := PlanAuto(Input{SharedContextTokens: 90_000}); got.PerWorkerContextTokens != 0 {
		t.Fatalf("zero target: per-worker context = %d, want 0", got.PerWorkerContextTokens)
	}
}

// TestPlaceWorkersBalancesLeastUtilizedFirst proves deterministic, balanced
// placement across a node roster.
func TestPlaceWorkersBalancesLeastUtilizedFirst(t *testing.T) {
	in := Input{
		EffectiveCap: 8, DistinctPools: 8, ReadyWork: 8,
		Nodes: []Node{
			{Name: "a", SeatCap: 4, Live: 1, Healthy: true}, // util 1/4
			{Name: "b", SeatCap: 2, Live: 0, Healthy: true}, // util 0/2
		},
	}
	got := PlanAuto(in)
	if got.Target != 5 || got.Refill != 5 {
		t.Fatalf("target/refill = %d/%d, want 5/5\nplan: %s", got.Target, got.Refill, got)
	}
	nodes := make([]string, len(got.Assignments))
	for i, a := range got.Assignments {
		nodes[i] = a.Node
	}
	// b(0/2) first, then a(1/4), then b(1/2) vs a(2/4) tie -> name order a,
	// then b(1/2), then a(3/4).
	want := []string{"b", "a", "a", "b", "a"}
	if !reflect.DeepEqual(nodes, want) {
		t.Fatalf("placement = %v, want %v", nodes, want)
	}
}

// TestPlanAutoNoRosterPlacesLocally proves the single-box default: no roster
// still yields launchable assignments on the implicit local node.
func TestPlanAutoNoRosterPlacesLocally(t *testing.T) {
	got := PlanAuto(Input{EffectiveCap: 2, DistinctPools: 2, ReadyWork: 2})
	if len(got.Assignments) != 2 {
		t.Fatalf("assignments = %d, want 2", len(got.Assignments))
	}
	for _, a := range got.Assignments {
		if a.Node != "local" {
			t.Fatalf("assignment on %q, want local", a.Node)
		}
	}
}

// TestPlanAutoIsDeterministic proves the fold is pure: identical input yields
// an identical plan, including map contents and placement order.
func TestPlanAutoIsDeterministic(t *testing.T) {
	in := Input{
		EffectiveCap: 5, DistinctPools: 4, ReadyWork: 6, RequiredWorkers: 5,
		LiveWorkers: 1, SharedContextTokens: 40_000,
		Nodes: []Node{
			{Name: "b", SeatCap: 3, Live: 1, Healthy: true},
			{Name: "a", SeatCap: 3, Live: 0, Healthy: true},
		},
	}
	first := PlanAuto(in)
	for i := 0; i < 5; i++ {
		if got := PlanAuto(in); !reflect.DeepEqual(got, first) {
			t.Fatalf("run %d diverged:\n got %+v\nwant %+v", i, got, first)
		}
	}
}
