package routing

import (
	"reflect"
	"strings"
	"testing"
)

// TestDefaultDegradationChainShape witnesses the standard ladder: three named
// rungs in order, the last being the maintenance rung, the local-only rungs
// pinned to the local surface, and the tag convention that keeps each rung
// selective. It validates against its own invariants.
//
// The tag assertions are load-bearing for demotion: the first two rungs carry a
// capability tag precisely so they do NOT match every engine, and the final
// maintenance rung is the ONLY unconstrained catch-all. A regression that
// dropped the primary/fallback tags would silently make rung 1 swallow the fleet
// again.
func TestDefaultDegradationChainShape(t *testing.T) {
	chain := DefaultDegradationChain()
	if err := chain.Validate(); err != nil {
		t.Fatalf("DefaultDegradationChain().Validate() = %v, want nil", err)
	}
	if chain.Version != ChainSchema {
		t.Errorf("chain version = %q, want %q", chain.Version, ChainSchema)
	}
	wantNames := []string{"primary", "fallback-small", "maintenance"}
	if len(chain.Rungs) != len(wantNames) {
		t.Fatalf("chain has %d rungs, want %d", len(chain.Rungs), len(wantNames))
	}
	for i, want := range wantNames {
		if chain.Rungs[i].Name != want {
			t.Errorf("rung %d name = %q, want %q", i, chain.Rungs[i].Name, want)
		}
	}

	last, ok := chain.MaintenanceRung()
	if !ok {
		t.Fatal("MaintenanceRung() reported no rung for a non-empty chain")
	}
	if last.Name != "maintenance" {
		t.Errorf("maintenance rung name = %q, want %q", last.Name, "maintenance")
	}
	if chain.Rungs[1].Surface != SurfaceLocal || chain.Rungs[2].Surface != SurfaceLocal {
		t.Errorf("fallback/maintenance surfaces = %q/%q, want %q/%q",
			chain.Rungs[1].Surface, chain.Rungs[2].Surface, SurfaceLocal, SurfaceLocal)
	}
	if !reflect.DeepEqual(chain.Rungs[0].Tags, []string{"primary"}) {
		t.Errorf("primary rung tags = %v, want [primary]", chain.Rungs[0].Tags)
	}
	if !reflect.DeepEqual(chain.Rungs[1].Tags, []string{"small"}) {
		t.Errorf("fallback-small rung tags = %v, want [small]", chain.Rungs[1].Tags)
	}
	if len(chain.Rungs[2].Tags) != 0 {
		t.Errorf("maintenance rung tags = %v, want none (it is the catch-all final slot)", chain.Rungs[2].Tags)
	}
	// The selective rungs must carry a capability tag; a bare surface constraint
	// alone would let a local primary model be mistaken for the small fallback.
	for i := 0; i < len(chain.Rungs)-1; i++ {
		if len(chain.Rungs[i].Tags) == 0 {
			t.Errorf("non-final rung %q has no tag constraint, so it cannot be selective", chain.Rungs[i].Name)
		}
	}
	if empty, ok := (DegradationChain{}).MaintenanceRung(); ok || empty.Name != "" {
		t.Errorf("empty chain MaintenanceRung() = (%+v, %t), want (zero, false)", empty, ok)
	}
}

// TestDegradationChainValidateFailsLoud is the fail-loud witness: a ladder too
// short to degrade, an unnamed rung, and an unknown rung surface each return an
// error prefixed "routing: " rather than silently routing.
func TestDegradationChainValidateFailsLoud(t *testing.T) {
	cases := []struct {
		name  string
		chain DegradationChain
	}{
		{
			name:  "single rung is not a ladder",
			chain: DegradationChain{Version: "x", Rungs: []Rung{{Name: "only"}}},
		},
		{
			name:  "unnamed rung",
			chain: DegradationChain{Version: "x", Rungs: []Rung{{Name: "primary"}, {Name: ""}}},
		},
		{
			name:  "unknown surface",
			chain: DegradationChain{Version: "x", Rungs: []Rung{{Name: "primary"}, {Name: "quantum", Surface: Surface("quantum")}}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.chain.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want a fail-loud error")
			}
			if !strings.HasPrefix(err.Error(), "routing: ") {
				t.Errorf("error %q lacks the %q prefix", err.Error(), "routing: ")
			}
		})
	}
}

// TestDegradationOrderIsPureAndStable is the LOAD-BEARING purity witness: Order
// deep-equals a pre-call copy (it never mutates its input) and two calls return
// the same slice, so it is a replayable stable partition. The engines carry the
// standard default-chain tags so each fills a distinct rung.
func TestDegradationOrderIsPureAndStable(t *testing.T) {
	chain := DefaultDegradationChain()
	engines := []Engine{
		{ID: "maint", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1},
		{ID: "primary", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1, Tags: []string{"primary"}},
		{ID: "fallback", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Tags: []string{"small"}},
	}
	before := make([]Engine, len(engines))
	copy(before, engines)

	first := chain.Order(engines)
	if !reflect.DeepEqual(engines, before) {
		t.Fatalf("Order mutated its input:\n got %+v\nwant %+v", engines, before)
	}
	second := chain.Order(engines)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("Order is not deterministic:\n first %+v\nsecond %+v", first, second)
	}
	if !reflect.DeepEqual(engines, before) {
		t.Fatalf("second Order call mutated its input:\n got %+v\nwant %+v", engines, before)
	}
}

// TestDegradationOrderInvokesLadder is the LOAD-BEARING invoked-ordering witness:
// given engines declared with the maintenance engine FIRST, Order moves the
// primary-capable engine to the front and the maintenance engine to the back,
// and an engine matching no rung lands ahead of the maintenance engine. That the
// declared order changes proves the ladder is INVOKED, not inert.
//
// The ladder here is deliberately constrained per rung. The default chain's
// "primary" rung is unconstrained by design ("any engine may serve it"), so every
// engine matches rung 0 and the partition is invisible; a constrained ladder is
// the honest exercise of the ordering rule.
func TestDegradationOrderInvokesLadder(t *testing.T) {
	chain := DegradationChain{
		Version: "test-ordered-ladder",
		Rungs: []Rung{
			{Name: "primary", Tags: []string{"big"}},
			{Name: "fallback-small", Tags: []string{"small"}},
			{Name: "maintenance", Surface: SurfaceLocal},
		},
	}
	if err := chain.Validate(); err != nil {
		t.Fatalf("test chain invalid: %v", err)
	}
	engines := []Engine{
		{ID: "maint", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1},
		{ID: "primary", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1, Tags: []string{"big"}},
		{ID: "fallback", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Tags: []string{"small"}},
	}

	got := chain.Order(engines)
	ids := make([]string, 0, len(got))
	for _, e := range got {
		ids = append(ids, e.ID)
	}

	if ids[0] != "primary" {
		t.Errorf("first ordered engine = %q, want %q", ids[0], "primary")
	}
	if ids[len(ids)-1] != "maint" {
		t.Errorf("last ordered engine = %q, want %q (maintenance is the last resort)", ids[len(ids)-1], "maint")
	}
	if ids[0] == "maint" {
		t.Fatal("ladder had no effect: maintenance engine stayed first - Order is an inert seam")
	}

	// An engine matching no rung sits ahead of the maintenance rung, because an
	// unclassified engine is presumed more capable than the deliberately small
	// maintenance model.
	withUnknown := chain.Order([]Engine{
		{ID: "maint", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1},
		{ID: "unclassified", Surface: Surface("nope"), Healthy: true, CapacityHeadroom: 1},
	})
	if withUnknown[len(withUnknown)-1].ID != "maint" {
		t.Errorf("last engine with an unclassified peer = %q, want maintenance %q",
			withUnknown[len(withUnknown)-1].ID, "maint")
	}
}

// TestDegradationPlanIsLegible witnesses the receipt: one RungPlan per chain
// rung, Maintenance true ONLY on the last, Eligible listing the engine IDs whose
// first-matching rung is that rung, and a non-empty audit naming every rung.
//
// The default chain's "primary" rung is unconstrained, so it would claim every
// engine; the constrained ladder here makes each engine's first matching rung
// distinct, which is what makes the per-rung attribution observable.
func TestDegradationPlanIsLegible(t *testing.T) {
	chain := DegradationChain{
		Version: "test-plan-ladder",
		Rungs: []Rung{
			{Name: "primary", Surface: SurfaceHosted},
			{Name: "fallback-small", Surface: SurfaceLocal, Tags: []string{"small"}},
			{Name: "maintenance", Surface: SurfaceLocal, Tags: []string{"maintenance"}},
		},
	}
	if err := chain.Validate(); err != nil {
		t.Fatalf("test chain invalid: %v", err)
	}
	engines := []Engine{
		{ID: "big", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1},
		{ID: "small", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Tags: []string{"small"}},
		{ID: "kiosk", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Tags: []string{"maintenance"}},
	}

	plan := chain.Plan(engines)
	if plan.Schema != ChainSchema {
		t.Errorf("plan schema = %q, want %q", plan.Schema, ChainSchema)
	}
	if plan.Chain != chain.Version {
		t.Errorf("plan chain = %q, want %q", plan.Chain, chain.Version)
	}
	if len(plan.Rungs) != len(chain.Rungs) {
		t.Fatalf("plan has %d rung plans, want %d", len(plan.Rungs), len(chain.Rungs))
	}
	for i, rp := range plan.Rungs {
		wantMaint := i == len(chain.Rungs)-1
		if rp.Maintenance != wantMaint {
			t.Errorf("rung %d Maintenance = %t, want %t", i, rp.Maintenance, wantMaint)
		}
	}
	if !reflect.DeepEqual(plan.Rungs[0].Eligible, []string{"big"}) {
		t.Errorf("primary rung eligible = %v, want [big]", plan.Rungs[0].Eligible)
	}
	if !reflect.DeepEqual(plan.Rungs[1].Eligible, []string{"small"}) {
		t.Errorf("fallback-small rung eligible = %v, want [small]", plan.Rungs[1].Eligible)
	}
	if !reflect.DeepEqual(plan.Rungs[2].Eligible, []string{"kiosk"}) {
		t.Errorf("maintenance rung eligible = %v, want [kiosk]", plan.Rungs[2].Eligible)
	}
	if len(plan.Audit) != len(chain.Rungs) {
		t.Fatalf("plan audit has %d lines, want %d", len(plan.Audit), len(chain.Rungs))
	}
	for _, r := range chain.Rungs {
		found := false
		for _, line := range plan.Audit {
			if strings.Contains(line, r.Name) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("plan audit does not name rung %q: %v", r.Name, plan.Audit)
		}
	}
}
