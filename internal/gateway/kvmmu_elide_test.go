package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// kvmmu_elide_test.go is the gateway-level integration witness for the SECOND half of issue #579:
// the model-side PLANNED-ELISION residency BRIDGE (internal/kvmmu's ApplyPlan) wired onto the LIVE
// serve path. Where kvmmu_evict_test.go drives a trust QUARANTINE through the REAL InKernelPlanner,
// these drive a CAPACITY plan: a context-planner elision evicts the over-budget spans' K/V so the
// kernel-owned residency shrinks to the O(1) view — asserted bit-exact end-to-end (in the provable
// direction), not via the synthetic-model unit witness.

// TestLiveKVResidencyShrinksToView is the #579 planned-elision deliverable: a transcript whose
// over-budget tail spans are elided by the planner drives a real model.KVCache.Evict over those
// spans, and the resulting cache is BIT-IDENTICAL to a session that only ever prefilled the
// resident prefix (the O(1)-residency invariant) — on the REAL in-kernel planner, end-to-end. The
// elided spans are the positional SUFFIX (the provable direction the user scoped #579 to): the
// early spans stay resident and the later low-density candidates are shed.
func TestLiveKVResidencyShrinksToView(t *testing.T) {
	planner := liveInKernelPlanner(t)

	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "the resident question the model keeps"},
		{Role: agent.RoleAssistant, Content: "an over-budget candidate that lost the knapsack"},
		{Role: agent.RoleUser, Content: "another over-budget candidate shed by the planner"},
	}
	// Provable direction: keep the first two spans resident, elide the later two (the over-budget
	// tail). SegElisionPlan keys the plan by the same segIDFor ids the bridge lowers into.
	elided := []bool{false, false, true, true}
	plan := agent.SegElisionPlan(messages, elided)

	elider, ok := any(planner).(agent.KVSpanElider)
	if !ok {
		t.Fatal("InKernelPlanner must implement agent.KVSpanElider (the gateway type-asserts it)")
	}
	freed, exact := elider.ElideKVSpans(messages, plan)
	if freed <= 0 {
		t.Fatalf("live planned-elision freed %d positions, want > 0 (the elided tail must be evicted)", freed)
	}
	if !exact {
		t.Fatalf("post-elision cache is NOT bit-identical to the resident-prefix view — the O(1)-residency invariant failed in the provable direction")
	}
	t.Logf("LIVE #579 planned-elision: %d segments evicted on the real in-kernel path; KV residency bit-exact to the O(1) view", freed)
}

// TestLiveKVResidencyPrefixElisionShrinksNotExact is the honesty control for the OTHER direction
// (issue #579, "bit-exact provable direction only"): eliding an old PREFIX that the resident tail
// already attended to still shrinks residency (the K/V is evicted) but is NOT reported bit-exact —
// a re-RoPE cannot un-see attention a surviving later token already absorbed, so the bridge
// reports reposition_exact=false rather than overclaiming.
func TestLiveKVResidencyPrefixElisionShrinksNotExact(t *testing.T) {
	planner := liveInKernelPlanner(t)
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "an old cold turn the planner drops"},
		{Role: agent.RoleAssistant, Content: "an old cold answer the planner drops"},
		{Role: agent.RoleUser, Content: "the recent resident question"},
	}
	// Non-provable direction: elide the leading PREFIX, keep the recent tail resident.
	elided := []bool{true, true, false, false}
	plan := agent.SegElisionPlan(messages, elided)
	elider, _ := any(planner).(agent.KVSpanElider)
	freed, exact := elider.ElideKVSpans(messages, plan)
	if freed <= 0 {
		t.Fatalf("prefix elision freed %d, want > 0 (residency must still shrink)", freed)
	}
	if exact {
		t.Fatalf("prefix elision must NOT be reported bit-exact (a re-RoPE cannot un-see attended history); got exact=true")
	}
	t.Logf("LIVE #579: prefix elision shrank residency (%d evicted) but is honestly NOT bit-exact", freed)
}

// TestLiveKVResidencyDefaultOff is the posture guard: with the bridge flag OFF (the default), the
// SAME live planner does NOT drive a residency eviction — the served path is byte-for-byte the
// pre-bridge behavior until an operator opts in.
func TestLiveKVResidencyDefaultOff(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "") // explicitly OFF (the default)
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	m := model.NewSynthetic(kvmmuSynthCfg())
	planner := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "synthetic-elide-off", false, nil, false)

	ev, ok := any(planner).(agent.KVSpanElider)
	if !ok {
		t.Fatal("InKernelPlanner must implement agent.KVSpanElider")
	}
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "resident"},
		{Role: agent.RoleAssistant, Content: "cold"},
	}
	plan := agent.SegElisionPlan(messages, []bool{false, false, true})
	if freed, exact := ev.ElideKVSpans(messages, plan); freed != 0 || exact {
		t.Fatalf("bridge OFF must be inert: got freed=%d exact=%v, want 0/false", freed, exact)
	}
}

// TestLiveKVResidencyNoOpWhenNothingElided is the boundary control: a plan that elides nothing is a
// no-op — eviction is driven by the plan, not fired unconditionally.
func TestLiveKVResidencyNoOpWhenNothingElided(t *testing.T) {
	planner := liveInKernelPlanner(t)
	ev, _ := any(planner).(agent.KVSpanElider)
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "the only turn"},
	}
	plan := agent.SegElisionPlan(messages, []bool{false, false}) // nothing elided
	if freed, exact := ev.ElideKVSpans(messages, plan); freed != 0 || exact {
		t.Fatalf("all-resident plan must elide nothing: got freed=%d exact=%v, want 0/false", freed, exact)
	}
}

// TestElidedPrefixMaskSuffixGuard checks the gateway's pre-filter: it recovers the elided-prefix
// mask only when the planned view is a clean trailing suffix of the full history. A rewritten or
// reordered plan yields ok=false, so the residency bridge never fires on the wrong spans.
func TestElidedPrefixMaskSuffixGuard(t *testing.T) {
	full := []agent.Message{
		{Role: agent.RoleSystem, Content: "sys"},
		{Role: agent.RoleUser, Content: "a"},
		{Role: agent.RoleAssistant, Content: "b"},
	}
	cases := []struct {
		name     string
		planned  []agent.Message
		wantOK   bool
		wantMask []bool
	}{
		{"clean trailing suffix drops prefix", full[1:], true, []bool{true, false, false}},
		{"drops two-message prefix", full[2:], true, []bool{true, true, false}},
		{"full identity (nothing elided)", full, false, nil},
		{"rewritten last content", []agent.Message{{Role: agent.RoleAssistant, Content: "DIFFERENT"}}, false, nil},
		{"reordered role", []agent.Message{{Role: agent.RoleUser, Content: "b"}}, false, nil},
		{"empty plan", nil, false, nil},
	}
	for _, tc := range cases {
		mask, ok := elidedPrefixMask(full, tc.planned)
		if ok != tc.wantOK {
			t.Errorf("%s: ok = %v, want %v", tc.name, ok, tc.wantOK)
			continue
		}
		if ok {
			if len(mask) != len(tc.wantMask) {
				t.Errorf("%s: mask len %d, want %d", tc.name, len(mask), len(tc.wantMask))
				continue
			}
			for i := range mask {
				if mask[i] != tc.wantMask[i] {
					t.Errorf("%s: mask[%d] = %v, want %v", tc.name, i, mask[i], tc.wantMask[i])
				}
			}
		}
	}
}
