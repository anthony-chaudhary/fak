package radixkv

import (
	"testing"
)

// Prior-art: vLLM PagedAttention; SGLang RadixAttention.

type mockDemotionDecider struct {
	consultedCount int
	action         string
	tier           string
}

func (m *mockDemotionDecider) DecideDemotion(tokens int, hits int, bytes int64) DemotionDecision {
	m.consultedCount++
	return DemotionDecision{Action: m.action, Tier: m.tier}
}

func insertPureTokens(tree *Tree, req []int) {
	b, m := tree.Lookup(req)
	leaf := tree.Insert(b, req[m:], nil)
	tree.Done(leaf)
}

// TestEvictToBudgetConsultsDemotionPlan verifies the radixkv half of #3414:
// When demote-before-drop is enabled, evictToBudget consults the demotion decider
// before dropping an over-budget victim, routing to spill and updating t.Demotions().
// When disabled (default), the demotion decider is not consulted and behavior is drop-only.
func TestEvictToBudgetConsultsDemotionPlan(t *testing.T) {
	// 1. Default-off behavior: demotion decider set, but EnableDemoteBeforeDrop is false
	treeDefault := New(100)
	plannerDefault := &mockDemotionDecider{action: "spill", tier: "host"}
	treeDefault.SetDemotionDecider(plannerDefault)

	// Insert into tree beyond budget
	for i := 0; i < 20; i++ {
		insertPureTokens(treeDefault, []int{100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 600 + i, 700 + i, 800 + i})
	}

	if plannerDefault.consultedCount != 0 {
		t.Fatalf("planner consulted when demoteBeforeDrop=false! count=%d", plannerDefault.consultedCount)
	}
	if treeDefault.Demotions() != 0 {
		t.Fatalf("treeDefault.Demotions() = %d, want 0", treeDefault.Demotions())
	}

	// 2. Enabled behavior: EnableDemoteBeforeDrop(true)
	treeEnabled := New(50)
	plannerEnabled := &mockDemotionDecider{action: "spill", tier: "host"}
	treeEnabled.SetDemotionDecider(plannerEnabled)
	treeEnabled.EnableDemoteBeforeDrop(true)

	// Insert over budget
	for i := 0; i < 20; i++ {
		insertPureTokens(treeEnabled, []int{i * 10, 100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 600 + i, 700 + i})
	}

	if plannerEnabled.consultedCount == 0 {
		t.Fatalf("expected demotion decider to be consulted when demoteBeforeDrop=true")
	}
	if treeEnabled.Demotions() != plannerEnabled.consultedCount {
		t.Fatalf("treeEnabled.Demotions() = %d, want %d", treeEnabled.Demotions(), plannerEnabled.consultedCount)
	}
}
