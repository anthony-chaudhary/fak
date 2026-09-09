package radixkv

import (
	"errors"
	"testing"
)

// TestPriorityRootPrefixPreservedUnderSeverePressure is the issue #12327 acceptance witness:
// verifying that a Tier 0 coordinator prompt prefix is pinned and immune from eviction (seg = 3)
// even when subagent branches collapse under severe memory pressure.
func TestPriorityRootPrefixPreservedUnderSeverePressure(t *testing.T) {
	// Budget of 50 tokens: sufficient for coordinator prefix (40 tokens),
	// but under pressure once subagents attach (40 + 3*20 = 100 tokens).
	tree := New(50)

	coordPrompt := seqTokens(1, 40)
	coordNode := tree.PinPrefix(coordPrompt)
	if coordNode == nil {
		t.Fatal("PinPrefix returned nil")
	}
	if !tree.IsNodePinned(coordNode) {
		t.Fatal("coordNode should report IsNodePinned=true")
	}
	if tier := tree.NodeTier(coordNode); tier != Tier0PinnedRoot {
		t.Fatalf("coordNode NodeTier=%v, want Tier0PinnedRoot", tier)
	}

	// Attach 3 distinct subagent branches extending the coordinator prompt.
	subA := cat(coordPrompt, seqTokens(10, 20))
	subB := cat(coordPrompt, seqTokens(20, 20))
	subC := cat(coordPrompt, seqTokens(30, 20))

	for _, sub := range [][]int{subA, subB, subC} {
		b, m := tree.Lookup(sub)
		if m != len(coordPrompt) {
			t.Fatalf("subagent lookup matched %d, want %d (full coord prompt)", m, len(coordPrompt))
		}
		leaf := tree.Insert(b, sub[m:], nil)
		tree.Done(leaf)
	}

	// Under severe budget pressure (100 tokens cached > 50 budget), evictToBudget runs.
	// Reduce budget further to 20 tokens: strictly LESS than the coordinator prompt (40 tokens).
	tree.SetRetention(20)

	// All subagent branches must be pruned away, collapsing the tree back to the coordinator prefix.
	// Even though 40 tokens > 20 budget and the coordinator node has now become a leaf,
	// it must be completely preserved because Tier 0 is immune (seg = 3).
	if m := tree.MatchLen(coordPrompt); m != len(coordPrompt) {
		t.Fatalf("coordinator prefix must survive severe budget pressure: matched %d/%d", m, len(coordPrompt))
	}

	// Verify subagent branch suffixes were evicted.
	if m := tree.MatchLen(subA); m != len(coordPrompt) {
		t.Errorf("subA suffix should be evicted, but matched %d (expected %d)", m, len(coordPrompt))
	}
	if m := tree.MatchLen(subB); m != len(coordPrompt) {
		t.Errorf("subB suffix should be evicted, but matched %d (expected %d)", m, len(coordPrompt))
	}
	if m := tree.MatchLen(subC); m != len(coordPrompt) {
		t.Errorf("subC suffix should be evicted, but matched %d (expected %d)", m, len(coordPrompt))
	}

	// Even with budget 0 (keep-none), pinned Tier 0 nodes must remain immune.
	tree.SetRetention(0)
	if m := tree.MatchLen(coordPrompt); m != len(coordPrompt) {
		t.Fatalf("coordinator prefix must survive keep-none retention: matched %d/%d", m, len(coordPrompt))
	}
}

// TestPriorityTieredEvictionOrder verifies the strict eviction hierarchy:
// Tier 3 (probationary tool outputs) -> Tier 2 (idle subagents) -> Tier 1 (active subagents) -> Tier 0 (pinned root, immune).
func TestPriorityTieredEvictionOrder(t *testing.T) {
	// Initialize tree with sufficient budget so all subagent branches can be populated
	// and tiered before applying budget pressure.
	tree := New(100)

	coordPrompt := seqTokens(1, 20)
	tree.PinPrefix(coordPrompt)

	// Subagent 1: Tier 1 (active subagent)
	sub1Prompt := cat(coordPrompt, seqTokens(10, 10))
	b1, m1 := tree.Lookup(sub1Prompt)
	leaf1 := tree.Insert(b1, sub1Prompt[m1:], nil)
	tree.Done(leaf1)
	tree.SetNodeTier(leaf1, Tier1ActiveSubagent)

	// Subagent 2: Tier 2 (idle subagent)
	sub2Prompt := cat(coordPrompt, seqTokens(20, 10))
	b2, m2 := tree.Lookup(sub2Prompt)
	leaf2 := tree.Insert(b2, sub2Prompt[m2:], nil)
	tree.Done(leaf2)
	tree.SetNodeTier(leaf2, Tier2IdleSubagent)

	// Subagent 3: Tier 3 (probationary tool outputs)
	sub3Prompt := cat(coordPrompt, seqTokens(30, 10))
	b3, m3 := tree.Lookup(sub3Prompt)
	leaf3 := tree.Insert(b3, sub3Prompt[m3:], nil)
	tree.Done(leaf3)
	tree.SetNodeTier(leaf3, Tier3Probationary)

	// Total tokens cached: 20 (coord) + 3*10 = 50 tokens.
	// Step 1: Reduce budget to 40 tokens (evict 1 leaf).
	tree.SetRetention(40)

	// Tier 3 (probationary tool outputs) must be the first victim evicted.
	if m := tree.MatchLen(sub3Prompt); m != len(coordPrompt) {
		t.Fatalf("Tier 3 probationary leaf should have been evicted first, matched %d/%d", m, len(sub3Prompt))
	}
	if m := tree.MatchLen(sub2Prompt); m != len(sub2Prompt) {
		t.Fatalf("Tier 2 idle leaf should still be resident, matched %d/%d", m, len(sub2Prompt))
	}
	if m := tree.MatchLen(sub1Prompt); m != len(sub1Prompt) {
		t.Fatalf("Tier 1 active leaf should still be resident, matched %d/%d", m, len(sub1Prompt))
	}

	// Step 2: Reduce budget to 30 tokens (evict another leaf).
	tree.SetRetention(30)

	// Tier 2 (idle subagent) must be the next victim evicted.
	if m := tree.MatchLen(sub2Prompt); m != len(coordPrompt) {
		t.Fatalf("Tier 2 idle leaf should have been evicted next, matched %d/%d", m, len(sub2Prompt))
	}
	if m := tree.MatchLen(sub1Prompt); m != len(sub1Prompt) {
		t.Fatalf("Tier 1 active leaf should still be resident, matched %d/%d", m, len(sub1Prompt))
	}

	// Step 3: Reduce budget to 20 tokens (evict remaining subagent).
	tree.SetRetention(20)

	// Tier 1 (active subagent) is evicted when no Tier 2 or 3 leaves remain.
	if m := tree.MatchLen(sub1Prompt); m != len(coordPrompt) {
		t.Fatalf("Tier 1 active leaf should have been evicted, matched %d/%d", m, len(sub1Prompt))
	}

	// Coordinator prefix (Tier 0) must remain resident and immune.
	if m := tree.MatchLen(coordPrompt); m != len(coordPrompt) {
		t.Fatalf("Tier 0 pinned root must be immune, matched %d/%d", m, len(coordPrompt))
	}
}

// TestPrioritySetNodeRetentionValidation tests validation and tier mapping of RetentionRequest.
func TestPrioritySetNodeRetentionValidation(t *testing.T) {
	tree := New(100)
	prompt := seqTokens(1, 10)
	b, m := tree.Lookup(prompt)
	leaf := tree.Insert(b, prompt[m:], nil)
	tree.Done(leaf)

	// Out-of-range priorities fail closed.
	if err := tree.SetNodeRetention(leaf, RetentionRequest{Priority: -1}); !errors.Is(err, ErrRetentionPriorityRange) {
		t.Errorf("expected ErrRetentionPriorityRange for priority -1, got %v", err)
	}
	if err := tree.SetNodeRetention(leaf, RetentionRequest{Priority: 101}); !errors.Is(err, ErrRetentionPriorityRange) {
		t.Errorf("expected ErrRetentionPriorityRange for priority 101, got %v", err)
	}

	// Negative TTL fails closed.
	if err := tree.SetNodeRetention(leaf, RetentionRequest{Priority: 50, TTL: -1}); !errors.Is(err, ErrRetentionTTLNegative) {
		t.Errorf("expected ErrRetentionTTLNegative, got %v", err)
	}

	// Negative Admitted tick fails closed.
	if err := tree.SetNodeRetention(leaf, RetentionRequest{Priority: 50, Admitted: -1}); !errors.Is(err, ErrRetentionAdmittedNegative) {
		t.Errorf("expected ErrRetentionAdmittedNegative, got %v", err)
	}

	// Valid priority 100 sets Tier 0 and pinned.
	req := RetentionRequest{Priority: MaxRetentionPriority, TTL: RetainForever, Admitted: 0}
	if err := tree.SetNodeRetention(leaf, req); err != nil {
		t.Fatalf("SetNodeRetention failed: %v", err)
	}
	if !tree.IsNodePinned(leaf) {
		t.Error("Priority 100 should mark node as pinned")
	}
	if tier := tree.NodeTier(leaf); tier != Tier0PinnedRoot {
		t.Errorf("expected Tier0PinnedRoot, got %v", tier)
	}

	gotReq, ok := tree.NodeRetention(leaf)
	if !ok || gotReq.Priority != MaxRetentionPriority {
		t.Errorf("NodeRetention mismatch: ok=%v, got=%+v", ok, gotReq)
	}
}

// TestPriorityRetentionTTLExpiration verifies that a TTL-expired retention request
// demotes to Tier 3 (probationary) and is evicted before unexpired nodes.
func TestPriorityRetentionTTLExpiration(t *testing.T) {
	tree := New(50)

	// Leaf 1: High priority (Tier 1, priority 75) but with a short TTL window that has expired.
	prompt1 := seqTokens(1, 10)
	b1, m1 := tree.Lookup(prompt1)
	leaf1 := tree.Insert(b1, prompt1[m1:], nil)
	tree.Done(leaf1)
	if err := tree.SetNodeRetention(leaf1, RetentionRequest{
		Priority: 75,
		TTL:      5,
		Admitted: 1, // expired once clock > 6
	}); err != nil {
		t.Fatalf("SetNodeRetention leaf1: %v", err)
	}

	// Leaf 2: Default priority (Tier 2, priority 35) with forever retention (unexpired).
	prompt2 := seqTokens(2, 10)
	b2, m2 := tree.Lookup(prompt2)
	leaf2 := tree.Insert(b2, prompt2[m2:], nil)
	tree.Done(leaf2)
	if err := tree.SetNodeRetention(leaf2, RetentionRequest{
		Priority: DefaultRetentionPriority,
		TTL:      RetainForever,
		Admitted: 1,
	}); err != nil {
		t.Fatalf("SetNodeRetention leaf2: %v", err)
	}

	// Advance clock past leaf1's TTL.
	for i := 0; i < 10; i++ {
		tree.tick()
	}

	// Verify leaf1 has expired and demoted to Tier 3 probationary.
	if tier := tree.NodeTier(leaf1); tier != Tier3Probationary {
		t.Fatalf("leaf1 should demote to Tier3Probationary on expiration, got %v", tier)
	}

	// Under budget pressure, the expired leaf1 (now Tier 3) must be evicted before leaf2 (Tier 2).
	tree.SetRetention(10)

	if m := tree.MatchLen(prompt1); m != 0 {
		t.Fatalf("expired leaf1 should have been evicted first, matched %d", m)
	}
	if m := tree.MatchLen(prompt2); m != len(prompt2) {
		t.Fatalf("unexpired leaf2 should still be resident, matched %d/%d", m, len(prompt2))
	}
}

// TestPriorityEdgeSplitPreservesPinning verifies that splitting an edge of a pinned coordinator
// prefix preserves Tier 0 pinning across both intermediate and leaf boundaries.
func TestPriorityEdgeSplitPreservesPinning(t *testing.T) {
	tree := New(20)

	coordPrompt := seqTokens(1, 20)
	coordNode := tree.PinPrefix(coordPrompt)
	if !tree.IsNodePinned(coordNode) {
		t.Fatal("coordNode should be pinned")
	}

	// Diverge at offset 10 to cause a mid-edge split.
	divergent := cat(coordPrompt[:10], seqTokens(99, 10))
	b, m := tree.Lookup(divergent)
	if m != 10 {
		t.Fatalf("lookup matched %d, want 10", m)
	}
	leafDiv := tree.Insert(b, divergent[m:], nil)
	tree.Done(leafDiv)

	// Both the split intermediate node and the original tail node must remain pinned.
	if !tree.IsNodePinned(b) {
		t.Error("intermediate split boundary node should inherit pinned status")
	}

	// Apply severe pressure: budget 10 tokens (less than the 20 coordinator tokens).
	tree.SetRetention(10)

	// The unpinned divergent branch should be evicted.
	if m := tree.MatchLen(divergent); m != 10 {
		t.Errorf("divergent suffix should be evicted, matched %d (expected 10)", m)
	}

	// The full 20-token coordinator prompt must still match completely.
	if m := tree.MatchLen(coordPrompt); m != len(coordPrompt) {
		t.Fatalf("full coordinator prompt must survive edge split: matched %d/%d", m, len(coordPrompt))
	}
}

// TestPriorityStrategyNameResolution verifies that "priority" and "priority-tier" strategies
// are registered in the eviction strategy factory.
func TestPriorityStrategyNameResolution(t *testing.T) {
	tree, err := NewWithEvictionStrategy(50, "priority")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(priority): %v", err)
	}
	if st := tree.Stats().EvictionPolicy; st != "priority" {
		t.Errorf("EvictionPolicy=%q, want 'priority'", st)
	}

	tree2, err := NewWithEvictionStrategy(50, "priority-tier")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(priority-tier): %v", err)
	}
	if st := tree2.Stats().EvictionPolicy; st != "priority" {
		t.Errorf("EvictionPolicy=%q, want 'priority'", st)
	}
}

// TestPriorityInsertWithTierAndRetention verifies InsertWithTier and InsertWithRetention convenience methods.
func TestPriorityInsertWithTierAndRetention(t *testing.T) {
	tree := New(100)
	coordPrompt := seqTokens(1, 10)
	tree.PinPrefix(coordPrompt)

	sub1 := cat(coordPrompt, seqTokens(10, 10))
	b1, m1 := tree.Lookup(sub1)
	leaf1 := tree.InsertWithTier(b1, sub1[m1:], nil, Tier1ActiveSubagent)
	tree.Done(leaf1)

	if tier := tree.NodeTier(leaf1); tier != Tier1ActiveSubagent {
		t.Fatalf("expected Tier1ActiveSubagent, got %v", tier)
	}

	sub2 := cat(coordPrompt, seqTokens(20, 10))
	b2, m2 := tree.Lookup(sub2)
	req := RetentionRequest{Priority: 35, TTL: RetainForever, Admitted: 0}
	leaf2, err := tree.InsertWithRetention(b2, sub2[m2:], nil, req)
	if err != nil {
		t.Fatalf("InsertWithRetention failed: %v", err)
	}
	tree.Done(leaf2)

	if tier := tree.NodeTier(leaf2); tier != Tier2IdleSubagent {
		t.Fatalf("expected Tier2IdleSubagent, got %v", tier)
	}
}
