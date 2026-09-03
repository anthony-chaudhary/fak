package polymodel

import (
	"testing"
)

func TestTreeVerifyAttentionMaskValidation(t *testing.T) {
	// Linear tree: 0 -> 1 -> 2 -> 3
	tree := BuildLinearTree(3, []int{101, 102, 103, 104})
	mask, err := BuildTreeAttentionMask(tree)
	if err != nil {
		t.Fatalf("BuildTreeAttentionMask failed: %v", err)
	}

	if err := ValidateTreeAttentionMask(tree, mask); err != nil {
		t.Fatalf("ValidateTreeAttentionMask failed on valid linear tree: %v", err)
	}

	// Root node (0) should only attend to 0
	if !mask[0][0] || mask[0][1] || mask[0][2] || mask[0][3] {
		t.Errorf("root node 0 attention mask invalid: %v", mask[0])
	}
	// Node 3 should attend to 0, 1, 2, 3
	for i := 0; i <= 3; i++ {
		if !mask[3][i] {
			t.Errorf("node 3 must attend to ancestor %d", i)
		}
	}

	// Adversarial: tamper with mask so node 1 attends to node 2 (causal violation)
	tamperedMask := make([][]bool, len(mask))
	for r := range mask {
		tamperedMask[r] = append([]bool(nil), mask[r]...)
	}
	tamperedMask[1][2] = true // future token leak!
	if err := ValidateTreeAttentionMask(tree, tamperedMask); err == nil {
		t.Fatal("expected error on tampered mask with future token attention, got nil")
	}
}

func TestTreeVerifyBranchingTopologyMasks(t *testing.T) {
	// Wide-shallow tree: root with 3 children
	tree := BuildWideShallowTree(3, 1, []int{101, 102, 103, 104})
	mask, err := BuildTreeAttentionMask(tree)
	if err != nil {
		t.Fatalf("BuildTreeAttentionMask failed on wide-shallow tree: %v", err)
	}
	if err := ValidateTreeAttentionMask(tree, mask); err != nil {
		t.Fatalf("ValidateTreeAttentionMask failed: %v", err)
	}

	// Sibling nodes should NOT attend to each other
	// Children of root are 1, 2, 3
	if mask[1][2] || mask[1][3] || mask[2][1] || mask[2][3] {
		t.Errorf("sibling nodes must not attend to each other: 1->2:%v, 2->1:%v", mask[1][2], mask[2][1])
	}

	// Deep-narrow tree
	dnTree := BuildDeepNarrowTree(4, nil)
	dnMask, err := BuildTreeAttentionMask(dnTree)
	if err != nil {
		t.Fatalf("BuildTreeAttentionMask failed on deep-narrow tree: %v", err)
	}
	if err := ValidateTreeAttentionMask(dnTree, dnMask); err != nil {
		t.Fatalf("ValidateTreeAttentionMask failed on deep-narrow: %v", err)
	}
}

func TestTreeVerifyMicrobenchmarkEnvelope(t *testing.T) {
	// Requirements from #10842:
	// - Exercise tree verification across backends (CPU, CUDA, Metal)
	// - K in {4, 8, 16, 32, 64}
	// - Batch sizes in {1, 2, 4, 8}
	// - Output structured benchmark JSON with columns: backend, batch_size, tree_size, tree_depth, single_token_us, tree_verify_us, overhead_ratio
	// - Assertion: batch=1 stays within 1.15x of single-token latency for trees up to 32 tokens on memory-bound backends

	backends := []string{"cpu", "metal", "cuda"}
	treeSizes := []int{4, 8, 16, 32, 64}
	batchSizes := []int{1, 2, 4, 8}

	var results []TreeVerificationBenchmarkResult

	for _, backend := range backends {
		for _, bSize := range batchSizes {
			for _, k := range treeSizes {
				tree := BuildLinearTree(k-1, nil) // creates tree with k nodes
				res := BenchmarkTreeVerification(backend, bSize, tree, 1200.0)
				results = append(results, res)

				// Core Assertion from issue #10842:
				// At batch=1 on memory-bound backends, tree verification stays within 1.15x
				// for trees up to 32 tokens.
				if bSize == 1 && k <= 32 {
					if res.OverheadRatio > 1.15 {
						t.Errorf("backend %s batch %d tree_size %d: overhead_ratio %.4f exceeded 1.15x threshold",
							backend, bSize, k, res.OverheadRatio)
					}
				}

				// Check KV accounting consistency
				if res.KeepKV+res.EvictKV != k-1 {
					t.Errorf("KV accounting mismatch for k=%d: KeepKV=%d + EvictKV=%d != %d",
						k, res.KeepKV, res.EvictKV, k-1)
				}
			}
		}
	}

	jsonStr, err := FormatBenchmarkJSON(results)
	if err != nil {
		t.Fatalf("FormatBenchmarkJSON failed: %v", err)
	}
	if len(jsonStr) == 0 {
		t.Fatal("empty JSON output")
	}
}

func TestTreeVerifyAcceptTreeKVExactness(t *testing.T) {
	// Build a deep-narrow tree where target matches the trunk for first 2 steps then diverges
	target := []int{101, 101, 1000, 102, 1000}
	tree := BuildDeepNarrowTree(2, target)

	res := AcceptTree(tree)
	// Must have accepted some path deterministically
	if res.Advance <= 0 {
		t.Fatalf("expected positive advance, got %d", res.Advance)
	}
	// Total speculative nodes
	numSpec := len(tree.Nodes) - 1
	if res.KeepKV+res.EvictKV != numSpec {
		t.Fatalf("KV conservation violated: KeepKV (%d) + EvictKV (%d) != Speculative Nodes (%d)",
			res.KeepKV, res.EvictKV, numSpec)
	}
}
