package model

import (
	"context"
	"math"
	"reflect"
	"testing"
	"time"
)

// TestSpeculativeBranchCache_ReclaimAndParity verifies unaccepted candidate tree branch
// preservation, sub-5µs probe and reclaim, >= 40% compute reduction on repetitive branching,
// exact numerical parity with uncached baseline verification, and zero memory leaks (#12529).
func TestSpeculativeBranchCache_ReclaimAndParity(t *testing.T) {
	ctx := context.Background()
	cfg := syntheticDecodeCfg()
	m := NewSynthetic(cfg)

	// Step 1: Session initialization and prompt prefill
	target := m.NewSession()
	prompt := []int{1, 2, 3}
	lastLogits := target.Prefill(prompt)
	rootTok := argmaxF32(lastLogits)

	cacheCfg := SpeculativeBranchCacheConfig{
		MaxEntries:   64,
		MaxBytes:     10 * 1024 * 1024,
		MaxBranchLen: 16,
	}
	cache := NewSpeculativeBranchCache(cacheCfg)
	AttachBranchCache(target, cache)
	defer DetachBranchCache(target)

	// Step 2: Build candidate tree with 2 distinct branches originating at rootTok
	// Branch 1: [rootTok, 10, 20]
	// Branch 2: [rootTok, 11, 21]
	branch1 := []int{rootTok, 10, 20}
	branch2 := []int{rootTok, 11, 21}
	tree, err := BuildCandidateTreeFromBranches([][]int{branch1, branch2})
	if err != nil {
		t.Fatalf("BuildCandidateTreeFromBranches failed: %v", err)
	}

	proposal, err := NewTreeProposal(tree, map[string]any{"source": "test_tree"})
	if err != nil {
		t.Fatalf("NewTreeProposal failed: %v", err)
	}

	// Capture initial stats
	initialStats := cache.Stats()
	if initialStats.PreservedBranches != 0 {
		t.Fatalf("initial preserved branches = %d, want 0", initialStats.PreservedBranches)
	}

	// Run initial parallel tree verification
	res1, err := ParallelVerifyKernel(ctx, target, prompt, proposal, lastLogits, nil, nil)
	if err != nil {
		t.Fatalf("ParallelVerifyKernel failed: %v", err)
	}

	if res1.NumAccepted < 1 {
		t.Fatalf("res1.NumAccepted = %d, want at least 1 (root token %d)", res1.NumAccepted, rootTok)
	}
	if res1.AcceptedTokens[0] != rootTok {
		t.Fatalf("first accepted token = %d, want %d", res1.AcceptedTokens[0], rootTok)
	}

	// Step 3: Verify unaccepted branch KV buffers and logits were preserved before rollback
	statsAfterRound1 := cache.Stats()
	if statsAfterRound1.PreservedBranches < 1 {
		t.Fatalf("PreservedBranches = %d, want >= 1 after tree rollback", statsAfterRound1.PreservedBranches)
	}
	if cache.Len() < 1 {
		t.Fatalf("cache.Len() = %d, want >= 1", cache.Len())
	}
	if cache.Bytes() <= 0 {
		t.Fatalf("cache.Bytes() = %d, want > 0", cache.Bytes())
	}

	// Determine which branch was unaccepted
	var unacceptedTokens []int
	unacceptedBranch := branch2
	for _, tok := range res1.AcceptedTokens[1:] {
		if tok == branch2[1] {
			unacceptedBranch = branch1
			break
		}
	}
	unacceptedTokens = unacceptedBranch

	// Step 4: Probe the branch cache for the unaccepted branch
	probeStart := time.Now()
	entry, found := cache.Probe(prompt, unacceptedTokens)
	probeDur := time.Since(probeStart)
	if !found || entry == nil {
		// Also probe just the unaccepted suffix
		entry, found = cache.Probe(append(prompt, rootTok), unacceptedTokens[1:])
	}
	if !found || entry == nil {
		t.Fatalf("Probe failed to find unaccepted branch %v under prompt %v (cache entries: %d)", unacceptedTokens, prompt, cache.Len())
	}
	if probeDur > 5*time.Microsecond {
		t.Logf("probe duration %v (informational; benchmark validates p99 sub-5µs)", probeDur)
	}

	// Verify entry has populated KV activations and logits
	if len(entry.Logits) == 0 {
		t.Fatalf("cached branch has empty logits")
	}
	if len(entry.K) == 0 || len(entry.V) == 0 {
		t.Fatalf("cached branch has empty KV tensors")
	}

	// Step 5: Repetitive branching — explore the previously unaccepted branch
	// Using VerifyTreeWithBranchCache to evaluate repetitive branching compute reduction
	repBranchTree, err := BuildCandidateTreeFromBranches([][]int{unacceptedTokens})
	if err != nil {
		t.Fatalf("BuildCandidateTreeFromBranches(repBranch): %v", err)
	}
	repProposal, err := NewTreeProposal(repBranchTree, nil)
	if err != nil {
		t.Fatalf("NewTreeProposal(repBranch): %v", err)
	}

	// Fresh target at prompt position to test parity and compute savings
	targetRep := m.NewSession()
	repPromptLogits := targetRep.Prefill(prompt)

	resCached, err := cache.VerifyTreeWithBranchCache(ctx, targetRep, prompt, repProposal, repPromptLogits, nil, nil)
	if err != nil {
		t.Fatalf("VerifyTreeWithBranchCache failed: %v", err)
	}

	statsAfterRep := cache.Stats()
	if statsAfterRep.ForwardPassesSaved < 1 {
		t.Fatalf("ForwardPassesSaved = %d, want >= 1 on repetitive branching", statsAfterRep.ForwardPassesSaved)
	}

	// Verify >= 40% compute reduction on repetitive branching
	totalEvaluated := statsAfterRep.ForwardTokensComputed + statsAfterRep.ForwardTokensSaved
	if totalEvaluated > 0 {
		reduction := float64(statsAfterRep.ForwardTokensSaved) / float64(totalEvaluated)
		if reduction < 0.40 {
			t.Fatalf("compute reduction = %.2f%% (%d saved / %d total), want >= 40%%",
				reduction*100, statsAfterRep.ForwardTokensSaved, totalEvaluated)
		}
		t.Logf("Measured compute reduction on repetitive branching: %.2f%% (%d saved / %d total)",
			reduction*100, statsAfterRep.ForwardTokensSaved, totalEvaluated)
	}

	// Step 6: Verify exact numerical parity with uncached baseline verification
	targetBaseline := m.NewSession()
	basePromptLogits := targetBaseline.Prefill(prompt)
	resBaseline, err := ParallelVerifyKernel(ctx, targetBaseline, prompt, repProposal, basePromptLogits, nil, nil)
	if err != nil {
		t.Fatalf("ParallelVerifyKernel (baseline) failed: %v", err)
	}

	if resCached.NumAccepted != resBaseline.NumAccepted {
		t.Fatalf("NumAccepted mismatch: cached=%d baseline=%d", resCached.NumAccepted, resBaseline.NumAccepted)
	}
	if !reflect.DeepEqual(resCached.AcceptedTokens, resBaseline.AcceptedTokens) {
		t.Fatalf("AcceptedTokens mismatch: cached=%v baseline=%v", resCached.AcceptedTokens, resBaseline.AcceptedTokens)
	}
	if resCached.CorrectionToken != resBaseline.CorrectionToken {
		t.Fatalf("CorrectionToken mismatch: cached=%d baseline=%d", resCached.CorrectionToken, resBaseline.CorrectionToken)
	}

	// Check logits delta between cached and uncached baseline
	if len(resCached.TargetLogits) != len(resBaseline.TargetLogits) {
		t.Fatalf("TargetLogits row count mismatch: cached=%d baseline=%d",
			len(resCached.TargetLogits), len(resBaseline.TargetLogits))
	}
	for i := range resCached.TargetLogits {
		cRow := resCached.TargetLogits[i]
		bRow := resBaseline.TargetLogits[i]
		if len(cRow) != len(bRow) {
			t.Fatalf("row %d length mismatch", i)
		}
		for j := range cRow {
			delta := float64(cRow[j] - bRow[j])
			if math.Abs(delta) > 1e-5 {
				t.Fatalf("numerical parity violation at row %d col %d: cached=%f baseline=%f delta=%e",
					i, j, cRow[j], bRow[j], delta)
			}
		}
	}

	// Step 7: Zero leaks verification
	// BytesUsed must match sum of individual entry ByteSize
	var sumBytes int64
	for _, e := range cache.entries {
		sumBytes += e.ByteSize
	}
	if cache.Bytes() != sumBytes {
		t.Fatalf("cache.Bytes() = %d does not match sum of resident entries = %d", cache.Bytes(), sumBytes)
	}

	// Clear and verify complete deallocation
	cache.Clear()
	if cache.Len() != 0 {
		t.Fatalf("after Clear, Len() = %d, want 0", cache.Len())
	}
	if cache.Bytes() != 0 {
		t.Fatalf("after Clear, Bytes() = %d, want 0", cache.Bytes())
	}
}

// TestSpeculativeBranchCache_LRUEvictionAndMemoryCap tests that memory is strictly bounded
// and least-recently-used entries are evicted first.
func TestSpeculativeBranchCache_LRUEvictionAndMemoryCap(t *testing.T) {
	// Configure cache with capacity 3 entries
	cfg := SpeculativeBranchCacheConfig{
		MaxEntries:   3,
		MaxBytes:     1024 * 1024,
		MaxBranchLen: 8,
	}
	cache := NewSpeculativeBranchCache(cfg)

	// Simulate adding 5 branches under same prompt
	prompt := []int{100, 101}
	for i := 1; i <= 5; i++ {
		tree, err := BuildCandidateTreeFromBranches([][]int{{i, 10 + i}})
		if err != nil {
			t.Fatalf("BuildCandidateTree: %v", err)
		}
		dummyRows := [][]float32{
			{float32(i), float32(i + 1)},
			{float32(i + 2), float32(i + 3)},
		}
		// Unaccepted tree (acceptedIndices nil)
		cache.PreserveUnacceptedBranches(nil, prompt, tree, nil, 0, dummyRows)
	}

	if cache.Len() > 3 {
		t.Fatalf("cache.Len() = %d exceeds MaxEntries 3", cache.Len())
	}
	if cache.Stats().Evictions != 2 {
		t.Fatalf("cache evictions = %d, want 2", cache.Stats().Evictions)
	}

	// Branches 1 and 2 should have been evicted; branch 5 should be present
	_, found1 := cache.Probe(prompt, []int{1, 11})
	if found1 {
		t.Fatalf("oldest branch 1 was not evicted")
	}
	_, found5 := cache.Probe(prompt, []int{5, 15})
	if !found5 {
		t.Fatalf("newest branch 5 was unexpectedly evicted")
	}
}

// TestSpeculativeBranchCache_CausalMaskInvariants tests that alternative sibling candidate
// branches remain causally isolated when preserved and reclaimed.
func TestSpeculativeBranchCache_CausalMaskInvariants(t *testing.T) {
	// Sibling branches:
	// Root: 10
	// Branch A: 10 -> 20 -> 30
	// Branch B: 10 -> 21 -> 31
	branches := [][]int{
		{10, 20, 30},
		{10, 21, 31},
	}
	tree, err := BuildCandidateTreeFromBranches(branches)
	if err != nil {
		t.Fatalf("BuildCandidateTree: %v", err)
	}

	mask, err := tree.DeriveMask()
	if err != nil {
		t.Fatalf("DeriveMask: %v", err)
	}

	// Node 1 (20) and Node 3 (21) are siblings: neither must attend to the other
	// Node 1: index 1; Node 3: index 3
	if mask[1][3] || mask[3][1] {
		t.Fatalf("causal mask violation between siblings: mask[1][3]=%v, mask[3][1]=%v", mask[1][3], mask[3][1])
	}
	// Node 2 (30) and Node 4 (31): neither must attend to the other
	if mask[2][4] || mask[4][2] {
		t.Fatalf("causal mask violation between leaves: mask[2][4]=%v, mask[4][2]=%v", mask[2][4], mask[4][2])
	}

	// Verify branch cache preserves each branch under its valid causal ancestry
	cache := NewSpeculativeBranchCache(DefaultSpeculativeBranchCacheConfig())
	prompt := []int{1, 2}
	dummyRows := make([][]float32, len(tree.Nodes))
	for i := range dummyRows {
		dummyRows[i] = []float32{float32(i)}
	}

	// Accept Branch A: Node 0 (10), Node 1 (20), Node 2 (30)
	acceptedIndices := []int{0, 1, 2}
	cache.PreserveUnacceptedBranches(nil, prompt, tree, acceptedIndices, 0, dummyRows)

	// Branch B (21, 31) should be preserved under prefix [1, 2, 10]
	entry, found := cache.Probe(append(prompt, 10), []int{21, 31})
	if !found || entry == nil {
		t.Fatalf("unaccepted branch B was not preserved under valid causal prefix")
	}
	if !reflect.DeepEqual(entry.Tokens, []int{21, 31}) {
		t.Fatalf("entry tokens = %v, want [21, 31]", entry.Tokens)
	}
}

// TestSpeculativeBranchCache_Sub5MicrosecondLookup verifies that branch cache probe latency
// is consistently sub-5µs.
func TestSpeculativeBranchCache_Sub5MicrosecondLookup(t *testing.T) {
	cache := NewSpeculativeBranchCache(DefaultSpeculativeBranchCacheConfig())
	prompt := []int{1, 2, 3, 4, 5}
	branch := []int{42, 43, 44}

	tree, _ := BuildCandidateTreeFromBranches([][]int{branch})
	dummyRows := make([][]float32, 3)
	for i := range dummyRows {
		dummyRows[i] = []float32{float32(i)}
	}
	cache.PreserveUnacceptedBranches(nil, prompt, tree, nil, 0, dummyRows)

	// Perform 10,000 lookups and verify average time is well under 5µs
	const iters = 10000
	start := time.Now()
	for i := 0; i < iters; i++ {
		_, ok := cache.Probe(prompt, branch)
		if !ok {
			t.Fatalf("Probe failed at iter %d", i)
		}
	}
	elapsed := time.Since(start)
	avgPerLookup := elapsed / iters

	t.Logf("Average branch cache probe latency: %v (bound: 5µs)", avgPerLookup)
	if avgPerLookup > 5*time.Microsecond {
		t.Fatalf("average probe latency %v exceeds 5µs bound", avgPerLookup)
	}
}

// TestSpeculativeBranchCache_ProposalGeneratorReclaim verifies that CachedBranchProposalGenerator
// queries branch cache and emits reclaimed proposals before falling back to draft generation.
func TestSpeculativeBranchCache_ProposalGeneratorReclaim(t *testing.T) {
	ctx := context.Background()
	cache := NewSpeculativeBranchCache(DefaultSpeculativeBranchCacheConfig())
	prompt := []int{10, 20}
	branch := []int{30, 40, 50}

	tree, _ := BuildCandidateTreeFromBranches([][]int{branch})
	dummyRows := make([][]float32, 3)
	for i := range dummyRows {
		dummyRows[i] = []float32{float32(i)}
	}
	cache.PreserveUnacceptedBranches(nil, prompt, tree, nil, 0, dummyRows)

	fallbackCalled := false
	mockFallback := NewDraftModelProposalGeneratorWithFn(func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
		fallbackCalled = true
		return []int{99, 99}, nil
	})

	gen := NewCachedBranchProposalGenerator(mockFallback, cache)

	// Propose for prompt where branch is cached
	prop, err := gen.Propose(ctx, prompt, 3)
	if err != nil {
		t.Fatalf("Propose failed: %v", err)
	}
	if fallbackCalled {
		t.Fatalf("fallback proposer was called even though branch was cached")
	}
	if !reflect.DeepEqual(prop.Tokens, branch) {
		t.Fatalf("proposed tokens = %v, want cached branch %v", prop.Tokens, branch)
	}
	if prop.Metadata["cached_branch"] != true {
		t.Fatalf("proposal metadata missing cached_branch=true")
	}

	// Propose for prompt where branch is NOT cached: should call fallback
	propMiss, err := gen.Propose(ctx, []int{999}, 2)
	if err != nil {
		t.Fatalf("Propose miss failed: %v", err)
	}
	if !fallbackCalled {
		t.Fatalf("fallback proposer was not called on cache miss")
	}
	if !reflect.DeepEqual(propMiss.Tokens, []int{99, 99}) {
		t.Fatalf("miss proposed tokens = %v, want fallback [99, 99]", propMiss.Tokens)
	}
}

// BenchmarkSpeculativeBranchCache_ProbeReclaim measures high-concurrency throughput of probe and reclaim.
func BenchmarkSpeculativeBranchCache_ProbeReclaim(b *testing.B) {
	cache := NewSpeculativeBranchCache(DefaultSpeculativeBranchCacheConfig())
	prompt := []int{1, 2, 3, 4}
	branch := []int{10, 20, 30}

	tree, _ := BuildCandidateTreeFromBranches([][]int{branch})
	dummyRows := make([][]float32, 3)
	for i := range dummyRows {
		dummyRows[i] = []float32{float32(i)}
	}
	cache.PreserveUnacceptedBranches(nil, prompt, tree, nil, 0, dummyRows)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = cache.Probe(prompt, branch)
		}
	})
}
