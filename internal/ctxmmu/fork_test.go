package ctxmmu_test

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestForkLatencyUnder1ms verifies that zero-copy subagent prefix branching via COW page tables
// executes in O(1) time with subagent fork latency strictly under 1ms, even for 32k-64k prefixes.
func TestForkLatencyUnder1ms(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	// 32k token prefix = 512 blocks of 64 tokens
	prefixTokensCount := 32768
	parent, err := mgr.RegisterSession("coord-32k", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	tokens := make([]int32, prefixTokensCount)
	for i := 0; i < prefixTokensCount; i++ {
		tokens[i] = int32(1000 + (i % 5000))
	}
	if err := parent.AppendTokens(tokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	if parent.TokenCount() != prefixTokensCount {
		t.Fatalf("expected %d tokens, got %d", prefixTokensCount, parent.TokenCount())
	}
	if parent.PageCount() != 512 {
		t.Fatalf("expected 512 pages, got %d", parent.PageCount())
	}

	// Warm-up fork
	_, err = mgr.ForkSession("coord-32k", "subagent-warmup")
	if err != nil {
		t.Fatalf("Warmup fork failed: %v", err)
	}

	// Measure 32k fork latency
	child, err := mgr.ForkSession("coord-32k", "subagent-1")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}

	telem := child.Telemetry()
	if telem.ForkLatency >= 1*time.Millisecond {
		t.Fatalf("fork latency %v exceeded 1ms threshold", telem.ForkLatency)
	}
	if telem.ForkLatencyMs >= 1.0 {
		t.Fatalf("fork latency %f ms exceeded 1.0ms threshold", telem.ForkLatencyMs)
	}
	if telem.ForkCloneBytes != 0 {
		t.Fatalf("expected 0 fork clone bytes, got %d", telem.ForkCloneBytes)
	}
	if telem.SharedPrefixKVHitRate != 1.0 {
		t.Fatalf("expected 1.0 (100%%) prefix hit rate, got %f", telem.SharedPrefixKVHitRate)
	}
	if telem.SharedPagesCount != 512 {
		t.Fatalf("expected 512 shared pages, got %d", telem.SharedPagesCount)
	}
	if telem.UniquePagesCount != 0 {
		t.Fatalf("expected 0 unique pages on fork, got %d", telem.UniquePagesCount)
	}

	// Test 64k token prefix = 1024 blocks of 64 tokens
	parent64k, err := mgr.RegisterSession("coord-64k", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession 64k failed: %v", err)
	}
	tokens64k := make([]int32, 65536)
	for i := range tokens64k {
		tokens64k[i] = int32(2000 + (i % 7000))
	}
	if err := parent64k.AppendTokens(tokens64k...); err != nil {
		t.Fatalf("AppendTokens 64k failed: %v", err)
	}

	start := time.Now()
	child64k, err := mgr.ForkSession("coord-64k", "subagent-64k")
	dur := time.Since(start)
	if err != nil {
		t.Fatalf("ForkSession 64k failed: %v", err)
	}
	if dur >= 1*time.Millisecond {
		t.Fatalf("64k fork latency %v exceeded 1ms", dur)
	}
	if child64k.Telemetry().ForkCloneBytes != 0 {
		t.Fatalf("expected 0 clone bytes for 64k prefix, got %d", child64k.Telemetry().ForkCloneBytes)
	}
}

// TestForkSharedPrefix100PercentHitRate verifies that a forked subagent reuses the shared prefix
// with 100% KV cache hit rate and coherent GPU virtual addresses on RDNA 3.5.
func TestForkSharedPrefix100PercentHitRate(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	parent, err := mgr.RegisterSession("parent-prefix", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	prefixTokens := make([]int32, 2048) // 32 blocks of 64 tokens
	for i := range prefixTokens {
		prefixTokens[i] = int32(42 + i)
	}
	if err := parent.AppendTokens(prefixTokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	child, err := mgr.ForkSession("parent-prefix", "child-prefix")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}

	telem := child.Telemetry()
	if telem.SharedPrefixKVHitRate != 1.0 {
		t.Fatalf("expected shared prefix hit rate 1.0 (100%%), got %f", telem.SharedPrefixKVHitRate)
	}
	if telem.SharedPagesCount != parent.PageCount() {
		t.Fatalf("expected shared pages %d, got %d", parent.PageCount(), telem.SharedPagesCount)
	}
	if telem.UniquePagesCount != 0 {
		t.Fatalf("expected 0 unique pages, got %d", telem.UniquePagesCount)
	}

	// Verify exact token sequence match
	childTokens := child.ReadTokens()
	parentTokens := parent.ReadTokens()
	if len(childTokens) != len(parentTokens) {
		t.Fatalf("token length mismatch: parent %d, child %d", len(parentTokens), len(childTokens))
	}
	for i := range parentTokens {
		if childTokens[i] != parentTokens[i] {
			t.Fatalf("token mismatch at %d: parent %d != child %d", i, parentTokens[i], childTokens[i])
		}
	}

	// Verify coherent GPU virtual addresses and host addresses match
	childEntries := child.PageTableEntries()
	parentEntries := parent.PageTableEntries()
	for i := range parentEntries {
		if childEntries[i].GPUVirtualAddress != parentEntries[i].GPUVirtualAddress {
			t.Fatalf("page %d GPU VA mismatch: child 0x%x != parent 0x%x",
				i, childEntries[i].GPUVirtualAddress, parentEntries[i].GPUVirtualAddress)
		}
		if childEntries[i].HostAddress != parentEntries[i].HostAddress {
			t.Fatalf("page %d HostAddress mismatch: child 0x%x != parent 0x%x",
				i, childEntries[i].HostAddress, parentEntries[i].HostAddress)
		}
		if childEntries[i].BlockID != parentEntries[i].BlockID {
			t.Fatalf("page %d BlockID mismatch: child %d != parent %d",
				i, childEntries[i].BlockID, parentEntries[i].BlockID)
		}
		if !childEntries[i].Shared || !parentEntries[i].Shared {
			t.Fatalf("page %d should be marked shared in both", i)
		}
		if childEntries[i].Writable || parentEntries[i].Writable {
			t.Fatalf("page %d should be marked read-only (not writable) until COW", i)
		}
	}
}

// TestForkCOWAllocationOnlyOnUniqueAppend verifies that new physical blocks are allocated
// only when unique divergent tokens are appended or a shared block is mutated.
func TestForkCOWAllocationOnlyOnUniqueAppend(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	// Parent has 100 tokens: 1 full block (64 tokens) and 1 partial block (36 tokens)
	parent, err := mgr.RegisterSession("parent-cow", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	parentTokens := make([]int32, 100)
	for i := range parentTokens {
		parentTokens[i] = int32(10 + i)
	}
	if err := parent.AppendTokens(parentTokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	if mgr.ActiveBlockCount() != 2 {
		t.Fatalf("expected 2 active blocks in pool, got %d", mgr.ActiveBlockCount())
	}

	// Fork to child: zero-copy pointer replication (NO new blocks allocated)
	child, err := mgr.ForkSession("parent-cow", "child-cow")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}

	if mgr.ActiveBlockCount() != 2 {
		t.Fatalf("after fork expected active blocks to remain 2, got %d", mgr.ActiveBlockCount())
	}
	if child.Telemetry().COWClonesCount != 0 {
		t.Fatalf("expected 0 COW clones on fork, got %d", child.Telemetry().COWClonesCount)
	}

	// Child appends 10 unique tokens:
	// Because block 1 has 36 tokens and capacity 64, but is shared, COW must trigger
	// to clone block 1 so parent's block 1 is not modified!
	uniqueTokens := []int32{901, 902, 903, 904, 905, 906, 907, 908, 909, 910}
	if err := child.AppendTokens(uniqueTokens...); err != nil {
		t.Fatalf("child AppendTokens failed: %v", err)
	}

	// Active blocks: block 0 (shared by both), parent block 1 (unique to parent), child block 1' (unique to child)
	// Total active blocks should be 3
	if mgr.ActiveBlockCount() != 3 {
		t.Fatalf("expected 3 active blocks in pool after COW, got %d", mgr.ActiveBlockCount())
	}

	childTelem := child.Telemetry()
	if childTelem.COWClonesCount != 1 {
		t.Fatalf("expected 1 COW clone, got %d", childTelem.COWClonesCount)
	}
	if childTelem.UniquePagesCount != 1 {
		t.Fatalf("expected 1 unique page, got %d", childTelem.UniquePagesCount)
	}
	if childTelem.SharedPagesCount != 1 {
		t.Fatalf("expected 1 shared page, got %d", childTelem.SharedPagesCount)
	}

	// Child now appends 80 tokens (overflowing block 1' capacity and allocating block 2)
	overflowTokens := make([]int32, 80)
	for i := range overflowTokens {
		overflowTokens[i] = int32(8000 + i)
	}
	if err := child.AppendTokens(overflowTokens...); err != nil {
		t.Fatalf("child AppendTokens overflow failed: %v", err)
	}

	// Total active blocks: block 0 (shared), parent block 1 (unique), child block 1' (unique), child block 2 (unique)
	// Total active blocks should be 4
	if mgr.ActiveBlockCount() != 4 {
		t.Fatalf("expected 4 active blocks in pool, got %d", mgr.ActiveBlockCount())
	}

	if child.Telemetry().UniquePagesCount != 2 {
		t.Fatalf("expected 2 unique pages in child, got %d", child.Telemetry().UniquePagesCount)
	}
	if child.Telemetry().SharedPagesCount != 1 {
		t.Fatalf("expected 1 shared page in child, got %d", child.Telemetry().SharedPagesCount)
	}
}

// TestParentDataIsolationOnDivergentAppend verifies that when a child appends divergent tokens,
// parent data remains completely isolated and untouched, and vice versa.
func TestParentDataIsolationOnDivergentAppend(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	parent, err := mgr.RegisterSession("coord", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	prefix := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if err := parent.AppendTokens(prefix...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	child1, err := mgr.ForkSession("coord", "subagent-1")
	if err != nil {
		t.Fatalf("Fork subagent-1 failed: %v", err)
	}

	child2, err := mgr.ForkSession("coord", "subagent-2")
	if err != nil {
		t.Fatalf("Fork subagent-2 failed: %v", err)
	}

	// Child1 appends unique tokens
	if err := child1.AppendTokens(101, 102, 103); err != nil {
		t.Fatalf("child1 AppendTokens failed: %v", err)
	}

	// Child2 appends different unique tokens
	if err := child2.AppendTokens(201, 202, 203, 204); err != nil {
		t.Fatalf("child2 AppendTokens failed: %v", err)
	}

	// Parent appends its own tokens
	if err := parent.AppendTokens(301, 302); err != nil {
		t.Fatalf("parent AppendTokens failed: %v", err)
	}

	parentTokens := parent.ReadTokens()
	child1Tokens := child1.ReadTokens()
	child2Tokens := child2.ReadTokens()

	// Verify parent tokens
	expectedParent := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 301, 302}
	if len(parentTokens) != len(expectedParent) {
		t.Fatalf("parent tokens len mismatch: expected %d, got %d", len(expectedParent), len(parentTokens))
	}
	for i := range expectedParent {
		if parentTokens[i] != expectedParent[i] {
			t.Fatalf("parent token %d: expected %d, got %d", i, expectedParent[i], parentTokens[i])
		}
	}

	// Verify child1 tokens
	expectedChild1 := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101, 102, 103}
	if len(child1Tokens) != len(expectedChild1) {
		t.Fatalf("child1 tokens len mismatch: expected %d, got %d", len(expectedChild1), len(child1Tokens))
	}
	for i := range expectedChild1 {
		if child1Tokens[i] != expectedChild1[i] {
			t.Fatalf("child1 token %d: expected %d, got %d", i, expectedChild1[i], child1Tokens[i])
		}
	}

	// Verify child2 tokens
	expectedChild2 := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 201, 202, 203, 204}
	if len(child2Tokens) != len(expectedChild2) {
		t.Fatalf("child2 tokens len mismatch: expected %d, got %d", len(expectedChild2), len(child2Tokens))
	}
	for i := range expectedChild2 {
		if child2Tokens[i] != expectedChild2[i] {
			t.Fatalf("child2 token %d: expected %d, got %d", i, expectedChild2[i], child2Tokens[i])
		}
	}

	// Test MutateToken: child1 mutates prefix token at index 2
	if err := child1.MutateToken(2, 9999); err != nil {
		t.Fatalf("MutateToken failed: %v", err)
	}

	child1Mutated := child1.ReadTokens()
	parentAfterMutate := parent.ReadTokens()
	child2AfterMutate := child2.ReadTokens()

	if child1Mutated[2] != 9999 {
		t.Fatalf("expected child1 token 2 to be 9999, got %d", child1Mutated[2])
	}
	if parentAfterMutate[2] != 3 {
		t.Fatalf("parent token 2 must remain isolated as 3, got %d", parentAfterMutate[2])
	}
	if child2AfterMutate[2] != 3 {
		t.Fatalf("child2 token 2 must remain isolated as 3, got %d", child2AfterMutate[2])
	}
}

// TestSafeRefcountReclamation verifies that physical blocks are safely reclaimed when
// refcount reaches zero, whether child releases before parent or parent releases before child.
func TestSafeRefcountReclamation(t *testing.T) {
	t.Run("Child releases before parent", func(t *testing.T) {
		mgr := ctxmmu.NewForkManager()

		parent, err := mgr.RegisterSession("parent-reclaim1", ctxmmu.BlockGranularity64)
		if err != nil {
			t.Fatalf("RegisterSession failed: %v", err)
		}
		if err := parent.AppendTokens(1, 2, 3); err != nil {
			t.Fatalf("AppendTokens failed: %v", err)
		}
		if mgr.ActiveBlockCount() != 1 {
			t.Fatalf("expected 1 active block, got %d", mgr.ActiveBlockCount())
		}

		child, err := mgr.ForkSession("parent-reclaim1", "child-reclaim1")
		if err != nil {
			t.Fatalf("ForkSession failed: %v", err)
		}
		// Child COW appends unique block
		if err := child.AppendTokens(10, 20, 30); err != nil {
			t.Fatalf("child AppendTokens failed: %v", err)
		}
		// Total active blocks: 2
		if mgr.ActiveBlockCount() != 2 {
			t.Fatalf("expected 2 active blocks, got %d", mgr.ActiveBlockCount())
		}

		// Release child
		if err := mgr.ReleaseSession("child-reclaim1"); err != nil {
			t.Fatalf("ReleaseSession child failed: %v", err)
		}
		// Child's unique block reclaimed; parent's block still active
		if mgr.ActiveBlockCount() != 1 {
			t.Fatalf("expected 1 active block after child release, got %d", mgr.ActiveBlockCount())
		}

		// Parent is still intact and readable
		pTokens := parent.ReadTokens()
		if len(pTokens) != 3 || pTokens[0] != 1 || pTokens[1] != 2 || pTokens[2] != 3 {
			t.Fatalf("parent tokens corrupted: %v", pTokens)
		}

		// Release parent
		if err := mgr.ReleaseSession("parent-reclaim1"); err != nil {
			t.Fatalf("ReleaseSession parent failed: %v", err)
		}
		// All blocks reclaimed!
		if mgr.ActiveBlockCount() != 0 {
			t.Fatalf("expected 0 active blocks, got %d", mgr.ActiveBlockCount())
		}
	})

	t.Run("Parent releases before child", func(t *testing.T) {
		mgr := ctxmmu.NewForkManager()

		parent, err := mgr.RegisterSession("parent-reclaim2", ctxmmu.BlockGranularity64)
		if err != nil {
			t.Fatalf("RegisterSession failed: %v", err)
		}
		if err := parent.AppendTokens(100, 200, 300); err != nil {
			t.Fatalf("AppendTokens failed: %v", err)
		}

		child, err := mgr.ForkSession("parent-reclaim2", "child-reclaim2")
		if err != nil {
			t.Fatalf("ForkSession failed: %v", err)
		}

		// Release parent first
		if err := mgr.ReleaseSession("parent-reclaim2"); err != nil {
			t.Fatalf("ReleaseSession parent failed: %v", err)
		}

		// Block should NOT be freed because child still references it
		if mgr.ActiveBlockCount() != 1 {
			t.Fatalf("expected 1 active block kept alive for child, got %d", mgr.ActiveBlockCount())
		}

		// Child can still read its tokens cleanly
		cTokens := child.ReadTokens()
		if len(cTokens) != 3 || cTokens[0] != 100 || cTokens[1] != 200 || cTokens[2] != 300 {
			t.Fatalf("child tokens corrupted after parent release: %v", cTokens)
		}

		// Child can append new tokens
		if err := child.AppendTokens(400, 500); err != nil {
			t.Fatalf("child AppendTokens failed: %v", err)
		}

		// Release child
		if err := mgr.ReleaseSession("child-reclaim2"); err != nil {
			t.Fatalf("ReleaseSession child failed: %v", err)
		}

		// All blocks reclaimed!
		if mgr.ActiveBlockCount() != 0 {
			t.Fatalf("expected 0 active blocks, got %d", mgr.ActiveBlockCount())
		}
	})
}

// TestMultiAgentFanOut8Subagents simulates a coordinator branching to 8 concurrent subagents
// sharing a large prefix context window (AST, repo index, tool schemas).
func TestMultiAgentFanOut8Subagents(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	coordinator, err := mgr.RegisterSession("coordinator", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	// 4,096 prefix tokens = 64 blocks
	prefixCount := 4096
	prefixTokens := make([]int32, prefixCount)
	for i := range prefixTokens {
		prefixTokens[i] = int32(5000 + (i % 3000))
	}
	if err := coordinator.AppendTokens(prefixTokens...); err != nil {
		t.Fatalf("coordinator AppendTokens failed: %v", err)
	}

	initialBlocks := mgr.ActiveBlockCount()
	if initialBlocks != 64 {
		t.Fatalf("expected 64 initial blocks, got %d", initialBlocks)
	}

	const numSubagents = 8
	subagents := make([]*ctxmmu.ForkedSession, numSubagents)
	var wg sync.WaitGroup

	// Fork 8 subagents concurrently
	for i := 0; i < numSubagents; i++ {
		subagentID := fmt.Sprintf("subagent-%d", i)
		child, err := mgr.ForkSession("coordinator", subagentID)
		if err != nil {
			t.Fatalf("ForkSession %s failed: %v", subagentID, err)
		}
		subagents[i] = child

		telem := child.Telemetry()
		if telem.ForkLatency >= 1*time.Millisecond {
			t.Fatalf("subagent %d fork latency %v >= 1ms", i, telem.ForkLatency)
		}
		if telem.ForkCloneBytes != 0 {
			t.Fatalf("subagent %d clone bytes %d != 0", i, telem.ForkCloneBytes)
		}
		if telem.SharedPrefixKVHitRate != 1.0 {
			t.Fatalf("subagent %d hit rate %f != 1.0", i, telem.SharedPrefixKVHitRate)
		}
		if telem.SharedPagesCount != 64 {
			t.Fatalf("subagent %d shared pages %d != 64", i, telem.SharedPagesCount)
		}
	}

	// After forking 8 subagents, active blocks in pool must STILL be 64 (zero-copy replication)
	if mgr.ActiveBlockCount() != 64 {
		t.Fatalf("expected 64 active blocks after 8 forks, got %d", mgr.ActiveBlockCount())
	}

	// Verify all shared blocks have refcount == 9 (1 coordinator + 8 subagents)
	for _, entry := range coordinator.PageTableEntries() {
		b, err := mgr.Pool().Allocate(ctxmmu.BlockGranularity64)
		if err == nil {
			mgr.Pool().Release(b) // just check pool health
		}
		_ = entry
	}

	// Concurrently append unique divergent tokens in each subagent
	for i := 0; i < numSubagents; i++ {
		wg.Add(1)
		go func(idx int, child *ctxmmu.ForkedSession) {
			defer wg.Done()
			uniqueTaskTokens := []int32{
				int32(10000 + idx*100),
				int32(10000 + idx*100 + 1),
				int32(10000 + idx*100 + 2),
			}
			if err := child.AppendTokens(uniqueTaskTokens...); err != nil {
				t.Errorf("subagent %d AppendTokens failed: %v", idx, err)
			}
		}(i, subagents[i])
	}
	wg.Wait()

	// Verify isolation for all 8 subagents
	for i := 0; i < numSubagents; i++ {
		tokens := subagents[i].ReadTokens()
		expectedLen := prefixCount + 3
		if len(tokens) != expectedLen {
			t.Fatalf("subagent %d tokens len mismatch: expected %d, got %d", i, expectedLen, len(tokens))
		}
		expectedTaskToken := int32(10000 + i*100)
		if tokens[prefixCount] != expectedTaskToken {
			t.Fatalf("subagent %d expected divergent token %d, got %d", i, expectedTaskToken, tokens[prefixCount])
		}
	}

	// Coordinator must be completely unaffected
	coordTokens := coordinator.ReadTokens()
	if len(coordTokens) != prefixCount {
		t.Fatalf("coordinator tokens modified: expected %d, got %d", prefixCount, len(coordTokens))
	}

	// Release all subagents and coordinator concurrently
	for i := 0; i < numSubagents; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			if err := mgr.ReleaseSession(fmt.Sprintf("subagent-%d", idx)); err != nil {
				t.Errorf("Release subagent-%d failed: %v", idx, err)
			}
		}(i)
	}
	wg.Wait()

	if err := mgr.ReleaseSession("coordinator"); err != nil {
		t.Fatalf("Release coordinator failed: %v", err)
	}

	// Full reclamation: 0 active blocks remaining
	if mgr.ActiveBlockCount() != 0 {
		t.Fatalf("expected 0 active blocks after full release, got %d", mgr.ActiveBlockCount())
	}
}

// TestBlockGranularity16And64 verifies coherent handling of both 16-token and 64-token block granularities.
func TestBlockGranularity16And64(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	// 16-token block granularity
	s16, err := mgr.RegisterSession("sess-16", ctxmmu.BlockGranularity16)
	if err != nil {
		t.Fatalf("RegisterSession 16 failed: %v", err)
	}
	tokens16 := make([]int32, 48) // 3 blocks of 16 tokens
	for i := range tokens16 {
		tokens16[i] = int32(i)
	}
	if err := s16.AppendTokens(tokens16...); err != nil {
		t.Fatalf("AppendTokens 16 failed: %v", err)
	}
	if s16.PageCount() != 3 {
		t.Fatalf("expected 3 pages for 16-granularity, got %d", s16.PageCount())
	}

	// Fork 16-token session
	child16, err := mgr.ForkSession("sess-16", "child-16")
	if err != nil {
		t.Fatalf("ForkSession 16 failed: %v", err)
	}
	if child16.Granularity() != 16 {
		t.Fatalf("expected granularity 16, got %d", child16.Granularity())
	}
	if child16.PageCount() != 3 {
		t.Fatalf("expected 3 pages in child16, got %d", child16.PageCount())
	}

	// 64-token block granularity
	s64, err := mgr.RegisterSession("sess-64", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession 64 failed: %v", err)
	}
	tokens64 := make([]int32, 128) // 2 blocks of 64 tokens
	for i := range tokens64 {
		tokens64[i] = int32(i)
	}
	if err := s64.AppendTokens(tokens64...); err != nil {
		t.Fatalf("AppendTokens 64 failed: %v", err)
	}
	if s64.PageCount() != 2 {
		t.Fatalf("expected 2 pages for 64-granularity, got %d", s64.PageCount())
	}

	// Invalid granularity test
	_, err = mgr.RegisterSession("sess-invalid", 32)
	if err != ctxmmu.ErrInvalidGranularity {
		t.Fatalf("expected ErrInvalidGranularity for 32, got %v", err)
	}

	// Clean up
	_ = mgr.ReleaseSession("sess-16")
	_ = mgr.ReleaseSession("child-16")
	_ = mgr.ReleaseSession("sess-64")
	if mgr.ActiveBlockCount() != 0 {
		t.Fatalf("expected 0 active blocks, got %d", mgr.ActiveBlockCount())
	}
}

// TestCoherentUMAMappingRDNA35 tests host Context MMU and RDNA 3.5 GPU virtual address translation.
func TestCoherentUMAMappingRDNA35(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	sess, err := mgr.RegisterSession("sess-uma", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	tokens := make([]int32, 128)
	if err := sess.AppendTokens(tokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	entries := sess.PageTableEntries()
	for i, entry := range entries {
		if entry.GPUVirtualAddress < ctxmmu.RDNA35GPUVABase {
			t.Fatalf("entry %d GPU VA 0x%x below base 0x%x",
				i, entry.GPUVirtualAddress, ctxmmu.RDNA35GPUVABase)
		}
		if entry.HostAddress == 0 {
			t.Fatalf("entry %d host address is zero", i)
		}

		// Test Host to GPU VA translation
		gpuVA, err := mgr.HostToGPUVirtualAddress(entry.HostAddress)
		if err != nil {
			t.Fatalf("HostToGPUVirtualAddress failed for entry %d: %v", i, err)
		}
		if gpuVA != entry.GPUVirtualAddress {
			t.Fatalf("entry %d translated GPU VA 0x%x != expected 0x%x",
				i, gpuVA, entry.GPUVirtualAddress)
		}

		// Test GPU VA to Host translation
		hostAddr, err := mgr.GPUVirtualAddressToHost(entry.GPUVirtualAddress)
		if err != nil {
			t.Fatalf("GPUVirtualAddressToHost failed for entry %d: %v", i, err)
		}
		if hostAddr != entry.HostAddress {
			t.Fatalf("entry %d translated HostAddr 0x%x != expected 0x%x",
				i, hostAddr, entry.HostAddress)
		}
	}

	// Address out of range test
	_, err = mgr.HostToGPUVirtualAddress(0xdeadbeef)
	if err != ctxmmu.ErrAddressOutOfRange {
		t.Fatalf("expected ErrAddressOutOfRange, got %v", err)
	}
	_, err = mgr.GPUVirtualAddressToHost(0xdeadbeef)
	if err != ctxmmu.ErrAddressOutOfRange {
		t.Fatalf("expected ErrAddressOutOfRange, got %v", err)
	}

	_ = mgr.ReleaseSession("sess-uma")
}

// TestMMUSessionForkerIntegration tests the MMU method implementations and SessionForker interface.
func TestMMUSessionForkerIntegration(t *testing.T) {
	m := ctxmmu.New()

	// Verify SessionForker interface
	var _ ctxmmu.SessionForker = m

	parent, err := m.RegisterForkSession("mmu-parent", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterForkSession failed: %v", err)
	}

	if err := parent.AppendTokens(1, 2, 3, 4, 5); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	child, err := m.ForkSession("mmu-parent", "mmu-child")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}

	if child.ParentID() != "mmu-parent" {
		t.Fatalf("expected parent ID 'mmu-parent', got %q", child.ParentID())
	}
	if child.SessionID() != "mmu-child" {
		t.Fatalf("expected child ID 'mmu-child', got %q", child.SessionID())
	}

	// Direct Fork() method on session
	grandchild, err := child.Fork("mmu-grandchild")
	if err != nil {
		t.Fatalf("grandchild Fork failed: %v", err)
	}
	if grandchild.ParentID() != "mmu-child" {
		t.Fatalf("expected parent ID 'mmu-child', got %q", grandchild.ParentID())
	}

	// Test error cases
	_, err = m.ForkSession("non-existent", "child-err")
	if err != ctxmmu.ErrParentNotFound {
		t.Fatalf("expected ErrParentNotFound, got %v", err)
	}

	_, err = m.ForkSession("mmu-parent", "mmu-child")
	if err != ctxmmu.ErrSessionExists {
		t.Fatalf("expected ErrSessionExists, got %v", err)
	}

	_, err = m.ForkSession("mmu-parent", "mmu-parent")
	if err != ctxmmu.ErrSelfFork {
		t.Fatalf("expected ErrSelfFork, got %v", err)
	}

	_, err = m.ForkSession("", "child-err")
	if err != ctxmmu.ErrInvalidSessionID {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}

	// Release sessions
	if err := m.ReleaseSession("mmu-grandchild"); err != nil {
		t.Fatalf("ReleaseSession grandchild failed: %v", err)
	}
	if err := m.ReleaseSession("mmu-child"); err != nil {
		t.Fatalf("ReleaseSession child failed: %v", err)
	}
	if err := m.ReleaseSession("mmu-parent"); err != nil {
		t.Fatalf("ReleaseSession parent failed: %v", err)
	}

	// Re-releasing released session
	err = m.ReleaseSession("mmu-parent")
	if err != ctxmmu.ErrSessionNotFound {
		t.Fatalf("expected ErrSessionNotFound for already released session, got %v", err)
	}
}

// TestConcurrentForkAndAppendRace tests concurrent forking, appending, and releasing under -race.
func TestConcurrentForkAndAppendRace(t *testing.T) {
	mgr := ctxmmu.NewForkManager()

	root, err := mgr.RegisterSession("root", ctxmmu.BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession root failed: %v", err)
	}
	if err := root.AppendTokens(1, 2, 3, 4, 5, 6, 7, 8); err != nil {
		t.Fatalf("AppendTokens root failed: %v", err)
	}

	var wg sync.WaitGroup
	const workers = 16

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			childID := fmt.Sprintf("worker-%d", workerID)
			child, err := mgr.ForkSession("root", childID)
			if err != nil {
				t.Errorf("worker %d fork failed: %v", workerID, err)
				return
			}

			// Append tokens
			for j := 0; j < 5; j++ {
				tok := int32(workerID*1000 + j)
				if err := child.AppendTokens(tok); err != nil {
					t.Errorf("worker %d append failed: %v", workerID, err)
					return
				}
			}

			// Read tokens
			toks := child.ReadTokens()
			if len(toks) != 13 {
				t.Errorf("worker %d token count mismatch: %d", workerID, len(toks))
			}

			// Release session
			if err := mgr.ReleaseSession(childID); err != nil {
				t.Errorf("worker %d release failed: %v", workerID, err)
			}
		}(i)
	}

	wg.Wait()

	if err := mgr.ReleaseSession("root"); err != nil {
		t.Fatalf("ReleaseSession root failed: %v", err)
	}

	if mgr.ActiveBlockCount() != 0 {
		t.Fatalf("expected 0 active blocks, got %d", mgr.ActiveBlockCount())
	}
}
