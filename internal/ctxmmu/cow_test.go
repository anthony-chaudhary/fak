package ctxmmu_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestCOWPageTable_ForkSession verifies:
// 1. Subagent fork latency is strictly under 1 ms (pointer copy only).
// 2. 100% prefix hit rate across 8 children sharing a large prefix (32k tokens).
// 3. 0 duplicate physical pages allocated in UMA DRAM.
func TestCOWPageTable_ForkSession(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	// 32k tokens prefix = 512 blocks of 64 tokens
	prefixTokensCount := 32768
	prefixTokens := make([]int, prefixTokensCount)
	for i := 0; i < prefixTokensCount; i++ {
		prefixTokens[i] = 1000 + (i % 5000)
	}

	parent, err := table.CreateSession("coord-32k")
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	if err := table.AppendTokens("coord-32k", prefixTokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	if parent.PageCount() != 512 {
		t.Fatalf("expected 512 pages for 32k tokens, got %d", parent.PageCount())
	}
	if parent.TokenCount != prefixTokensCount {
		t.Fatalf("expected %d tokens, got %d", prefixTokensCount, parent.TokenCount)
	}

	initialPhysicalBlocks := table.PhysicalBlockCount()
	initialAllocatedBytes := table.TotalAllocatedBytes()
	if initialPhysicalBlocks != 512 {
		t.Fatalf("expected 512 initial physical blocks, got %d", initialPhysicalBlocks)
	}

	// Warmup fork
	if _, err := table.ForkSession("coord-32k", "subagent-warmup"); err != nil {
		t.Fatalf("Warmup fork failed: %v", err)
	}
	if err := table.ReleaseSession("subagent-warmup"); err != nil {
		t.Fatalf("Warmup release failed: %v", err)
	}

	// Fork 8 child subagents
	const numChildren = 8
	children := make([]*ctxmmu.SessionBranch, numChildren)

	for i := 0; i < numChildren; i++ {
		childID := fmt.Sprintf("subagent-%d", i)

		start := time.Now()
		child, err := table.ForkSession("coord-32k", childID)
		dur := time.Since(start)

		if err != nil {
			t.Fatalf("ForkSession %s failed: %v", childID, err)
		}

		// Latency requirement: < 1 ms (pointer copy only)
		if dur >= 5*time.Millisecond {
			t.Fatalf("child %s fork wall duration %v exceeded 5ms threshold", childID, dur)
		}
		if child.ForkLatency >= 1*time.Millisecond {
			t.Fatalf("child %s fork latency telemetry %v exceeded 1ms threshold", childID, child.ForkLatency)
		}

		// 100% prefix hit rate requirement
		if child.PrefixHitRate != 1.0 {
			t.Fatalf("expected prefix hit rate 1.0 (100%%), got %f", child.PrefixHitRate)
		}
		telem := child.Telemetry()
		if telem.SharedPrefixKVHitRate != 1.0 {
			t.Fatalf("expected telemetry hit rate 1.0 (100%%), got %f", telem.SharedPrefixKVHitRate)
		}
		if telem.ForkCloneBytes != 0 {
			t.Fatalf("expected 0 fork clone bytes, got %d", telem.ForkCloneBytes)
		}
		if telem.SharedPagesCount != 512 {
			t.Fatalf("expected 512 shared pages, got %d", telem.SharedPagesCount)
		}
		if telem.UniquePagesCount != 0 {
			t.Fatalf("expected 0 unique pages on fork, got %d", telem.UniquePagesCount)
		}

		children[i] = child
	}

	// Verify 0 duplicate physical pages allocated in UMA DRAM
	currentPhysicalBlocks := table.PhysicalBlockCount()
	if currentPhysicalBlocks != initialPhysicalBlocks {
		t.Fatalf("expected 0 duplicate physical pages allocated, got physical block count %d (was %d)",
			currentPhysicalBlocks, initialPhysicalBlocks)
	}
	if table.DuplicatePhysicalPagesAllocated() != 0 {
		t.Fatalf("expected 0 duplicate physical pages, got %d", table.DuplicatePhysicalPagesAllocated())
	}
	if table.TotalAllocatedBytes() != initialAllocatedBytes {
		t.Fatalf("expected total allocated bytes unchanged at %d, got %d",
			initialAllocatedBytes, table.TotalAllocatedBytes())
	}

	// Verify pointer identity for shared blocks across parent and children
	for _, child := range children {
		if len(child.Blocks) != len(parent.Blocks) {
			t.Fatalf("block count mismatch: child %d != parent %d", len(child.Blocks), len(parent.Blocks))
		}
		for b := 0; b < len(parent.Blocks); b++ {
			if child.Blocks[b] != parent.Blocks[b] {
				t.Fatalf("expected identical block pointer at index %d (shallow copy)", b)
			}
			if child.Blocks[b].ID != parent.Blocks[b].ID {
				t.Fatalf("expected block ID match at index %d: child %d != parent %d",
					b, child.Blocks[b].ID, parent.Blocks[b].ID)
			}
			// 1 parent + 8 children = 9 references
			if child.Blocks[b].RefCount() != int32(numChildren+1) {
				t.Fatalf("expected block %d refcount %d, got %d",
					b, numChildren+1, child.Blocks[b].RefCount())
			}
		}
	}

	// Verify metrics
	metrics := table.Metrics()
	if metrics.ActiveBranches != numChildren+1 {
		t.Fatalf("expected %d active branches, got %d", numChildren+1, metrics.ActiveBranches)
	}
	// DedupRatio = 1.0 - (1/9) ≈ 88.89% (target >= 85%)
	if metrics.DedupRatio < 0.85 {
		t.Fatalf("expected DedupRatio >= 0.85, got %f", metrics.DedupRatio)
	}
	if metrics.DeduplicatedBytes <= 0 {
		t.Fatalf("expected DeduplicatedBytes > 0, got %d", metrics.DeduplicatedBytes)
	}
}

// TestCOWPageTable_AppendTriggersCOW verifies:
// 1. Child token append creates a private physical copy (COW) when writing to shared blocks.
// 2. Parent blocks remain completely immutable and untouched.
// 3. Sibling sessions remain uncorrupted and isolated.
func TestCOWPageTable_AppendTriggersCOW(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	// 100 tokens: block 0 has 64 tokens (full), block 1 has 36 tokens (partial)
	parentTokens := make([]int, 100)
	for i := range parentTokens {
		parentTokens[i] = i + 1
	}

	parent, err := table.CreateSession("parent-coord")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := table.AppendTokens("parent-coord", parentTokens); err != nil {
		t.Fatalf("AppendTokens parent: %v", err)
	}

	// Verify parent has 2 blocks (block 0 full, block 1 partial)
	if parent.PageCount() != 2 {
		t.Fatalf("expected 2 blocks for parent, got %d", parent.PageCount())
	}
	origBlock0ID := parent.Blocks[0].ID
	origBlock1ID := parent.Blocks[1].ID

	// Fork two sibling sessions
	child1, err := table.ForkSession("parent-coord", "child-worker-1")
	if err != nil {
		t.Fatalf("ForkSession child-1: %v", err)
	}
	child2, err := table.ForkSession("parent-coord", "child-worker-2")
	if err != nil {
		t.Fatalf("ForkSession child-2: %v", err)
	}

	// At this point, all 3 sessions share block 0 and block 1 (refcount 3)
	if parent.Blocks[1].RefCount() != 3 {
		t.Fatalf("expected block 1 refcount 3 before append, got %d", parent.Blocks[1].RefCount())
	}

	// Child 1 appends 3 tokens: writing into shared block 1 must trigger COW!
	child1Appended := []int{1001, 1002, 1003}
	if err := table.AppendTokens("child-worker-1", child1Appended); err != nil {
		t.Fatalf("AppendTokens child-1: %v", err)
	}

	// Verify Child 1 got a new private physical block for block 1
	if child1.Blocks[1].ID == origBlock1ID {
		t.Fatalf("expected child1 block 1 to have new private ID, got original %d", origBlock1ID)
	}
	// Block 0 was full and unmodified, so it MUST remain shared
	if child1.Blocks[0].ID != origBlock0ID {
		t.Fatalf("expected child1 block 0 to remain shared, got %d (orig %d)", child1.Blocks[0].ID, origBlock0ID)
	}

	// Child 2 appends 5 different tokens: writing into shared block 1 must trigger COW!
	child2Appended := []int{2001, 2002, 2003, 2004, 2005}
	if err := table.AppendTokens("child-worker-2", child2Appended); err != nil {
		t.Fatalf("AppendTokens child-2: %v", err)
	}

	// Verify Child 2 got a separate private block for block 1
	if child2.Blocks[1].ID == origBlock1ID {
		t.Fatalf("expected child2 block 1 to have new private ID, got original %d", origBlock1ID)
	}
	if child2.Blocks[1].ID == child1.Blocks[1].ID {
		t.Fatalf("expected child1 and child2 to have distinct private blocks, both have %d", child1.Blocks[1].ID)
	}
	// Block 0 remains shared across parent, child-1, and child-2
	if child2.Blocks[0].ID != origBlock0ID {
		t.Fatalf("expected child2 block 0 to remain shared, got %d", child2.Blocks[0].ID)
	}

	// Invariant 1: Parent tokens are 100% untouched and uncorrupted!
	parentRead := parent.Tokens()
	if len(parentRead) != 100 {
		t.Fatalf("parent corrupted: expected 100 tokens, got %d", len(parentRead))
	}
	for i := 0; i < 100; i++ {
		if parentRead[i] != parentTokens[i] {
			t.Fatalf("parent corrupted at index %d: expected %d, got %d", i, parentTokens[i], parentRead[i])
		}
	}
	if parent.Blocks[1].ID != origBlock1ID {
		t.Fatalf("parent block 1 ID corrupted: got %d, want %d", parent.Blocks[1].ID, origBlock1ID)
	}
	if parent.Blocks[1].TokenCount != 36 {
		t.Fatalf("parent block 1 token count corrupted: got %d, want 36", parent.Blocks[1].TokenCount)
	}

	// Invariant 2: Child 1 tokens are prefix + child1Appended
	child1Read := child1.Tokens()
	if len(child1Read) != 103 {
		t.Fatalf("child1 expected 103 tokens, got %d", len(child1Read))
	}
	for i := 0; i < 100; i++ {
		if child1Read[i] != parentTokens[i] {
			t.Fatalf("child1 prefix corrupted at %d", i)
		}
	}
	for i := 0; i < 3; i++ {
		if child1Read[100+i] != child1Appended[i] {
			t.Fatalf("child1 appended token mismatch at %d: got %d, want %d", i, child1Read[100+i], child1Appended[i])
		}
	}

	// Invariant 3: Child 2 tokens are prefix + child2Appended
	child2Read := child2.Tokens()
	if len(child2Read) != 105 {
		t.Fatalf("child2 expected 105 tokens, got %d", len(child2Read))
	}
	for i := 0; i < 100; i++ {
		if child2Read[i] != parentTokens[i] {
			t.Fatalf("child2 prefix corrupted at %d", i)
		}
	}
	for i := 0; i < 5; i++ {
		if child2Read[100+i] != child2Appended[i] {
			t.Fatalf("child2 appended token mismatch at %d: got %d, want %d", i, child2Read[100+i], child2Appended[i])
		}
	}

	// Invariant 4: Mutating a token inside a shared block triggers COW and preserves other sessions
	if err := table.MutateToken("child-worker-1", 10, 9999); err != nil {
		t.Fatalf("MutateToken failed: %v", err)
	}
	// Child 1 block 0 must now be private (new ID != origBlock0ID)
	if child1.Blocks[0].ID == origBlock0ID {
		t.Fatalf("expected block 0 COW mutation to allocate new private block for child 1")
	}
	// Parent and Child 2 still have origBlock0ID and token at 10 is untouched (parentTokens[10] = 11)
	if parent.Tokens()[10] != 11 {
		t.Fatalf("parent block 0 corrupted by child 1 mutate: got %d, want 11", parent.Tokens()[10])
	}
	if child2.Tokens()[10] != 11 {
		t.Fatalf("child 2 block 0 corrupted by child 1 mutate: got %d, want 11", child2.Tokens()[10])
	}
	if child1.Tokens()[10] != 9999 {
		t.Fatalf("child 1 mutate token mismatch: got %d, want 9999", child1.Tokens()[10])
	}
}

// TestCOWPageTable_ReleaseSession verifies:
// 1. Releasing a session decrements refcounts on all owned page blocks.
// 2. Physical blocks are kept alive as long as other sessions reference them.
// 3. Physical blocks reaching refcount 0 are safely freed and memory is reclaimed.
func TestCOWPageTable_ReleaseSession(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	tokens := make([]int, 128) // 2 blocks of 64
	for i := range tokens {
		tokens[i] = i + 10
	}

	parent, err := table.CreateSession("parent")
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if err := table.AppendTokens("parent", tokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	child, err := table.ForkSession("parent", "child")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}

	// Both blocks have refcount 2
	for i, blk := range parent.Blocks {
		if blk.RefCount() != 2 {
			t.Fatalf("block %d expected refcount 2, got %d", i, blk.RefCount())
		}
	}

	allocatedBeforeRelease := table.TotalAllocatedBytes()
	physBlocksBefore := table.PhysicalBlockCount()
	if physBlocksBefore != 2 {
		t.Fatalf("expected 2 physical blocks, got %d", physBlocksBefore)
	}

	// Release child session
	if err := table.ReleaseSession("child"); err != nil {
		t.Fatalf("ReleaseSession child failed: %v", err)
	}

	if !child.IsReleased() {
		t.Fatalf("expected child to report released")
	}

	// Blocks are NOT freed because parent still holds them; refcount must be 1
	for i, blk := range parent.Blocks {
		if blk.RefCount() != 1 {
			t.Fatalf("block %d expected refcount 1 after child release, got %d", i, blk.RefCount())
		}
	}
	if table.PhysicalBlockCount() != physBlocksBefore {
		t.Fatalf("physical blocks count changed prematurely: got %d, want %d",
			table.PhysicalBlockCount(), physBlocksBefore)
	}
	if table.TotalAllocatedBytes() != allocatedBeforeRelease {
		t.Fatalf("allocated bytes changed prematurely: got %d, want %d",
			table.TotalAllocatedBytes(), allocatedBeforeRelease)
	}
	if table.ActiveBranches() != 1 {
		t.Fatalf("expected 1 active branch, got %d", table.ActiveBranches())
	}

	// Now release parent session
	if err := table.ReleaseSession("parent"); err != nil {
		t.Fatalf("ReleaseSession parent failed: %v", err)
	}

	if !parent.IsReleased() {
		t.Fatalf("expected parent to report released")
	}

	// All blocks reach refcount 0: memory must be completely reclaimed!
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks after all sessions released, got %d", table.PhysicalBlockCount())
	}
	if table.TotalAllocatedBytes() != 0 {
		t.Fatalf("expected 0 allocated bytes after memory reclamation, got %d", table.TotalAllocatedBytes())
	}
	if table.ActiveBranches() != 0 {
		t.Fatalf("expected 0 active branches, got %d", table.ActiveBranches())
	}
	if table.DedupRatio() != 0.0 {
		t.Fatalf("expected 0.0 dedup ratio, got %f", table.DedupRatio())
	}

	// Releasing an already released session must return error
	if err := table.ReleaseSession("parent"); err != ctxmmu.ErrSessionNotFound && err != ctxmmu.ErrSessionReleased {
		t.Fatalf("expected ErrSessionNotFound or ErrSessionReleased, got %v", err)
	}
}

// TestCOWPageTable_Concurrency verifies:
// 1. Thread-safe concurrent forking across multiple subagents.
// 2. Thread-safe concurrent appending and COW branching without race conditions.
// 3. Clean concurrent memory reclamation.
func TestCOWPageTable_Concurrency(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	prefixCount := 1024
	prefixTokens := make([]int, prefixCount)
	for i := range prefixTokens {
		prefixTokens[i] = i * 7
	}

	parent, err := table.CreateSession("coord-concurrent")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := table.AppendTokens("coord-concurrent", prefixTokens); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}

	const numWorkers = 24
	var wg sync.WaitGroup
	errCh := make(chan error, numWorkers*3)

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			childID := fmt.Sprintf("subagent-worker-%d", workerID)

			child, err := table.ForkSession("coord-concurrent", childID)
			if err != nil {
				errCh <- fmt.Errorf("fork %s failed: %w", childID, err)
				return
			}

			// Append worker-specific tokens
			newTokens := []int{workerID * 1000, workerID*1000 + 1, workerID*1000 + 2}
			if err := table.AppendTokens(childID, newTokens); err != nil {
				errCh <- fmt.Errorf("append %s failed: %w", childID, err)
				return
			}

			// Read and verify token count
			readBack := child.Tokens()
			expectedLen := prefixCount + len(newTokens)
			if len(readBack) != expectedLen {
				errCh <- fmt.Errorf("worker %s token len mismatch: got %d, want %d",
					childID, len(readBack), expectedLen)
				return
			}

			// Verify prefix integrity
			if readBack[0] != 0 || readBack[1] != 7 || readBack[100] != 700 {
				errCh <- fmt.Errorf("worker %s prefix corrupted", childID)
				return
			}
		}(i)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	// Verify parent was not corrupted during concurrent worker activity
	parentTokens := parent.Tokens()
	if len(parentTokens) != prefixCount {
		t.Fatalf("parent corrupted: expected %d tokens, got %d", prefixCount, len(parentTokens))
	}
	for i := range prefixTokens {
		if parentTokens[i] != prefixTokens[i] {
			t.Fatalf("parent corrupted at %d", i)
		}
	}

	// Concurrently release all child workers
	var releaseWg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		releaseWg.Add(1)
		go func(workerID int) {
			defer releaseWg.Done()
			childID := fmt.Sprintf("subagent-worker-%d", workerID)
			if err := table.ReleaseSession(childID); err != nil {
				t.Errorf("release %s failed: %v", childID, err)
			}
		}(i)
	}
	releaseWg.Wait()

	// Release coordinator
	if err := table.ReleaseSession("coord-concurrent"); err != nil {
		t.Fatalf("release coord failed: %v", err)
	}

	// Verify 0 leaks after concurrent execution
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks after release, got %d", table.PhysicalBlockCount())
	}
	if table.TotalAllocatedBytes() != 0 {
		t.Fatalf("expected 0 allocated bytes, got %d", table.TotalAllocatedBytes())
	}
	if table.ActiveBranches() != 0 {
		t.Fatalf("expected 0 active branches, got %d", table.ActiveBranches())
	}
}

// TestCOWPageTable_Granularity16 verifies COW page tables configured with 16-token granularity.
func TestCOWPageTable_Granularity16(t *testing.T) {
	table := ctxmmu.NewCOWPageTable(
		ctxmmu.WithBlockCapacity(16),
		ctxmmu.WithBytesPerToken(64),
	)

	if table.BlockCapacity() != 16 {
		t.Fatalf("expected block capacity 16, got %d", table.BlockCapacity())
	}
	if table.BlockSize() != 16*64 {
		t.Fatalf("expected block size 1024, got %d", table.BlockSize())
	}

	tokens := make([]int, 48) // 3 blocks of 16
	for i := range tokens {
		tokens[i] = i
	}

	if err := table.AppendTokens("sess-16", tokens); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}

	sess, err := table.GetSession("sess-16")
	if err != nil {
		t.Fatalf("GetSession: %v", err)
	}
	if sess.PageCount() != 3 {
		t.Fatalf("expected 3 blocks, got %d", sess.PageCount())
	}

	child, err := table.ForkSession("sess-16", "sess-16-child")
	if err != nil {
		t.Fatalf("ForkSession: %v", err)
	}
	if child.PrefixHitRate != 1.0 {
		t.Fatalf("expected prefix hit rate 1.0, got %f", child.PrefixHitRate)
	}

	_ = table.ReleaseSession("sess-16-child")
	_ = table.ReleaseSession("sess-16")
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 blocks after release, got %d", table.PhysicalBlockCount())
	}
}

// TestCOWPageTable_ValidationAndErrors verifies edge cases and parameter validations.
func TestCOWPageTable_ValidationAndErrors(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	// Empty session ID
	if _, err := table.CreateSession(""); err != ctxmmu.ErrInvalidSessionID {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}
	if err := table.AppendTokens("", []int{1, 2}); err != ctxmmu.ErrInvalidSessionID {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}
	if err := table.ReleaseSession(""); err != ctxmmu.ErrInvalidSessionID {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}

	// Parent not found
	if _, err := table.ForkSession("non-existent", "child"); err != ctxmmu.ErrParentNotFound {
		t.Fatalf("expected ErrParentNotFound, got %v", err)
	}

	// Self fork
	if _, err := table.CreateSession("self"); err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if _, err := table.ForkSession("self", "self"); err != ctxmmu.ErrSelfFork {
		t.Fatalf("expected ErrSelfFork, got %v", err)
	}

	// Duplicate session
	if _, err := table.CreateSession("self"); err != ctxmmu.ErrSessionExists {
		t.Fatalf("expected ErrSessionExists, got %v", err)
	}

	// Mutate out of bounds
	if err := table.MutateToken("self", 999, 42); err != ctxmmu.ErrIndexOutOfBounds {
		t.Fatalf("expected ErrIndexOutOfBounds, got %v", err)
	}

	// Release
	if err := table.ReleaseSession("self"); err != nil {
		t.Fatalf("ReleaseSession: %v", err)
	}
}

// -----------------------------------------------------------------------------
// Lock-Free Candidate Tree Speculation Tests (#12319)
// -----------------------------------------------------------------------------

// TestCOWTreeSpeculation_ForkAndSquashLatency verifies:
// 1. Fork + append + squash latency across 16 concurrent candidate branches is strictly < 20µs per branch.
// 2. 0 duplicate physical pages are allocated on candidate forks.
// 3. Parent session remains uncorrupted and memory is cleanly reclaimed after squashing losing branches.
func TestCOWTreeSpeculation_ForkAndSquashLatency(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	// 512 prefix tokens = 8 blocks of 64 tokens
	prefixCount := 512
	prefixTokens := make([]int, prefixCount)
	for i := range prefixTokens {
		prefixTokens[i] = 1000 + i
	}

	_, err := table.CreateSession("parent-spec")
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if err := table.AppendTokens("parent-spec", prefixTokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	initialPhysBlocks := table.PhysicalBlockCount()
	if initialPhysBlocks != 8 {
		t.Fatalf("expected 8 physical blocks, got %d", initialPhysBlocks)
	}

	// Warmup 1 candidate fork & squash to prime candidatePool & blockPool
	warmupCand, err := table.ForkCandidate("parent-spec", "cand-warmup")
	if err != nil {
		t.Fatalf("warmup ForkCandidate failed: %v", err)
	}
	if err := warmupCand.AppendTokens([]int{9999}); err != nil {
		t.Fatalf("warmup AppendTokens failed: %v", err)
	}
	if err := table.SquashCandidate("cand-warmup"); err != nil {
		t.Fatalf("warmup SquashCandidate failed: %v", err)
	}

	const numCandidates = 16
	var wg sync.WaitGroup
	errCh := make(chan error, numCandidates*3)
	latencies := make([]time.Duration, numCandidates)

	for i := 0; i < numCandidates; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			candID := fmt.Sprintf("cand-%d", idx)

			start := time.Now()

			cand, err := table.ForkCandidate("parent-spec", candID)
			if err != nil {
				errCh <- fmt.Errorf("branch %d fork failed: %w", idx, err)
				return
			}

			if cand.PageCount() != 8 {
				errCh <- fmt.Errorf("branch %d page count mismatch: got %d, want 8", idx, cand.PageCount())
				return
			}
			if cand.TokenCount != prefixCount {
				errCh <- fmt.Errorf("branch %d token count mismatch: got %d, want %d", idx, cand.TokenCount, prefixCount)
				return
			}

			// Draft 4 tokens
			draftTokens := []int{2000 + idx*10, 2000 + idx*10 + 1, 2000 + idx*10 + 2, 2000 + idx*10 + 3}
			if err := cand.AppendTokens(draftTokens); err != nil {
				errCh <- fmt.Errorf("branch %d append failed: %w", idx, err)
				return
			}

			// Squash losing branch
			if err := table.SquashCandidate(candID); err != nil {
				errCh <- fmt.Errorf("branch %d squash failed: %w", idx, err)
				return
			}

			dur := time.Since(start)
			latencies[idx] = dur
		}(i)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	var totalDuration time.Duration
	for _, lat := range latencies {
		totalDuration += lat
	}
	avgLatency := totalDuration / numCandidates

	t.Logf("16 candidate branches: avg latency per branch = %v", avgLatency)
	for i, lat := range latencies {
		t.Logf("  branch %2d: %v", i, lat)
	}

	// Sub-20µs requirement per branch on average
	if avgLatency >= 20*time.Microsecond {
		t.Fatalf("average branch latency %v exceeded 20µs threshold", avgLatency)
	}

	// Verify 0 duplicate physical pages allocated
	if table.DuplicatePhysicalPagesAllocated() != 0 {
		t.Fatalf("expected 0 duplicate physical pages, got %d", table.DuplicatePhysicalPagesAllocated())
	}

	// Verify parent session tokens are completely intact
	parentSess, err := table.GetSession("parent-spec")
	if err != nil {
		t.Fatalf("GetSession parent-spec failed: %v", err)
	}
	parentTokens := parentSess.Tokens()
	if len(parentTokens) != prefixCount {
		t.Fatalf("parent tokens count corrupted: got %d, want %d", len(parentTokens), prefixCount)
	}
	for i := 0; i < prefixCount; i++ {
		if parentTokens[i] != prefixTokens[i] {
			t.Fatalf("parent token %d corrupted: got %d, want %d", i, parentTokens[i], prefixTokens[i])
		}
	}

	// Verify candidate table is clean (0 active candidates)
	if table.CandidateCount() != 0 {
		t.Fatalf("expected 0 active candidates after squash, got %d", table.CandidateCount())
	}
}

// TestCOWTreeSpeculation_ZeroDuplicatePhysicalPages verifies:
// Forking candidate branches shares physical page blocks and allocates 0 redundant physical DRAM pages.
func TestCOWTreeSpeculation_ZeroDuplicatePhysicalPages(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	prefixCount := 1024
	prefixTokens := make([]int, prefixCount)
	for i := range prefixTokens {
		prefixTokens[i] = 5000 + i
	}

	parent, err := table.CreateSession("parent-1024")
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if err := table.AppendTokens("parent-1024", prefixTokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	initialPhysBlocks := table.PhysicalBlockCount()
	if initialPhysBlocks != 16 {
		t.Fatalf("expected 16 physical blocks, got %d", initialPhysBlocks)
	}

	const numCandidates = 16
	candidates := make([]*ctxmmu.ForkedCandidate, numCandidates)

	for i := 0; i < numCandidates; i++ {
		candID := fmt.Sprintf("cand-zero-%d", i)
		cand, err := table.ForkCandidate("parent-1024", candID)
		if err != nil {
			t.Fatalf("ForkCandidate %s failed: %v", candID, err)
		}
		candidates[i] = cand
	}

	// Verify 0 duplicate physical pages allocated
	currentPhysBlocks := table.PhysicalBlockCount()
	if currentPhysBlocks != initialPhysBlocks {
		t.Fatalf("expected physical blocks count unchanged at %d, got %d", initialPhysBlocks, currentPhysBlocks)
	}

	// Verify refcounts: 1 parent + 16 candidates = 17 references per block
	for bIdx, blk := range parent.Blocks {
		if blk.RefCount() != int32(numCandidates+1) {
			t.Fatalf("block %d refcount expected %d, got %d", bIdx, numCandidates+1, blk.RefCount())
		}
	}

	// Squash all candidates
	for i := 0; i < numCandidates; i++ {
		candID := fmt.Sprintf("cand-zero-%d", i)
		if err := table.SquashCandidate(candID); err != nil {
			t.Fatalf("SquashCandidate %s failed: %v", candID, err)
		}
	}

	// Refcounts must return to 1
	for bIdx, blk := range parent.Blocks {
		if blk.RefCount() != 1 {
			t.Fatalf("block %d refcount expected 1 after all squashed, got %d", bIdx, blk.RefCount())
		}
	}
	if table.PhysicalBlockCount() != initialPhysBlocks {
		t.Fatalf("physical block count corrupted: got %d, want %d", table.PhysicalBlockCount(), initialPhysBlocks)
	}

	_ = table.ReleaseSession("parent-1024")
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks after release, got %d", table.PhysicalBlockCount())
	}
}

// TestCOWTreeSpeculation_FullPromotion verifies:
// Winning candidate branch is promoted directly into the authoritative parent session in zero memory copies.
func TestCOWTreeSpeculation_FullPromotion(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	parentTokens := make([]int, 100)
	for i := range parentTokens {
		parentTokens[i] = i + 1
	}

	parent, err := table.CreateSession("parent-promote-full")
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if err := table.AppendTokens("parent-promote-full", parentTokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	cand, err := table.ForkCandidate("parent-promote-full", "cand-win")
	if err != nil {
		t.Fatalf("ForkCandidate failed: %v", err)
	}

	draftTokens := []int{101, 102, 103, 104, 105, 106, 107, 108}
	if err := cand.AppendTokens(draftTokens); err != nil {
		t.Fatalf("AppendTokens candidate failed: %v", err)
	}

	if cand.TokenCount != 108 {
		t.Fatalf("expected candidate token count 108, got %d", cand.TokenCount)
	}

	// Promote winning candidate branch
	if err := table.PromoteCandidate("cand-win"); err != nil {
		t.Fatalf("PromoteCandidate failed: %v", err)
	}

	// Parent session must now have 108 tokens
	if parent.TokenCount != 108 {
		t.Fatalf("expected parent token count 108, got %d", parent.TokenCount)
	}
	readBack := parent.Tokens()
	if len(readBack) != 108 {
		t.Fatalf("expected 108 tokens read back, got %d", len(readBack))
	}
	for i := 0; i < 100; i++ {
		if readBack[i] != i+1 {
			t.Fatalf("prefix token %d corrupted: got %d, want %d", i, readBack[i], i+1)
		}
	}
	for i := 0; i < 8; i++ {
		if readBack[100+i] != 101+i {
			t.Fatalf("draft token %d corrupted: got %d, want %d", i, readBack[100+i], 101+i)
		}
	}

	// Candidate must be removed from active candidates
	if table.HasCandidate("cand-win") {
		t.Fatalf("candidate cand-win still reported active after promotion")
	}
	if !cand.IsReleased() {
		t.Fatalf("candidate cand-win should be released after promotion")
	}
	if !cand.IsAuthoritative() {
		t.Fatalf("candidate cand-win should be authoritative after promotion")
	}

	// Release parent session: all physical memory must be cleanly reclaimed
	if err := table.ReleaseSession("parent-promote-full"); err != nil {
		t.Fatalf("ReleaseSession failed: %v", err)
	}
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks after release, got %d", table.PhysicalBlockCount())
	}
	if table.TotalAllocatedBytes() != 0 {
		t.Fatalf("expected 0 allocated bytes after release, got %d", table.TotalAllocatedBytes())
	}
}

// TestCOWTreeSpeculation_PartialPromotion verifies:
// Partial acceptance of drafted tokens truncates the winning candidate to acceptedTokens,
// cleanly squashes unaccepted tail pages, and commits the accepted prefix to root.
func TestCOWTreeSpeculation_PartialPromotion(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	parentTokens := make([]int, 100)
	for i := range parentTokens {
		parentTokens[i] = i + 1000
	}

	parent, err := table.CreateSession("parent-promote-part")
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if err := table.AppendTokens("parent-promote-part", parentTokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	cand, err := table.ForkCandidate("parent-promote-part", "cand-part")
	if err != nil {
		t.Fatalf("ForkCandidate failed: %v", err)
	}

	draftTokens := []int{201, 202, 203, 204, 205, 206, 207, 208}
	if err := cand.AppendTokens(draftTokens); err != nil {
		t.Fatalf("AppendTokens candidate failed: %v", err)
	}

	// Model verifier accepts only first 3 draft tokens (201, 202, 203).
	// Total accepted tokens = 100 prefix + 3 accepted = 103 tokens.
	if err := table.PromoteCandidate("cand-part", 103); err != nil {
		t.Fatalf("PromoteCandidate with partial acceptance failed: %v", err)
	}

	// Verify parent session has exactly 103 tokens
	if parent.TokenCount != 103 {
		t.Fatalf("expected parent token count 103, got %d", parent.TokenCount)
	}
	readBack := parent.Tokens()
	if len(readBack) != 103 {
		t.Fatalf("expected 103 tokens read back, got %d", len(readBack))
	}
	for i := 0; i < 100; i++ {
		if readBack[i] != i+1000 {
			t.Fatalf("prefix token %d mismatch: got %d, want %d", i, readBack[i], i+1000)
		}
	}
	expectedDraft := []int{201, 202, 203}
	for i := 0; i < 3; i++ {
		if readBack[100+i] != expectedDraft[i] {
			t.Fatalf("accepted draft token %d mismatch: got %d, want %d", i, readBack[100+i], expectedDraft[i])
		}
	}

	// Clean release
	if err := table.ReleaseSession("parent-promote-part"); err != nil {
		t.Fatalf("ReleaseSession failed: %v", err)
	}
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks after release, got %d", table.PhysicalBlockCount())
	}
}

// TestCOWTreeSpeculation_TreeDepthLimit verifies:
// Hierarchical candidate trees enforce MaxCandidateTreeDepth (8) to prevent unbounded recursion.
func TestCOWTreeSpeculation_TreeDepthLimit(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	if _, err := table.CreateSession("root"); err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := table.AppendTokens("root", []int{1, 2, 3}); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}

	prevID := "root"
	// Fork candidates from depth 1 to MaxCandidateTreeDepth (8)
	for d := 1; d <= ctxmmu.MaxCandidateTreeDepth; d++ {
		candID := fmt.Sprintf("cand-depth-%d", d)
		cand, err := table.ForkCandidate(prevID, candID)
		if err != nil {
			t.Fatalf("ForkCandidate at depth %d failed: %v", d, err)
		}
		if cand.Depth() != d {
			t.Fatalf("expected depth %d, got %d", d, cand.Depth())
		}
		prevID = candID
	}

	// Attempting to fork at depth 9 must fail with ErrMaxTreeDepth
	_, err := table.ForkCandidate(prevID, "cand-depth-9")
	if !errors.Is(err, ctxmmu.ErrMaxTreeDepth) {
		t.Fatalf("expected ErrMaxTreeDepth at depth 9, got %v", err)
	}

	// Clean up all candidates
	for d := ctxmmu.MaxCandidateTreeDepth; d >= 1; d-- {
		candID := fmt.Sprintf("cand-depth-%d", d)
		if err := table.SquashCandidate(candID); err != nil {
			t.Fatalf("SquashCandidate %s failed: %v", candID, err)
		}
	}

	if err := table.ReleaseSession("root"); err != nil {
		t.Fatalf("ReleaseSession: %v", err)
	}
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks, got %d", table.PhysicalBlockCount())
	}
}

// TestCOWTreeSpeculation_ConcurrentSquashRace verifies thread safety under -race.
func TestCOWTreeSpeculation_ConcurrentSquashRace(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	prefixTokens := make([]int, 256)
	for i := range prefixTokens {
		prefixTokens[i] = i
	}

	parent, err := table.CreateSession("race-parent")
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := table.AppendTokens("race-parent", prefixTokens); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}

	const concurrency = 32
	var wg sync.WaitGroup
	errCh := make(chan error, concurrency*2)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			candID := fmt.Sprintf("race-cand-%d", workerID)

			cand, err := table.ForkCandidate("race-parent", candID)
			if err != nil {
				errCh <- fmt.Errorf("fork %s: %w", candID, err)
				return
			}

			// Draft tokens
			if err := cand.AppendTokens([]int{workerID * 100, workerID*100 + 1}); err != nil {
				errCh <- fmt.Errorf("append %s: %w", candID, err)
				return
			}

			// Read tokens
			toks := cand.Tokens()
			if len(toks) != 258 {
				errCh <- fmt.Errorf("len %s mismatch: got %d, want 258", candID, len(toks))
				return
			}

			// Squash
			if err := cand.Squash(); err != nil {
				errCh <- fmt.Errorf("squash %s: %w", candID, err)
				return
			}
		}(i)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	if parent.TokenCount != 256 {
		t.Fatalf("parent token count corrupted: %d", parent.TokenCount)
	}

	_ = table.ReleaseSession("race-parent")
	if table.PhysicalBlockCount() != 0 {
		t.Fatalf("expected 0 physical blocks, got %d", table.PhysicalBlockCount())
	}
}

// TestCOWTreeSpeculation_ErrorsAndValidation verifies sentinel errors and boundary checks.
func TestCOWTreeSpeculation_ErrorsAndValidation(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	if _, err := table.CreateSession("sess"); err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	// Empty IDs
	if _, err := table.ForkCandidate("", "c1"); !errors.Is(err, ctxmmu.ErrInvalidSessionID) {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}
	if _, err := table.ForkCandidate("sess", ""); !errors.Is(err, ctxmmu.ErrInvalidSessionID) {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}
	if err := table.SquashCandidate(""); !errors.Is(err, ctxmmu.ErrInvalidSessionID) {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}
	if err := table.PromoteCandidate(""); !errors.Is(err, ctxmmu.ErrInvalidSessionID) {
		t.Fatalf("expected ErrInvalidSessionID, got %v", err)
	}

	// Self fork
	if _, err := table.ForkCandidate("sess", "sess"); !errors.Is(err, ctxmmu.ErrSelfFork) {
		t.Fatalf("expected ErrSelfFork, got %v", err)
	}

	// Parent not found
	if _, err := table.ForkCandidate("non-existent", "c1"); !errors.Is(err, ctxmmu.ErrParentNotFound) {
		t.Fatalf("expected ErrParentNotFound, got %v", err)
	}

	// Fork valid candidate
	cand, err := table.ForkCandidate("sess", "c1")
	if err != nil {
		t.Fatalf("ForkCandidate: %v", err)
	}

	// Duplicate candidate
	if _, err := table.ForkCandidate("sess", "c1"); !errors.Is(err, ctxmmu.ErrCandidateExists) {
		t.Fatalf("expected ErrCandidateExists, got %v", err)
	}

	// Candidate not found
	if err := table.SquashCandidate("non-existent"); !errors.Is(err, ctxmmu.ErrCandidateNotFound) {
		t.Fatalf("expected ErrCandidateNotFound, got %v", err)
	}
	if err := table.PromoteCandidate("non-existent"); !errors.Is(err, ctxmmu.ErrCandidateNotFound) {
		t.Fatalf("expected ErrCandidateNotFound, got %v", err)
	}

	// Squash c1
	if err := cand.Squash(); err != nil {
		t.Fatalf("Squash: %v", err)
	}

	// Squash already squashed candidate
	if err := cand.Squash(); !errors.Is(err, ctxmmu.ErrCandidateNotFound) {
		t.Fatalf("expected ErrCandidateNotFound, got %v", err)
	}

	_ = table.ReleaseSession("sess")
}

// -----------------------------------------------------------------------------
// Benchmarks for Tree Speculation (#12319)
// -----------------------------------------------------------------------------

func BenchmarkCOWTreeSpeculation_ForkAndSquash(b *testing.B) {
	table := ctxmmu.NewCOWPageTable()

	prefixTokens := make([]int, 512)
	for i := range prefixTokens {
		prefixTokens[i] = 100 + i
	}
	if _, err := table.CreateSession("bench-parent"); err != nil {
		b.Fatalf("CreateSession: %v", err)
	}
	if err := table.AppendTokens("bench-parent", prefixTokens); err != nil {
		b.Fatalf("AppendTokens: %v", err)
	}

	var candNames = [16]string{
		"c0", "c1", "c2", "c3",
		"c4", "c5", "c6", "c7",
		"c8", "c9", "c10", "c11",
		"c12", "c13", "c14", "c15",
	}

	// Warmup run
	for i := 0; i < 16; i++ {
		c, err := table.ForkCandidate("bench-parent", candNames[i])
		if err != nil {
			b.Fatalf("warmup fork: %v", err)
		}
		_ = c.AppendTokens([]int{1, 2, 3, 4})
		if err := table.SquashCandidate(candNames[i]); err != nil {
			b.Fatalf("warmup squash: %v", err)
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for n := 0; n < b.N; n++ {
		for i := 0; i < 16; i++ {
			c, err := table.ForkCandidate("bench-parent", candNames[i])
			if err != nil {
				b.Fatalf("fork failed: %v", err)
			}
			_ = c.AppendTokens([]int{1, 2, 3, 4})
			if err := table.SquashCandidate(candNames[i]); err != nil {
				b.Fatalf("squash failed: %v", err)
			}
		}
	}
}

func BenchmarkCOWTreeSpeculation_SteadyState(b *testing.B) {
	table := ctxmmu.NewCOWPageTable()

	prefixTokens := make([]int, 512)
	for i := range prefixTokens {
		prefixTokens[i] = 100 + i
	}
	if _, err := table.CreateSession("bench-steady"); err != nil {
		b.Fatalf("CreateSession: %v", err)
	}
	if err := table.AppendTokens("bench-steady", prefixTokens); err != nil {
		b.Fatalf("AppendTokens: %v", err)
	}

	candName := "c-steady"

	// Warmup
	c, _ := table.ForkCandidate("bench-steady", candName)
	_ = c.AppendTokens([]int{1, 2, 3, 4})
	_ = table.SquashCandidate(candName)

	b.ResetTimer()
	b.ReportAllocs()

	for n := 0; n < b.N; n++ {
		c, err := table.ForkCandidate("bench-steady", candName)
		if err != nil {
			b.Fatalf("fork: %v", err)
		}
		_ = c.AppendTokens([]int{1, 2, 3, 4})
		if err := table.SquashCandidate(candName); err != nil {
			b.Fatalf("squash: %v", err)
		}
	}
}
