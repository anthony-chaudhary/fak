package ctxmmu_test

import (
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
