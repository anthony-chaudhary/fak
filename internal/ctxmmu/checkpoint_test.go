package ctxmmu

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestInPlaceCheckpoint_SaveAndRestore verifies end-to-end saving and restoring
// of in-place prompt cache checkpoints with physical block pinning, sequence descriptors,
// and zero-copy resumption.
func TestInPlaceCheckpoint_SaveAndRestore(t *testing.T) {
	mmu := New()
	pool := NewSharedTokenPool()
	forkMgr := NewForkManager()
	cowTable := NewCOWPageTable()
	cm := NewCheckpointManager(mmu, pool, forkMgr, cowTable)

	sessionID := "agent-stream-save-restore-1"

	// Register session in fork manager and append tokens
	forkedSess, err := forkMgr.RegisterSession(sessionID, BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	tokenCount := 128
	tokens := make([]int32, tokenCount)
	for i := 0; i < tokenCount; i++ {
		tokens[i] = int32(1000 + i)
	}
	if err := forkedSess.AppendTokens(tokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	// Also register session in COW table and append tokens
	cowSess, err := cowTable.RegisterSession(sessionID)
	if err != nil {
		t.Fatalf("COW RegisterSession failed: %v", err)
	}
	cowTokens := make([]int, 64)
	for i := 0; i < 64; i++ {
		cowTokens[i] = 2000 + i
	}
	if err := cowSess.AppendTokens(cowTokens); err != nil {
		t.Fatalf("COW AppendTokens failed: %v", err)
	}

	// Reserve and commit in shared token pool
	if err := pool.Reserve(sessionID, 1000); err != nil {
		t.Fatalf("pool.Reserve failed: %v", err)
	}
	if err := pool.Commit(sessionID, 400); err != nil {
		t.Fatalf("pool.Commit failed: %v", err)
	}

	// Inject a quarantined result into MMU to verify quarantined ID harvesting
	mmu.mu.Lock()
	mmu.held["q101"] = abi.Ref{Kind: abi.RefBlob, Digest: "q101"}
	mmu.held["q102"] = abi.Ref{Kind: abi.RefBlob, Digest: "q102"}
	mmu.mu.Unlock()

	// Initial check on block refcount
	initialEntries := forkedSess.PageTableEntries()
	if len(initialEntries) == 0 {
		t.Fatalf("expected page table entries in forked session")
	}
	initialRef := initialEntries[0].PhysicalBlock.RefCount()

	// Step 1: Save in-place checkpoint
	desc, err := cm.SaveInPlaceCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("SaveInPlaceCheckpoint failed: %v", err)
	}
	if desc == nil {
		t.Fatalf("expected non-nil descriptor")
	}

	// Verify sequence descriptor fields
	if desc.SessionID != sessionID {
		t.Errorf("expected session ID %q, got %q", sessionID, desc.SessionID)
	}
	if desc.CommittedTokens != 400 {
		t.Errorf("expected 400 committed tokens, got %d", desc.CommittedTokens)
	}
	if desc.HeadroomTokens != 600 {
		t.Errorf("expected 600 headroom tokens, got %d", desc.HeadroomTokens)
	}

	// Verify output headroom was reclaimed in the pool while committed tokens remained pinned
	committedUsage, reservedUsage := pool.StreamUsage(sessionID)
	if committedUsage != 400 {
		t.Errorf("expected committed usage 400, got %d", committedUsage)
	}
	if reservedUsage != 0 {
		t.Errorf("expected reserved headroom 0 after pause, got %d", reservedUsage)
	}

	// Verify physical page blocks were retained (blk.Retain())
	if len(desc.PinnedBlocks) == 0 {
		t.Fatalf("expected pinned physical blocks in descriptor")
	}
	retainedRef := desc.PinnedBlocks[0].RefCount()
	if retainedRef != initialRef+1 {
		t.Errorf("expected retained ref count %d, got %d", initialRef+1, retainedRef)
	}

	// Verify AttentionMask
	if desc.AttentionMask.Geometry != GeometryCausal {
		t.Errorf("expected GeometryCausal, got %v", desc.AttentionMask.Geometry)
	}
	if !desc.AttentionMask.Causal {
		t.Errorf("expected causal mask")
	}
	if len(desc.AttentionMask.QuarantinedIDs) != 2 {
		t.Errorf("expected 2 quarantined IDs, got %d", len(desc.AttentionMask.QuarantinedIDs))
	}

	// Verify DeltaNet state descriptor (~110 MB)
	if desc.DeltaNetState.NumLayers != 48 {
		t.Errorf("expected 48 layers, got %d", desc.DeltaNetState.NumLayers)
	}
	if desc.DeltaNetState.TotalStateBytes < 100*1024*1024 {
		t.Errorf("expected ~110 MB DeltaNet total state bytes, got %d", desc.DeltaNetState.TotalStateBytes)
	}

	// Verify RoPE offsets
	if desc.RoPE.PositionIndex != int64(tokenCount) {
		t.Errorf("expected RoPE position index %d, got %d", tokenCount, desc.RoPE.PositionIndex)
	}
	if desc.RoPE.BaseFrequency != 1000000.0 {
		t.Errorf("expected RoPE base frequency 1000000.0, got %f", desc.RoPE.BaseFrequency)
	}

	// Verify MTP draft state
	if desc.MTPState.DraftDepth != 4 {
		t.Errorf("expected MTP draft depth 4, got %d", desc.MTPState.DraftDepth)
	}

	// Verify PrefixHash
	if desc.PrefixHash == [32]byte{} {
		t.Errorf("expected non-empty prefix hash")
	}
	if desc.PrefixHashHex == "" {
		t.Errorf("expected non-empty prefix hash hex")
	}

	// Step 2: GetCheckpoint lookup
	gotDesc, err := cm.GetCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("GetCheckpoint failed: %v", err)
	}
	if gotDesc != desc {
		t.Errorf("GetCheckpoint returned mismatched descriptor")
	}

	// Step 3: Zero-copy resumption via RestoreInPlaceCheckpoint
	if err := cm.RestoreInPlaceCheckpoint(sessionID, desc); err != nil {
		t.Fatalf("RestoreInPlaceCheckpoint failed: %v", err)
	}

	if !desc.Restored {
		t.Errorf("expected descriptor to be marked restored")
	}
	if desc.PhysicalBytesTransferred != 0 {
		t.Errorf("expected 0 physical bytes transferred (zero-copy), got %d", desc.PhysicalBytesTransferred)
	}

	// Verify headroom was re-reserved in the pool
	committedAfter, reservedAfter := pool.StreamUsage(sessionID)
	if committedAfter != 400 {
		t.Errorf("expected committed usage 400, got %d", committedAfter)
	}
	if reservedAfter != 600 {
		t.Errorf("expected re-reserved headroom 600, got %d", reservedAfter)
	}

	// Verify physical block retain was released
	restoredRef := desc.PinnedBlocks[0].RefCount()
	if restoredRef != initialRef {
		t.Errorf("expected block ref count returned to %d, got %d", initialRef, restoredRef)
	}

	// Verify restoring again returns ErrCheckpointAlreadyRestored
	if err := cm.RestoreInPlaceCheckpoint(sessionID, desc); !errors.Is(err, ErrCheckpointAlreadyRestored) {
		t.Errorf("expected ErrCheckpointAlreadyRestored on double restore, got %v", err)
	}
}

// TestInPlaceCheckpoint_PrefixHashIntegrity verifies that cryptographic SHA-256 Merkle root
// validation prevents resuming from tampered or corrupted checkpoints.
func TestInPlaceCheckpoint_PrefixHashIntegrity(t *testing.T) {
	forkMgr := NewForkManager()
	cm := NewCheckpointManager(nil, nil, forkMgr, nil)

	sessionID := "prefix-integrity-stream"
	sess, err := forkMgr.RegisterSession(sessionID, BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	tokens := []int32{10, 20, 30, 40, 50, 60, 70, 80}
	if err := sess.AppendTokens(tokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	desc, err := cm.SaveInPlaceCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("SaveInPlaceCheckpoint failed: %v", err)
	}

	// Verify initial valid hash calculation
	expectedHash := ComputePrefixMerkleRoot(desc.TokenSequence, desc.PinnedBlocks, desc.PinnedCOWBlocks)
	if desc.PrefixHash != expectedHash {
		t.Fatalf("initial prefix hash mismatch: got %x, want %x", desc.PrefixHash, expectedHash)
	}

	// Tamper 1: Corrupt PrefixHash directly
	originalHash := desc.PrefixHash
	desc.PrefixHash[0] ^= 0xff

	err = cm.RestoreInPlaceCheckpoint(sessionID, desc)
	if !errors.Is(err, ErrPrefixHashMismatch) {
		t.Fatalf("expected ErrPrefixHashMismatch for corrupted PrefixHash, got %v", err)
	}

	// Restore correct hash
	desc.PrefixHash = originalHash

	// Tamper 2: Corrupt token sequence
	desc.TokenSequence[0] = 999999
	err = cm.RestoreInPlaceCheckpoint(sessionID, desc)
	if !errors.Is(err, ErrPrefixHashMismatch) {
		t.Fatalf("expected ErrPrefixHashMismatch for corrupted token sequence, got %v", err)
	}

	// Restore correct token
	desc.TokenSequence[0] = 10

	// Tamper 3: Corrupt physical block digest
	if len(desc.PinnedBlocks) > 0 {
		origDigest := desc.PinnedBlocks[0].Digest
		desc.PinnedBlocks[0].Digest[0] ^= 0xff

		err = cm.RestoreInPlaceCheckpoint(sessionID, desc)
		if !errors.Is(err, ErrPrefixHashMismatch) {
			t.Fatalf("expected ErrPrefixHashMismatch for corrupted block digest, got %v", err)
		}

		desc.PinnedBlocks[0].Digest = origDigest
	}

	// Untampered restore must succeed
	if err := cm.RestoreInPlaceCheckpoint(sessionID, desc); err != nil {
		t.Fatalf("untampered restore failed: %v", err)
	}
}

// TestInPlaceCheckpoint_SharedTokenPoolPinning proves that when an agent stream pauses
// for tool execution, its committed tokens remain pinned in the SharedTokenPool while
// uncommitted headroom is released to allow concurrent streams to proceed, and headroom
// is re-reserved upon resumption.
func TestInPlaceCheckpoint_SharedTokenPoolPinning(t *testing.T) {
	// 5000 max tokens total capacity
	maxTokens := 5000
	pool := NewSharedTokenPoolWithConfig(maxTokens, int64(maxTokens)*DefaultPoolBytesPerToken, DefaultPoolBytesPerToken, 1.0)
	cm := NewCheckpointManager(nil, pool, nil, nil)

	streamA := "agent-stream-A"
	streamB := "agent-stream-B"

	// Stream A reserves 2000 tokens and commits 500 tokens (headroom = 1500)
	if err := pool.Reserve(streamA, 2000); err != nil {
		t.Fatalf("pool.Reserve streamA failed: %v", err)
	}
	if err := pool.Commit(streamA, 500); err != nil {
		t.Fatalf("pool.Commit streamA failed: %v", err)
	}

	if pool.CommittedTokens() != 500 || pool.ReservedTokens() != 1500 {
		t.Fatalf("unexpected pool tokens: committed=%d reserved=%d", pool.CommittedTokens(), pool.ReservedTokens())
	}

	// Stream B attempts to reserve 3500 tokens -> must fail because 500+1500+3500 = 5500 > 5000
	if err := pool.Reserve(streamB, 3500); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("expected ErrPoolExhausted for streamB before streamA pause, got %v", err)
	}

	// Stream A pauses for tool execution -> in-place checkpoint preserves committed tokens
	descA, err := cm.SaveInPlaceCheckpoint(streamA)
	if err != nil {
		t.Fatalf("SaveInPlaceCheckpoint streamA failed: %v", err)
	}

	// Verify uncommitted headroom (1500) was freed while committed tokens (500) stayed pinned
	comA, resA := pool.StreamUsage(streamA)
	if comA != 500 {
		t.Fatalf("expected streamA committed tokens pinned at 500, got %d", comA)
	}
	if resA != 0 {
		t.Fatalf("expected streamA reserved headroom freed to 0, got %d", resA)
	}
	if pool.CommittedTokens() != 500 {
		t.Fatalf("expected pool committed tokens 500, got %d", pool.CommittedTokens())
	}
	if pool.ReservedTokens() != 0 {
		t.Fatalf("expected pool reserved tokens 0, got %d", pool.ReservedTokens())
	}
	if pool.FreeTokens() != 4500 {
		t.Fatalf("expected 4500 free tokens in pool, got %d", pool.FreeTokens())
	}

	// Now Stream B can successfully reserve 3500 tokens
	if err := pool.Reserve(streamB, 3500); err != nil {
		t.Fatalf("streamB reservation failed while streamA was paused: %v", err)
	}
	if pool.ReservedTokens() != 3500 {
		t.Fatalf("expected pool reserved tokens 3500, got %d", pool.ReservedTokens())
	}

	// Stream B completes and releases its tokens
	pool.Release(streamB)
	if pool.ReservedTokens() != 0 || pool.CommittedTokens() != 500 {
		t.Fatalf("unexpected pool state after streamB release: com=%d res=%d", pool.CommittedTokens(), pool.ReservedTokens())
	}

	// Stream A finishes tool execution and resumes
	if err := cm.RestoreInPlaceCheckpoint(streamA, descA); err != nil {
		t.Fatalf("RestoreInPlaceCheckpoint streamA failed: %v", err)
	}

	// Verify stream A's headroom was re-reserved and committed tokens are still intact
	comAfter, resAfter := pool.StreamUsage(streamA)
	if comAfter != 500 {
		t.Fatalf("expected streamA committed tokens 500 after restore, got %d", comAfter)
	}
	if resAfter != 1500 {
		t.Fatalf("expected streamA headroom re-reserved to 1500 after restore, got %d", resAfter)
	}
	if pool.CommittedTokens() != 500 || pool.ReservedTokens() != 1500 {
		t.Fatalf("unexpected pool totals after restore: com=%d res=%d", pool.CommittedTokens(), pool.ReservedTokens())
	}
}

// TestInPlaceCheckpoint_ExpiryAndReaping verifies checkpoint TTL enforcement,
// rejection of expired checkpoints, and reclamation of pinned physical page blocks upon reaping.
func TestInPlaceCheckpoint_ExpiryAndReaping(t *testing.T) {
	forkMgr := NewForkManager()
	cm := NewCheckpointManager(nil, nil, forkMgr, nil)

	// Create 3 sessions with physical blocks
	sessions := []string{"sess-exp-1", "sess-exp-2", "sess-live-3"}
	descs := make([]*SessionDescriptor, len(sessions))

	for i, sid := range sessions {
		s, err := forkMgr.RegisterSession(sid, BlockGranularity64)
		if err != nil {
			t.Fatalf("RegisterSession %s failed: %v", sid, err)
		}
		if err := s.AppendTokens(100, 200, 300); err != nil {
			t.Fatalf("AppendTokens %s failed: %v", sid, err)
		}
		desc, err := cm.SaveInPlaceCheckpoint(sid)
		if err != nil {
			t.Fatalf("SaveInPlaceCheckpoint %s failed: %v", sid, err)
		}
		descs[i] = desc

		// Verify initial retain increment
		if descs[i].PinnedBlocks[0].RefCount() < 2 {
			t.Fatalf("expected pinned block refcount >= 2, got %d", descs[i].PinnedBlocks[0].RefCount())
		}
	}

	// Expire session 1 and 2
	pastTime := time.Now().Add(-5 * time.Minute)
	descs[0].SetExpiresAt(pastTime)
	descs[1].SetExpiresAt(pastTime)

	// Attempting to restore an expired checkpoint returns ErrCheckpointExpired
	err := cm.RestoreInPlaceCheckpoint(sessions[0], descs[0])
	if !errors.Is(err, ErrCheckpointExpired) {
		t.Fatalf("expected ErrCheckpointExpired for session 0, got %v", err)
	}

	// Reap expired checkpoints
	reaped := cm.ReapExpiredCheckpoints(time.Now())
	if reaped != 2 {
		t.Fatalf("expected 2 checkpoints reaped, got %d", reaped)
	}

	// Verify session 1 and 2 are removed from manager
	if _, err := cm.GetCheckpoint(sessions[0]); !errors.Is(err, ErrCheckpointNotFound) {
		t.Errorf("expected ErrCheckpointNotFound for session 0, got %v", err)
	}
	if _, err := cm.GetCheckpoint(sessions[1]); !errors.Is(err, ErrCheckpointNotFound) {
		t.Errorf("expected ErrCheckpointNotFound for session 1, got %v", err)
	}

	// Verify pinned physical blocks had their temporary retain released back to 1
	if descs[0].PinnedBlocks[0].RefCount() != 1 {
		t.Errorf("expected session 0 block refcount restored to 1, got %d", descs[0].PinnedBlocks[0].RefCount())
	}
	if descs[1].PinnedBlocks[0].RefCount() != 1 {
		t.Errorf("expected session 1 block refcount restored to 1, got %d", descs[1].PinnedBlocks[0].RefCount())
	}

	// Verify live session 3 still exists and can be restored
	gotLive, err := cm.GetCheckpoint(sessions[2])
	if err != nil {
		t.Fatalf("expected session 3 to be present, got err: %v", err)
	}
	if err := cm.RestoreInPlaceCheckpoint(sessions[2], gotLive); err != nil {
		t.Fatalf("failed to restore live session 3: %v", err)
	}
}

// TestInPlaceCheckpoint_ZeroCopyResumptionLatency verifies that zero-copy resumption
// of ~110 MB sequence descriptors executes strictly within deadline (< 2.5s) with 0 physical
// bytes transferred over the bus.
func TestInPlaceCheckpoint_ZeroCopyResumptionLatency(t *testing.T) {
	forkMgr := NewForkManager()
	pool := NewSharedTokenPool()
	cm := NewCheckpointManager(nil, pool, forkMgr, nil)

	sessionID := "large-latency-stream"
	sess, err := forkMgr.RegisterSession(sessionID, BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}

	// 2048 tokens across 32 physical blocks
	tokenCount := 2048
	tokens := make([]int32, tokenCount)
	for i := 0; i < tokenCount; i++ {
		tokens[i] = int32(5000 + i)
	}
	if err := sess.AppendTokens(tokens...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	if err := pool.Reserve(sessionID, 4000); err != nil {
		t.Fatalf("pool.Reserve failed: %v", err)
	}
	if err := pool.Commit(sessionID, 2048); err != nil {
		t.Fatalf("pool.Commit failed: %v", err)
	}

	desc, err := cm.SaveInPlaceCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("SaveInPlaceCheckpoint failed: %v", err)
	}

	// Verify ~110 MB DeltaNet descriptor footprint
	if desc.DeltaNetState.TotalStateBytes < 100*1024*1024 {
		t.Fatalf("expected ~110 MB DeltaNet state, got %d bytes", desc.DeltaNetState.TotalStateBytes)
	}

	// Allocate real buffers to confirm heavy descriptor handling
	desc.DeltaNetState.AllocateBuffers()

	start := time.Now()
	if err := cm.RestoreInPlaceCheckpoint(sessionID, desc); err != nil {
		t.Fatalf("RestoreInPlaceCheckpoint failed: %v", err)
	}
	elapsed := time.Since(start)

	// Verify zero-copy: 0 bytes transferred across bus
	if desc.PhysicalBytesTransferred != 0 {
		t.Errorf("expected 0 physical bytes transferred, got %d", desc.PhysicalBytesTransferred)
	}

	// Verify latency threshold < 2.5s (in practice < 50ms)
	maxLatency := 2500 * time.Millisecond
	if elapsed > maxLatency {
		t.Fatalf("resumption latency %v exceeded 2.5s deadline", elapsed)
	}
	if desc.ResumptionLatency > maxLatency {
		t.Fatalf("recorded descriptor resumption latency %v exceeded 2.5s deadline", desc.ResumptionLatency)
	}

	t.Logf("Zero-copy resumption of ~110 MB descriptor completed in %v (< 2.5s deadline, 0 physical bytes transferred)", elapsed)
}

// TestInPlaceCheckpoint_Concurrency tests concurrent saving, getting, restoring,
// and reaping of in-place checkpoints across multiple goroutines.
func TestInPlaceCheckpoint_Concurrency(t *testing.T) {
	forkMgr := NewForkManager()
	pool := NewSharedTokenPool()
	cm := NewCheckpointManager(nil, pool, forkMgr, nil)

	numWorkers := 16
	iterationsPerWorker := 10

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	stopReaper := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopReaper:
				return
			case <-ticker.C:
				cm.ReapExpiredCheckpoints(time.Now())
			}
		}
	}()

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()

			for iter := 0; iter < iterationsPerWorker; iter++ {
				sid := fmt.Sprintf("concurrent-sess-%d-%d", workerID, iter)

				sess, err := forkMgr.RegisterSession(sid, BlockGranularity16)
				if err != nil {
					t.Errorf("worker %d RegisterSession failed: %v", workerID, err)
					return
				}

				toks := []int32{int32(workerID*100 + iter), int32(workerID*100 + iter + 1)}
				if err := sess.AppendTokens(toks...); err != nil {
					t.Errorf("worker %d AppendTokens failed: %v", workerID, err)
					return
				}

				if err := pool.Reserve(sid, 100); err != nil {
					t.Errorf("worker %d pool.Reserve failed: %v", workerID, err)
					return
				}
				if err := pool.Commit(sid, 50); err != nil {
					t.Errorf("worker %d pool.Commit failed: %v", workerID, err)
					return
				}

				desc, err := cm.SaveInPlaceCheckpoint(sid)
				if err != nil {
					t.Errorf("worker %d SaveInPlaceCheckpoint failed: %v", workerID, err)
					return
				}

				gotDesc, err := cm.GetCheckpoint(sid)
				if err != nil || gotDesc != desc {
					t.Errorf("worker %d GetCheckpoint failed: %v", workerID, err)
					return
				}

				// Random sleep simulating tool execution duration
				time.Sleep(time.Duration(rand.Intn(5)) * time.Millisecond)

				if err := cm.RestoreInPlaceCheckpoint(sid, desc); err != nil {
					t.Errorf("worker %d RestoreInPlaceCheckpoint failed: %v", workerID, err)
					return
				}

				if desc.PhysicalBytesTransferred != 0 {
					t.Errorf("worker %d non-zero physical transfer: %d", workerID, desc.PhysicalBytesTransferred)
					return
				}

				pool.Release(sid)
				_ = forkMgr.ReleaseSession(sid)
			}
		}(w)
	}

	wg.Wait()
	close(stopReaper)
}

// TestInPlaceCheckpoint_EdgeCasesAndConvenienceFunctions verifies parameter validation,
// coherence failure handling, MMU integration, and package-level convenience functions.
func TestInPlaceCheckpoint_EdgeCasesAndConvenienceFunctions(t *testing.T) {
	mmu := New()
	pool := NewSharedTokenPool()
	cm := mmu.CheckpointManager(pool)

	// Validation checks
	if _, err := cm.SaveInPlaceCheckpoint(""); !errors.Is(err, ErrEmptySessionID) {
		t.Fatalf("expected ErrEmptySessionID for empty SaveInPlaceCheckpoint, got %v", err)
	}
	if err := cm.RestoreInPlaceCheckpoint("", nil); !errors.Is(err, ErrEmptySessionID) {
		t.Fatalf("expected ErrEmptySessionID for empty RestoreInPlaceCheckpoint, got %v", err)
	}
	if err := cm.RestoreInPlaceCheckpoint("s1", nil); !errors.Is(err, ErrNilDescriptor) {
		t.Fatalf("expected ErrNilDescriptor for nil descriptor, got %v", err)
	}
	if _, err := cm.GetCheckpoint(""); !errors.Is(err, ErrEmptySessionID) {
		t.Fatalf("expected ErrEmptySessionID for empty GetCheckpoint, got %v", err)
	}
	if _, err := cm.GetCheckpoint("non-existent"); !errors.Is(err, ErrCheckpointNotFound) {
		t.Fatalf("expected ErrCheckpointNotFound for unknown session, got %v", err)
	}

	// Session mismatch check
	desc := &SessionDescriptor{SessionID: "session-alpha"}
	if err := cm.RestoreInPlaceCheckpoint("session-beta", desc); !errors.Is(err, ErrSessionMismatch) {
		t.Fatalf("expected ErrSessionMismatch, got %v", err)
	}

	// Coherence failure check: physical block with zero host address
	incoherentBlock := &PhysicalKVBlock{
		ID:                9999,
		Granularity:       BlockGranularity64,
		HostAddress:       0, // null host address triggers coherence failure
		GPUVirtualAddress: RDNA35GPUVABase + 0x1000,
	}
	incoherentDesc := &SessionDescriptor{
		SessionID:    "incoherent-sess",
		PinnedBlocks: []*PhysicalKVBlock{incoherentBlock},
	}
	if err := cm.RestoreInPlaceCheckpoint("incoherent-sess", incoherentDesc); !errors.Is(err, ErrBlockIncoherent) {
		t.Fatalf("expected ErrBlockIncoherent, got %v", err)
	}

	// Package-level default checkpoint manager convenience functions
	defaultMgr := DefaultCheckpointManager()
	if defaultMgr == nil {
		t.Fatalf("expected non-nil default checkpoint manager")
	}

	// Test package-level Save/Restore/Get/Reap
	pkgSid := "pkg-level-test-sess"
	_ = pool.Reserve(pkgSid, 200)
	_ = pool.Commit(pkgSid, 100)

	pkgMgr := NewCheckpointManager(mmu, pool, nil, nil)
	SetDefaultCheckpointManager(pkgMgr)

	pkgDesc, err := SaveInPlaceCheckpoint(pkgSid)
	if err != nil {
		t.Fatalf("package-level SaveInPlaceCheckpoint failed: %v", err)
	}
	gotPkgDesc, err := GetCheckpoint(pkgSid)
	if err != nil || gotPkgDesc != pkgDesc {
		t.Fatalf("package-level GetCheckpoint failed: %v", err)
	}
	if err := RestoreInPlaceCheckpoint(pkgSid, pkgDesc); err != nil {
		t.Fatalf("package-level RestoreInPlaceCheckpoint failed: %v", err)
	}
	_ = ReapExpiredCheckpoints(time.Now())
}
