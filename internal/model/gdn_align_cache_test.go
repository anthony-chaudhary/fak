package model

import (
	"errors"
	"math"
	"testing"
	"time"
)

// TestQwenGDNAlignModePrefixCaching_Parity tests parity between uncached sequential prefill
// and cached prefix prefill using AlignStateManager recurrent checkpointing and PagedPrefixOwner forking.
// It verifies cosine similarity >= 0.999999, max difference <= 1e-6, zero clone bytes, and zero rematerialization.
func TestQwenGDNAlignModePrefixCaching_Parity(t *testing.T) {
	const (
		kHd          = 16
		vHd          = 16
		blockTokens  = 16
		prefixTokens = 512
		totalTokens  = 1024
	)

	type tokenInput struct {
		qn, kn, vh []float32
		bt, g      float32
	}

	// Generate deterministic inputs for all 1024 tokens
	inputs := make([]tokenInput, totalTokens)
	tokenIDs := make([]int, totalTokens)
	for tIdx := 0; tIdx < totalTokens; tIdx++ {
		tokenIDs[tIdx] = 1000 + tIdx
		qn := make([]float32, kHd)
		kn := make([]float32, kHd)
		vh := make([]float32, vHd)
		for i := 0; i < kHd; i++ {
			qn[i] = float32(math.Sin(float64(tIdx*17 + i*3 + 1)))
			kn[i] = float32(math.Cos(float64(tIdx*13 + i*5 + 2)))
		}
		for d := 0; d < vHd; d++ {
			vh[d] = float32(math.Sin(float64(tIdx*23 + d*7 + 3)))
		}
		bt := float32(0.5 + 0.1*math.Sin(float64(tIdx)))
		g := float32(0.92)
		inputs[tIdx] = tokenInput{qn: qn, kn: kn, vh: vh, bt: bt, g: g}
	}

	// Ground truth: uncached sequential prefill over all 1024 tokens
	stUncached := make([]float32, kHd*vHd)
	outUncached := make([][]float32, totalTokens)
	kvmem := make([]float32, vHd)
	delta := make([]float32, vHd)

	for tIdx := 0; tIdx < totalTokens; tIdx++ {
		od := make([]float32, vHd)
		ScalarHeadStep(stUncached, inputs[tIdx].qn, inputs[tIdx].kn, inputs[tIdx].vh, inputs[tIdx].bt, inputs[tIdx].g, od, kvmem, delta)
		outUncached[tIdx] = od
	}

	// Cached path: prefill first 512 tokens with AlignStateManager checkpointing
	alignMgr := NewAlignStateManager(blockTokens)
	stCached := make([]float32, kHd*vHd)

	for tIdx := 0; tIdx < prefixTokens; tIdx++ {
		od := make([]float32, vHd)
		ScalarHeadStep(stCached, inputs[tIdx].qn, inputs[tIdx].kn, inputs[tIdx].vh, inputs[tIdx].bt, inputs[tIdx].g, od, kvmem, delta)

		currentLayers := []GDNLayerState{
			{
				Layer:     0,
				Recurrent: append([]float32(nil), stCached...),
			},
		}
		checkpointed, blkIdx, err := alignMgr.Step(1, currentLayers, tokenIDs[:tIdx+1])
		if err != nil {
			t.Fatalf("alignMgr.Step failed at token %d: %v", tIdx, err)
		}
		if (tIdx+1)%blockTokens == 0 {
			expectedBlk := (tIdx+1)/blockTokens - 1
			if !checkpointed || blkIdx != expectedBlk {
				t.Fatalf("token %d: expected checkpointed=true, blkIdx=%d, got checkpointed=%v, blkIdx=%d",
					tIdx+1, expectedBlk, checkpointed, blkIdx)
			}
		}
	}

	// Construct attention K/Kraw/V test planes for prefix
	cfg := Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 16}
	stride := cfg.NumKVHeads * cfg.HeadDim
	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:       cfg,
		BlockTokens:  blockTokens,
		Tokens:       prefixTokens,
		TokenIDs:     tokenIDs[:prefixTokens],
		K:            kData,
		Kraw:         krawData,
		V:            vData,
		Sidecar:      []GDNLayerState{{Layer: 0, Recurrent: append([]float32(nil), stCached...)}},
		AlignManager: alignMgr,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	// Fork continuation session at 512 tokens
	sess, err := owner.ForkSessionAt("session-continuation-512", prefixTokens)
	if err != nil {
		t.Fatalf("ForkSessionAt 512 failed: %v", err)
	}
	defer sess.Release()

	// Verify telemetry: zero clone bytes, 32 shared pages (512 / 16), zero rematerialization
	telemetry := sess.Telemetry()
	if telemetry.ForkCloneBytes != 0 {
		t.Errorf("expected ForkCloneBytes == 0, got %d", telemetry.ForkCloneBytes)
	}
	if telemetry.SharedPages != 32 {
		t.Errorf("expected SharedPages == 32, got %d", telemetry.SharedPages)
	}
	if err := sess.VerifyZeroRematerialization(); err != nil {
		t.Errorf("VerifyZeroRematerialization failed: %v", err)
	}

	// Restore recurrent state from session sidecar
	layer0, ok := sess.LayerSidecar(0)
	if !ok {
		t.Fatalf("failed to retrieve layer 0 from continuation session sidecar")
	}
	stContinuation := append([]float32(nil), layer0.Recurrent...)

	// Continue forward pass for tokens 512..1023
	outCached := make([][]float32, totalTokens)
	for tIdx := prefixTokens; tIdx < totalTokens; tIdx++ {
		od := make([]float32, vHd)
		ScalarHeadStep(stContinuation, inputs[tIdx].qn, inputs[tIdx].kn, inputs[tIdx].vh, inputs[tIdx].bt, inputs[tIdx].g, od, kvmem, delta)
		outCached[tIdx] = od
	}

	// Compare outputs between uncached and cached paths for tokens 512..1023
	for tIdx := prefixTokens; tIdx < totalTokens; tIdx++ {
		uncachedVec := outUncached[tIdx]
		cachedVec := outCached[tIdx]

		var maxDiff float64
		var dot, normUncached, normCached float64
		for d := 0; d < vHd; d++ {
			u := float64(uncachedVec[d])
			c := float64(cachedVec[d])
			diff := math.Abs(u - c)
			if diff > maxDiff {
				maxDiff = diff
			}
			dot += u * c
			normUncached += u * u
			normCached += c * c
		}

		if maxDiff > 1e-6 {
			t.Fatalf("token %d: max difference %e exceeds 1e-6", tIdx, maxDiff)
		}

		cosSim := dot / (math.Sqrt(normUncached) * math.Sqrt(normCached))
		if cosSim < 0.999999 {
			t.Fatalf("token %d: cosine similarity %f < 0.999999", tIdx, cosSim)
		}
	}
}

// TestPagedPrefixOwner_AlignForkAtArbitraryBlockBoundaries verifies forking at arbitrary
// block boundaries (16, 32, 48 tokens) restores the correct slab and shares blocks,
// while non-aligned requests are rejected with ErrNonExactHybridPrefix.
func TestPagedPrefixOwner_AlignForkAtArbitraryBlockBoundaries(t *testing.T) {
	const (
		blockTokens  = 16
		prefixTokens = 64 // 4 blocks
	)

	cfg := Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 8}
	stride := cfg.NumKVHeads * cfg.HeadDim
	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)

	tokenIDs := make([]int, prefixTokens)
	for i := range tokenIDs {
		tokenIDs[i] = 2000 + i
	}

	alignMgr := NewAlignStateManager(blockTokens)
	for b := 0; b < 4; b++ {
		count := (b + 1) * blockTokens
		sidecar := []GDNLayerState{
			{
				Layer:     0,
				Recurrent: []float32{float32((b + 1) * 100)},
			},
		}
		if _, err := alignMgr.CheckpointBlock(b, count, tokenIDs[:count], sidecar); err != nil {
			t.Fatalf("CheckpointBlock %d failed: %v", b, err)
		}
	}

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:       cfg,
		BlockTokens:  blockTokens,
		Tokens:       prefixTokens,
		TokenIDs:     tokenIDs,
		K:            kData,
		Kraw:         krawData,
		V:            vData,
		Sidecar:      []GDNLayerState{{Layer: 0, Recurrent: []float32{400}}},
		AlignManager: alignMgr,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	// Fork at block 0 (16 tokens)
	s0, err := owner.ForkSessionAt("session-block-0", 16)
	if err != nil {
		t.Fatalf("ForkSessionAt 16 failed: %v", err)
	}
	defer s0.Release()
	if s0.Telemetry().SharedPages != 1 {
		t.Errorf("s0 SharedPages = %d, want 1", s0.Telemetry().SharedPages)
	}
	if s0.Telemetry().ForkCloneBytes != 0 {
		t.Errorf("s0 ForkCloneBytes = %d, want 0", s0.Telemetry().ForkCloneBytes)
	}
	l0, _ := s0.LayerSidecar(0)
	if len(l0.Recurrent) == 0 || l0.Recurrent[0] != 100 {
		t.Errorf("s0 Recurrent[0] = %v, want 100", l0.Recurrent)
	}

	// Fork at block 1 (32 tokens)
	s1, err := owner.ForkSessionAt("session-block-1", 32)
	if err != nil {
		t.Fatalf("ForkSessionAt 32 failed: %v", err)
	}
	defer s1.Release()
	if s1.Telemetry().SharedPages != 2 {
		t.Errorf("s1 SharedPages = %d, want 2", s1.Telemetry().SharedPages)
	}
	l1, _ := s1.LayerSidecar(0)
	if len(l1.Recurrent) == 0 || l1.Recurrent[0] != 200 {
		t.Errorf("s1 Recurrent[0] = %v, want 200", l1.Recurrent)
	}

	// Fork at block 2 (48 tokens)
	s2, err := owner.ForkSessionAt("session-block-2", 48)
	if err != nil {
		t.Fatalf("ForkSessionAt 48 failed: %v", err)
	}
	defer s2.Release()
	if s2.Telemetry().SharedPages != 3 {
		t.Errorf("s2 SharedPages = %d, want 3", s2.Telemetry().SharedPages)
	}
	l2, _ := s2.LayerSidecar(0)
	if len(l2.Recurrent) == 0 || l2.Recurrent[0] != 300 {
		t.Errorf("s2 Recurrent[0] = %v, want 300", l2.Recurrent)
	}

	// Fork at full prefix (64 tokens)
	s3, err := owner.ForkSessionAt("session-block-3", 64)
	if err != nil {
		t.Fatalf("ForkSessionAt 64 failed: %v", err)
	}
	defer s3.Release()
	if s3.Telemetry().SharedPages != 4 {
		t.Errorf("s3 SharedPages = %d, want 4", s3.Telemetry().SharedPages)
	}

	// Non-aligned fork request (20 tokens) should fail with ErrNonExactHybridPrefix
	_, errNonAligned := owner.ForkSessionAt("session-non-aligned", 20)
	if errNonAligned == nil {
		t.Fatalf("expected error for non-aligned ForkSessionAt 20, got nil")
	}
	if !errors.Is(errNonAligned, ErrNonExactHybridPrefix) {
		t.Errorf("expected ErrNonExactHybridPrefix, got %v", errNonAligned)
	}

	// ForkSessionExact with aligned prefix tokens
	sExact16, err := owner.ForkSessionExact("session-exact-16", tokenIDs[:16])
	if err != nil {
		t.Fatalf("ForkSessionExact 16 failed: %v", err)
	}
	defer sExact16.Release()
	if sExact16.Telemetry().SharedPages != 1 {
		t.Errorf("sExact16 SharedPages = %d, want 1", sExact16.Telemetry().SharedPages)
	}

	// ForkSessionExact with non-aligned prefix tokens (20 tokens)
	_, errExactNonAligned := owner.ForkSessionExact("session-exact-20", tokenIDs[:20])
	if errExactNonAligned == nil {
		t.Fatalf("expected error for non-aligned ForkSessionExact 20, got nil")
	}
	if !errors.Is(errExactNonAligned, ErrNonExactHybridPrefix) {
		t.Errorf("expected ErrNonExactHybridPrefix, got %v", errExactNonAligned)
	}

	// ForkSessionExact with mismatched token IDs
	mismatched := append([]int(nil), tokenIDs[:16]...)
	mismatched[0] = 9999
	_, errMismatch := owner.ForkSessionExact("session-mismatch", mismatched)
	if errMismatch == nil {
		t.Fatalf("expected error for mismatched tokens, got nil")
	}
	if !errors.Is(errMismatch, ErrNonExactHybridPrefix) {
		t.Errorf("expected ErrNonExactHybridPrefix, got %v", errMismatch)
	}
}

// TestQwenGDNAlign_RadixPrefixMatch verifies AlignRadixTree multi-block common prefix matching.
func TestQwenGDNAlign_RadixPrefixMatch(t *testing.T) {
	const blockSize = 16
	tree := NewAlignRadixTree(blockSize)

	// Create 3 dummy slabs for sequence A (48 tokens = 3 blocks)
	toksA := make([]int, 48)
	for i := range toksA {
		toksA[i] = 100 + i
	}
	slabsA := make([]*RecurrentStateSlab, 3)
	for b := 0; b < 3; b++ {
		slabsA[b] = NewRecurrentStateSlab(b, (b+1)*blockSize, toksA[:(b+1)*blockSize], []GDNLayerState{
			{Layer: 0, Recurrent: []float32{float32(b + 1)}},
		})
	}

	if err := tree.InsertSequence(toksA, slabsA); err != nil {
		t.Fatalf("InsertSequence A failed: %v", err)
	}

	// Sequence B: shares first 32 tokens (blocks 0 and 1) with A, divergent block 2
	toksB := make([]int, 48)
	copy(toksB[:32], toksA[:32])
	for i := 32; i < 48; i++ {
		toksB[i] = 900 + i
	}
	slabsB := make([]*RecurrentStateSlab, 3)
	slabsB[0] = slabsA[0]
	slabsB[1] = slabsA[1]
	slabsB[2] = NewRecurrentStateSlab(2, 48, toksB, []GDNLayerState{
		{Layer: 0, Recurrent: []float32{99.0}},
	})

	if err := tree.InsertSequence(toksB, slabsB); err != nil {
		t.Fatalf("InsertSequence B failed: %v", err)
	}

	// Query 1: full match sequence A (48 tokens)
	matchedBlocks, matchedTokens, slab := tree.MatchLongestPrefix(toksA)
	if matchedBlocks != 3 || matchedTokens != 48 || slab == nil {
		t.Fatalf("query A: expected 3 blocks, 48 tokens, got %d blocks, %d tokens, slab=%v",
			matchedBlocks, matchedTokens, slab)
	}
	if slab.BlockIndex != 2 || slab.Layers[0].Recurrent[0] != 3.0 {
		t.Errorf("query A slab mismatch: BlockIndex=%d, val=%v", slab.BlockIndex, slab.Layers[0].Recurrent[0])
	}

	// Query 2: full match sequence B (48 tokens)
	matchedBlocks, matchedTokens, slab = tree.MatchLongestPrefix(toksB)
	if matchedBlocks != 3 || matchedTokens != 48 || slab == nil {
		t.Fatalf("query B: expected 3 blocks, 48 tokens, got %d blocks, %d tokens", matchedBlocks, matchedTokens)
	}
	if slab.BlockIndex != 2 || slab.Layers[0].Recurrent[0] != 99.0 {
		t.Errorf("query B slab mismatch: BlockIndex=%d, val=%v", slab.BlockIndex, slab.Layers[0].Recurrent[0])
	}

	// Query 3: partial sequence matching 2 blocks (32 tokens)
	query2Blocks := toksA[:32]
	matchedBlocks, matchedTokens, slab = tree.MatchLongestPrefix(query2Blocks)
	if matchedBlocks != 2 || matchedTokens != 32 || slab == nil {
		t.Fatalf("query 2 blocks: expected 2 blocks, 32 tokens, got %d blocks, %d tokens", matchedBlocks, matchedTokens)
	}
	if slab.BlockIndex != 1 {
		t.Errorf("query 2 blocks slab index = %d, want 1", slab.BlockIndex)
	}

	// Query 4: sequence with unaligned extra tokens (25 tokens) -> should match exactly 1 block (16 tokens)
	queryUnaligned := toksA[:25]
	matchedBlocks, matchedTokens, slab = tree.MatchLongestPrefix(queryUnaligned)
	if matchedBlocks != 1 || matchedTokens != 16 || slab == nil {
		t.Fatalf("query unaligned: expected 1 block, 16 tokens, got %d blocks, %d tokens", matchedBlocks, matchedTokens)
	}
	if slab.BlockIndex != 0 {
		t.Errorf("query unaligned slab index = %d, want 0", slab.BlockIndex)
	}

	// Query 5: completely divergent sequence
	divergent := []int{555, 666, 777, 888, 999, 111, 222, 333, 444, 555, 666, 777, 888, 999, 111, 222}
	matchedBlocks, matchedTokens, slab = tree.MatchLongestPrefix(divergent)
	if matchedBlocks != 0 || matchedTokens != 0 || slab != nil {
		t.Fatalf("query divergent: expected 0 matches and nil slab, got %d blocks, %d tokens, slab=%v",
			matchedBlocks, matchedTokens, slab)
	}
}

// TestQwenGDNAlign_CopyForwardBoundary verifies that Step triggers checkpointing and copy-forward on boundary crossing.
func TestQwenGDNAlign_CopyForwardBoundary(t *testing.T) {
	const blockSize = 16
	mgr := NewAlignStateManager(blockSize)

	tokenIDs := make([]int, 32)
	for i := range tokenIDs {
		tokenIDs[i] = 3000 + i
	}

	layers := []GDNLayerState{
		{
			Layer:     0,
			Recurrent: []float32{1.0, 2.0, 3.0, 4.0},
		},
	}

	// Step tokens 1..15 (no boundary crossing)
	for step := 1; step < blockSize; step++ {
		checkpointed, blkIdx, err := mgr.Step(1, layers, tokenIDs[:step])
		if err != nil {
			t.Fatalf("Step %d failed: %v", step, err)
		}
		if checkpointed {
			t.Fatalf("Step %d: unexpected checkpoint before boundary", step)
		}
		if blkIdx != -1 {
			t.Fatalf("Step %d: expected blkIdx -1, got %d", step, blkIdx)
		}
	}

	// Step token 16 (crosses block 0 boundary)
	checkpointed, blkIdx, err := mgr.Step(1, layers, tokenIDs[:blockSize])
	if err != nil {
		t.Fatalf("Step 16 failed: %v", err)
	}
	if !checkpointed || blkIdx != 0 {
		t.Fatalf("Step 16: expected checkpointed=true, blkIdx=0, got checkpointed=%v, blkIdx=%d", checkpointed, blkIdx)
	}
	if !mgr.HasCheckpoint(0) {
		t.Fatalf("expected HasCheckpoint(0) == true")
	}

	slab0, ok := mgr.GetSlab(0)
	if !ok || slab0 == nil {
		t.Fatalf("GetSlab(0) failed")
	}
	if !slab0.IsFrozen {
		t.Errorf("expected slab0 to be frozen")
	}
	if slab0.Layers[0].Recurrent[0] != 1.0 {
		t.Errorf("slab0 recurrent val = %v, want 1.0", slab0.Layers[0].Recurrent[0])
	}

	// Mutate active layers for subsequent steps — ensure frozen slab is not corrupted (copy-forward invariant)
	layers[0].Recurrent[0] = 999.0
	checkpointed, blkIdx, err = mgr.Step(1, layers, tokenIDs[:blockSize+1])
	if err != nil {
		t.Fatalf("Step 17 failed: %v", err)
	}
	if checkpointed {
		t.Fatalf("Step 17: unexpected checkpoint")
	}

	// Verify slab 0 was not affected by mutation of active state
	slab0After, _ := mgr.GetSlab(0)
	if slab0After.Layers[0].Recurrent[0] != 1.0 {
		t.Fatalf("slab0 corrupted after active mutation: got %v, want 1.0", slab0After.Layers[0].Recurrent[0])
	}

	// Step remaining 15 tokens to reach 32 (block 1 boundary)
	checkpointed, blkIdx, err = mgr.Step(15, layers, tokenIDs[:32])
	if err != nil {
		t.Fatalf("Step to 32 failed: %v", err)
	}
	if !checkpointed || blkIdx != 1 {
		t.Fatalf("Step 32: expected checkpointed=true, blkIdx=1, got checkpointed=%v, blkIdx=%d", checkpointed, blkIdx)
	}
	if !mgr.HasCheckpoint(1) {
		t.Fatalf("expected HasCheckpoint(1) == true")
	}

	// Verify Stats
	stats := mgr.Stats()
	if stats.BlockSize != blockSize {
		t.Errorf("stats.BlockSize = %d, want %d", stats.BlockSize, blockSize)
	}
	if stats.CheckpointsCount != 2 {
		t.Errorf("stats.CheckpointsCount = %d, want 2", stats.CheckpointsCount)
	}
	if stats.TotalTokens != 32 {
		t.Errorf("stats.TotalTokens = %d, want 32", stats.TotalTokens)
	}
	if stats.TotalBytes <= 0 {
		t.Errorf("stats.TotalBytes = %d, want > 0", stats.TotalBytes)
	}

	// Verify RestoreAtTokens
	restored, err := mgr.RestoreAtTokens(16)
	if err != nil {
		t.Fatalf("RestoreAtTokens(16) failed: %v", err)
	}
	if len(restored) == 0 || restored[0].Recurrent[0] != 1.0 {
		t.Errorf("restored state mismatch: got %v, want 1.0", restored[0].Recurrent[0])
	}

	// Unaligned RestoreAtTokens should error
	if _, err := mgr.RestoreAtTokens(20); err == nil {
		t.Fatalf("expected error on unaligned RestoreAtTokens(20), got nil")
	}

	// Step with tokensDelta == 0 should be a no-op even if on a boundary
	checkpointed, blkIdx, err = mgr.Step(0, layers, nil)
	if err != nil {
		t.Fatalf("Step(0) failed: %v", err)
	}
	if checkpointed || blkIdx != -1 {
		t.Errorf("Step(0) returned checkpointed=%v, blkIdx=%d, want false, -1", checkpointed, blkIdx)
	}
}

// TestQwenGDNAlign_TTFTReduction_1024Tokens proves TTFT (Time To First Token) reduction
// on a repeated 1,024-token prompt prefix.
func TestQwenGDNAlign_TTFTReduction_1024Tokens(t *testing.T) {
	const (
		kHd          = 16
		vHd          = 16
		blockTokens  = 16
		prefixTokens = 1024
		totalTokens  = 1025 // 1024 prefix tokens + 1 continuation token (first token generation)
		iterations   = 20   // Average over iterations to dampen OS jitter
	)

	type tokenInput struct {
		qn, kn, vh []float32
		bt, g      float32
	}

	inputs := make([]tokenInput, totalTokens)
	tokenIDs := make([]int, totalTokens)
	for tIdx := 0; tIdx < totalTokens; tIdx++ {
		tokenIDs[tIdx] = 5000 + tIdx
		qn := make([]float32, kHd)
		kn := make([]float32, kHd)
		vh := make([]float32, vHd)
		for i := 0; i < kHd; i++ {
			qn[i] = float32(math.Sin(float64(tIdx*11 + i*5 + 1)))
			kn[i] = float32(math.Cos(float64(tIdx*7 + i*3 + 2)))
		}
		for d := 0; d < vHd; d++ {
			vh[d] = float32(math.Sin(float64(tIdx*13 + d*2 + 4)))
		}
		inputs[tIdx] = tokenInput{
			qn: qn,
			kn: kn,
			vh: vh,
			bt: 0.45,
			g:  0.95,
		}
	}

	// 1. Establish cached prefix state with AlignStateManager across 1024 tokens
	alignMgr := NewAlignStateManager(blockTokens)
	stPrefix := make([]float32, kHd*vHd)
	kvmem := make([]float32, vHd)
	delta := make([]float32, vHd)

	for tIdx := 0; tIdx < prefixTokens; tIdx++ {
		od := make([]float32, vHd)
		ScalarHeadStep(stPrefix, inputs[tIdx].qn, inputs[tIdx].kn, inputs[tIdx].vh, inputs[tIdx].bt, inputs[tIdx].g, od, kvmem, delta)

		currentLayers := []GDNLayerState{
			{Layer: 0, Recurrent: append([]float32(nil), stPrefix...)},
		}
		if _, _, err := alignMgr.Step(1, currentLayers, tokenIDs[:tIdx+1]); err != nil {
			t.Fatalf("alignMgr.Step failed at token %d: %v", tIdx, err)
		}
	}

	cfg := Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 16}
	stride := cfg.NumKVHeads * cfg.HeadDim
	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:       cfg,
		BlockTokens:  blockTokens,
		Tokens:       prefixTokens,
		TokenIDs:     tokenIDs[:prefixTokens],
		K:            kData,
		Kraw:         krawData,
		V:            vData,
		Sidecar:      []GDNLayerState{{Layer: 0, Recurrent: append([]float32(nil), stPrefix...)}},
		AlignManager: alignMgr,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	// 2. Measure uncached TTFT: full re-prefill of 1024 prompt tokens + 1 continuation token
	odUncached := make([]float32, vHd)
	var uncachedDuration int64
	for it := 0; it < iterations; it++ {
		stUncached := make([]float32, kHd*vHd)
		start := makeTestTimestamp()
		for tIdx := 0; tIdx < totalTokens; tIdx++ {
			clear(odUncached)
			ScalarHeadStep(stUncached, inputs[tIdx].qn, inputs[tIdx].kn, inputs[tIdx].vh, inputs[tIdx].bt, inputs[tIdx].g, odUncached, kvmem, delta)
		}
		uncachedDuration += makeTestTimestamp() - start
	}

	// 3. Measure cached TTFT: ForkSessionAt(1024) and evaluate ONLY 1 continuation token
	odCached := make([]float32, vHd)
	var cachedDuration int64
	var lastSession *PagedPrefixSession

	for it := 0; it < iterations; it++ {
		start := makeTestTimestamp()
		sess, err := owner.ForkSessionAt("ttft-session", prefixTokens)
		if err != nil {
			t.Fatalf("ForkSessionAt failed: %v", err)
		}
		layer0, ok := sess.LayerSidecar(0)
		if !ok {
			t.Fatalf("missing layer 0 sidecar")
		}
		stContinuation := append([]float32(nil), layer0.Recurrent...)
		clear(odCached)
		// Only evaluate token 1024
		ScalarHeadStep(stContinuation, inputs[1024].qn, inputs[1024].kn, inputs[1024].vh, inputs[1024].bt, inputs[1024].g, odCached, kvmem, delta)
		cachedDuration += makeTestTimestamp() - start

		if lastSession != nil {
			lastSession.Release()
		}
		lastSession = sess
	}
	if lastSession != nil {
		defer lastSession.Release()
	}

	// 4. Verify numerical parity on the first generated continuation token (token 1024)
	for d := 0; d < vHd; d++ {
		diff := math.Abs(float64(odUncached[d] - odCached[d]))
		if diff > 1e-6 {
			t.Fatalf("continuation token output mismatch at dim %d: uncached=%f, cached=%f, diff=%e",
				d, odUncached[d], odCached[d], diff)
		}
	}

	// 5. Verify zero-copy telemetry
	telemetry := lastSession.Telemetry()
	if telemetry.ForkCloneBytes != 0 {
		t.Errorf("expected ForkCloneBytes == 0, got %d", telemetry.ForkCloneBytes)
	}
	expectedBlocks := prefixTokens / blockTokens // 1024 / 16 = 64 blocks
	if telemetry.SharedPages != expectedBlocks {
		t.Errorf("expected SharedPages == %d, got %d", expectedBlocks, telemetry.SharedPages)
	}
	if err := lastSession.VerifyZeroRematerialization(); err != nil {
		t.Errorf("VerifyZeroRematerialization failed: %v", err)
	}

	// 6. Prove TTFT reduction: cached TTFT must be strictly faster than uncached prefill
	if cachedDuration >= uncachedDuration {
		t.Fatalf("TTFT reduction unmet: cachedDuration (%d ns) >= uncachedDuration (%d ns)",
			cachedDuration, uncachedDuration)
	}

	speedup := float64(uncachedDuration) / float64(cachedDuration)
	t.Logf("TTFT Benchmark on 1,024-token prompt prefix (%d iterations):", iterations)
	t.Logf("  Uncached prefill TTFT total: %d ns (avg: %.2f µs/op)", uncachedDuration, float64(uncachedDuration)/(float64(iterations)*1000.0))
	t.Logf("  Cached align-mode TTFT total: %d ns (avg: %.2f µs/op)", cachedDuration, float64(cachedDuration)/(float64(iterations)*1000.0))
	t.Logf("  Measured TTFT Speedup: %.2fx (recurrent steps avoided: %d per request)", speedup, prefixTokens)
}

func makeTestTimestamp() int64 {
	return time.Now().UnixNano()
}
