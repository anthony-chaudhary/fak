package model

import (
	"errors"
	"slices"
	"testing"
)

// Helper to construct predictable float32 K/Kraw/V test patterns.
func makeTestTokenPlanes(nLayers, stride, pos int) (k, kraw, v [][]float32) {
	k = make([][]float32, nLayers)
	kraw = make([][]float32, nLayers)
	v = make([][]float32, nLayers)
	for l := 0; l < nLayers; l++ {
		k[l] = make([]float32, stride)
		kraw[l] = make([]float32, stride)
		v[l] = make([]float32, stride)
		for d := 0; d < stride; d++ {
			base := float32(pos*1000 + l*10 + d)
			k[l][d] = base + 1.0
			kraw[l][d] = base + 2.0
			v[l][d] = -(base + 1.0)
		}
	}
	return k, kraw, v
}

func makeTestPrefixData(nTokens, nLayers, stride int) (kData, krawData, vData [][][]float32) {
	kData = make([][][]float32, nTokens)
	krawData = make([][][]float32, nTokens)
	vData = make([][][]float32, nTokens)
	for pos := 0; pos < nTokens; pos++ {
		kData[pos], krawData[pos], vData[pos] = makeTestTokenPlanes(nLayers, stride, pos)
	}
	return kData, krawData, vData
}

func makeTestSidecar() []GDNLayerState {
	return []GDNLayerState{
		{
			Layer:     0,
			Conv:      []float32{1.1, 1.2, 1.3},
			Recurrent: []float32{10.1, 10.2, 10.3, 10.4},
		},
		{
			Layer:     2,
			Conv:      []float32{2.1, 2.2, 2.3},
			Recurrent: []float32{20.1, 20.2, 20.3, 20.4},
		},
	}
}

// Test zero-copy fork: verify shared block identity and fork_clone_bytes == 0.
func TestPagedPrefixCOW_ZeroCopyFork(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const blockTokens = 16
	const prefixTokens = 32 // exactly 2 blocks
	stride := cfg.NumKVHeads * cfg.HeadDim

	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)
	sidecar := makeTestSidecar()

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:      cfg,
		BlockTokens: blockTokens,
		Tokens:      prefixTokens,
		K:           kData,
		Kraw:        krawData,
		V:           vData,
		Sidecar:     sidecar,
		IsMetal:     true,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	if owner.PrefixTokens() != prefixTokens {
		t.Fatalf("PrefixTokens = %d, want %d", owner.PrefixTokens(), prefixTokens)
	}
	if len(owner.PageTable()) != 2 {
		t.Fatalf("PageTable len = %d, want 2", len(owner.PageTable()))
	}

	// Fork child session
	child, err := owner.ForkSession("session-agent-1")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}
	defer child.Release()

	// 1. Verify shared block identity
	if child.Len() != prefixTokens {
		t.Fatalf("child.Len = %d, want %d", child.Len(), prefixTokens)
	}
	childTable := child.PageTable()
	ownerTable := owner.PageTable()
	if len(childTable) != len(ownerTable) {
		t.Fatalf("page table length mismatch: child=%d, owner=%d", len(childTable), len(ownerTable))
	}
	for i := range childTable {
		if childTable[i] != ownerTable[i] {
			t.Fatalf("block ID mismatch at logical page %d: child=%d, owner=%d", i, childTable[i], ownerTable[i])
		}
		if child.Block(i) != owner.Block(i) {
			t.Fatalf("block pointer mismatch at logical page %d", i)
		}
		if child.Block(i).ID() != owner.Block(i).ID() {
			t.Fatalf("block ID method mismatch at logical page %d", i)
		}
		// Refcount should be 2 (owner + child)
		if ref := child.Block(i).Refcount(); ref != 2 {
			t.Fatalf("block %d refcount = %d, want 2", childTable[i], ref)
		}
	}

	// 2. Verify telemetry: fork_clone_bytes == 0
	telemetry := child.Telemetry()
	if telemetry.ForkCloneBytes != 0 {
		t.Fatalf("ForkCloneBytes = %d, want 0", telemetry.ForkCloneBytes)
	}
	if telemetry.SharedPages != 2 {
		t.Fatalf("SharedPages = %d, want 2", telemetry.SharedPages)
	}
	if telemetry.COWBytes != 0 {
		t.Fatalf("COWBytes = %d, want 0", telemetry.COWBytes)
	}
	wantSidecarBytes := int64((len(sidecar[0].Conv) + len(sidecar[0].Recurrent) + len(sidecar[1].Conv) + len(sidecar[1].Recurrent)) * 4)
	if telemetry.RecurrentSidecarBytes != wantSidecarBytes {
		t.Fatalf("RecurrentSidecarBytes = %d, want %d", telemetry.RecurrentSidecarBytes, wantSidecarBytes)
	}
	if err := child.VerifyZeroRematerialization(); err != nil {
		t.Fatalf("VerifyZeroRematerialization failed: %v", err)
	}

	// 3. Verify Gather reads correct values through shared blocks
	gatheredK := child.GatherK(0)
	if len(gatheredK) != prefixTokens*stride {
		t.Fatalf("GatherK len = %d, want %d", len(gatheredK), prefixTokens*stride)
	}
	for pos := 0; pos < prefixTokens; pos++ {
		for d := 0; d < stride; d++ {
			wantVal := float32(pos*1000 + 0*10 + d + 1)
			gotVal := gatheredK[pos*stride+d]
			if gotVal != wantVal {
				t.Fatalf("pos %d dim %d: got %f, want %f", pos, d, gotVal, wantVal)
			}
		}
	}
}

// Test tail-only COW: verify mutation past prefix writes only new blocks while shared blocks retain identical content.
func TestPagedPrefixCOW_TailOnlyCOW(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const blockTokens = 16
	const prefixTokens = 32 // 2 full blocks: block 0 and block 1
	stride := cfg.NumKVHeads * cfg.HeadDim

	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)
	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:      cfg,
		BlockTokens: blockTokens,
		Tokens:      prefixTokens,
		K:           kData,
		Kraw:        krawData,
		V:           vData,
		Sidecar:     makeTestSidecar(),
		IsMetal:     true,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	// Snapshot owner's shared block 0 and block 1 data
	ownerBlk0Copy := slices.Clone(owner.Block(0).Data())
	ownerBlk1Copy := slices.Clone(owner.Block(1).Data())

	child, err := owner.ForkSession("session-tail-cow")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}
	defer child.Release()

	// 1. Continuation: child appends 8 tokens past prefix boundary (tokens 32..39)
	for pos := 32; pos < 40; pos++ {
		k, kraw, v := makeTestTokenPlanes(cfg.NumLayers, stride, pos)
		if err := child.Append(k, kraw, v); err != nil {
			t.Fatalf("Append token %d failed: %v", pos, err)
		}
	}

	if child.Len() != 40 {
		t.Fatalf("child.Len = %d, want 40", child.Len())
	}
	childTable := child.PageTable()
	if len(childTable) != 3 {
		t.Fatalf("child table length = %d, want 3", len(childTable))
	}
	// Shared blocks 0 and 1 remain identical in identity
	if childTable[0] != owner.PageTable()[0] || childTable[1] != owner.PageTable()[1] {
		t.Fatalf("shared prefix blocks diverged during tail append: child=%v, owner=%v", childTable, owner.PageTable())
	}
	// Tail block 2 is brand new
	tailBlockID := childTable[2]
	if tailBlockID == childTable[0] || tailBlockID == childTable[1] {
		t.Fatalf("tail block %d collided with prefix blocks", tailBlockID)
	}
	if child.Block(2).Refcount() != 1 {
		t.Fatalf("tail block refcount = %d, want 1", child.Block(2).Refcount())
	}

	// Verify owner still has only 2 blocks and its data is untouched
	if len(owner.PageTable()) != 2 {
		t.Fatalf("owner page table modified: len = %d", len(owner.PageTable()))
	}
	if !slices.Equal(owner.Block(0).Data(), ownerBlk0Copy) {
		t.Fatalf("owner block 0 was modified by child tail append")
	}
	if !slices.Equal(owner.Block(1).Data(), ownerBlk1Copy) {
		t.Fatalf("owner block 1 was modified by child tail append")
	}

	// Tail append on block boundary requires 0 COW bytes (only new allocation)
	if child.Telemetry().COWBytes != 0 {
		t.Fatalf("tail append COWBytes = %d, want 0", child.Telemetry().COWBytes)
	}

	// 2. Mutation: child mutates token 5 (inside shared block 0)
	mutatedK, mutatedKraw, mutatedV := makeTestTokenPlanes(cfg.NumLayers, stride, 9999)
	if err := child.MutateToken(5, mutatedK, mutatedKraw, mutatedV); err != nil {
		t.Fatalf("MutateToken failed: %v", err)
	}

	// Now COW must have triggered on logical block 0
	newChildTable := child.PageTable()
	if newChildTable[0] == owner.PageTable()[0] {
		t.Fatalf("child block 0 still shares ID with owner after mutation")
	}
	if child.Telemetry().COWBytes <= 0 {
		t.Fatalf("child COWBytes = %d, want > 0 after mutation", child.Telemetry().COWBytes)
	}

	// Block 1 is STILL shared
	if newChildTable[1] != owner.PageTable()[1] {
		t.Fatalf("block 1 should remain shared after mutating block 0")
	}

	// Owner's block 0 remains completely UNTOUCHED
	if !slices.Equal(owner.Block(0).Data(), ownerBlk0Copy) {
		t.Fatalf("owner block 0 data changed after child mutation")
	}

	// Verify child has mutated token 5
	kPlane := child.Block(0).Get(0, planeK, 5)
	if kPlane[0] != mutatedK[0][0] {
		t.Fatalf("child token 5 not mutated: got %f, want %f", kPlane[0], mutatedK[0][0])
	}
	// Verify owner still has original token 5
	ownerKPlane := owner.Block(0).Get(0, planeK, 5)
	if ownerKPlane[0] == mutatedK[0][0] {
		t.Fatalf("owner token 5 was corrupted by child mutation")
	}
}

// Test refcount cleanup: verify all blocks are released cleanly after child and parent sessions close.
func TestPagedPrefixCOW_RefcountCleanup(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const blockTokens = 16
	const prefixTokens = 32
	stride := cfg.NumKVHeads * cfg.HeadDim

	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)
	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:      cfg,
		BlockTokens: blockTokens,
		Tokens:      prefixTokens,
		K:           kData,
		Kraw:        krawData,
		V:           vData,
		Sidecar:     makeTestSidecar(),
		IsMetal:     true,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}

	pool := owner.Pool()
	if pool.ActivePages() != 2 {
		t.Fatalf("initial ActivePages = %d, want 2", pool.ActivePages())
	}
	if pool.MetalPagesAllocated() != 2 {
		t.Fatalf("MetalPagesAllocated = %d, want 2", pool.MetalPagesAllocated())
	}
	if pool.MetalPagesFreed() != 0 {
		t.Fatalf("MetalPagesFreed = %d, want 0", pool.MetalPagesFreed())
	}

	// Fork child 1
	child1, err := owner.ForkSession("child-1")
	if err != nil {
		t.Fatalf("ForkSession child1 failed: %v", err)
	}
	if pool.ActivePages() != 2 {
		t.Fatalf("ActivePages after child1 fork = %d, want 2", pool.ActivePages())
	}

	// Fork child 2
	child2, err := owner.ForkSession("child-2")
	if err != nil {
		t.Fatalf("ForkSession child2 failed: %v", err)
	}
	if pool.ActivePages() != 2 {
		t.Fatalf("ActivePages after child2 fork = %d, want 2", pool.ActivePages())
	}

	// Shared blocks should now have refcount 3
	for i := 0; i < 2; i++ {
		if ref := owner.Block(i).Refcount(); ref != 3 {
			t.Fatalf("block %d refcount = %d, want 3", i, ref)
		}
	}

	// Child 1 appends into a new tail block (block 2)
	k, kraw, v := makeTestTokenPlanes(cfg.NumLayers, stride, 32)
	if err := child1.Append(k, kraw, v); err != nil {
		t.Fatalf("child1 Append failed: %v", err)
	}
	if pool.ActivePages() != 3 {
		t.Fatalf("ActivePages after tail alloc = %d, want 3", pool.ActivePages())
	}

	// 1. Release Child 1: private tail block 2 should be freed, shared blocks drop to refcount 2
	if err := child1.Release(); err != nil {
		t.Fatalf("child1 Release failed: %v", err)
	}
	if !child1.IsReleased() {
		t.Fatalf("child1 should be marked released")
	}
	if pool.ActivePages() != 2 {
		t.Fatalf("ActivePages after child1 release = %d, want 2", pool.ActivePages())
	}
	if pool.MetalPagesFreed() != 1 {
		t.Fatalf("MetalPagesFreed after child1 release = %d, want 1", pool.MetalPagesFreed())
	}
	for i := 0; i < 2; i++ {
		if ref := owner.Block(i).Refcount(); ref != 2 {
			t.Fatalf("block %d refcount = %d, want 2", i, ref)
		}
	}

	// 2. Release Child 2: shared blocks drop to refcount 1, nothing freed yet
	if err := child2.Release(); err != nil {
		t.Fatalf("child2 Release failed: %v", err)
	}
	if pool.ActivePages() != 2 {
		t.Fatalf("ActivePages after child2 release = %d, want 2", pool.ActivePages())
	}
	if pool.MetalPagesFreed() != 1 {
		t.Fatalf("MetalPagesFreed after child2 release = %d, want 1", pool.MetalPagesFreed())
	}
	for i := 0; i < 2; i++ {
		if ref := owner.Block(i).Refcount(); ref != 1 {
			t.Fatalf("block %d refcount = %d, want 1", i, ref)
		}
	}

	// 3. Release Owner: shared blocks drop to refcount 0 and are freed
	if err := owner.Release(); err != nil {
		t.Fatalf("owner Release failed: %v", err)
	}
	if !owner.IsReleased() {
		t.Fatalf("owner should be marked released")
	}
	if pool.ActivePages() != 0 {
		t.Fatalf("ActivePages after owner release = %d, want 0", pool.ActivePages())
	}
	if pool.MetalPagesFreed() != 3 {
		t.Fatalf("MetalPagesFreed after owner release = %d, want 3", pool.MetalPagesFreed())
	}
}

// Test sidecar isolation: verify recurrent state in child does not mutate parent.
func TestPagedPrefixCOW_SidecarIsolation(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const blockTokens = 16
	const prefixTokens = 16
	stride := cfg.NumKVHeads * cfg.HeadDim

	kData, krawData, vData := makeTestPrefixData(prefixTokens, cfg.NumLayers, stride)
	sidecar := makeTestSidecar()

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:      cfg,
		BlockTokens: blockTokens,
		Tokens:      prefixTokens,
		K:           kData,
		Kraw:        krawData,
		V:           vData,
		Sidecar:     sidecar,
		IsMetal:     false,
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	child, err := owner.ForkSession("sidecar-test")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}
	defer child.Release()

	// Mutate child's sidecar
	newConv := []float32{99.1, 99.2, 99.3}
	newRec := []float32{999.1, 999.2, 999.3, 999.4}
	if err := child.UpdateLayerSidecar(0, newConv, newRec); err != nil {
		t.Fatalf("UpdateLayerSidecar failed: %v", err)
	}

	// Verify owner sidecar is completely UNCHANGED
	ownerState, ok := owner.LayerSidecar(0)
	if !ok {
		t.Fatalf("owner missing layer 0 sidecar")
	}
	if slices.Equal(ownerState.Conv, newConv) {
		t.Fatalf("owner Conv mutated to match child: %v", ownerState.Conv)
	}
	if slices.Equal(ownerState.Recurrent, newRec) {
		t.Fatalf("owner Recurrent mutated to match child: %v", ownerState.Recurrent)
	}
	if !slices.Equal(ownerState.Conv, sidecar[0].Conv) {
		t.Fatalf("owner Conv changed from initial: got %v, want %v", ownerState.Conv, sidecar[0].Conv)
	}
	if !slices.Equal(ownerState.Recurrent, sidecar[0].Recurrent) {
		t.Fatalf("owner Recurrent changed from initial: got %v, want %v", ownerState.Recurrent, sidecar[0].Recurrent)
	}

	// Verify child has mutated values
	childState, ok := child.LayerSidecar(0)
	if !ok {
		t.Fatalf("child missing layer 0 sidecar")
	}
	if !slices.Equal(childState.Conv, newConv) {
		t.Fatalf("child Conv = %v, want %v", childState.Conv, newConv)
	}
	if !slices.Equal(childState.Recurrent, newRec) {
		t.Fatalf("child Recurrent = %v, want %v", childState.Recurrent, newRec)
	}
}

// Test non-exact hybrid fallback: verify typed error is returned for non-exact boundaries.
func TestPagedPrefixCOW_NonExactFallback(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const blockTokens = 16
	const prefixTokens = 32
	tokenIDs := make([]int, prefixTokens)
	for i := range tokenIDs {
		tokenIDs[i] = 1000 + i
	}

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:      cfg,
		BlockTokens: blockTokens,
		Tokens:      prefixTokens,
		TokenIDs:    tokenIDs,
		Sidecar:     makeTestSidecar(),
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	// 1. Fork at non-exact token count (e.g. 20 instead of 32)
	_, err = owner.ForkSessionAt("non-exact-tokens", 20)
	if err == nil {
		t.Fatalf("ForkSessionAt(20) succeeded, want error")
	}
	if !errors.Is(err, ErrNonExactHybridPrefix) {
		t.Fatalf("err is not ErrNonExactHybridPrefix: %v", err)
	}
	var fallbackErr *NonExactHybridFallbackError
	if !errors.As(err, &fallbackErr) {
		t.Fatalf("err is not *NonExactHybridFallbackError: %T", err)
	}
	if fallbackErr.ExpectedTokens != 32 || fallbackErr.RequestedTokens != 20 {
		t.Fatalf("fallbackErr = %+v, want 32 vs 20", fallbackErr)
	}

	// 2. Fork with divergent token IDs
	divergentTokens := append([]int(nil), tokenIDs...)
	divergentTokens[15] = 9999
	_, err = owner.ForkSessionExact("divergent", divergentTokens)
	if err == nil {
		t.Fatalf("ForkSessionExact with divergent tokens succeeded, want error")
	}
	if !errors.Is(err, ErrNonExactHybridPrefix) {
		t.Fatalf("err is not ErrNonExactHybridPrefix: %v", err)
	}

	// 3. Fork with exact matching token IDs succeeds
	exactChild, err := owner.ForkSessionExact("exact", tokenIDs)
	if err != nil {
		t.Fatalf("ForkSessionExact with exact token IDs failed: %v", err)
	}
	defer exactChild.Release()
	if exactChild.Len() != prefixTokens {
		t.Fatalf("exactChild.Len = %d, want %d", exactChild.Len(), prefixTokens)
	}
}

// Test zero rematerialization verification and refusal of full cache materialization.
func TestPagedPrefixCOW_ZeroRematerialization(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	const blockTokens = 16
	const prefixTokens = 16

	owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
		Config:      cfg,
		BlockTokens: blockTokens,
		Tokens:      prefixTokens,
		Sidecar:     makeTestSidecar(),
	})
	if err != nil {
		t.Fatalf("NewPagedPrefixOwner failed: %v", err)
	}
	defer owner.Release()

	child, err := owner.ForkSession("remat-test")
	if err != nil {
		t.Fatalf("ForkSession failed: %v", err)
	}
	defer child.Release()

	// RematerializeContiguous must return ErrFullMaterializationForbidden
	_, err = child.RematerializeContiguous()
	if !errors.Is(err, ErrFullMaterializationForbidden) {
		t.Fatalf("RematerializeContiguous error = %v, want ErrFullMaterializationForbidden", err)
	}

	// Receipt bridge
	receipt := child.ToPrefixShareReceipt(10)
	if receipt.ForkCloneBytes != 0 {
		t.Fatalf("receipt ForkCloneBytes = %d, want 0", receipt.ForkCloneBytes)
	}
	if receipt.SharedBlocks != 1 {
		t.Fatalf("receipt SharedBlocks = %d, want 1", receipt.SharedBlocks)
	}
	if receipt.Validation != "shared-zero-clone" {
		t.Fatalf("receipt Validation = %q, want 'shared-zero-clone'", receipt.Validation)
	}
}
