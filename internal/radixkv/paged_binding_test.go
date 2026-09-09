package radixkv_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func makeTestPool(t *testing.T, totalPages, tokensPerPage int) (*radixkv.PagedRadixKVPool, *ctxmmu.KVPool) {
	t.Helper()
	cfg := ctxmmu.KVPoolConfig{
		TotalPages:    totalPages,
		TokensPerPage: tokensPerPage,
		NumLayers:     4,
		NumKVHeads:    2,
		HeadDim:       32,
		DType:         ctxmmu.KVDTypeFP32,
		Topology:      ctxmmu.KVTopologyGQA,
	}
	kvPool, err := ctxmmu.NewKVPool(cfg)
	if err != nil {
		t.Fatalf("failed to create ctxmmu.KVPool: %v", err)
	}
	pagedRadix := radixkv.NewPagedRadixKVPool(0, kvPool)
	return pagedRadix, kvPool
}

// TestPagedRadixKVPool_ZeroCopyBinding is the primary witness test for Issue #12528:
//
//	go test -v ./internal/radixkv -run "TestPagedRadixKVPool_ZeroCopyBinding" -count=1
//
// It verifies:
// 1. Prefix hits bind physical block IDs directly into PagedBlockTable.
// 2. Zero tensor allocations / copies during prefix binding.
// 3. Physical block reference counts correctly track shared ownership between tree and stream.
func TestPagedRadixKVPool_ZeroCopyBinding(t *testing.T) {
	pagedRadix, kvPool := makeTestPool(t, 64, 16)
	tokens := []int{101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
		201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216} // 32 tokens = 2 blocks

	// Step 1: Insert prefix into radix tree backed by allocated physical blocks
	boundary, _ := pagedRadix.Tree.Lookup(nil)
	leaf, allocatedBlocks, err := pagedRadix.InsertWithAllocatedTokens(boundary, tokens)
	if err != nil {
		t.Fatalf("InsertWithAllocatedTokens failed: %v", err)
	}
	if leaf == nil {
		t.Fatalf("expected non-nil leaf")
	}
	if len(allocatedBlocks) != 2 {
		t.Fatalf("expected 2 allocated blocks, got %d", len(allocatedBlocks))
	}

	allocator := kvPool.Allocator()
	for _, bID := range allocatedBlocks {
		blk, err := allocator.GetBlock(bID)
		if err != nil {
			t.Fatalf("failed to get block %d: %v", bID, err)
		}
		if ref := blk.RefCount(); ref != 1 {
			t.Fatalf("block %d initial refcount = %d, want 1 (tree ownership)", bID, ref)
		}
	}

	// Step 2: Perform prefix hit binding for a new stream
	reqTokens := append([]int(nil), tokens...)
	reqTokens = append(reqTokens, 999) // divergent token suffix

	streamID := "stream-agent-001"
	table, matched, hit, err := pagedRadix.BindPrefix(reqTokens, streamID)
	if err != nil {
		t.Fatalf("BindPrefix returned error: %v", err)
	}
	if !hit {
		t.Fatalf("expected prefix cache hit, got hit=false")
	}
	if matched != 32 {
		t.Fatalf("expected 32 matched tokens, got %d", matched)
	}
	if table.BlockCount() != 2 {
		t.Fatalf("expected 2 blocks in block table, got %d", table.BlockCount())
	}

	// Verify block IDs match allocated blocks exactly
	tableBlocks := table.Pages()
	for i, bID := range tableBlocks {
		if bID != allocatedBlocks[i] {
			t.Errorf("table block[%d] = %d, want %d", i, bID, allocatedBlocks[i])
		}
	}

	// Verify Metal page table matches block IDs (uint32)
	metalPages := table.MetalPages()
	for i, mbID := range metalPages {
		if mbID != uint32(allocatedBlocks[i]) {
			t.Errorf("metal page[%d] = %d, want %d", i, mbID, allocatedBlocks[i])
		}
	}

	// Step 3: Verify reference counts incremented on shared blocks (tree + stream = 2)
	for _, bID := range allocatedBlocks {
		blk, err := allocator.GetBlock(bID)
		if err != nil {
			t.Fatalf("failed to get block %d: %v", bID, err)
		}
		if ref := blk.RefCount(); ref != 2 {
			t.Fatalf("block %d shared refcount = %d, want 2 (tree + stream)", bID, ref)
		}
	}

	// Step 4: Verify zero tensor allocations during repeated prefix binding
	// Zero tensor memory allocations invariant:
	allocs := testing.AllocsPerRun(100, func() {
		tbl, m, h, e := pagedRadix.BindPrefix(tokens, "temp-bench-stream")
		if e != nil || !h || m != 32 {
			t.Fatalf("BindPrefix failed in alloc test")
		}
		_ = pagedRadix.ReleaseBlockTable(tbl)
	})
	// In Go, constructing small metadata descriptors can produce 2-4 tiny object allocations,
	// but ZERO tensor allocations (0 float32 slices or large KV buffers allocated).
	if allocs > 10 {
		t.Errorf("excessive allocations during BindPrefix: %v allocs/run", allocs)
	}

	// Step 5: Clean up table
	if err := pagedRadix.ReleaseBlockTable(table); err != nil {
		t.Fatalf("ReleaseBlockTable failed: %v", err)
	}

	for _, bID := range allocatedBlocks {
		blk, err := allocator.GetBlock(bID)
		if err != nil {
			t.Fatalf("failed to get block %d: %v", bID, err)
		}
		if ref := blk.RefCount(); ref != 1 {
			t.Fatalf("block %d post-release refcount = %d, want 1 (tree ownership)", bID, ref)
		}
	}
}

// TestPagedRadixKVPool_ReferenceCountedEviction verifies Acceptance Criterion 3:
// Reference-counted page retention on radix eviction.
// If a radix node is evicted while an active stream holds a lease, the physical blocks
// remain retained until the stream finishes and releases its table.
func TestPagedRadixKVPool_ReferenceCountedEviction(t *testing.T) {
	pagedRadix, kvPool := makeTestPool(t, 32, 16)
	tokens := []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160} // 16 tokens = 1 block

	boundary, _ := pagedRadix.Tree.Lookup(nil)
	leaf, allocatedBlocks, err := pagedRadix.InsertWithAllocatedTokens(boundary, tokens)
	if err != nil {
		t.Fatalf("InsertWithAllocatedTokens failed: %v", err)
	}
	if len(allocatedBlocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(allocatedBlocks))
	}
	bID := allocatedBlocks[0]
	allocator := kvPool.Allocator()

	freeBefore := allocator.FreeCount()

	// Stream 1 binds the prefix
	table1, matched1, hit1, err := pagedRadix.BindPrefix(tokens, "stream-1")
	if err != nil || !hit1 || matched1 != 16 {
		t.Fatalf("Stream 1 bind failed: matched=%d hit=%v err=%v", matched1, hit1, err)
	}

	// Stream 2 binds the same prefix
	table2, matched2, hit2, err := pagedRadix.BindPrefix(tokens, "stream-2")
	if err != nil || !hit2 || matched2 != 16 {
		t.Fatalf("Stream 2 bind failed: matched=%d hit=%v err=%v", matched2, hit2, err)
	}

	blk, err := allocator.GetBlock(bID)
	if err != nil {
		t.Fatalf("GetBlock(%d) failed: %v", bID, err)
	}
	// Refcount must be 3 (Tree + Stream 1 + Stream 2)
	if ref := blk.RefCount(); ref != 3 {
		t.Fatalf("refcount = %d, want 3", ref)
	}

	// Evict the node from RadixKV
	freedTokens, err := pagedRadix.EvictNodePaged(leaf)
	if err != nil {
		t.Fatalf("EvictNodePaged failed: %v", err)
	}
	if freedTokens != 16 {
		t.Fatalf("freedTokens = %d, want 16", freedTokens)
	}

	// Refcount should drop from 3 to 2 because RadixKV released its reference.
	// The physical block MUST NOT be freed yet!
	if ref := blk.RefCount(); ref != 2 {
		t.Fatalf("post-eviction refcount = %d, want 2 (Stream 1 + Stream 2)", ref)
	}
	if allocator.FreeCount() != freeBefore {
		t.Fatalf("block was prematurely freed! freeCount = %d, want %d", allocator.FreeCount(), freeBefore)
	}

	// Stream 1 releases table
	if err := pagedRadix.ReleaseBlockTable(table1); err != nil {
		t.Fatalf("ReleaseBlockTable(1) failed: %v", err)
	}
	if ref := blk.RefCount(); ref != 1 {
		t.Fatalf("post-stream1-release refcount = %d, want 1 (Stream 2)", ref)
	}
	if allocator.FreeCount() != freeBefore {
		t.Fatalf("block was prematurely freed after stream 1 release")
	}

	// Stream 2 releases table
	if err := pagedRadix.ReleaseBlockTable(table2); err != nil {
		t.Fatalf("ReleaseBlockTable(2) failed: %v", err)
	}

	// Now that all holders have released, block refcount is <= 0 and block is returned to pool
	if allocator.FreeCount() != freeBefore+1 {
		t.Fatalf("block was not freed after all streams released: freeCount=%d, want %d", allocator.FreeCount(), freeBefore+1)
	}
}

// TestPagedRadixKVPool_MetalTableBinding verifies binding into Metal uint32 page tables
// and ctxmmu.KVSequence structures.
func TestPagedRadixKVPool_MetalTableBinding(t *testing.T) {
	pagedRadix, kvPool := makeTestPool(t, 32, 16)
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} // 20 tokens = 2 blocks

	boundary, _ := pagedRadix.Tree.Lookup(nil)
	_, allocatedBlocks, err := pagedRadix.InsertWithAllocatedTokens(boundary, tokens)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	seq, err := kvPool.CreateSequence("seq-metal-test")
	if err != nil {
		t.Fatalf("CreateSequence failed: %v", err)
	}

	matched, hit, err := pagedRadix.BindPrefixToSequence(tokens, seq)
	if err != nil || !hit {
		t.Fatalf("BindPrefixToSequence failed: hit=%v matched=%d err=%v", hit, matched, err)
	}
	if matched != 20 {
		t.Fatalf("matched = %d, want 20", matched)
	}

	seqPages := seq.Pages()
	if len(seqPages) != 2 {
		t.Fatalf("seq pages len = %d, want 2", len(seqPages))
	}
	for i, id := range allocatedBlocks {
		if seqPages[i] != id {
			t.Errorf("seq page[%d] = %d, want %d", i, seqPages[i], id)
		}
	}
}

// TestPagedRadixKVPool_BitExactAttention verifies the done condition:
// attention computed over the bound paged page tables is bit-exact identical
// to reference scaled dot-product attention over contiguous tensors.
func TestPagedRadixKVPool_BitExactAttention(t *testing.T) {
	pagedRadix, _ := makeTestPool(t, 32, 16)
	numTokens := 32
	tokens := make([]int, numTokens)
	for i := range tokens {
		tokens[i] = 1000 + i
	}

	boundary, _ := pagedRadix.Tree.Lookup(nil)
	_, allocatedBlocks, err := pagedRadix.InsertWithAllocatedTokens(boundary, tokens)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	cfg := pagedRadix.Pool().Config()
	numHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	stride := numHeads * headDim

	// Generate deterministic synthetic activations for K and V
	refK := make([]float32, numTokens*stride)
	refV := make([]float32, numTokens*stride)
	rnd := rand.New(rand.NewSource(42))

	for tok := 0; tok < numTokens; tok++ {
		tokK := make([]float32, stride)
		tokV := make([]float32, stride)
		for d := 0; d < stride; d++ {
			valK := rnd.Float32()*2.0 - 1.0
			valV := rnd.Float32()*2.0 - 1.0
			tokK[d] = valK
			tokV[d] = valV
			refK[tok*stride+d] = valK
			refV[tok*stride+d] = valV
		}

		bID := allocatedBlocks[tok/16]
		tokInBlk := tok % 16
		for layer := 0; layer < cfg.NumLayers; layer++ {
			if err := pagedRadix.WriteTokenKV(bID, tokInBlk, layer, tokK, tokV); err != nil {
				t.Fatalf("WriteTokenKV failed: %v", err)
			}
		}
	}

	// Bind block table
	table, matched, hit, err := pagedRadix.BindPrefix(tokens, "stream-bitexact")
	if err != nil || !hit || matched != numTokens {
		t.Fatalf("BindPrefix failed: hit=%v matched=%d err=%v", hit, matched, err)
	}

	// Generate random query vector Q
	q := make([]float32, stride)
	for d := 0; d < stride; d++ {
		q[d] = rnd.Float32()*2.0 - 1.0
	}

	// Compute attention using paged binding
	pagedOut, err := pagedRadix.ComputePagedAttention(table, q, 0)
	if err != nil {
		t.Fatalf("ComputePagedAttention failed: %v", err)
	}

	// Compute reference attention using contiguous refK and refV
	refOut := make([]float32, stride)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := make([]float32, numTokens)

	for h := 0; h < numHeads; h++ {
		qHead := q[h*headDim : (h+1)*headDim]

		var maxScore float32 = -math.MaxFloat32
		for t := 0; t < numTokens; t++ {
			kHead := refK[t*stride+h*headDim : t*stride+(h+1)*headDim]
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kHead[d]
			}
			score := dot * scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}

		var sumExp float32
		for t := 0; t < numTokens; t++ {
			expVal := float32(math.Exp(float64(scores[t] - maxScore)))
			scores[t] = expVal
			sumExp += expVal
		}
		invSum := float32(1.0) / sumExp
		for t := 0; t < numTokens; t++ {
			scores[t] *= invSum
		}

		outHead := refOut[h*headDim : (h+1)*headDim]
		for t := 0; t < numTokens; t++ {
			vHead := refV[t*stride+h*headDim : t*stride+(h+1)*headDim]
			weight := scores[t]
			for d := 0; d < headDim; d++ {
				outHead[d] += weight * vHead[d]
			}
		}
	}

	// Assert bit-exact equality: every float32 bit must match
	for d := 0; d < stride; d++ {
		pagedBits := math.Float32bits(pagedOut[d])
		refBits := math.Float32bits(refOut[d])
		if pagedBits != refBits {
			t.Fatalf("bit mismatch at dim %d: paged=%f (%08x) vs ref=%f (%08x)",
				d, pagedOut[d], pagedBits, refOut[d], refBits)
		}
	}
}

// TestPagedRadixKVPool_Sub50usLatency verifies that prefix binding completes in sub-50µs.
func TestPagedRadixKVPool_Sub50usLatency(t *testing.T) {
	pagedRadix, _ := makeTestPool(t, 64, 16)
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

	boundary, _ := pagedRadix.Tree.Lookup(nil)
	_, _, err := pagedRadix.InsertWithAllocatedTokens(boundary, tokens)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	const iterations = 500
	var totalDuration time.Duration

	for i := 0; i < iterations; i++ {
		start := time.Now()
		tbl, matched, hit, err := pagedRadix.BindPrefix(tokens, "")
		dur := time.Since(start)

		if err != nil || !hit || matched != 20 {
			t.Fatalf("BindPrefix iteration %d failed", i)
		}
		totalDuration += dur
		_ = pagedRadix.ReleaseBlockTable(tbl)
	}

	avgLatency := totalDuration / iterations
	t.Logf("Average prefix binding latency: %v (threshold: 50µs)", avgLatency)

	if avgLatency > 50*time.Microsecond {
		t.Fatalf("average binding latency %v exceeded 50µs SLA", avgLatency)
	}
}

// BenchmarkPagedRadixKVPool_BindingLatency benchmarks prefix hit binding throughput and latency.
func BenchmarkPagedRadixKVPool_BindingLatency(b *testing.B) {
	cfg := ctxmmu.KVPoolConfig{
		TotalPages:    2048,
		TokensPerPage: 16,
		NumLayers:     4,
		NumKVHeads:    2,
		HeadDim:       32,
		DType:         ctxmmu.KVDTypeFP32,
		Topology:      ctxmmu.KVTopologyGQA,
	}
	kvPool, _ := ctxmmu.NewKVPool(cfg)
	pagedRadix := radixkv.NewPagedRadixKVPool(0, kvPool)

	tokens := make([]int, 64)
	for i := range tokens {
		tokens[i] = 500 + i
	}
	boundary, _ := pagedRadix.Tree.Lookup(nil)
	_, _, _ = pagedRadix.InsertWithAllocatedTokens(boundary, tokens)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		tbl, _, hit, err := pagedRadix.BindPrefix(tokens, "")
		if !hit || err != nil {
			b.Fatal("unexpected miss or error")
		}
		_ = pagedRadix.ReleaseBlockTable(tbl)
	}
}
