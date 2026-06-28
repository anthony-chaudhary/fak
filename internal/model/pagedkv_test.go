package model

import "testing"

// pagedkv_test.go — the gate for the PagedAttention block allocator (pagedkv.go, #277).
// It pins the four properties the issue's acceptance is measured against, all on real
// float32 KV bytes with no GPU: variable-length sequences cost ceil(len/blockTokens)
// blocks, a paged gather is bit-identical to a contiguous layout, copy-on-write Fork
// shares a prefix with zero byte copies and diverges one block at a time, and aggregate
// internal-fragmentation overhead stays ≤ 20% on an agent-shaped workload.

func pagedTestCfg() Config { return Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8} }

// pagedTokenKV builds a token's per-layer K/V with a value that is a unique, reproducible
// function of (pos, layer, dim) so a gather can be checked exactly. V is the negation of K
// so K and V layouts can never be silently swapped.
func pagedTokenKV(p *PagedKVPool, pos int) (k, v [][]float32) {
	k = make([][]float32, p.nLayers)
	v = make([][]float32, p.nLayers)
	for l := 0; l < p.nLayers; l++ {
		k[l] = make([]float32, p.stride)
		v[l] = make([]float32, p.stride)
		for d := 0; d < p.stride; d++ {
			val := float32(pos*1000 + l*10 + d)
			k[l][d] = val
			v[l][d] = -val
		}
	}
	return k, v
}

func pagedFillSeq(p *PagedKVPool, n int) *PagedKV {
	s := p.NewSequence()
	for pos := 0; pos < n; pos++ {
		k, v := pagedTokenKV(p, pos)
		s.Append(k, v)
	}
	return s
}

// TestPagedKVVariableLengthAllocation proves page-based allocation is demand-driven: a
// sequence of any length costs exactly ceil(len/blockTokens) blocks — never a worst-case
// reservation — which is what makes variable-length sequences efficient.
func TestPagedKVVariableLengthAllocation(t *testing.T) {
	const blockTokens = 16
	p := NewPagedKVPool(pagedTestCfg(), blockTokens)
	for _, n := range []int{0, 1, 15, 16, 17, 100, 256} {
		s := pagedFillSeq(p, n)
		want := (n + blockTokens - 1) / blockTokens
		if got := s.Blocks(); got != want {
			t.Fatalf("len %d: Blocks()=%d, want ceil(%d/%d)=%d", n, got, n, blockTokens, want)
		}
		if s.Len() != n {
			t.Fatalf("len %d: Len()=%d", n, s.Len())
		}
		s.Free()
	}
}

// TestPagedKVGatherBitExact proves the page table reconstructs the exact contiguous K/V a
// non-paged cache would hold: gathering each layer returns precisely the values appended,
// in logical order, across block boundaries. Without this, paging would change numerics.
func TestPagedKVGatherBitExact(t *testing.T) {
	cfg := pagedTestCfg()
	p := NewPagedKVPool(cfg, 16)
	const n = 40 // spans 3 blocks (16+16+8)
	s := pagedFillSeq(p, n)

	stride := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		gotK, gotV := s.GatherK(l), s.GatherV(l)
		if len(gotK) != n*stride || len(gotV) != n*stride {
			t.Fatalf("layer %d: gather lens K=%d V=%d, want %d", l, len(gotK), len(gotV), n*stride)
		}
		for pos := 0; pos < n; pos++ {
			for d := 0; d < stride; d++ {
				want := float32(pos*1000 + l*10 + d)
				if g := gotK[pos*stride+d]; g != want {
					t.Fatalf("K[layer=%d pos=%d d=%d]=%v, want %v", l, pos, d, g, want)
				}
				if g := gotV[pos*stride+d]; g != -want {
					t.Fatalf("V[layer=%d pos=%d d=%d]=%v, want %v", l, pos, d, g, -want)
				}
			}
		}
	}
}

// TestPagedKVCopyOnWriteShareAndDiverge is the cache-sharing witness. A Fork shares every
// physical block with zero copies (the pool's distinct-block count does not rise); a write
// into the fork copies exactly ONE block, leaving the rest shared and the parent byte-for-
// byte unchanged. The shared physical-block count stays below the sum of the two page
// tables — the memory win RadixAttention's full-prefix clone does not get.
func TestPagedKVCopyOnWriteShareAndDiverge(t *testing.T) {
	cfg := pagedTestCfg()
	p := NewPagedKVPool(cfg, 16)
	const n = 40 // 3 blocks
	a := pagedFillSeq(p, n)

	beforeForkBlocks := p.PhysicalBlocks()
	if beforeForkBlocks != 3 {
		t.Fatalf("pre-fork PhysicalBlocks=%d, want 3", beforeForkBlocks)
	}
	b := a.Fork()
	if got := p.PhysicalBlocks(); got != 3 {
		t.Fatalf("Fork copied bytes: PhysicalBlocks=%d, want 3 (share, no copy)", got)
	}

	// Snapshot the parent's K for every layer so we can prove the COW write never touches it.
	parentK := make([][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		parentK[l] = a.GatherK(l)
	}

	// Write a new token into the fork: it lands in logical block 2 (the shared tail), which
	// must be copied before the write — exactly one new physical block.
	k, v := pagedTokenKV(p, n) // pos 40, a value the parent never held
	b.Append(k, v)
	if got := p.PhysicalBlocks(); got != 4 {
		t.Fatalf("post-write PhysicalBlocks=%d, want 4 (one block copied)", got)
	}
	if a.Len() != n || b.Len() != n+1 {
		t.Fatalf("lengths after diverge: a=%d b=%d, want %d and %d", a.Len(), b.Len(), n, n+1)
	}

	// The shared physical-block count (4) is strictly below the page-table sum (3+3=6): the
	// copy-on-write share is cheaper than cloning the whole prefix.
	if sum := a.Blocks() + b.Blocks(); p.PhysicalBlocks() >= sum {
		t.Fatalf("no sharing win: PhysicalBlocks=%d not < page-table sum %d", p.PhysicalBlocks(), sum)
	}

	// Parent unchanged by the fork's write.
	for l := 0; l < cfg.NumLayers; l++ {
		got := a.GatherK(l)
		for i := range parentK[l] {
			if got[i] != parentK[l][i] {
				t.Fatalf("layer %d: COW write mutated the parent at %d (%v != %v)", l, i, got[i], parentK[l][i])
			}
		}
	}

	// The fork shares the parent's first 40 tokens and owns the 41st.
	for l := 0; l < cfg.NumLayers; l++ {
		gb := b.GatherK(l)
		for i := range parentK[l] {
			if gb[i] != parentK[l][i] {
				t.Fatalf("layer %d: fork prefix diverged at %d", l, i)
			}
		}
		stride := cfg.NumKVHeads * cfg.HeadDim
		for d := 0; d < stride; d++ {
			if want := float32(n*1000 + l*10 + d); gb[n*stride+d] != want {
				t.Fatalf("layer %d: fork token %d d=%d=%v, want %v", l, n, d, gb[n*stride+d], want)
			}
		}
	}
}

// TestPagedKVAggregateOverheadUnder20Pct measures the only memory cost paging adds —
// internal fragmentation, the unused slots in each sequence's partial tail block — across
// an agent-shaped batch of variable-length sequences, and holds it to the issue's ≤ 20%
// bar. (A single sub-page sequence is necessarily higher overhead; the acceptance is an
// aggregate-workload property, which is how vLLM states it.)
func TestPagedKVAggregateOverheadUnder20Pct(t *testing.T) {
	const blockTokens = 16
	p := NewPagedKVPool(pagedTestCfg(), blockTokens)
	lengths := []int{320, 96, 512, 1024, 200, 448, 768, 130}
	var liveTokens, allocatedSlots int
	for _, n := range lengths {
		s := pagedFillSeq(p, n)
		liveTokens += n
		allocatedSlots += s.Blocks() * blockTokens
	}
	overhead := float64(allocatedSlots-liveTokens) / float64(liveTokens)
	if overhead > 0.20 {
		t.Fatalf("aggregate KV memory overhead %.4f exceeds the 0.20 bar", overhead)
	}
	// The bound is B/avgLen; on this workload it is a few percent. Assert it is genuinely
	// small so a regression that quietly doubled allocation would still trip the gate.
	if overhead > 0.05 {
		t.Fatalf("aggregate overhead %.4f unexpectedly high for avg len %d", overhead, liveTokens/len(lengths))
	}
}

// TestPagedKVFreeReusesBlocks proves the pool recycles freed blocks instead of growing
// without bound: after freeing a sequence, an equal-sized sequence reuses the same backing
// storage (the physical block count returns to its prior level).
func TestPagedKVFreeReusesBlocks(t *testing.T) {
	p := NewPagedKVPool(pagedTestCfg(), 16)
	a := pagedFillSeq(p, 100) // 7 blocks
	peak := len(p.blocks)
	if p.PhysicalBlocks() != 7 {
		t.Fatalf("PhysicalBlocks=%d, want 7", p.PhysicalBlocks())
	}
	a.Free()
	if p.PhysicalBlocks() != 0 {
		t.Fatalf("after Free PhysicalBlocks=%d, want 0", p.PhysicalBlocks())
	}
	b := pagedFillSeq(p, 100)
	if p.PhysicalBlocks() != 7 {
		t.Fatalf("reuse: PhysicalBlocks=%d, want 7", p.PhysicalBlocks())
	}
	if grew := len(p.blocks) - peak; grew != 0 {
		t.Fatalf("pool allocated %d new blocks instead of reusing freed ones", grew)
	}
	_ = b
}

// BenchmarkPagedKVForkShare documents the cache-sharing cost: forking a long shared prefix
// is O(blocks) refcount bumps with no KV byte copies — the paged alternative to cloning a
// whole-prefix KVCache. Run: go test ./internal/model -run x -bench BenchmarkPagedKVForkShare
func BenchmarkPagedKVForkShare(b *testing.B) {
	p := NewPagedKVPool(pagedTestCfg(), 16)
	base := pagedFillSeq(p, 2048) // 128 blocks of shared prefix
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := base.Fork()
		f.Free()
	}
}
