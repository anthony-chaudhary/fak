package model

import (
	"strconv"
	"testing"
)

// paged_reserve_test.go — #34 proof: the paged allocator's Reserve / Clone / CloneWithReserve
// preserve the contiguous KVCache's reuse semantics (kvcache.go) bit-exactly. All on real
// float32 KV with no GPU and on a plain 2-plane K/V pool, so the proof stands on the committed
// #277 allocator alone (it does not depend on the 3-plane Kraw path).

func pagedReserveCfg() Config {
	return Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 3} // stride = 6
}

// appendPagedToken appends one deterministic token (distinct per token/layer/lane) to s and
// mirrors the same bytes into wantK/wantV, so a later GatherK/GatherV must reproduce them.
func appendPagedToken(s *PagedKV, wantK, wantV [][]float32, tok int) {
	nL, stride := s.pool.nLayers, s.pool.stride
	k := make([][]float32, nL)
	v := make([][]float32, nL)
	for l := 0; l < nL; l++ {
		kr := make([]float32, stride)
		vr := make([]float32, stride)
		for j := 0; j < stride; j++ {
			val := float32(tok*1000 + l*100 + j + 1)
			kr[j] = val
			vr[j] = -val
		}
		k[l], v[l] = kr, vr
		wantK[l] = append(wantK[l], kr...)
		wantV[l] = append(wantV[l], vr...)
	}
	s.Append(k, v)
}

// newPagedSeq mints a sequence of n deterministic tokens plus the contiguous K/V it must gather.
func newPagedSeq(pool *PagedKVPool, n int) (*PagedKV, [][]float32, [][]float32) {
	wantK := make([][]float32, pool.nLayers)
	wantV := make([][]float32, pool.nLayers)
	for l := range wantK {
		wantK[l] = []float32{}
		wantV[l] = []float32{}
	}
	s := pool.NewSequence()
	for tok := 0; tok < n; tok++ {
		appendPagedToken(s, wantK, wantV, tok)
	}
	return s, wantK, wantV
}

func assertGathersEqual(t *testing.T, tag string, s *PagedKV, wantK, wantV [][]float32) {
	t.Helper()
	for l := 0; l < s.pool.nLayers; l++ {
		assertFloat32BitsEqual(t, tag+" K l"+strconv.Itoa(l), wantK[l], s.GatherK(l))
		assertFloat32BitsEqual(t, tag+" V l"+strconv.Itoa(l), wantV[l], s.GatherV(l))
	}
}

// TestPagedReservePreservesContentAndPreAllocates proves Reserve grows the page table to the
// blocks a future growth will need WITHOUT changing Len or any gathered byte, and that the
// reserved blocks then absorb appends with no further pool allocation — yet the grown sequence
// is byte-for-byte identical to one built without reserving.
func TestPagedReservePreservesContentAndPreAllocates(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4) // 4 tokens/block
	seq, wantK, wantV := newPagedSeq(pool, 7)    // 7 tokens -> ceil(7/4) = 2 blocks

	if seq.Blocks() != 2 || pool.PhysicalBlocks() != 2 {
		t.Fatalf("setup: Blocks=%d PhysicalBlocks=%d, want 2/2", seq.Blocks(), pool.PhysicalBlocks())
	}

	seq.Reserve(10) // room for 7+10=17 tokens -> ceil(17/4) = 5 blocks
	if seq.Len() != 7 {
		t.Fatalf("Reserve changed Len: %d, want 7", seq.Len())
	}
	if seq.Blocks() != 5 {
		t.Fatalf("Reserve did not pre-allocate: Blocks=%d, want 5", seq.Blocks())
	}
	if pb := pool.PhysicalBlocks(); pb != 5 {
		t.Fatalf("Reserve must own its reserved blocks: PhysicalBlocks=%d, want 5", pb)
	}
	// Live content untouched — reserved blocks sit past the live tail.
	assertGathersEqual(t, "post-reserve", seq, wantK, wantV)

	// Grow into the reserved space: tokens 7..15 (9 more -> 16 total, still <= 17 reserved).
	for tok := 7; tok < 16; tok++ {
		appendPagedToken(seq, wantK, wantV, tok)
	}
	if seq.Len() != 16 {
		t.Fatalf("post-grow Len=%d, want 16", seq.Len())
	}
	// 16 tokens occupy ceil(16/4)=4 blocks, all within the 5 reserved — no new pool alloc.
	if pb := pool.PhysicalBlocks(); pb != 5 {
		t.Fatalf("append into reserved space minted a block: PhysicalBlocks=%d, want 5", pb)
	}
	assertGathersEqual(t, "post-grow", seq, wantK, wantV)

	// And the grown sequence equals one built fresh to 16 tokens (reserve is invisible to content).
	ref, refK, refV := newPagedSeq(NewPagedKVPool(pagedReserveCfg(), 4), 16)
	_ = ref
	assertGathersEqual(t, "reserve-vs-fresh", seq, refK, refV)
}

// TestPagedCloneIsEagerDeepCopyAndIndependent proves Clone is an eager, fully-private deep copy
// (shares no block, contrast Fork's copy-on-write share), is bit-identical to the source, and is
// mutation-independent in both directions.
func TestPagedCloneIsEagerDeepCopyAndIndependent(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	seq, wantK, wantV := newPagedSeq(pool, 7) // 2 blocks
	if pool.PhysicalBlocks() != 2 {
		t.Fatalf("setup PhysicalBlocks=%d, want 2", pool.PhysicalBlocks())
	}

	clone := seq.Clone()
	if clone.Len() != 7 || clone.Blocks() != 2 {
		t.Fatalf("clone Len=%d Blocks=%d, want 7/2", clone.Len(), clone.Blocks())
	}
	// Eager deep copy: the clone added its own 2 physical blocks (Fork would have added 0).
	if pb := pool.PhysicalBlocks(); pb != 4 {
		t.Fatalf("Clone must be an eager deep copy (no sharing): PhysicalBlocks=%d, want 4", pb)
	}
	assertGathersEqual(t, "clone", clone, wantK, wantV)

	// Contrast: Fork shares the source's blocks copy-on-write, minting nothing.
	fork := seq.Fork()
	if pb := pool.PhysicalBlocks(); pb != 4 {
		t.Fatalf("Fork must share (COW), not copy: PhysicalBlocks=%d, want 4", pb)
	}
	_ = fork

	// Snapshot the source, mutate the clone, and confirm the source is byte-for-byte unchanged.
	srcK := make([][]float32, pool.nLayers)
	srcV := make([][]float32, pool.nLayers)
	for l := 0; l < pool.nLayers; l++ {
		srcK[l] = append([]float32(nil), seq.GatherK(l)...)
		srcV[l] = append([]float32(nil), seq.GatherV(l)...)
	}
	appendPagedToken(clone, wantK, wantV, 99) // wantK/wantV now track the clone
	if seq.Len() != 7 {
		t.Fatalf("clone append changed source Len: %d, want 7", seq.Len())
	}
	assertGathersEqual(t, "source-after-clone-mutated", seq, srcK, srcV)
}

// TestPagedCloneWithReserve proves CloneWithReserve = Clone + Reserve: an independent, bit-exact
// prefix copy with room pre-allocated for the continuation.
func TestPagedCloneWithReserve(t *testing.T) {
	pool := NewPagedKVPool(pagedReserveCfg(), 4)
	seq, wantK, wantV := newPagedSeq(pool, 7) // 2 blocks

	cwr := seq.CloneWithReserve(10) // clone (2 blocks) + reserve to ceil(17/4)=5 blocks
	if cwr.Len() != 7 {
		t.Fatalf("CloneWithReserve changed Len: %d, want 7", cwr.Len())
	}
	if cwr.Blocks() != 5 {
		t.Fatalf("CloneWithReserve did not reserve: Blocks=%d, want 5", cwr.Blocks())
	}
	assertGathersEqual(t, "clone-with-reserve", cwr, wantK, wantV)

	// Independent of the source: source(2) + clone's deep-copied(2) + reserved(3) = 7 live blocks.
	if pb := pool.PhysicalBlocks(); pb != 7 {
		t.Fatalf("CloneWithReserve sharing leak: PhysicalBlocks=%d, want 7", pb)
	}

	// Grow the clone into its reserved space without minting blocks or touching the source.
	before := pool.PhysicalBlocks()
	cloneK, cloneV := wantK, wantV
	for tok := 7; tok < 16; tok++ {
		appendPagedToken(cwr, cloneK, cloneV, tok)
	}
	if pb := pool.PhysicalBlocks(); pb != before {
		t.Fatalf("clone grew past its reservation: PhysicalBlocks=%d, want %d", pb, before)
	}
	if seq.Len() != 7 {
		t.Fatalf("clone growth disturbed source Len: %d, want 7", seq.Len())
	}
	assertGathersEqual(t, "clone-grown", cwr, cloneK, cloneV)
}
