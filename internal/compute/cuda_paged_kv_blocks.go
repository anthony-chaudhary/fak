package compute

import (
	"fmt"
	"os"
	"strings"
)

// CUDAPagedKVBlockTokens is the page size of the opt-in CUDA paged KV store: the
// PagedAttention default the vendored vLLM kernel is instantiated for, and the page size
// of the device planner's paged prefix ledger.
const CUDAPagedKVBlockTokens = 16

// cudaPagedKVHeadDims are the head widths the vendored paged-attention kernel is
// instantiated for (vLLM's own list); a store for any other width stays contiguous.
var cudaPagedKVHeadDims = map[int]bool{32: true, 64: true, 80: true, 96: true, 112: true, 120: true, 128: true, 192: true, 256: true}

// CUDAPagedKVEnabled reports whether FAK_CUDA_PAGED_KV opts CUDA KV stores into the
// page-granular store. It is off by default: the contiguous store keeps graph capture,
// the prompt panel's direct append and the resident Qwen sequence path.
func CUDAPagedKVEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_CUDA_PAGED_KV"))) {
	case "1", "true", "on", "yes":
		return true
	}
	return false
}

// CUDAPagedKVEligible reports whether cfg's geometry can use the paged store: f32 KV and a
// head width the vendored kernel is instantiated for.
func CUDAPagedKVEligible(cfg KVConfig) bool {
	return cfg.Precision == KVPrecisionF32 && cfg.NumLayers > 0 && cfg.NumKVHeads > 0 &&
		cudaPagedKVHeadDims[cfg.HeadDim]
}

// pagedKVBlocks is the refcounted physical block allocator behind the paged store. One
// block id addresses the same page in every layer's K, pre-RoPE K and V arena, so a
// session's block table is one []int32 shared by all layers. A clone retains every block
// it maps (zero-copy prefix sharing); a writer copies a block it does not own alone first.
//
// compute.CUDABlockTable is not reused here: its ids come from a per-table counter that a
// clone inherits, so two diverging clones mint the same id for different pages.
type pagedKVBlocks struct {
	refs []int32 // len == capacity; 0 = free
	free []int32 // stack of free ids
	live int
}

// grow raises the capacity to n blocks; existing ids are unchanged.
func (a *pagedKVBlocks) grow(n int) {
	old := len(a.refs)
	if n <= old {
		return
	}
	a.refs = append(a.refs, make([]int32, n-old)...)
	for id := n - 1; id >= old; id-- {
		a.free = append(a.free, int32(id))
	}
}

// alloc returns a fresh block with one reference, or false when the arena is full.
func (a *pagedKVBlocks) alloc() (int32, bool) {
	if len(a.free) == 0 {
		return -1, false
	}
	id := a.free[len(a.free)-1]
	a.free = a.free[:len(a.free)-1]
	a.refs[id] = 1
	a.live++
	return id, true
}

func (a *pagedKVBlocks) check(id int32) {
	if id < 0 || int(id) >= len(a.refs) || a.refs[id] <= 0 {
		panic(fmt.Sprintf("compute: paged KV block %d is not live", id))
	}
}

// retain adds a reference to a live block.
func (a *pagedKVBlocks) retain(id int32) {
	a.check(id)
	a.refs[id]++
}

// release drops one reference and reports whether the block returned to the free list.
func (a *pagedKVBlocks) release(id int32) bool {
	a.check(id)
	a.refs[id]--
	if a.refs[id] > 0 {
		return false
	}
	a.free = append(a.free, id)
	a.live--
	return true
}

// shared reports whether more than one table maps id.
func (a *pagedKVBlocks) shared(id int32) bool {
	a.check(id)
	return a.refs[id] > 1
}

// pagedKVGrowBlocks is the next arena capacity when need blocks do not fit cur: 1.5x
// growth with a 64-block floor, so arena copies stay amortized.
func pagedKVGrowBlocks(cur, need int) int {
	n := cur + cur/2
	if n < 64 {
		n = 64
	}
	if n < need {
		n = need
	}
	return n
}

// pagedKVRun is one contiguous write inside a single page: rows [src, src+n) of an
// append land at in-block offset off of logical block blk.
type pagedKVRun struct {
	blk, off, src, n int
}

// pagedKVRuns splits an append of n rows starting at logical position start into runs
// that never cross a page boundary.
func pagedKVRuns(start, n, blockTokens int) []pagedKVRun {
	var runs []pagedKVRun
	src := 0
	for n > 0 {
		pos := start + src
		blk, off := pos/blockTokens, pos%blockTokens
		take := blockTokens - off
		if take > n {
			take = n
		}
		runs = append(runs, pagedKVRun{blk: blk, off: off, src: src, n: take})
		src += take
		n -= take
	}
	return runs
}
