package radixkv

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// Regression tests for the boundary-page free invariants borrowed from the
// SGLang unified radix cache correctness fix (sgl-project/sglang
// python/sglang/srt/mem_cache/unified_radix_cache.py:998,1073-1078 and
// unified_cache/components/base.py:387 @b02e16a; upstream PRs #38982 and
// #39426 hit the same boundary-page bug twice). See the invariant block above
// ReleaseBlockTable in paged_binding.go.

// TestBoundaryPageTwoTouchingRangesAbortShape reproduces the abort shape the
// upstream fix removed: two free ranges of one physical row that touch at the
// truncation boundary share that boundary page and must be rejected unless
// merged into the single union range.
func TestBoundaryPageTwoTouchingRangesAbortShape(t *testing.T) {
	// Pre-fix shape: unaligned-tail half [200,250) plus deferred
	// truncation-tail half [250,400) — touching at 250, sharing the boundary
	// page that contains token 250.
	touching := []freeRange{{start: 200, end: 250}, {start: 250, end: 400}}
	if err := validateRowFreeRanges(touching); !errors.Is(err, ErrTouchingFreeRanges) {
		t.Fatalf("validateRowFreeRanges(touching) = %v, want ErrTouchingFreeRanges", err)
	}

	// Post-fix plan: the same inputs merged into one union range.
	planned := planTailFreeRanges(200, 250, 250, 400)
	want := []freeRange{{start: 200, end: 400}}
	if len(planned) != len(want) {
		t.Fatalf("planTailFreeRanges = %+v, want exactly one merged range %+v", planned, want)
	}
	if planned[0] != want[0] {
		t.Fatalf("planTailFreeRanges = %+v, want %+v", planned, want)
	}
	if err := validateRowFreeRanges(planned); err != nil {
		t.Fatalf("merged plan must validate, got %v", err)
	}
}

// TestBoundaryPageComponentFloorAfterCrossComponentMin pins invariant 1: a
// length reduced across components by min must be re-floored to each
// component's alignment, or the reducing component's boundary page is left
// partially owned.
func TestBoundaryPageComponentFloorAfterCrossComponentMin(t *testing.T) {
	// Component caps: full=400 (identity alignment), swa=250 (identity),
	// c128=384 (128-token group alignment).
	caps := []int{400, 250, 384}
	alignments := []int{1, 1, 128}
	m := caps[0]
	for _, c := range caps[1:] {
		if c < m {
			m = c
		}
	}
	if m != 250 {
		t.Fatalf("cross-component min = %d, want 250", m)
	}
	// Without the re-floor step the min lands between two c128 group
	// boundaries: the boundary page is mis-owned.
	if m%alignments[2] == 0 {
		t.Fatalf("pre-fix min %d is unexpectedly aligned to c128 groups", m)
	}
	// With the re-floor step the result is a multiple of every component's
	// alignment.
	floored := m
	for _, a := range alignments {
		floored = floorToAlignment(floored, a)
	}
	if floored != 128 {
		t.Fatalf("re-floored length = %d, want 128", floored)
	}
	for i, a := range alignments {
		if a > 1 && floored%a != 0 {
			t.Fatalf("re-floored length %d not aligned to component %d alignment %d", floored, i, a)
		}
	}

	// Closed value contract for the floor primitive itself.
	if got := floorToAlignment(384, 128); got != 384 {
		t.Errorf("floorToAlignment(384,128) = %d, want 384", got)
	}
	if got := floorToAlignment(0, 128); got != 0 {
		t.Errorf("floorToAlignment(0,128) = %d, want 0", got)
	}
	if got := floorToAlignment(250, 1); got != 250 {
		t.Errorf("floorToAlignment(250,1) = %d, want 250 (identity alignment)", got)
	}
}

// TestBoundaryPageNonAbortingFreesUnchanged pins that the guards leave
// non-abort frees untouched: disjoint ranges still validate, a gapped tail
// plan keeps both halves, and the pool's real free path still releases
// exactly the bound blocks.
func TestBoundaryPageNonAbortingFreesUnchanged(t *testing.T) {
	disjoint := []freeRange{{start: 100, end: 200}, {start: 300, end: 400}}
	if err := validateRowFreeRanges(disjoint); err != nil {
		t.Fatalf("disjoint ranges must validate, got %v", err)
	}

	gapped := planTailFreeRanges(200, 250, 300, 400)
	want := []freeRange{{start: 200, end: 250}, {start: 300, end: 400}}
	if len(gapped) != len(want) || gapped[0] != want[0] || gapped[1] != want[1] {
		t.Fatalf("gapped plan = %+v, want %+v", gapped, want)
	}
	if err := validateRowFreeRanges(gapped); err != nil {
		t.Fatalf("gapped plan must validate, got %v", err)
	}

	// Protected prefix running past the truncated cache length drops the
	// empty first half instead of emitting an inverted range.
	past := planTailFreeRanges(192, 128, 200, 400)
	wantPast := []freeRange{{start: 200, end: 400}}
	if len(past) != len(wantPast) || past[0] != wantPast[0] {
		t.Fatalf("past-truncation plan = %+v, want %+v", past, wantPast)
	}

	// Real free path: bind then release returns the tree to sole ownership
	// and the allocator to its prior free count (no abort, no drift).
	cfg := ctxmmu.KVPoolConfig{
		TotalPages:    64,
		TokensPerPage: 16,
		NumLayers:     4,
		NumKVHeads:    2,
		HeadDim:       32,
		DType:         ctxmmu.KVDTypeFP32,
		Topology:      ctxmmu.KVTopologyGQA,
	}
	kvPool, err := ctxmmu.NewKVPool(cfg)
	if err != nil {
		t.Fatalf("NewKVPool: %v", err)
	}
	pool := NewPagedRadixKVPool(0, kvPool)
	tokens := make([]int, 32)
	for i := range tokens {
		tokens[i] = 1000 + i
	}
	boundary, _ := pool.Tree.Lookup(nil)
	leaf, allocated, err := pool.InsertWithAllocatedTokens(boundary, tokens)
	if err != nil || leaf == nil {
		t.Fatalf("InsertWithAllocatedTokens: %v", err)
	}
	table, matched, hit, err := pool.BindPrefix(tokens, "boundary-page-stream")
	if err != nil || !hit || matched != 32 {
		t.Fatalf("BindPrefix: hit=%v matched=%d err=%v", hit, matched, err)
	}
	alloc := kvPool.Allocator()
	freeBefore := alloc.FreeCount()
	if err := pool.ReleaseBlockTable(table); err != nil {
		t.Fatalf("ReleaseBlockTable: %v", err)
	}
	if freeAfter := alloc.FreeCount(); freeAfter != freeBefore {
		t.Fatalf("FreeCount changed %d -> %d on a non-abort free", freeBefore, freeAfter)
	}
	for _, bID := range allocated {
		blk, err := alloc.GetBlock(bID)
		if err != nil {
			t.Fatalf("GetBlock(%d): %v", bID, err)
		}
		if ref := blk.RefCount(); ref != 1 {
			t.Fatalf("block %d refcount = %d after release, want 1 (tree ownership)", bID, ref)
		}
	}
}
