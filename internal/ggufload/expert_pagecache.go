package ggufload

// expert_pagecache.go — the page-cache/mmap-aware EXPERT SOURCE seam (#1302), the loader-side
// half of the Tier-3 SSD offload foundation (gguf_mmap.go, #2726/#2722).
//
// What it is for. FAK_GGUF_MMAP already retains a read-only memory map of each shard instead of
// an os.File + pread (gguf_mmap.go), and WeightSource already carries that mapped region as
// data/dataFor. Until this slice nothing on the routed-expert path READ those bytes: the R5
// checkpoint tier (#5616) faults each expert through the shard's io.ReaderAt, so an mmap-backed
// shard pays a page-cache copy anyway and the map buys nothing for the expert bulk. This seam
// lets the model tier adopt a ZERO-COPY read-through over the mapped file for the fused routed
// expert slabs specifically — the bytes the ladder exists to keep off the heap and out of a
// second buffer.
//
// Default OFF, gate FAK_EXPERT_PAGECACHE. The gate is deliberately separate from (and narrower
// than) FAK_GGUF_MMAP: mapping a shard is a load-wide storage decision, while adopting the map
// on the expert path is the behaviour change that can alter expert fault semantics, so it needs
// its own explicit opt-in. Unset, or any spelling other than the documented truthy set, keeps
// the historical ReadAt path byte-for-byte — including on every platform without an mmap impl
// (Windows), where data/dataFor are always nil and pageCacheAdoption therefore never fires.

import (
	"bytes"
	"os"
	"strings"
	"sync"
)

// expertPageCacheOnce/expertPageCacheOn cache the FAK_EXPERT_PAGECACHE decision once per process,
// mirroring the ggufMmapEnabled idiom (gguf_mmap.go) so loader goroutines never re-read process
// environment and a single load has one immutable adoption contract even across split shards
// opened at different times. Tests reset the pair directly (same package) to exercise both arms
// in one process.
var (
	expertPageCacheOnce sync.Once
	expertPageCacheOn   bool
)

// expertPageCacheEnabled reports the FAK_EXPERT_PAGECACHE opt-in. Only the documented truthy
// spellings enable it; every other value (including unset) preserves the historical ReadAt path
// byte-for-byte.
func expertPageCacheEnabled() bool {
	expertPageCacheOnce.Do(func() {
		switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_EXPERT_PAGECACHE"))) {
		case "1", "on", "true":
			expertPageCacheOn = true
		}
	})
	return expertPageCacheOn
}

// resetExpertPageCacheEnv clears the cached gate so a test can re-evaluate the environment in
// one process. It is the package-local analogue of assigning ggufMmapOnce/ggufMmapOn directly
// and is used the same way (see expert_pagecache_test.go).
func resetExpertPageCacheEnv() {
	expertPageCacheOnce = sync.Once{}
	expertPageCacheOn = false
}

// pageCacheAdoption decides whether the model tier may adopt the zero-copy read-through path over
// a shard's mapped region. It returns (data, true) only when the gate is on, the mapped region is
// non-empty, AND the region is exactly the shard's declared size — adopting a shorter or longer
// buffer would let the tier arithmetic index outside the real mapping. Any other combination
// returns (nil, false), which is the historical ReadAt path byte-for-byte.
//
// ok=true is the model tier's licence to serve expert faults as sub-slices of data instead of
// copying through Reader; ok=false means every existing behaviour is preserved unchanged. The
// function is pure and total so both arms are unit-testable without a real mmap.
func pageCacheAdoption(f FusedExpertShard, enabled bool) (data []byte, ok bool) {
	if !enabled || len(f.Data) == 0 || int64(len(f.Data)) != f.Size {
		return nil, false
	}
	return f.Data, true
}

// sameBytes reports whether two byte slices are identical, treating two nils as equal. It backs
// FusedExpertTensors' guard that every tensor sharing a shard reader resolves to the same mapped
// region (expert_checkpoint_source.go).
func sameBytes(a, b []byte) bool {
	return bytes.Equal(a, b)
}
