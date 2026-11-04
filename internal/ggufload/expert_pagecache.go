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
// Default OFF, gate WithExpertPageCache (quant_q4k_loader.go). The gate is deliberately separate
// from (and narrower than) FAK_GGUF_MMAP: mapping a shard is a load-wide storage decision, while
// adopting the map on the expert path is the behaviour change that can alter expert fault
// semantics, so it needs its own explicit opt-in. The retired FAK_EXPERT_PAGECACHE ambient read
// (#13605 leaf A-cfg-int4) cached its decision in a process-global sync.Once; the replacement is
// resolved once per load request from the caller's Q4K load options and passed explicitly to
// FusedExpertTensors and buildExpertCheckpointTier, so one load has one immutable adoption
// contract even across split shards opened at different times. Unset options, or any request
// without WithExpertPageCache(true), keep the historical ReadAt path byte-for-byte — including
// on every platform without an mmap impl (Windows), where data/dataFor are always nil and
// pageCacheAdoption therefore never fires.

import "bytes"

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
