package model

import "fmt"

// expert_readahead.go — two no-op-safe deltas on the demand-paged expert read path,
// clean-room-inspired by colibri (#4359, epic #2726; colibri@1bdaeee c/glm.c:1235 +
// c/st.h:219; Apache-2.0 <-> Apache-2.0, no bytes vendored). Both target the disk-bandwidth
// bound of cold MoE decode, where a layer's fused [E, out, in] expert block dwarfs the top-k
// the router actually picks:
//
//  1. readExpertSlice — read ONLY expert e's [base+e*stride, +stride) sub-range, so a top-k
//     route over the demand-paged path moves k*stride bytes, not the whole E*stride layer
//     (read-amplification avoidance).
//  2. willneedExpertSlice — issue a best-effort MADV_WILLNEED over the NEXT picked expert's
//     sub-range while the caller computes the CURRENT expert's GEMM, so the later synchronous
//     read hits warm page cache (read/compute overlap). Fires only on the mmap path; a pure
//     hint, a no-op on the ReadAt fallback and on non-unix platforms.
//
// GENERATION FRAME (gen/second-next, #4359 — architectural option, never default exposure):
// these are a prototype behind the strongest possible gate — nothing in the default
// load/decode path calls them, so no default behaviour changes. PROMOTION evidence: the
// compatibility test (expert_readahead_test.go) proves the slice read is byte-identical to a
// whole-tensor-then-slice on both the mmap and ReadAt paths and quantifies the k/E
// read-amplification win; promote by wiring willneedExpertSlice(pick[i+1]) into the
// demand-paged expert loop and readExpertSlice into the fused-block reads at moe_split.go and
// the ggufload GLM seam (gguf_glm_tensors.go:376, a separate lane), each behind a measured
// bytes/token-before-vs-after witness. DEMOTION/retirement: retire if a resident-weights or
// fully-mmapped deployment makes the fused block never disk-bound — the slice read then only
// adds bookkeeping and the WILLNEED hint has nothing to overlap. INVALIDATING assumption:
// that OS readahead is NOT already prefetching the next block for free; on a
// sequential-scan-friendly filesystem with generous readahead the explicit WILLNEED can be a
// wash or worse, so any promotion must land behind the measured witness, not a default flip.

// expertSliceBounds returns the [start, end) byte range of expert e inside a fused expert
// tensor whose data begins at `base` and gives each of `expertCount` experts `stride`
// contiguous bytes, validated against the file's data region [dataBase, size). It is the
// shared bounds check for readExpertSlice and willneedExpertSlice.
func (sf *safetensorsFile) expertSliceBounds(base int64, e, expertCount int, stride int64) (start, end int64, err error) {
	if e < 0 || expertCount <= 0 || e >= expertCount {
		return 0, 0, fmt.Errorf("model: expert %d out of range [0,%d)", e, expertCount)
	}
	if base < 0 || stride <= 0 {
		return 0, 0, fmt.Errorf("model: invalid expert slice base=%d stride=%d", base, stride)
	}
	start = base + int64(e)*stride
	end = start + stride
	if end < start || start < sf.dataBase || end > sf.size {
		return 0, 0, fmt.Errorf("model: expert slice [%d,%d) outside data region [%d,%d)", start, end, sf.dataBase, sf.size)
	}
	return start, end, nil
}

// readExpertSlice reads exactly expert e's byte sub-range out of a fused expert tensor, never
// the whole E*stride layer. On the mmap path it returns a zero-copy three-index-capped slice
// into the mapped region; otherwise it ReadAts a single expert-sized buffer.
func (sf *safetensorsFile) readExpertSlice(base int64, e, expertCount int, stride int64) ([]byte, error) {
	start, end, err := sf.expertSliceBounds(base, e, expertCount, stride)
	if err != nil {
		return nil, err
	}
	if sf.data != nil {
		return sf.data[start:end:end], nil
	}
	b := make([]byte, end-start)
	if _, err := sf.r.ReadAt(b, start); err != nil {
		return nil, fmt.Errorf("model: expert slice read: %w", err)
	}
	return b, nil
}

// willneedExpertSlice issues a best-effort MADV_WILLNEED readahead over expert e's byte
// sub-range so the kernel warms those pages while the caller computes another expert's GEMM.
// It fires only on the mmap path (there is a mapped region to advise), swallows every error
// (a failed hint only forfeits the overlap, never correctness), and returns whether a hint
// was actually issued — for a bytes/token witness, never for control flow.
func (sf *safetensorsFile) willneedExpertSlice(base int64, e, expertCount int, stride int64) bool {
	if sf.data == nil {
		return false
	}
	start, end, err := sf.expertSliceBounds(base, e, expertCount, stride)
	if err != nil || end > int64(len(sf.data)) {
		return false
	}
	return madviseWillneed(sf.data, int(start), int(end-start))
}
