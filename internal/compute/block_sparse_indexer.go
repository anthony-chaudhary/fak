package compute

// block_sparse_indexer.go — DeepSeek-V4.1 two-level block-sparse indexer, host-side.
//
// PROVENANCE (DIRECT-PORT, ticket oss-port-sglang-block-sparse-indexer):
//   Source:  sgl-project/sglang @ faaff1eca8876b8b77d8704f080c3ee2064b8b65
//            ("dsv4.1: Top-k kernels and candidate selection", PR #39648);
//            license and notices: THIRD_PARTY_NOTICES.md ("SGLang").
//   License: Apache-2.0 (SGLang root LICENSE; no per-file copyright header on the
//            pinned sources). Copyright SGLang authors.
//   Reuse:   DIRECT-PORT of the decomposition contract — the level-one block-amax
//            reducer (newest block forced +inf, rows with <= topk blocks skipped),
//            the exact level-two top-k over block maxima, and the candidate-block
//            table finaliser (ascending ids, INT32_MAX pad, pool-slot transform,
//            identity fast path). The CUDA control flow (CTAs, atomics, PDL) does not
//            transfer to Go; the semantics do, and the level-two selection reuses the
//            existing exact engine below instead of a parallel top-k.
//
// THE DECOMPOSITION (upstream pipeline, one row at a time):
//  1. Level one: amax[i] = max(scores[8i : 8i+8]) for
//     i < ceil(seqLen/8); the newest (last) block is forced to +inf so it is always
//     selected; rows with at most topk blocks are skipped entirely (every block is
//     selected anyway — the identity fast path). kBlockTokens = 8.
//  2. Level two: the EXACT top-k over the block maxima,
//     emitting block ids. Upstream runs its bf16 radix-select kernel; here the seam
//     is DeepSelectTopk (deepselect_topk.go) — the landed host ancestor of that same
//     kernel — so the selected key SET is byte-for-byte the flat engine's set on the
//     identically prepared keys, for every input.
//  3. Finalise: the selected block ids (any order, -1
//     padded) become ascending with INT32_MAX past the count, plus each block mapped
//     to its index-pool slot pageTable[id/bpp]*bpp + id%bpp (bpp = pageSize/8 blocks
//     per pool page). Rows with at most topk blocks get the identity table without
//     reading their input.
//
// EXACTNESS (the acceptance invariant): TwoLevelBlockKeys' selected block-key set
// equals the flat DeepSelectTopk set on the identically prepared (newest-forced)
// block-maxima keys, for EVERY input. The argument has three cases:
//   - nblocks > topk: level two IS DeepSelectTopk over the forced amax array, so the
//     sets agree by construction (the engine is exact under (value desc, id asc)).
//   - nblocks <= topk: both paths select all blocks — the fast path emits them
//     directly, the flat engine returns every item when k >= n.
//   - ties: the forced +inf key maps to the largest candidateKey, finite ties resolve
//     by lower block id at both levels, and the all-equal / engineered-tie inputs are
//     pinned by TestBlockAmaxMatchesFlatDeepSelect.
//
// TOKEN-LEVEL CAVEAT (inherited from upstream, documented so nobody "fixes" it): the
// forced newest block occupies one of the topk slots. When it displaces a block that
// a flat token-level top-k would have covered, the covered token set diverges from the
// flat token top-k — that is the upstream recency guarantee (always attend the newest
// context), not a defect. TestBlockAmaxNewestBlockAlwaysSelected pins the steal.

import (
	"errors"
	"math"
	"sort"
)

// BlockSparseTokensPerBlock is the two-level indexer's kBlockTokens: each level-one
// key covers this many consecutive token scores.
const BlockSparseTokensPerBlock = 8

// BlockSparsePadID is the candidate-block-table pad value past a row's count:
// INT32_MAX, matching the upstream pad value.
const BlockSparsePadID = int32(math.MaxInt32)

// ErrBlockSparsePageSize is returned when a pool page size is not a power of two of
// at least BlockSparseTokensPerBlock tokens (upstream: "page_size must be a power of
// two of at least 8").
var ErrBlockSparsePageSize = errors.New("compute: block-sparse page size must be a power of two of at least 8")

// blockSparseSelectBlock is the DeepSelect input block size and compaction slack for
// level-two selection over block maxima. B=B2=1024 matches the DSA indexer sizing
// (dsaIndexSelectBlock); the engine stays exact for any geometry, the choice only
// changes pass counts, never the selected set.
const blockSparseSelectBlock = 1024

// blockSparsePageBits derives bpp-exponent from a pool page size: pageSize =
// 8 << bits tokens per page. ok=false means pageSize is not a power of two >= 8.
func blockSparsePageBits(pageSize int) (bits int, ok bool) {
	if pageSize < BlockSparseTokensPerBlock || pageSize&(pageSize-1) != 0 {
		return 0, false
	}
	for p := pageSize; p > BlockSparseTokensPerBlock; p >>= 1 {
		bits++
	}
	return bits, true
}

// blockAmaxReduce computes one block's amax over scores[lo:hi] with NaN ignored:
// CUDA fmaxf returns the non-NaN argument, so a NaN score never poisons the block
// key (torch.amax would propagate it). An all-NaN block reduces to -Inf, which the
// level-two selection then never picks — the same effective behaviour as upstream's
// NaN block key, which its ordered compares likewise never select.
func blockAmaxReduce(scores []float64, lo, hi int) float64 {
	m := math.Inf(-1)
	for i := lo; i < hi; i++ {
		v := scores[i]
		if math.IsNaN(v) {
			continue
		}
		if v > m {
			m = v
		}
	}
	return m
}

// BlockAmax is the level-one reducer: one row of token scores becomes its block keys.
// out[i] = max(scores[8i : min(8i+8, seqLen)]) for i < ceil(seqLen/8), and the newest
// (last) block is forced to +Inf so downstream selection always keeps it. A
// zero-length row yields nil. If seqLen exceeds len(scores), the reduction clamps to
// len(scores) (the caller owns the row's valid extent; the clamp is defensive, not a
// contract).
//
// Host divergence from the pinned CUDA kernel, documented: the kernel reads full
// 8-wide vectors even for the last partial block (safe because the caller sizes the
// buffer for the longest row), while this host contract reduces exactly the
// seqLen-valid scores — the Triton consumer's masked semantics.
func BlockAmax(scores []float64, seqLen int) []float64 {
	if seqLen <= 0 {
		return nil
	}
	if seqLen > len(scores) {
		seqLen = len(scores)
	}
	nblocks := (seqLen + BlockSparseTokensPerBlock - 1) / BlockSparseTokensPerBlock
	out := make([]float64, nblocks)
	for b := 0; b < nblocks; b++ {
		lo := b * BlockSparseTokensPerBlock
		hi := lo + BlockSparseTokensPerBlock
		if hi > seqLen {
			hi = seqLen
		}
		out[b] = blockAmaxReduce(scores, lo, hi)
	}
	// The newest block is always selected: force its key to +inf (upstream:
	// out[idx] = idx + 1 == num_keys ? inf : key).
	out[nblocks-1] = math.Inf(1)
	return out
}

// HierarchicalBlockTopk is the level-two selection: the exact top-k over level-one
// block keys, returning the selected block ids (positions into amax) highest key
// first, ties by lower block id — the DeepSelect total order applied to the block
// keys. It extends the existing seam rather than building a parallel top-k: the
// engine is DeepSelectTopk itself, the landed host ancestor of upstream's
// level-two top-k kernel. k <= 0 or an empty key row yields nil; k >= len(amax)
// yields
// every block id.
func HierarchicalBlockTopk(amax []float64, k int) []int {
	n := len(amax)
	if k <= 0 || n == 0 {
		return nil
	}
	pool := make([]Candidate, n)
	for i, v := range amax {
		pool[i] = Candidate{Value: v, Position: i}
	}
	sel := DeepSelectTopk(pool, k, blockSparseSelectBlock, blockSparseSelectBlock)
	out := make([]int, len(sel))
	for i := range sel {
		out[i] = sel[i].Position
	}
	return out
}

// TwoLevelBlockKeys is the new two-level path over one row of token scores: reduce
// to block keys (BlockAmax, newest block forced +Inf), then select the top-`topk`
// block keys (HierarchicalBlockTopk via the exact DeepSelect engine). Rows with at
// most topk blocks take the identity fast path — every block is selected, nothing is
// reduced or selected (upstream skips such rows entirely). Returns the selected
// block ids in ascending order (upstream leaves level two unordered; the table
// builder sorts, and folding the sort in here makes the composition self-contained).
//
// The selected key SET equals the flat DeepSelectTopk set on the identically
// prepared (newest-forced) block-maxima keys for every input — see the EXACTNESS
// note at the top of this file.
func TwoLevelBlockKeys(scores []float64, seqLen, topk int) []int {
	nblocks := (seqLen + BlockSparseTokensPerBlock - 1) / BlockSparseTokensPerBlock
	if topk <= 0 || nblocks == 0 {
		return nil
	}
	if nblocks <= topk {
		out := make([]int, nblocks)
		for i := range out {
			out[i] = i
		}
		return out
	}
	sel := HierarchicalBlockTopk(BlockAmax(scores, seqLen), topk)
	sort.Ints(sel)
	return sel
}

// CandidateBlockTable finalises a row's selected block ids into the two-level
// indexer's block table, mirroring the upstream finaliser:
//
//   - ids: the valid selected ids ascending, BlockSparsePadID (INT32_MAX) past the
//     count. Valid means 0 <= id < nblocks; -1 padding and out-of-range ids are
//     dropped. Duplicate ids collapse to one entry (upstream dedups through a
//     per-row bitmap).
//   - pages: the same blocks as index-pool slots, pageTable[id/bpp]*bpp + id%bpp
//     with bpp = pageSize/8 blocks per pool page (int32 arithmetic, matching the
//     kernel's (table[id >> page_bits] << page_bits) | (id & mask)), same padding.
//
// A row with at most topk blocks gets the identity table — ids t for t < nblocks,
// pages transform(t) — WITHOUT reading selected (upstream: "without reading its
// input"), so garbage or -1-filled selections still yield the correct table there.
// pageSize is the index pool's page size in tokens: a power of two >= 8, else
// ErrBlockSparsePageSize. topk <= 0 yields nil tables and no error. Both returned
// slices have length topk.
func CandidateBlockTable(selected []int, seqLen, topk, pageSize int, pageTable []int32) (ids, pages []int32, err error) {
	pageBits, ok := blockSparsePageBits(pageSize)
	if !ok {
		return nil, nil, ErrBlockSparsePageSize
	}
	if topk <= 0 {
		return nil, nil, nil
	}
	nblocks := (seqLen + BlockSparseTokensPerBlock - 1) / BlockSparseTokensPerBlock
	ids = make([]int32, topk)
	pages = make([]int32, topk)
	for r := range ids {
		ids[r] = BlockSparsePadID
		pages[r] = BlockSparsePadID
	}
	transform := func(id int32) int32 {
		return pageTable[id>>uint(pageBits)]<<uint(pageBits) | (id & int32((1<<pageBits)-1))
	}
	emit := func(rank int32, id int32) {
		ids[rank] = id
		pages[rank] = transform(id)
	}
	if nblocks <= topk {
		// Identity fast path: every block is selected regardless of the input.
		for t := int32(0); t < int32(nblocks); t++ {
			emit(t, t)
		}
		return ids, pages, nil
	}
	seen := make([]bool, nblocks)
	uniq := make([]int32, 0, topk)
	for _, id := range selected {
		if id < 0 || id >= nblocks {
			continue // -1 pad and out-of-range ids are dropped
		}
		i := int32(id)
		if seen[i] {
			continue
		}
		seen[i] = true
		uniq = append(uniq, i)
	}
	sort.Slice(uniq, func(a, b int) bool { return uniq[a] < uniq[b] })
	if len(uniq) > topk {
		uniq = uniq[:topk]
	}
	for r := range uniq {
		emit(int32(r), uniq[r])
	}
	return ids, pages, nil
}
