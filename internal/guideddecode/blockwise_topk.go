package guideddecode

// blockwise_topk.go — the streaming blockwise sparse top-k candidate ranker
// (borrow B2 from the GLiNER2 study, docs/research/STUDY-GLINER2-2026-09-18.md in
// the private research registry). Scoring every conditional (start, end) pair by
// materializing a full [Q, Ks, L] score tensor costs memory that grows with the
// product of the candidate sets; scoring the ends in streaming blocks of
// end_block_size and merging each block into a running per-start top-k keeps the
// largest tensor at [Q, Ks, end_block_size], so a fixed schema and budget make
// peak memory linear in sequence length.
//
// The reference (fastino-ai/GLiNER2 @ d7c727458bf6929bc9ef5ee04e13c3f717a7c455,
// Apache-2.0, gliner2/models/boundary/proposal.py:147-200 `_score_ends_blockwise`
// with `merge_running_topk` at :125-144; MASK_LOGIT = -1.0e4 from
// boundary/constants.py) streams end blocks, masks invalid pairs to the finite
// MASK_LOGIT sentinel, and merges [running || block] with a stable descending
// sort, taking the running top-k. Stable ties therefore retain concatenation
// order — the running entries win over a later block's equal score, and within a
// block the lower end index wins.
//
// This file is the CPU reference kernel and its equivalence witness only: no GPU
// kernel, no model architecture, and no existing decode implementation is
// touched. It stays stdlib-only, like the rest of the package.
//
// No consumer yet (issue #13273): the guided-decode lane currently constrains the
// tool-call envelope at the byte level (see guideddecode.go) and does not rank
// conditional (start, end) candidate spans, so there is no call site to wire this
// selector into. It lands as a standalone primitive — exactly as the sibling
// interval-score kernel did — so the blockwise ranking mechanism and its
// peak-memory bound are available (and witnessed) for the decode path that adopts
// span ranking later; the gold-plating boundary of #13273 excludes wiring it into
// an unbuilt path now.

// blockwiseMaskLogit is the finite invalid-pair sentinel, ported from the
// reference's MASK_LOGIT. It is deliberately finite (not -Inf): several score
// terms may be summed before masking, and a dtype minimum can overflow to -Inf
// and defeat a stable sort. A real candidate must score strictly above it; the
// sentinel can never be admitted because a masked pair is only ever compared
// against other masked pairs, and a full block of masked pairs still fills the
// running top-k with sentinel-scored placeholders rather than panicking.
const blockwiseMaskLogit float32 = -1.0e4

// BlockwisePair is one scored (start, end) candidate pair in the running top-k.
// End is the end-boundary index; Start is the start-boundary index it was scored
// against. Score is the summed conditional pair score.
type BlockwisePair struct {
	Start int
	End   int
	Score float32
}

// BlockwiseStartSlot is one start-boundary candidate: its index, its own score
// (a marginal added to every pair it spawns), and whether it is a real slot. A
// padded/dead slot (Valid == false) must not spawn candidates for a short sample
// batched with a long one — the reference's `start_valid` guard.
type BlockwiseStartSlot struct {
	Index int
	Score float32
	Valid bool
}

// BlockwiseTopK is the per-start running top-k accumulator. It holds Q start
// slots, each retaining at most Ks ends. The largest slice it ever materializes
// is one [Ks+1] merge window per slot, so a fixed Q/Ks/blockSize keeps peak
// memory linear in the number of ends rather than in Q*Ks*L.
type BlockwiseTopK struct {
	q  int
	ks int

	// scores[q][r] and ends[q][r] are the running top-k for start slot q, in
	// descending score order. ends[q][r] is the end index whose score is
	// scores[q][r]; a sentinel-filled slot has Score == blockwiseMaskLogit and
	// End == 0 (the reference initializes top_idx to zeros).
	scores [][]float32
	ends   [][]int
	starts []BlockwiseStartSlot

	conditionalElems int
	maxBlockElems    int
	scoredBlocks     int
}

// NewBlockwiseTopK returns an accumulator for Q start slots retaining Ks ends
// each. Q and Ks must be positive; a non-positive value returns nil so a caller
// that passed a degenerate shape gets an explicit refusal rather than a
// panicking accumulator.
func NewBlockwiseTopK(q, ks int) *BlockwiseTopK {
	if q <= 0 || ks <= 0 {
		return nil
	}
	b := &BlockwiseTopK{q: q, ks: ks}
	b.scores = make([][]float32, q)
	b.ends = make([][]int, q)
	b.starts = make([]BlockwiseStartSlot, q)
	for i := range b.scores {
		b.scores[i] = make([]float32, ks)
		b.ends[i] = make([]int, ks)
		for r := range b.scores[i] {
			b.scores[i][r] = blockwiseMaskLogit
		}
	}
	return b
}

// SetStart installs the start-boundary slot for position q. Valid slots have a
// real Index and are eligible to spawn candidate pairs; an invalid slot is
// carried but produces only sentinel-scored placeholders. Returns false on an
// out-of-range q or a nil receiver.
func (b *BlockwiseTopK) SetStart(q int, s BlockwiseStartSlot) bool {
	if b == nil || q < 0 || q >= b.q {
		return false
	}
	b.starts[q] = s
	return true
}

// endScoreFunc scores one (start slot, end index, end score) triple. It returns
// the conditional pair score and whether the pair is admissible. A caller
// supplies this so the blockwise ranker does not have to know the model's score
// terms; the reference computes `compat*scale + end_marginal + start_score`.
type endScoreFunc func(start BlockwiseStartSlot, endIndex int, endScore float32) (float32, bool)

// ScoreBlock streams one block of ends through the accumulator, merging each
// scored pair into the running top-k. endScores is the block's per-end score
// contribution and ends its end-boundary indices; both must have the same length.
// admiss is consulted for each (slot, end) pair after the score is computed and
// can veto a pair (the reference's `keep` mask: boundary valid, query valid,
// end > start, start valid). A bare `end > start` check is always applied on top,
// so an inadmissible span can never be admitted even if admiss admits it.
//
// It returns the number of pairs it scored (including vetoed ones) so the caller
// can account for the conditional work, or false if the block is malformed.
func (b *BlockwiseTopK) ScoreBlock(ends []int, endScores []float32, admiss endScoreFunc) (int, bool) {
	if b == nil || len(ends) != len(endScores) {
		return 0, false
	}
	scored := 0
	for q := range b.starts {
		start := b.starts[q]
		for i, e := range ends {
			score, ok := b.pairScore(start, e, endScores[i], admiss)
			scored++
			if !ok {
				score = blockwiseMaskLogit
			}
			b.merge(q, e, score)
		}
		b.conditionalElems += len(ends)
	}
	// Account the reference's [Q, Ks, e] block tensor budget (the shape the
	// upstream _score_ends_blockwise materializes) so MaxBlockElements equals
	// MaxBlockwiseIntermediateElems. The accumulator actually touches the block a
	// slot at a time, so its own largest live slice is a [Ks+1] merge window —
	// strictly smaller than this budget, never larger.
	if elems := MaxBlockwiseIntermediateElems(len(b.starts), b.ks, len(ends)); elems > b.maxBlockElems {
		b.maxBlockElems = elems
	}
	b.scoredBlocks++
	return scored, true
}

// pairScore mirrors the reference's per-pair validity conjunction. The score is
// computed even for a vetoed pair (the caller may have already paid for it), but
// ok is false unless the start slot is real, the end is strictly after the start
// index, and admiss agrees.
func (b *BlockwiseTopK) pairScore(start BlockwiseStartSlot, end int, endScore float32, admiss endScoreFunc) (float32, bool) {
	score := start.Score + endScore
	if admiss != nil {
		s, ok := admiss(start, end, endScore)
		if !ok {
			return s, false
		}
		score = s
	}
	if !start.Valid || end <= start.Index {
		return score, false
	}
	return score, true
}

// merge folds (end, score) into the running top-k for slot q with the reference's
// stable-descending semantics: the concatenation of the running entries and the
// one new entry is sorted by score descending, ties broken by concatenation
// order (running first, then the new entry), and the top Ks are retained. An
// equal-scoring new end therefore cannot displace an existing running end, and it
// lands after any earlier end with the same score, which is exactly the
// reference's "stable ties retain concatenation order" contract.
func (b *BlockwiseTopK) merge(q, end int, score float32) {
	cur := b.scores[q]
	// Fast path: the new entry is not better than the current worst, and ties
	// keep the running entry, so nothing moves.
	if score <= cur[len(cur)-1] {
		return
	}
	// Insertion into a descending run: walk left while the preceding entry has a
	// STRICTLY lower score, then insert. Stopping on an equal score is the whole
	// stable-tie contract: the new entry lands after every existing entry it ties
	// with, so an earlier end keeps the better rank.
	pos := len(cur) - 1
	for pos > 0 && cur[pos-1] < score {
		pos--
	}
	copy(cur[pos+1:], cur[pos:len(cur)-1])
	copy(b.ends[q][pos+1:], b.ends[q][pos:len(b.ends[q])-1])
	cur[pos] = score
	b.ends[q][pos] = end
}

// Results returns the per-start top-k in descending score order. Entries that
// never received a real candidate retain Score == blockwiseMaskLogit, so a caller
// can tell a filled slot from an unfilled one. The returned slices are copies; the
// accumulator is not mutated.
func (b *BlockwiseTopK) Results() [][]BlockwisePair {
	if b == nil {
		return nil
	}
	out := make([][]BlockwisePair, b.q)
	for q := range out {
		out[q] = make([]BlockwisePair, b.ks)
		for r := 0; r < b.ks; r++ {
			out[q][r] = BlockwisePair{Start: b.starts[q].Index, End: b.ends[q][r], Score: b.scores[q][r]}
		}
	}
	return out
}

// ConditionalElements is the running total of conditional pair scores computed
// across all blocks — the reference's `conditional_elems` work-accounting stat.
func (b *BlockwiseTopK) ConditionalElements() int {
	if b == nil {
		return 0
	}
	return b.conditionalElems
}

// MaxBlockElements is the largest single block tensor the accumulator
// materialized, in elements: Q*Ks*end_block_size for the widest block seen (the
// reference's [B,Q,Ks,e] block shape). It is the linear-memory witness — for a
// fixed Q and Ks it grows with end_block_size, not with the full end count L.
// The accumulator's own largest live slice is a [Ks+1] merge window per slot, so
// this is an upper bound on the true live set, never an understatement.
func (b *BlockwiseTopK) MaxBlockElements() int {
	if b == nil {
		return 0
	}
	return b.maxBlockElems
}

// BlocksScored reports how many ScoreBlock calls the accumulator has merged.
func (b *BlockwiseTopK) BlocksScored() int {
	if b == nil {
		return 0
	}
	return b.scoredBlocks
}

// MaxBlockwiseIntermediateElems returns the largest number of elements the
// blockwise ranker materializes for Q start slots, Ks retained ends, and a block
// of blockSize ends: Q*Ks*blockSize. It is the explicit linear-memory bound a
// test asserts against, mirroring the prefix kernel's MaxIntervalIntermediateLen;
// the full-matrix alternative a caller must not need is Q*Ks*L.
func MaxBlockwiseIntermediateElems(q, ks, blockSize int) int {
	if q <= 0 || ks <= 0 || blockSize <= 0 {
		return 0
	}
	return q * ks * blockSize
}

// BlockwiseTopKReference is the O(N log N) full-candidate reference the streaming
// ranker must equal: it scores every (start, end) pair once, then keeps each
// start slot's top Ks by score. It exists so a test (and a caller that needs
// ground truth) can check the blockwise result without reusing the streaming
// merge, and so the equivalence witness does not depend on the implementation's
// own ordering.
//
// It applies the same admission predicate as ScoreBlock (start valid, end >
// start index, admiss) and the same stable tie order: equal scores retain the
// (slot, end) enumeration order the reference would produce.
func BlockwiseTopKReference(starts []BlockwiseStartSlot, ends []int, endScores []float32, ks int, admiss endScoreFunc) [][]BlockwisePair {
	if ks <= 0 {
		return nil
	}
	out := make([][]BlockwisePair, len(starts))
	for q, start := range starts {
		kept := make([]BlockwisePair, 0, len(ends))
		for i, e := range ends {
			score, ok := refPairScore(start, e, endScores[i], admiss)
			if !ok {
				continue
			}
			kept = append(kept, BlockwisePair{Start: start.Index, End: e, Score: score})
		}
		// Stable selection sort of the top Ks by score descending: a strictly
		// greater score moves ahead, a tie keeps the earlier (lower-index)
		// candidate first, matching the reference's stable sort.
		for i := 0; i < len(kept) && i < ks; i++ {
			best := i
			for j := i + 1; j < len(kept); j++ {
				if kept[j].Score > kept[best].Score {
					best = j
				}
			}
			kept[i], kept[best] = kept[best], kept[i]
		}
		if len(kept) > ks {
			kept = kept[:ks]
		}
		out[q] = kept
	}
	return out
}

// refPairScore is the reference's per-pair score+admission, shared by the
// reference so the witness compares against one independent predicate rather
// than re-deriving it.
func refPairScore(start BlockwiseStartSlot, end int, endScore float32, admiss endScoreFunc) (float32, bool) {
	score := start.Score + endScore
	if admiss != nil {
		s, ok := admiss(start, end, endScore)
		if !ok {
			return s, false
		}
		score = s
	}
	if !start.Valid || end <= start.Index {
		return score, false
	}
	return score, true
}
