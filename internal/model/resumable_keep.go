package model

// Adapted from modular/modular max/python/max/pipelines/kv_cache/paged_kv_cache/jenga_block_manager.py@88ab8913af6b9298fe914e0f20d87a49e4172d0d (Apache-2.0 WITH LLVM-exception; Copyright (c) 2026 Modular Inc.). Adapted to fak-owned Go.
//
// Re-expressed in idiomatic Go as an engine-agnostic resumable-boundary and
// keep-trim policy for paged prefix caches; no upstream code blocks are copied.
// The hit rules follow the upstream shared prefix-hit core (rust_kv/kv-core
// prefix_hit.rs at the same pin): a full group's hit is the run from the root;
// a sliding-window group's hit is the deepest stopping point whose trailing
// window it holds complete; a recurrent leaf's hit is its deepest published
// boundary. The align-manager plumbing in paged_prefix_cow.go carries the state
// this policy selects over; wiring that seam is a later leaf.

// GroupShape classifies how a cache group reads its history, which selects the
// prefix-hit rule that validates a resumable boundary against it. Only
// cacheable groups participate: scratch leaves are never published and have no
// opinion on a hit.
type GroupShape int

const (
	// ShapeFull reads its whole history: every committed block is needed by
	// every later hit, so such leaves offload as each block commits.
	ShapeFull GroupShape = iota
	// ShapeSlidingWindow reads only the trailing window of blocks.
	ShapeSlidingWindow
	// ShapeRecurrent holds one state per published boundary rather than a run
	// of blocks; the deepest resident boundary is a complete hit on its own.
	ShapeRecurrent
)

// CacheGroup is one cache group's published-block view plus the hit rule its
// shape selects. Rows are positional: Published[i] reports whether block i of
// the request's row is published (committed into the prefix cache). A caller
// builds one value per group per call from its live rows; nothing here owns
// engine state.
type CacheGroup struct {
	// Shape selects the hit rule.
	Shape GroupShape
	// BlocksInWindow is the window in whole blocks, at least 1, for
	// ShapeSlidingWindow only; ignored by the other shapes.
	BlocksInWindow int
	// Published holds the per-block publication flags for the request's row.
	Published []bool
}

// LongestHit applies the group's prefix-hit rule to the first numBlocks
// blocks of its row and returns how many of them the group can serve.
func (g CacheGroup) LongestHit(numBlocks int) int {
	if numBlocks <= 0 {
		return 0
	}
	n := numBlocks
	if n > len(g.Published) {
		n = len(g.Published)
	}
	switch g.Shape {
	case ShapeFull:
		return fullLongestHit(g.Published[:n])
	case ShapeSlidingWindow:
		return slidingWindowLongestHit(g.Published[:n], g.BlocksInWindow)
	case ShapeRecurrent:
		return recurrentLongestHit(g.Published[:n])
	default:
		return 0
	}
}

// HasFullPrefixDependency reports whether every committed block of this group
// is needed by every later hit. Full-history groups stream: they offload as
// each block commits. The others offload only at checkpoints.
func (g CacheGroup) HasFullPrefixDependency() bool {
	return g.Shape == ShapeFull
}

// fullLongestHit returns the run of published blocks from the root, which is
// what a full group reads. The result is downward-closed: a group serving n
// also serves anything shorter.
func fullLongestHit(published []bool) int {
	for i, ok := range published {
		if !ok {
			return i
		}
	}
	return len(published)
}

// slidingWindowLongestHit returns the deepest stopping point whose window the
// group holds complete. The window sits at the END of the candidate, so the
// answer is found by walking backwards counting a consecutive run: the first
// place the run reaches blocksInWindow is the deepest stopping point that
// works. A shallower one may not -- its window covers different blocks -- so
// this cannot be found by shortening a full-attention answer. A run that
// survives to index 0 is a hit of just that run, with no window check:
// nothing sits below the root to be missing.
func slidingWindowLongestHit(published []bool, blocksInWindow int) int {
	if blocksInWindow < 1 {
		blocksInWindow = 1
	}
	run := 0
	for idx := len(published) - 1; idx >= 0; idx-- {
		if !published[idx] {
			run = 0
			continue
		}
		run++
		if run >= blocksInWindow {
			return idx + run
		}
	}
	return run
}

// recurrentLongestHit returns the deepest published boundary plus one. A
// recurrent state is published at one boundary rather than as a run of
// blocks, so any resident boundary is a complete hit on its own; a shallower
// candidate is answered from below it.
func recurrentLongestHit(published []bool) int {
	for idx := len(published) - 1; idx >= 0; idx-- {
		if published[idx] {
			return idx + 1
		}
	}
	return 0
}

// ResumableKeepPolicy tracks where a request can resume and selects the keep
// boundaries its caches trim to. It is pure policy: no engine state, no pools,
// no locks.
type ResumableKeepPolicy struct {
	// BlockSize is tokens per page/block.
	BlockSize int
	// HandlesEndOfPrompt reports whether the checkpointer handles the
	// end-of-prompt event. Only then does a replay boundary become a
	// candidate and recurrent-state publish get suppressed.
	HandlesEndOfPrompt bool
	// HandlesAnyCheckpoint reports whether the checkpointer takes any
	// checkpoint event at all. When it takes none, every cacheable leaf
	// streams instead of waiting for a checkpoint.
	HandlesAnyCheckpoint bool
}

// NewResumableKeepPolicy creates the policy. A non-positive blockSize is
// clamped to 1.
func NewResumableKeepPolicy(blockSize int, handlesEndOfPrompt, handlesAnyCheckpoint bool) *ResumableKeepPolicy {
	if blockSize <= 0 {
		blockSize = 1
	}
	return &ResumableKeepPolicy{
		BlockSize:            blockSize,
		HandlesEndOfPrompt:   handlesEndOfPrompt,
		HandlesAnyCheckpoint: handlesAnyCheckpoint,
	}
}

// ResumableState is where a request can currently resume.
type ResumableState struct {
	// Boundaries are page boundaries, counted in blocks, lowest first. Each
	// cache keeps the blocks needed to resume at any of them.
	Boundaries []int
	// CommittedIdx is the request's committed token index when Boundaries was
	// computed. If it has not changed, nothing new was committed, so
	// Boundaries is reused. It is -1 before the first step.
	CommittedIdx int
}

// NewResumableState returns the initial per-request state.
func NewResumableState() ResumableState {
	return ResumableState{CommittedIdx: -1}
}

// Step recomputes where the request can resume after a step that advanced its
// committed index to committedIdx tokens, and records it. It returns the
// validated keep boundaries, lowest first, which the caller passes to each
// group's advance/trim.
//
// The keep set is the committed boundary plus, on the step that commits past
// the end of a page-aligned prompt, the distinct boundary a replay of it
// resumes from; each candidate only if every cacheable group holds what
// resuming there needs. When nothing new was committed and at most one
// boundary was recorded, the previous boundaries are reused unchanged.
func (p *ResumableKeepPolicy) Step(req *ResumableState, committedIdx, promptLength int, groups ...CacheGroup) []int {
	if req.CommittedIdx == committedIdx && len(req.Boundaries) <= 1 {
		return req.Boundaries
	}
	committedBlocks := committedIdx / p.BlockSize
	candidates := make([]int, 0, 2)
	if p.HandlesEndOfPrompt {
		if replay, ok := DistinctReplayBoundary(promptLength, req.CommittedIdx, committedIdx, p.BlockSize); ok {
			candidates = append(candidates, replay)
		}
	}
	candidates = append(candidates, committedBlocks)
	keep := make([]int, 0, len(candidates))
	for _, b := range candidates {
		if p.ResumableAt(b, committedBlocks, groups...) {
			keep = append(keep, b)
		}
	}
	req.Boundaries = keep
	req.CommittedIdx = committedIdx
	return keep
}

// ResumableAt reports whether every cacheable group can resume after
// numBlocks blocks. Only published blocks count; each group applies the same
// rule as its prefix-cache lookup, and the candidate is accepted only when
// every group's longest published hit equals the candidate depth.
func (p *ResumableKeepPolicy) ResumableAt(numBlocks, committedBlocks int, groups ...CacheGroup) bool {
	if numBlocks <= 0 || numBlocks > committedBlocks {
		return false
	}
	for _, g := range groups {
		if g.LongestHit(numBlocks) != numBlocks {
			return false
		}
	}
	return true
}

// DistinctReplayBoundary returns the block boundary a replay of a page-aligned
// prompt resumes from, when the prompt ends within (start, end] tokens: the
// start of the page holding the prompt's last token, which a replay
// recomputes. Only a prompt that ends on a page boundary yields one: unless
// it does, that is the boundary its end commits, and there is nothing
// distinct. ok is false when there is no distinct replay boundary, including
// when the boundary would be zero (nothing below the first page).
func DistinctReplayBoundary(promptLength, start, end, blockSize int) (boundary int, ok bool) {
	if promptLength%blockSize != 0 || !(start < promptLength && promptLength <= end) {
		return 0, false
	}
	b := promptLength/blockSize - 1
	if b <= 0 {
		return 0, false
	}
	return b, true
}

// PublishesStates reports whether the forward that just ran may publish its
// recurrent state. Not when it ends a page-aligned prompt whose replay can
// already resume; the request keeps that state and continues in place.
// committedIdx is how far the published prefix reaches before this step's
// commit; processedLength includes the tokens the forward just wrote.
func (p *ResumableKeepPolicy) PublishesStates(req *ResumableState, committedIdx, processedLength, promptLength int) bool {
	if !p.HandlesEndOfPrompt {
		return true
	}
	replay, ok := DistinctReplayBoundary(promptLength, committedIdx, processedLength, p.BlockSize)
	if !ok {
		return true
	}
	for _, b := range req.Boundaries {
		if b == replay {
			return false
		}
	}
	return true
}

// SlidingWindowTrim returns the lowest block index a sliding-window group
// retains after advancing to committedDepth blocks with the given keep
// boundaries: the window ends at the lowest keep boundary, which also covers
// every higher one; with nothing in keep, it ends at the committed depth.
// Blocks below FirstNeeded are freed; the window [FirstNeeded, End) is kept.
func SlidingWindowTrim(blocksInWindow, committedDepth int, keep ...int) (firstNeeded, end int) {
	end = committedDepth
	for _, k := range keep {
		if k < end {
			end = k
		}
	}
	firstNeeded = end - blocksInWindow
	if firstNeeded < 0 {
		firstNeeded = 0
	}
	return firstNeeded, end
}

// FullGroupTrim reports what a full-history group frees after advancing:
// nothing. This group reads its whole history, so it keeps every block.
func FullGroupTrim(committedDepth int, keep ...int) (firstNeeded, end int) {
	return 0, committedDepth
}

// StreamedLeaves reports, per cacheable group, whether its leaves stream:
// offloaded as each block commits rather than only at a checkpoint. Every
// full-prefix-dependency group streams; every other group streams too when the
// checkpointer takes no checkpoint events at all.
func (p *ResumableKeepPolicy) StreamedLeaves(groups ...CacheGroup) []bool {
	streamed := make([]bool, len(groups))
	for i, g := range groups {
		streamed[i] = g.HasFullPrefixDependency() || !p.HandlesAnyCheckpoint
	}
	return streamed
}
