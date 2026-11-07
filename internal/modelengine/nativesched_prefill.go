package modelengine

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// The resident Qwen hybrid panel path requires a 16-token fresh chunk. Later
// append chunks may be smaller once the session owns prefix state. Refusing a
// smaller configured ceiling keeps the scheduler from silently selecting the
// token-loop fallback on the first chunk.
const nativeQwenPrefillMinChunkTokens = 16

// nativeServingPrefillTokensPerIteration is the fixed per-iteration prefill token
// ceiling the serving admission path arms for a resident-Qwen model (see
// Engine.nativeScheduler). It is a bounded serving default, not a tunable: it is
// deliberately below the serving prompt cap (maxPromptTokens) so an ordinary
// serving prompt can actually exceed it and be split across scheduler iterations
// rather than prefilled whole inside one. No configuration surface is added; the
// existing per-iteration mechanism — and its documented refusal below
// nativeQwenPrefillMinChunkTokens — is reused unchanged.
const nativeServingPrefillTokensPerIteration = 32

var errNativeSchedulerLaneNotDecodeReady = errors.New("modelengine: native scheduler lane is not decode-ready")

// residentQ4KPrefillCapability is minted from model-owned residency when the
// scheduler is constructed. Request/session q4k flags remain routing intent and
// cannot manufacture this scheduler-local proof. It also owns the shared
// per-step prefill token budget for the bounded chunked-prefill mechanism; the
// budget's lifetime matches the capability's (bounded prefill exists only when
// the capability is minted) and every access happens under NativeScheduler.mu.
type residentQ4KPrefillCapability struct {
	model  *model.Model
	budget nativePrefillStepBudget
}

type schedLaneState uint8

const (
	// Keep DECODE as the zero value so imported and test-constructed historical
	// lanes remain decode-ready without new initialization.
	schedLaneDecode schedLaneState = iota
	schedLanePrefilling
)

type nativeSchedulerEventKind uint8

const (
	nativeSchedulerEventPrefill nativeSchedulerEventKind = iota + 1
	nativeSchedulerEventTransition
	nativeSchedulerEventDecode
)

// nativeSchedulerEvent is a scheduler-local deterministic witness seam. It is
// intentionally unexported: production observability remains in the existing
// scheduler receipts, while focused tests can prove exact iteration ordering
// without process-global counters or timing.
type nativeSchedulerEvent struct {
	Iteration  uint64
	Kind       nativeSchedulerEventKind
	Lane       *schedLane
	State      schedLaneState
	ChunkStart int
	ChunkLen   int
	Token      int
}

// PrefixTree abstracts prefix residency queries for NativeScheduler.
// *radixkv.Tree satisfies this interface.
type PrefixTree interface {
	MatchLen(tokens []int) int
}

// PrefixStats records prefix lookup metrics.
type PrefixStats struct {
	Lookups       uint64
	Hits          uint64
	FullHits      uint64
	PartialHits   uint64
	Misses        uint64
	MatchedTokens uint64
}

type prefixHitInfo struct {
	matched    int
	rawMatched int // token-exact match before page-aligned truncation
	boundary   *radixkv.Node
	ownerTree  *radixkv.Tree // tree that acquired boundary, independent of later setters
	fullHit    bool          // raw match covered the entire prompt
	applied    bool
}

type nativePrefixState struct {
	mu            sync.RWMutex
	tree          *radixkv.Tree
	prefixTree    PrefixTree
	deterministic bool
	// qwenPrefillChunkAlign is the deterministic-mode prefill chunk boundary
	// alignment A (0 = off). It sits beside deterministic because both are
	// scheduler-level serving flags read while prefix state is live.
	qwenPrefillChunkAlign int
	stats                 PrefixStats
	chunkedExactReuses    uint64
	chunkedPrefixReuses   uint64
	holderLookups         map[*model.Session]*prefixHitInfo
	laneLookups           map[*schedLane]*prefixHitInfo
}

func (s *NativeScheduler) getPrefixState() *nativePrefixState {
	if s == nil {
		return nil
	}
	s.prefixStateMu.RLock()
	defer s.prefixStateMu.RUnlock()
	return s.prefixState
}

func (s *NativeScheduler) getOrCreatePrefixState() *nativePrefixState {
	if s == nil {
		return nil
	}
	s.prefixStateMu.Lock()
	defer s.prefixStateMu.Unlock()
	st := s.prefixState
	if st == nil {
		st = &nativePrefixState{
			holderLookups: make(map[*model.Session]*prefixHitInfo),
			laneLookups:   make(map[*schedLane]*prefixHitInfo),
		}
		s.prefixState = st
	}
	return st
}

// SetRadixKV installs a RadixKV prefix tree for prefix-cache-aware prefill scheduling.
func (s *NativeScheduler) SetRadixKV(tree *radixkv.Tree) {
	st := s.getOrCreatePrefixState()
	st.mu.Lock()
	defer st.mu.Unlock()
	st.tree = tree
	st.prefixTree = nil
	if tree != nil {
		st.prefixTree = tree
	}
}

// RadixKV returns the installed RadixKV prefix tree, if any.
func (s *NativeScheduler) RadixKV() *radixkv.Tree {
	st := s.getPrefixState()
	if st == nil {
		return nil
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.tree
}

// SetDeterministicMode toggles deterministic serving. When enabled the
// scheduler stops inserting finished prompts into the radix KV tree and skips
// prefix-hit application, because cache-hit sets that depend on eviction timing
// change float accumulation order. The zero value (false) preserves historical
// reuse. Adapted from SGLang python/sglang/srt/mem_cache/radix_cache.py:483-484
// (flag disable_finished_insert, cache_init_params.py:33@00a9a81b, Apache-2.0).
func (s *NativeScheduler) SetDeterministicMode(enabled bool) {
	st := s.getOrCreatePrefixState()
	if st == nil {
		return
	}
	st.mu.Lock()
	st.deterministic = enabled
	st.mu.Unlock()
}

// DeterministicMode reports whether deterministic serving is enabled.
func (s *NativeScheduler) DeterministicMode() bool {
	st := s.getPrefixState()
	if st == nil {
		return false
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.deterministic
}

// SetQwenPrefillChunkAlign configures the deterministic-mode prefill chunk
// boundary alignment A. Zero (default) preserves historical arbitrary-token
// chunk ends. A positive A constrains every non-final chunk end to a multiple
// of A measured from position 0, never below A; the final chunk always ends at
// the prompt tail. Negative values are refused as zero. Adapted from SGLang
// python/sglang/srt/managers/scheduler.py:1690-1722 (truncation_align_size,
// @00a9a81b, Apache-2.0).
func (s *NativeScheduler) SetQwenPrefillChunkAlign(align int) {
	st := s.getOrCreatePrefixState()
	if st == nil {
		return
	}
	if align < 0 {
		align = 0
	}
	st.mu.Lock()
	st.qwenPrefillChunkAlign = align
	st.mu.Unlock()
}

// QwenPrefillChunkAlign reports the configured deterministic-mode prefill
// chunk boundary alignment (0 = off).
func (s *NativeScheduler) QwenPrefillChunkAlign() int {
	st := s.getPrefixState()
	if st == nil {
		return 0
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.qwenPrefillChunkAlign
}

// deterministicChunkAlign reports the chunk boundary alignment active for the
// next prefill step: the configured A when deterministic serving is enabled,
// otherwise 0. Caller may hold s.mu; the read follows the established
// s.mu -> prefix-state lock order.
func (s *NativeScheduler) deterministicChunkAlign() int {
	st := s.getPrefixState()
	if st == nil {
		return 0
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	if !st.deterministic {
		return 0
	}
	return st.qwenPrefillChunkAlign
}

// alignedPrefillChunkEnd applies the deterministic chunk boundary alignment to
// one prefill step. Given the chunk start, its raw (unclamped) budget end,
// the prompt length, and alignment A > 0, it returns the effective end:
// the largest multiple of A at or below the raw end, measured from position 0
// and never below A. When that multiple cannot advance the cursor (alignment
// finer than the remaining distance), the next multiple past start is used so
// the lane stays live. An aligned end reaching or passing the prompt tail
// becomes the final chunk ending exactly at the tail. A <= 0 leaves the raw
// end unchanged.
func alignedPrefillChunkEnd(start, rawEnd, promptLen, align int) int {
	if align <= 0 {
		return rawEnd
	}
	aligned := rawEnd / align * align
	if aligned <= start {
		aligned = (start/align + 1) * align
	}
	if aligned >= promptLen {
		return promptLen
	}
	return aligned
}

// SetPrefixTree installs a generic PrefixTree for prefix-cache-aware prefill scheduling.
func (s *NativeScheduler) SetPrefixTree(pt PrefixTree) {
	st := s.getOrCreatePrefixState()
	st.mu.Lock()
	defer st.mu.Unlock()
	st.prefixTree = pt
	st.tree = nil
	if tree, ok := pt.(*radixkv.Tree); ok {
		st.tree = tree
		if tree == nil {
			st.prefixTree = nil
		}
	}
}

// PrefixTree returns the installed PrefixTree, if any.
func (s *NativeScheduler) PrefixTree() PrefixTree {
	st := s.getPrefixState()
	if st == nil {
		return nil
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.prefixTree
}

// PrefixStats returns metrics for prefix lookups performed by the scheduler.
func (s *NativeScheduler) PrefixStats() PrefixStats {
	st := s.getPrefixState()
	if st == nil {
		return PrefixStats{}
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.stats
}

// ChunkedExactPrefillReuses counts completed full-prompt copies between running
// host Qwen lanes. It is separate from synchronous prefix-flight role counters.
func (s *NativeScheduler) ChunkedExactPrefillReuses() uint64 {
	st := s.getPrefixState()
	if st == nil {
		return 0
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.chunkedExactReuses
}

func (s *NativeScheduler) lookupPrefix(prompt []int) (int, *radixkv.Node, *radixkv.Tree) {
	st := s.getPrefixState()
	if st == nil || len(prompt) == 0 {
		return 0, nil, nil
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.tree != nil {
		boundary, matched := st.tree.Lookup(prompt)
		st.stats.Lookups++
		if matched > 0 {
			st.stats.Hits++
			st.stats.MatchedTokens += uint64(matched)
			if matched == len(prompt) {
				st.stats.FullHits++
			} else {
				st.stats.PartialHits++
			}
		} else {
			st.stats.Misses++
		}
		if matched == 0 {
			// Lookup also leases the root on a miss; no holder will own it.
			st.tree.Done(boundary)
			return 0, nil, nil
		}
		return matched, boundary, st.tree
	}
	if st.prefixTree != nil {
		matched := st.prefixTree.MatchLen(prompt)
		st.stats.Lookups++
		if matched > 0 {
			st.stats.Hits++
			st.stats.MatchedTokens += uint64(matched)
			if matched == len(prompt) {
				st.stats.FullHits++
			} else {
				st.stats.PartialHits++
			}
		} else {
			st.stats.Misses++
		}
		return matched, nil, nil
	}
	return 0, nil, nil
}

// lookupPrefixPageAligned applies SGLang's page-aligned match truncation on top of
// lookupPrefix: paged KV is accounted per block, so a hit that ends mid-block cannot
// be reused and the reusable depth is floored to a multiple of the block size in
// tokens. A raw match shorter than one block is treated as a miss, and its boundary
// lease is released like any other miss. The raw (pre-truncation) match is
// returned alongside so admission can tell a floored full-prompt hit apart from
// a genuinely partial one. Adapted from SGLang
// python/sglang/srt/mem_cache/radix_cache.py:440-443,467@00a9a81
// (key.page_aligned(page_size), Apache-2.0).
func (s *NativeScheduler) lookupPrefixPageAligned(prompt []int, page int) (int, *radixkv.Node, *radixkv.Tree, int) {
	matched, boundary, tree := s.lookupPrefix(prompt)
	if matched == 0 || page <= 1 {
		return matched, boundary, tree, matched
	}
	aligned := matched / page * page
	if aligned == 0 {
		if tree != nil && boundary != nil {
			tree.Done(boundary)
		}
		return 0, nil, nil, matched
	}
	return aligned, boundary, tree, matched
}

func (s *NativeScheduler) recordPrefixLookup(sess *model.Session, prompt []int, matched, rawMatched int, boundary *radixkv.Node, ownerTree *radixkv.Tree) {
	st := s.getPrefixState()
	if st == nil || sess == nil {
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	info := &prefixHitInfo{
		matched:    matched,
		rawMatched: rawMatched,
		boundary:   boundary,
		ownerTree:  ownerTree,
		fullHit:    rawMatched == len(prompt),
	}
	st.holderLookups[sess] = info
}

func (s *NativeScheduler) getPrefixHitInfo(ln *schedLane) *prefixHitInfo {
	st := s.getPrefixState()
	if st == nil || ln == nil {
		return nil
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if info, ok := st.laneLookups[ln]; ok {
		return info
	}
	if ln.sess != nil {
		if info, ok := st.holderLookups[ln.sess]; ok {
			st.laneLookups[ln] = info
			return info
		}
	}
	return nil
}

func (s *NativeScheduler) clearPrefixLookup(sess *model.Session) {
	st := s.getPrefixState()
	if st == nil || sess == nil {
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if info, ok := st.holderLookups[sess]; ok {
		// A retired holder must not leave its lane/history rooted in prefix
		// metadata. Preserve every association owned by another live holder.
		for lane, associated := range st.laneLookups {
			if associated == info {
				delete(st.laneLookups, lane)
			}
		}
		if info.boundary != nil && info.ownerTree != nil {
			info.ownerTree.Done(info.boundary)
			info.boundary = nil
		}
		delete(st.holderLookups, sess)
	}
}

func (s *NativeScheduler) maybeInsertRadixKV(prompt []int, sess *model.Session, logits []float32) {
	st := s.getPrefixState()
	if st == nil || sess == nil || sess.Cache == nil {
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.tree == nil || st.deterministic {
		return
	}
	boundary, matched := st.tree.Lookup(prompt)
	if matched < len(prompt) {
		leaf := st.tree.InsertWithLogits(boundary, prompt[matched:], sess.Cache, logits)
		st.tree.Done(leaf)
	} else if len(logits) > 0 && boundary != nil && boundary.KV() != nil && boundary.KV().Len() == len(prompt) && len(boundary.Logits()) == 0 {
		// Complete a state-only boundary after bounded replay so subsequent
		// exact hits reuse its terminal logits rather than replaying again.
		leaf := st.tree.InsertWithLogits(boundary, nil, sess.Cache, logits)
		st.tree.Done(leaf)
	} else {
		st.tree.Done(boundary)
	}
}

// SetQwenPrefillMaxTokensPerIteration enables bounded scheduler-owned prefill for
// supported resident Qwen Q4_K admissions. The zero value and tokens<=0 preserve
// synchronous admission. A positive ceiling below the resident fresh-panel minimum
// is refused and leaves the feature disabled rather than silently falling back to
// token-loop prefill. Configure this before the first Admit.
func (s *NativeScheduler) SetQwenPrefillMaxTokensPerIteration(tokens int) error {
	if tokens > 0 && tokens < nativeQwenPrefillMinChunkTokens {
		s.mu.Lock()
		s.qwenPrefillTokens = 0
		s.mu.Unlock()
		return fmt.Errorf("modelengine: Qwen prefill max tokens per iteration %d is below resident minimum %d", tokens, nativeQwenPrefillMinChunkTokens)
	}
	if tokens < 0 {
		tokens = 0
	}
	s.mu.Lock()
	s.qwenPrefillTokens = tokens
	s.mu.Unlock()
	s.signal()
	return nil
}

// qwenPrefillStepBudgetLocked returns the shared step budget for iteration,
// re-arming it from the live ceiling when the step advances. A nil capability
// (unsupported model) reports ok=false so callers keep the historical lane-only
// sizing. Caller holds s.mu.
func (s *NativeScheduler) qwenPrefillStepBudgetLocked(iteration uint64) (*nativePrefillStepBudget, bool) {
	if s.qwenPrefillCap == nil {
		return nil, false
	}
	b := &s.qwenPrefillCap.budget
	if b.step != iteration {
		b.reset(iteration, s.qwenPrefillTokens)
	}
	return b, true
}

func (s *NativeScheduler) qwenPrefillChunkBudget(prep schedPrepare, sess *model.Session, promptLen int) int {
	s.mu.Lock()
	budget := s.qwenPrefillTokens
	page := s.blockTokensLocked()
	s.mu.Unlock()
	if budget < nativeQwenPrefillMinChunkTokens ||
		!prep.q4k ||
		sess == nil ||
		sess.M != s.m ||
		!sess.Q4K ||
		s.qwenPrefillCap == nil ||
		s.qwenPrefillCap.model != s.m ||
		s.qwenPrefillCap.model != sess.M ||
		sess.Backend != nil ||
		sess.Cache == nil ||
		sess.Cache.Len() != 0 ||
		!nativeQwenResidentAppendEligible(s.m.Cfg, promptLen) {
		return 0
	}

	// Acceptance criteria 1: query prefix cache depth during admission, truncated
	// to whole blocks so a mid-block hit can never end a reusable span.
	matched, boundary, ownerTree, rawMatched := s.lookupPrefixPageAligned(prep.prompt, page)
	if matched > 0 {
		s.recordPrefixLookup(sess, prep.prompt, matched, rawMatched, boundary, ownerTree)
	}

	// Calculate chunked prefill step sizes based on uncached suffix length.
	suffixLen := promptLen - matched
	if suffixLen <= 0 {
		return budget
	}

	stepSize := budget
	if suffixLen < stepSize && suffixLen >= nativeQwenPrefillMinChunkTokens {
		stepSize = suffixLen
	}
	return stepSize
}

// nativeQwenResidentAppendEligible exactly mirrors the model-owned
// q4kQwen35HybridPrefillOK architecture and diagnostic gate. Keeping the scheduler
// qualification no broader than the operation it intends to schedule ensures every
// unsupported Qwen configuration stays on historical synchronous admission. The
// qwen3_5_moe hybrid is admitted like the model gate admits it: each chunk's Prefill
// still applies the model's MoE session limits and falls back to the token loop there.
func nativeQwenResidentAppendEligible(cfg model.Config, promptLen int) bool {
	if os.Getenv("FAK_QWEN35_PREFILL_TOKEN_LOOP") != "" {
		return false
	}
	if !cfg.IsQwen35Hybrid() ||
		promptLen < nativeQwenPrefillMinChunkTokens ||
		!cfg.AttnOutputGate ||
		cfg.DenseMLP ||
		cfg.Alibi ||
		!cfg.NormGain1p ||
		cfg.LayerNorm ||
		cfg.BlockTopology != model.PreNorm {
		return false
	}
	for layer := 0; layer < cfg.NumLayers && layer < len(cfg.RopeThetaPerLayer); layer++ {
		if theta := cfg.RopeThetaPerLayer[layer]; theta != 0 && theta != cfg.RopeTheta {
			return false
		}
	}
	return true
}

// enforcePreemptionPreservingPrefillLocked keeps the existing block-budget loop
// live, but limits victim selection to decode-ready lanes. A partially-prefilled
// lane cannot be swapped or recomputed from prompt+generated history because its
// prompt cursor and recurrent state are not represented by that restore contract.
func (s *NativeScheduler) enforcePreemptionPreservingPrefillLocked() {
	s.applyPrefixHitsLocked()
	if !s.preemptionEnabledLocked() {
		return
	}
	for s.usedKVBlocksLocked() > s.preemption.MaxBlocks {
		idx := s.prefillSafePreemptibleLaneLocked()
		if idx < 0 {
			return
		}
		ln := s.lanes[idx]
		if err := s.preemptLaneLocked(ln); err != nil {
			ln.finish(nil, err)
		}
		s.lanes = append(s.lanes[:idx], s.lanes[idx+1:]...)
	}
}

func (s *NativeScheduler) applyPrefixHitsLocked() {
	if st := s.getPrefixState(); st != nil {
		st.mu.RLock()
		deterministic := st.deterministic
		st.mu.RUnlock()
		if deterministic {
			return
		}
	}
	for _, ln := range s.lanes {
		if ln == nil || ln.state != schedLanePrefilling || ln.promptCursor != 0 {
			continue
		}
		info := s.getPrefixHitInfo(ln)
		if info == nil || info.applied {
			continue
		}
		depth := installPrefixHitLocked(ln, info)
		if info.fullHit && depth == len(ln.prompt) {
			if err := s.setLaneStateLocked(ln, schedLaneDecode); err != nil {
				ln.finish(nil, err)
				continue
			}
			ln.promptCursor = len(ln.prompt)
			ln.promptLen = len(ln.prompt)
			ln.prefillChunkTokens = 0
			if info.boundary != nil && len(info.boundary.Logits()) > 0 {
				ln.logits = copyF32(info.boundary.Logits())
			}
			s.observeEvent(nativeSchedulerEvent{
				Iteration: s.iteration + 1,
				Kind:      nativeSchedulerEventTransition,
				Lane:      ln,
				State:     schedLaneDecode,
			})
		} else if depth > 0 {
			ln.promptCursor = depth
			ln.promptLen = depth
		}
	}
}

// installPrefixHitLocked marks info applied, installs a clone of its boundary KV
// into the lane session, and returns the prompt depth the lane may skip:
// min(info.matched, installed KV length), or 0 (cold prefill) when no KV was
// installed. The admitted match is a ceiling and is never grown later; adapted
// from SGLang python/sglang/srt/disaggregation/decode_hicache_mixin.py:256@f2de51e2
// (PR #41450, Apache-2.0).
func installPrefixHitLocked(ln *schedLane, info *prefixHitInfo) int {
	info.applied = true
	if info.boundary == nil || info.boundary.KV() == nil || ln.sess == nil {
		return 0
	}
	depth := min(info.matched, info.boundary.KV().Len())
	if info.fullHit && depth == len(ln.prompt) && len(info.boundary.Logits()) == 0 {
		// Recurrent KV cannot be rewound to recover terminal logits. Decline
		// before installing it so ordinary bounded prefill stays outside s.mu.
		return 0
	}
	if info.fullHit && len(info.boundary.Logits()) > 0 && info.boundary.KV().Len() == len(ln.prompt) {
		// Full-prompt hit whose boundary holds the complete span plus cached
		// terminal logits: consume it whole. Lookup splits edges so a match
		// always ends at a node boundary whose KV covers the matched prefix
		// from the root, so KV().Len() == len(prompt) proves the stored span IS
		// this prompt. Nothing is appended, so a hybrid Gated-DeltaNet
		// recurrence is used at its natural length and page alignment does not
		// shrink a span the tree holds in full.
		ln.sess.Cache = info.boundary.KV().Clone()
		return info.boundary.KV().Len()
	}
	if kvLen := info.boundary.KV().Len(); depth < kvLen {
		// The admitted depth ends mid-span (page-aligned truncation floors it
		// below the boundary node's stored KV). The lane appends at
		// promptCursor, so the installed cache must represent exactly depth
		// tokens. Mirror radixkv.split: a rewindable softmax-KV cache is
		// truncated; a hybrid Gated-DeltaNet recurrence holds one accumulated
		// state with no per-token journal and cannot be rolled back, so decline
		// and let the lane cold-prefill instead of appending onto a stale
		// recurrence.
		if err := info.boundary.KV().CanEvict(); err != nil {
			return 0
		}
	}
	ln.sess.Cache = info.boundary.KV().Clone()
	if depth < ln.sess.Cache.Len() {
		ln.sess.Cache.Truncate(depth)
	}
	return depth
}

// prefillSafePreemptibleLaneLocked reuses the configured historical victim picker
// against the decode-ready subset, then maps its answer back to the full running
// set. The temporary slice substitution is lock-local and preserves both the
// most-recent and cost-aware policies without making PREFILLING a valid victim.
func (s *NativeScheduler) prefillSafePreemptibleLaneLocked() int {
	original := s.lanes
	decode := make([]*schedLane, 0, len(original))
	originalIndexes := make([]int, 0, len(original))
	for i, ln := range original {
		if ln == nil || ln.state != schedLaneDecode {
			continue
		}
		decode = append(decode, ln)
		originalIndexes = append(originalIndexes, i)
	}
	if len(decode) == len(original) {
		return s.preemptibleLaneLocked()
	}
	if len(decode) == 0 {
		return -1
	}
	// The historical picker refuses a one-lane running set so preemption never
	// empties the scheduler. A protected PREFILLING lane means the scheduler still
	// has live work, so add an ineligible sentinel and let the same picker evaluate
	// the sole decode lane (including pin/cost-aware rules).
	selection := decode
	if len(selection) == 1 {
		selection = append(selection, nil)
	}
	s.lanes = selection
	defer func() { s.lanes = original }()
	idx := s.preemptibleLaneLocked()
	if idx < 0 || idx >= len(originalIndexes) {
		return -1
	}
	return originalIndexes[idx]
}

// advanceQwenPrefill performs at most one Session append call for one lane. It
// temporarily replaces the mutating session with a non-nil immutable stats shell
// so lock-taking observers count the lane without reaching Cache through
// laneKVBlocksLocked. The last published promptLen remains the synchronized live-KV
// accounting authority during that window. Model execution stays outside s.mu;
// only ownership transfer and completed-state publication take the lock.
func (s *NativeScheduler) advanceQwenPrefill(ln *schedLane, iteration uint64) {
	s.mu.Lock()
	if !s.livePrefillLaneLocked(ln) {
		s.mu.Unlock()
		return
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		s.mu.Unlock()
		return
	}

	// Apply any pending prefix cache hit before starting chunks.
	if ln.promptCursor == 0 {
		info := s.getPrefixHitInfo(ln)
		if info != nil && !info.applied {
			depth := installPrefixHitLocked(ln, info)
			if info.fullHit && depth == len(ln.prompt) {
				if err := s.setLaneStateLocked(ln, schedLaneDecode); err != nil {
					ln.finish(nil, err)
					s.mu.Unlock()
					return
				}
				ln.promptCursor = len(ln.prompt)
				ln.promptLen = len(ln.prompt)
				ln.prefillChunkTokens = 0
				if info.boundary != nil && len(info.boundary.Logits()) > 0 {
					ln.logits = copyF32(info.boundary.Logits())
				}
				s.mu.Unlock()
				s.observeEvent(nativeSchedulerEvent{
					Iteration: iteration,
					Kind:      nativeSchedulerEventTransition,
					Lane:      ln,
					State:     schedLaneDecode,
				})
				return
			} else if depth > 0 {
				ln.promptCursor = depth
				ln.promptLen = depth
			}
		}
	}

	start := ln.promptCursor
	if start < 0 || start >= len(ln.prompt) || ln.prefillChunkTokens < nativeQwenPrefillMinChunkTokens {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		s.mu.Unlock()
		return
	}
	// Shared step budget (mini-sglang PrefillAdder.token_budget@9a91cfa): the
	// chunk this lane may run now is capped by what the current step has not
	// yet spent across prefill lanes. A disarmed budget (zero ceiling) keeps
	// the historical lane-only sizing so an operator zero-disable never strands
	// an in-flight prefill. A lane that can draw neither its nominal chunk nor
	// a full resident minimum yields this step unchanged and resumes from the
	// same cursor next step - the resumable chunked_req, carried here by the
	// PREFILLING lane's cursor and resident KV.
	chunkTokens := ln.prefillChunkTokens
	if remain := len(ln.prompt) - start; chunkTokens > remain {
		chunkTokens = remain
	}
	if budget, ok := s.qwenPrefillStepBudgetLocked(iteration); ok && budget.total > 0 {
		take, _, err := budget.take(chunkTokens)
		if err != nil || (take < nativeQwenPrefillMinChunkTokens && take < chunkTokens) {
			s.mu.Unlock()
			return
		}
		chunkTokens = take
	}
	rawEnd := start + chunkTokens
	end := rawEnd
	// Deterministic serving constrains chunk boundaries, not only cache reuse:
	// align the non-final end before the finality decision so a remainder
	// smaller than one tile still terminates at the prompt tail.
	if align := s.deterministicChunkAlign(); align > 0 {
		end = alignedPrefillChunkEnd(start, rawEnd, len(ln.prompt), align)
	}
	chunk := ln.prompt[start:end]
	final := end == len(ln.prompt)
	sess := ln.takeSessionForModelLocked()
	if sess == nil {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		s.mu.Unlock()
		return
	}
	// laneKVBlocksLocked is unchanged and reads Cache whenever sess is exposed.
	// The shell has Cache=nil, so concurrent stats use the last published promptLen
	// instead of racing the real session append that follows.
	s.mu.Unlock()

	s.beforeModelExecution(nativeSchedulerEventPrefill, ln)
	prefillStarted := s.now()
	var logits []float32
	if final {
		logits = sess.Prefill(chunk)
	} else {
		sess.PrefillNoLogits(chunk)
	}
	prefillElapsed := s.now().Sub(prefillStarted)
	s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhasePrefill, prefillElapsed)

	s.mu.Lock()
	if !s.livePrefillLaneLocked(ln) || !ln.restoreSessionFromModelLocked(sess) {
		if ln != nil && ln.inflightSess == sess {
			ln.inflightSess = nil
		}
		s.mu.Unlock()
		s.closeLaneSession(sess)
		return
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		s.mu.Unlock()
		return
	}
	if final && len(logits) == 0 {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		s.mu.Unlock()
		return
	}
	ln.promptCursor = end
	// promptLen doubles as the existing preemption accountant's live token count.
	// It reaches the full input length only as chunks become resident in KV.
	ln.promptLen = end
	var exactTwins []*schedLane
	if final {
		if err := s.setLaneStateLocked(ln, schedLaneDecode); err != nil {
			ln.finish(nil, err)
			s.mu.Unlock()
			return
		}
		ln.logits = logits
		s.maybeInsertRadixKV(ln.prompt, ln.sess, logits)
		exactTwins = s.reuseExactQwenPrefillLocked(ln)
	}
	s.reuseQwenPrefillExtensionsLocked(ln)
	s.mu.Unlock()

	// Count only this successfully computed and published chunk, never a cursor
	// delta that includes cached prefix/twin reuse. This is model-call wall time,
	// not GPU completion or the aggregate PhasePrefill recorded by runIteration.
	// Keep recorder callbacks outside mu. Sub saturates on overflow; omit an
	// invalid injected interval instead of inventing a valid duration.
	if prefillElapsed >= 0 && prefillElapsed < time.Duration(1<<63-1) {
		s.recorder.ObservePrefillChunk(len(chunk), prefillElapsed)
	}

	s.observeEvent(nativeSchedulerEvent{
		Iteration:  iteration,
		Kind:       nativeSchedulerEventPrefill,
		Lane:       ln,
		State:      schedLanePrefilling,
		ChunkStart: start,
		ChunkLen:   len(chunk),
	})

	for _, twin := range exactTwins {
		s.mu.Lock()
		live := s.liveDecodeLaneLocked(twin)
		if live && twin.ctx.Err() != nil {
			twin.finish(nil, twin.ctx.Err())
			live = false
		}
		s.mu.Unlock()
		if live {
			s.observeEvent(nativeSchedulerEvent{Iteration: iteration, Kind: nativeSchedulerEventTransition, Lane: twin, State: schedLaneDecode})
		}
	}

	// A test observer or Close may cancel immediately after the chunk publication.
	// Retire under the same lock used to publish sess so cleanup owns it exactly once.
	s.mu.Lock()
	live := s.livePrefillLaneLocked(ln)
	if final {
		live = s.liveDecodeLaneLocked(ln)
	}
	if !live {
		s.mu.Unlock()
		return
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	if !final {
		return
	}
	s.observeEvent(nativeSchedulerEvent{
		Iteration: iteration,
		Kind:      nativeSchedulerEventTransition,
		Lane:      ln,
		State:     schedLaneDecode,
	})

}

// reuseExactQwenPrefillLocked publishes detached full-cache copies only at the
// donor's successful final-chunk boundary, before any decode can advance it.
// executor and s.mu are held. No recurrent cache is ever truncated.
func (s *NativeScheduler) reuseExactQwenPrefillLocked(donor *schedLane) []*schedLane {
	if !s.inBatchDedup || s.captureQwenState || donor == nil || donor.ctx.Err() != nil ||
		donor.terminal || donor.state != schedLaneDecode || donor.preemptRound != 0 ||
		donor.promptCursor != len(donor.prompt) || len(donor.logits) == 0 ||
		donor.prefillChunkTokens < nativeQwenPrefillMinChunkTokens || s.getPrefixHitInfo(donor) != nil {
		return nil
	}
	host := func(ln *schedLane) bool {
		if ln == nil || ln.sess == nil || ln.inflightSess != nil || !ln.q4k {
			return false
		}
		sess := ln.sess
		return sess.M == s.m && sess.Cache != nil && sess.Backend == nil && sess.Q4K &&
			!sess.Quant && !sess.Q4 && !sess.GPTQ && !sess.Metal && !sess.MetalQ4K &&
			!sess.Q4KGateUpOutputSlab && sess.PrecisionPolicy == nil
	}
	if !host(donor) || donor.sess.Cache.Len() != len(donor.prompt) {
		return nil
	}
	var copied []*schedLane
	blocks := s.laneKVBlocksLocked(donor)
	for _, twin := range s.lanes {
		if twin == donor || !host(twin) || twin.terminal || twin.ctx.Err() != nil ||
			twin.state != schedLanePrefilling || twin.promptCursor != 0 || twin.preemptRound != 0 ||
			twin.sess.Cache.Len() != 0 || twin.prefillChunkTokens != donor.prefillChunkTokens ||
			!slices.Equal(twin.prompt, donor.prompt) || s.getPrefixHitInfo(twin) != nil {
			continue
		}
		used := s.usedKVBlocksLocked()
		added := blocks - s.laneKVBlocksLocked(twin)
		if s.preemption.MaxBlocks > 0 && (added > s.preemption.MaxBlocks || used > s.preemption.MaxBlocks-added) {
			continue
		}
		if err := s.setLaneStateLocked(twin, schedLaneDecode); err != nil {
			twin.finish(nil, err)
			continue
		}
		twin.sess.Cache = donor.sess.Cache.Clone()
		twin.logits = copyF32(donor.logits)
		twin.promptCursor = len(twin.prompt)
		twin.promptLen = len(twin.prompt)
		copied = append(copied, twin)
		if used+added > s.maxObservedKVBlocks {
			s.maxObservedKVBlocks = used + added
		}
	}
	if len(copied) > 0 {
		st := s.getOrCreatePrefixState()
		st.mu.Lock()
		st.chunkedExactReuses += uint64(len(copied))
		st.mu.Unlock()
	}
	return copied
}

func (s *NativeScheduler) livePrefillLaneLocked(ln *schedLane) bool {
	if ln == nil || ln.terminal || ln.state != schedLanePrefilling {
		return false
	}
	return s.runningLaneLocked(ln)
}

func (s *NativeScheduler) liveDecodeLaneLocked(ln *schedLane) bool {
	if ln == nil || ln.terminal || ln.state != schedLaneDecode || ln.sess == nil {
		return false
	}
	return s.runningLaneLocked(ln)
}

func (s *NativeScheduler) runningLaneLocked(want *schedLane) bool {
	for _, ln := range s.lanes {
		if ln == want {
			return true
		}
	}
	return false
}

func (s *NativeScheduler) observeEvent(event nativeSchedulerEvent) {
	if s != nil && s.observeNativeEvent != nil {
		s.observeNativeEvent(event)
	}
}

func (s *NativeScheduler) closeLaneSession(sess *model.Session) {
	if sess == nil {
		return
	}
	s.clearPrefixLookup(sess)
	if s != nil && s.closeSession != nil {
		s.closeSession(sess)
		return
	}
	sess.Close()
}
