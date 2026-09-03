package modelengine

import (
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// The resident Qwen hybrid panel path requires a 16-token fresh chunk. Later
// append chunks may be smaller once the session owns prefix state. Refusing a
// smaller configured ceiling keeps the scheduler from silently selecting the
// token-loop fallback on the first chunk.
const nativeQwenPrefillMinChunkTokens = 16

var errNativeSchedulerLaneNotDecodeReady = errors.New("modelengine: native scheduler lane is not decode-ready")

// residentQ4KPrefillCapability is minted from model-owned residency when the
// scheduler is constructed. Request/session q4k flags remain routing intent and
// cannot manufacture this scheduler-local proof.
type residentQ4KPrefillCapability struct {
	model *model.Model
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

func (s *NativeScheduler) qwenPrefillChunkBudget(prep schedPrepare, sess *model.Session, promptLen int) int {
	s.mu.Lock()
	budget := s.qwenPrefillTokens
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
	return budget
}

// nativeQwenResidentAppendEligible exactly mirrors the model-owned
// q4kQwen35HybridPrefillOK architecture and diagnostic gate. Keeping the scheduler
// qualification no broader than the operation it intends to schedule ensures every
// unsupported Qwen configuration stays on historical synchronous admission.
func nativeQwenResidentAppendEligible(cfg model.Config, promptLen int) bool {
	if os.Getenv("FAK_QWEN35_PREFILL_TOKEN_LOOP") != "" {
		return false
	}
	if !cfg.IsQwen35Hybrid() ||
		promptLen < nativeQwenPrefillMinChunkTokens ||
		!cfg.AttnOutputGate ||
		cfg.IsMoE() ||
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
	start := ln.promptCursor
	if start < 0 || start >= len(ln.prompt) || ln.prefillChunkTokens < nativeQwenPrefillMinChunkTokens {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		s.mu.Unlock()
		return
	}
	end := start + ln.prefillChunkTokens
	if end > len(ln.prompt) {
		end = len(ln.prompt)
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
	s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhasePrefill, s.now().Sub(prefillStarted))

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
	if final {
		ln.logits = logits
		ln.state = schedLaneDecode
	}
	s.mu.Unlock()

	s.observeEvent(nativeSchedulerEvent{
		Iteration:  iteration,
		Kind:       nativeSchedulerEventPrefill,
		Lane:       ln,
		State:      schedLanePrefilling,
		ChunkStart: start,
		ChunkLen:   len(chunk),
	})

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
	if s != nil && s.closeSession != nil {
		s.closeSession(sess)
		return
	}
	sess.Close()
}
