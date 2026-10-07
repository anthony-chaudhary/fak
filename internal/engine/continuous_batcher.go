package engine

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Hardware constants and reference geometry for AMD Strix Halo APU (Ryzen AI MAX+ 395 / gfx1151)
// with LPDDR5X-8533 256-bit Unified Memory Architecture (UMA).
const (
	// StrixHaloDefaultBandwidthGBs is sustained GEMV decode bandwidth across 256-bit LPDDR5X-8533 bus.
	StrixHaloDefaultBandwidthGBs = 204.2
	// StrixHaloPeakBandwidthGBs is theoretical peak bandwidth (256-bit * 8.533 GHz).
	StrixHaloPeakBandwidthGBs = 273.0
	// StrixHaloComputePeakTFLOPs is peak matrix compute capability (FP16/INT8).
	StrixHaloComputePeakTFLOPs = 60.0

	// Qwen38_14B_Params is reference parameter count for Qwen3.8-14B (~14.7B).
	Qwen38_14B_Params = int64(14_700_000_000)
	// Qwen38_14B_WeightsBytes is model weight byte footprint in Q4_K_M (~4.5 b/w + scales = ~8.9 GB).
	Qwen38_14B_WeightsBytes = int64(8_900_000_000)
	// Qwen38_14B_FLOPsPerToken is 2 * params = 29.4 GFLOPs per token.
	Qwen38_14B_FLOPsPerToken = 2 * Qwen38_14B_Params
	// Qwen38_14B_SingleAgentTokSec is baseline single-agent decode throughput (~19 tok/s).
	Qwen38_14B_SingleAgentTokSec = 19.0

	// DefaultMaxSlots is the maximum concurrent subagent stream capacity (B = 1..32).
	DefaultMaxSlots = 32
	MinSlots        = 1
	MaxSlots        = 32

	// DefaultPrefillBudgetTokens is the default per-step prompt-token budget for
	// budgeted chunked prefill. This is a fak-derived default (not the upstream
	// 8192): a moderate budget keeps TTFT bounded without monopolizing a step.
	DefaultPrefillBudgetTokens = 512
)

// SlotState captures the lifecycle phase of an individual continuous batching slot.
type SlotState string

const (
	SlotStateEmpty        SlotState = "empty"
	SlotStateActiveDecode SlotState = "active_decode"
	SlotStatePrefilling   SlotState = "prefilling"
	SlotStateYieldedIO    SlotState = "yielded_io"
	SlotStateFinished     SlotState = "finished"
)

// BatchPhase names which scheduler arm a step ran: a PREFILL step advances prompt
// chunks for slots in SlotStatePrefilling; a DECODE step advances resident decodable
// slots; IDLE is a step with neither. Prefill is attempted first each StepPhase.
type BatchPhase string

const (
	PhaseIdle    BatchPhase = "idle"
	PhasePrefill BatchPhase = "prefill"
	PhaseDecode  BatchPhase = "decode"
)

var (
	ErrBatcherClosed        = errors.New("continuous_batcher: batcher closed")
	ErrInvalidSlots         = errors.New("continuous_batcher: slots must be between 1 and 32")
	ErrNilRequest           = errors.New("continuous_batcher: nil request")
	ErrInvalidTargetTokens  = errors.New("continuous_batcher: target tokens must be positive")
	ErrSessionNotFound      = errors.New("continuous_batcher: session not found")
	ErrSessionAlreadyExists = errors.New("continuous_batcher: session already exists")
	ErrSlotNotActive        = errors.New("continuous_batcher: slot is not in active decode state")
	ErrSlotNotYielded       = errors.New("continuous_batcher: slot is not in yielded IO state")
)

// SubagentRequest defines the parameters for an asynchronous subagent stream.
type SubagentRequest struct {
	SessionID       string
	PromptTokens    []int
	TargetTokens    int
	Priority        int
	Metadata        map[string]string
	ExecutionDepth  int   // Recurrent execution depth multiplier (default 1 if <= 0)
	RecurrentLoops  int   // Recurrent loops count (used if > ExecutionDepth)
	KVBytesPerToken int64 // KV cache footprint in bytes per token (uses batcher default if <= 0)

	// SubmissionSeq is the stable, monotonic submission-order identity assigned by
	// the batcher in Submit (before any queueing decision). It is the M2 ordering key
	// for decode residency, independent of Go map iteration and slot-index reuse.
	SubmissionSeq uint64
	// ChunkedPrefill asks the batcher to admit this request into SlotStatePrefilling
	// (budgeted, resumable prompt chunks) rather than eagerly prefill+activate it.
	// Only honored when ContinuousBatcherConfig.PrefillBudget > 0.
	ChunkedPrefill bool
}

func (req *SubagentRequest) effectiveDepth() int {
	if req == nil {
		return 1
	}
	depth := req.ExecutionDepth
	if req.RecurrentLoops > depth {
		depth = req.RecurrentLoops
	}
	if depth <= 0 {
		depth = 1
	}
	return depth
}

// Slot represents one execution lane in the continuous batcher.
type Slot struct {
	Index             int
	SessionID         string
	State             SlotState
	Request           *SubagentRequest
	PromptTokens      []int
	GeneratedTokens   []int
	TargetTokens      int
	ExecutionDepth    int // Configured recurrent depth limit (default 1)
	CurrentDepth      int // Current recurrent depth reached (0..ExecutionDepth)
	RecurrentPasses   int // Recurrent execution passes completed
	YieldCount        int
	ResumeCount       int
	LastToken         int
	EvictionDuration  time.Duration // 0 ms in UMA
	BytesSwapped      int64         // 0 bytes in UMA
	ReprefillTokens   int           // 0 re-prefill tokens on resume
	KVCacheStationary bool          // Remains resident in UMA
	KVCacheBytes      int64         // Allocated KV cache memory in bytes

	// PrefixHits / PrefixMatchedTokens record whether this slot's admission
	// reused a cached GDN recurrent boundary and how many prompt tokens it
	// skipped re-folding.
	PrefixHits          int
	PrefixMatchedTokens int

	// SubmissionSeq is the request's stable submission-order identity (M2 ordering key).
	SubmissionSeq uint64
	// PendingPrompt is the full prompt copy still being consumed by budgeted prefill;
	// it is nil for eager-active slots.
	PendingPrompt []int
	// PrefillPos is the count of PendingPrompt tokens already consumed by prefill.
	PrefillPos int

	sess        *model.Session
	tokenCh     chan int
	doneCh      chan struct{}
	err         error
	AdmittedAt  time.Time
	CompletedAt time.Time

	mu sync.Mutex
}

// Tokens returns the streaming channel for tokens generated by this slot.
func (s *Slot) Tokens() <-chan int {
	return s.tokenCh
}

// Done returns a channel that closes when the subagent turn completes.
func (s *Slot) Done() <-chan struct{} {
	return s.doneCh
}

// Err returns any error encountered during generation.
func (s *Slot) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// TokensGenerated returns the count of decode tokens produced so far.
func (s *Slot) TokensGenerated() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.GeneratedTokens)
}

// CurrentExecutionDepth returns the current recurrent execution depth reached by this slot.
func (s *Slot) CurrentExecutionDepth() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.CurrentDepth
}

// RecurrentPassCount returns the count of completed recurrent passes.
func (s *Slot) RecurrentPassCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.RecurrentPasses
}

// Depth returns the current execution depth reached by this slot.
func (s *Slot) Depth() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.CurrentDepth
}

// RecurrentDepth returns the configured recurrent execution depth limit.
func (s *Slot) RecurrentDepth() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ExecutionDepth
}

func (s *Slot) snapshot() *Slot {
	s.mu.Lock()
	defer s.mu.Unlock()

	cp := &Slot{
		Index:               s.Index,
		SessionID:           s.SessionID,
		State:               s.State,
		Request:             s.Request,
		TargetTokens:        s.TargetTokens,
		ExecutionDepth:      s.ExecutionDepth,
		CurrentDepth:        s.CurrentDepth,
		RecurrentPasses:     s.RecurrentPasses,
		YieldCount:          s.YieldCount,
		ResumeCount:         s.ResumeCount,
		LastToken:           s.LastToken,
		EvictionDuration:    s.EvictionDuration,
		BytesSwapped:        s.BytesSwapped,
		ReprefillTokens:     s.ReprefillTokens,
		KVCacheStationary:   s.KVCacheStationary,
		KVCacheBytes:        s.KVCacheBytes,
		SubmissionSeq:       s.SubmissionSeq,
		PrefillPos:          s.PrefillPos,
		PrefixHits:          s.PrefixHits,
		PrefixMatchedTokens: s.PrefixMatchedTokens,
		sess:                s.sess,
		tokenCh:             s.tokenCh,
		doneCh:              s.doneCh,
		err:                 s.err,
		AdmittedAt:          s.AdmittedAt,
		CompletedAt:         s.CompletedAt,
	}
	if s.GeneratedTokens != nil {
		cp.GeneratedTokens = append([]int(nil), s.GeneratedTokens...)
	}
	if s.PromptTokens != nil {
		cp.PromptTokens = append([]int(nil), s.PromptTokens...)
	}
	if s.PendingPrompt != nil {
		cp.PendingPrompt = append([]int(nil), s.PendingPrompt...)
	}
	return cp
}

func (s *Slot) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.SessionID = ""
	s.State = SlotStateEmpty
	s.Request = nil
	s.PromptTokens = nil
	s.GeneratedTokens = nil
	s.TargetTokens = 0
	s.ExecutionDepth = 0
	s.CurrentDepth = 0
	s.RecurrentPasses = 0
	s.YieldCount = 0
	s.ResumeCount = 0
	s.LastToken = 0
	s.EvictionDuration = 0
	s.BytesSwapped = 0
	s.ReprefillTokens = 0
	s.KVCacheStationary = false
	s.KVCacheBytes = 0
	s.SubmissionSeq = 0
	s.PendingPrompt = nil
	s.PrefillPos = 0
	s.PrefixHits = 0
	s.PrefixMatchedTokens = 0
	s.sess = nil
	s.tokenCh = nil
	s.doneCh = nil
	s.err = nil
	s.AdmittedAt = time.Time{}
	s.CompletedAt = time.Time{}
}

// ContinuousBatcherConfig configures continuous batching capacity and hardware parameters.
type ContinuousBatcherConfig struct {
	// MaxSlots caps concurrent active decode lanes (1..32, default 32).
	MaxSlots int

	// Model is optional pointer to underlying model. If provided, real forward passes run.
	Model *model.Model

	// Hardware and Roofline Parameters
	MemoryBandwidthGBs   float64
	ComputePeakTFLOPs    float64
	ModelWeightsBytes    int64
	ModelParams          int64
	SingleAgentTokPerSec float64
	FixedOverheadSec     float64

	// MaxKVCacheBytes caps total KV cache memory capacity across active slots (0 = unconstrained).
	MaxKVCacheBytes int64

	// KVBytesPerToken is the default KV cache footprint in bytes per token (default if <= 0 is 1024).
	KVBytesPerToken int64

	// PrefillBudget is the per-step prompt-token budget for budgeted chunked prefill.
	// 0 (default) preserves the legacy eager-prefill, decode-only stepping behavior;
	// > 0 enables a PREFILL arm for slots admitted with ChunkedPrefill.
	PrefillBudget int

	// DisableRecurrentPrefixReuse turns off GDN recurrent prefix reuse. By default
	// (false) a hybrid model enables an extend-only RecurrentPrefixCache so a
	// multi-turn conversation prefills only the new suffix instead of re-folding
	// the whole prompt from token 0. Non-hybrid models never build the cache.
	DisableRecurrentPrefixReuse bool

	// RecurrentPrefixCapacity bounds retained conversation boundaries when reuse
	// is enabled (0 → DefaultRecurrentPrefixCapacity).
	RecurrentPrefixCapacity int

	// With decode resident, StepPhase prefills only if prefilling >= floor(WaitingServedRatio*
	// decoding) or MaxWaitingDecodeSteps decodes ran since the last prefill; zeros: prefill-first.
	WaitingServedRatio    float64
	MaxWaitingDecodeSteps int
}

// DefaultContinuousBatcherConfig returns calibrated defaults for Strix Halo and Qwen3.8-14B.
func DefaultContinuousBatcherConfig() ContinuousBatcherConfig {
	return ContinuousBatcherConfig{
		MaxSlots:             DefaultMaxSlots,
		MemoryBandwidthGBs:   StrixHaloDefaultBandwidthGBs,
		ComputePeakTFLOPs:    StrixHaloComputePeakTFLOPs,
		ModelWeightsBytes:    Qwen38_14B_WeightsBytes,
		ModelParams:          Qwen38_14B_Params,
		SingleAgentTokPerSec: Qwen38_14B_SingleAgentTokSec,
		FixedOverheadSec:     0.0025, // 2.5 ms attention & kernel dispatch overhead
		MaxKVCacheBytes:      0,      // 0 = unconstrained
		KVBytesPerToken:      1024,
	}
}

// BatchStepResult encapsulates iteration-level execution output and telemetry.
type BatchStepResult struct {
	Iteration            uint64
	ActiveSlots          int
	YieldedSlots         int
	FinishedSlots        int
	EmptySlots           int
	TotalSlots           int
	GeneratedTokens      map[string]int
	TokensGenerated      int
	OperationalIntensity float64
	ArithmeticIntensity  float64
	AggregateThroughput  float64
	StepDuration         time.Duration
	StallDuration        time.Duration // 0s on yield/resume
	RetiredSessionIDs    []string
	PromotedSessionIDs   []string
	KVCacheBytesUsed     int64
	SlotDepths           map[string]int // Current execution depth per session ID

	// Phase names which scheduler arm this step ran (idle/prefill/decode).
	Phase BatchPhase
	// PrefillTokens is the number of prompt tokens consumed this step (prefill arm).
	PrefillTokens int
	// DecodeTokens is the number of decode tokens produced this step (decode arm).
	DecodeTokens int
	// DecodeResidentUIDs is the stable-UID roster of resident decodable slots observed
	// at the start of the step (M2 order: SubmissionSeq asc, tie-break SessionID).
	DecodeResidentUIDs []uint64
	// PrefixReuseTokens is the number of prompt tokens skipped by GDN recurrent
	// prefix reuse observed since the previous step (0 when nothing reused).
	PrefixReuseTokens int
	// PrefixHits is the number of admissions observed since the previous step
	// that reused a cached recurrent boundary.
	PrefixHits int
	// PrefixQueriedTokens is the number of prompt tokens looked up in the GDN
	// recurrent prefix cache since the previous step, hit or miss; it is the
	// denominator of the prefix hit rate (vLLM prefix_cache_queries).
	PrefixQueriedTokens int
}

// ContinuousBatcher manages dynamic iteration-level continuous batching for subagent turn loops.
type ContinuousBatcher struct {
	mu           sync.Mutex
	cfg          ContinuousBatcherConfig
	slots        []*Slot
	sessionMap   map[string]*Slot
	completedMap map[string]*Slot
	waitingQueue []*SubagentRequest
	iteration    uint64
	totalTokens  int64
	seqCounter   uint64
	uidSeq       uint64
	closed       bool

	// prefixCache is the extend-only GDN recurrent boundary cache. It is non-nil
	// only for a hybrid model with reuse enabled (and not explicitly disabled).
	prefixCache *RecurrentPrefixCache

	// pendingPrefixHits / pendingPrefixReuseTokens are admissions observed since
	// the last emitted step. They are incremented once at slot admission (when a
	// cached recurrent boundary is reused) and captured-and-reset by the next
	// step, so a reused slot resident for N steps is reported exactly once.
	pendingPrefixHits        int
	pendingPrefixReuseTokens int
	// pendingPrefixQueriedTokens counts prompt tokens looked up (hit or miss)
	// since the last emitted step, with the same capture-and-reset discipline.
	pendingPrefixQueriedTokens int
	decodeSinceRefill          int // decode steps since the last prefill step
}

// NewContinuousBatcher constructs a scheduler with the specified configuration.
func NewContinuousBatcher(cfg ...ContinuousBatcherConfig) (*ContinuousBatcher, error) {
	c := DefaultContinuousBatcherConfig()
	if len(cfg) > 0 {
		c = cfg[0]
		if c.MaxSlots == 0 {
			c.MaxSlots = DefaultMaxSlots
		}
		if c.MemoryBandwidthGBs <= 0 {
			c.MemoryBandwidthGBs = StrixHaloDefaultBandwidthGBs
		}
		if c.ComputePeakTFLOPs <= 0 {
			c.ComputePeakTFLOPs = StrixHaloComputePeakTFLOPs
		}
		if c.ModelWeightsBytes <= 0 {
			c.ModelWeightsBytes = Qwen38_14B_WeightsBytes
		}
		if c.ModelParams <= 0 {
			c.ModelParams = Qwen38_14B_Params
		}
		if c.SingleAgentTokPerSec <= 0 {
			c.SingleAgentTokPerSec = Qwen38_14B_SingleAgentTokSec
		}
		if c.FixedOverheadSec <= 0 {
			c.FixedOverheadSec = 0.0025
		}
		if c.KVBytesPerToken <= 0 {
			c.KVBytesPerToken = 1024
		}
		if c.MaxKVCacheBytes < 0 {
			c.MaxKVCacheBytes = 0
		}
	}

	if c.MaxSlots < MinSlots || c.MaxSlots > MaxSlots {
		return nil, ErrInvalidSlots
	}

	slots := make([]*Slot, c.MaxSlots)
	for i := 0; i < c.MaxSlots; i++ {
		slots[i] = &Slot{
			Index: i,
			State: SlotStateEmpty,
		}
	}

	cb := &ContinuousBatcher{
		cfg:          c,
		slots:        slots,
		sessionMap:   make(map[string]*Slot),
		completedMap: make(map[string]*Slot),
		waitingQueue: make([]*SubagentRequest, 0),
	}
	if c.Model != nil && c.Model.Cfg.IsHybrid() && !c.DisableRecurrentPrefixReuse {
		cb.prefixCache = NewRecurrentPrefixCache(c.RecurrentPrefixCapacity)
	}
	return cb, nil
}

// PrefixCacheStats returns the recurrent prefix-cache telemetry and whether the
// cache is enabled for this batcher.
func (cb *ContinuousBatcher) PrefixCacheStats() (RecurrentPrefixCacheStats, bool) {
	cb.mu.Lock()
	cache := cb.prefixCache
	cb.mu.Unlock()
	if cache == nil {
		return RecurrentPrefixCacheStats{}, false
	}
	return cache.Stats(), true
}

// PrefixCacheLen returns the number of retained conversation boundaries, or 0
// when the cache is disabled.
func (cb *ContinuousBatcher) PrefixCacheLen() int {
	cb.mu.Lock()
	cache := cb.prefixCache
	cb.mu.Unlock()
	if cache == nil {
		return 0
	}
	return cache.Len()
}

// Submit enqueues a subagent request into an available slot or waiting queue.
func (cb *ContinuousBatcher) Submit(req *SubagentRequest) (string, error) {
	if req == nil {
		return "", ErrNilRequest
	}
	if req.TargetTokens <= 0 {
		if req.ExecutionDepth > 0 || req.RecurrentLoops > 0 {
			req.TargetTokens = req.effectiveDepth()
		} else {
			return "", ErrInvalidTargetTokens
		}
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return "", ErrBatcherClosed
	}

	sessionID := req.SessionID
	if sessionID == "" {
		id := atomic.AddUint64(&cb.seqCounter, 1)
		sessionID = fmt.Sprintf("subagent-%d", id)
		req.SessionID = sessionID
	}

	if slot, exists := cb.sessionMap[sessionID]; exists && slot.State != SlotStateFinished {
		return "", ErrSessionAlreadyExists
	}
	for _, q := range cb.waitingQueue {
		if q.SessionID == sessionID {
			return "", ErrSessionAlreadyExists
		}
	}

	// Assign the stable submission-order identity only once the request is known
	// to be accepted (past the duplicate-session rejections) and before any
	// queueing decision, so a rejected duplicate does not burn a sequence number
	// while the decode-residency roster (M2) still orders by true submission time
	// even for requests that wait behind capacity.
	cb.uidSeq++
	req.SubmissionSeq = cb.uidSeq

	// Look for an available empty slot
	emptyIdx := -1
	for i, slot := range cb.slots {
		if slot.State == SlotStateEmpty {
			emptyIdx = i
			break
		}
	}

	reqBytes := cb.RequiredKVCacheBytes(req)
	canAdmit := emptyIdx != -1 && len(cb.waitingQueue) == 0 &&
		(cb.cfg.MaxKVCacheBytes <= 0 || cb.currentKVCacheBytesLocked()+reqBytes <= cb.cfg.MaxKVCacheBytes)

	if canAdmit {
		cb.initSlot(emptyIdx, req)
	} else {
		cb.waitingQueue = append(cb.waitingQueue, req)
	}

	return sessionID, nil
}

func (cb *ContinuousBatcher) initSlot(index int, req *SubagentRequest) *Slot {
	reqKVBytes := cb.RequiredKVCacheBytes(req)
	depth := req.effectiveDepth()
	slot := &Slot{
		Index:             index,
		SessionID:         req.SessionID,
		State:             SlotStateActiveDecode,
		Request:           req,
		PromptTokens:      append([]int(nil), req.PromptTokens...),
		GeneratedTokens:   make([]int, 0, req.TargetTokens),
		TargetTokens:      req.TargetTokens,
		ExecutionDepth:    depth,
		KVCacheStationary: true,
		KVCacheBytes:      reqKVBytes,
		SubmissionSeq:     req.SubmissionSeq,
		AdmittedAt:        time.Now(),
		tokenCh:           make(chan int, req.TargetTokens+16),
		doneCh:            make(chan struct{}),
	}

	// GDN recurrent prefix reuse (hybrid models only). model.PrefixSnapshot is the
	// only safe reuse unit for a Gated DeltaNet fold: the linear recurrence is an
	// irreversible fold over the whole prefix, so a whole-prefix boundary captured
	// at the token position that produced it — and carrying the full linear state
	// via KVCache.Clone → linear.clone() plus softmax KV — is the sole
	// mathematically valid reuse unit. Lookup is a take: on a hit ownership of the
	// snapshot transfers here and we prefill only req.PromptTokens[reused:].
	var (
		reused       int
		restored     bool
		restoredSess *model.Session
	)
	if cb.prefixCache != nil && cb.cfg.Model != nil && len(req.PromptTokens) > 0 {
		cb.pendingPrefixQueriedTokens += len(req.PromptTokens)
		if matched, snap, hit := cb.prefixCache.Lookup(req.SessionID, req.PromptTokens); hit {
			// Reuse is intentionally host-only: a device session would need a
			// matching Backend, or Restore refuses closed→miss.
			sess := &model.Session{
				M:     cb.cfg.Model,
				Cache: model.NewKVCache(cb.cfg.Model.Cfg),
			}
			if err := snap.Restore(sess); err != nil {
				// A failed restore is a miss, not a hit: release the snapshot and
				// fall back to a full fresh-session prefill.
				snap.Close()
			} else {
				reused = matched
				restored = true
				restoredSess = sess
				// Once-only admission accounting: report this reuse on the next
				// emitted step, not on every step the slot stays resident.
				cb.pendingPrefixHits++
				cb.pendingPrefixReuseTokens += reused
			}
		}
	}

	if cb.cfg.PrefillBudget > 0 && req.ChunkedPrefill {
		// Budgeted chunked prefill: defer activation. The slot consumes its prompt
		// copy in budget-sized chunks across PREFILL steps, then transitions to
		// SlotStateActiveDecode. Upstream mini-sglang prefill.py:65-151 treats an
		// over-budget prompt as resumable chunks; the in-repo precedent is
		// internal/modelengine/nativesched_prefill.go (PrefillNoLogits for
		// intermediate chunks, Prefill for the final chunk). When a real model is
		// configured the session is built here exactly as the eager branch does,
		// and its KV cache is advanced in lockstep with PrefillPos so no consumed
		// token is ever re-prefilled.
		slot.State = SlotStatePrefilling
		slot.PendingPrompt = append([]int(nil), req.PromptTokens...)
		slot.PrefillPos = 0
		slot.LastToken = 0
		if cb.cfg.Model != nil {
			if restored {
				// The restored session already holds the recurrent fold for the
				// first `reused` tokens; PendingPrompt stays the full prompt so the
				// existing PREFILL arm consumes only the suffix via
				// PendingPrompt[PrefillPos:...].
				slot.sess = restoredSess
				slot.PrefillPos = reused
			} else {
				slot.sess = &model.Session{
					M:     cb.cfg.Model,
					Cache: model.NewKVCache(cb.cfg.Model.Cfg),
				}
			}
		}
		if restored {
			slot.PrefixHits = 1
			slot.PrefixMatchedTokens = reused
		}
		cb.slots[index] = slot
		cb.sessionMap[req.SessionID] = slot
		return slot
	}

	if cb.cfg.Model != nil {
		sess := restoredSess
		if sess == nil {
			sess = &model.Session{
				M:     cb.cfg.Model,
				Cache: model.NewKVCache(cb.cfg.Model.Cfg),
			}
		}
		if len(req.PromptTokens) > 0 {
			if restored && reused >= len(req.PromptTokens) {
				// Strict extend guarantees a remaining suffix, but guard the
				// impossible case rather than call Prefill with an empty slice.
				slot.LastToken = req.PromptTokens[len(req.PromptTokens)-1]
			} else {
				logits := sess.Prefill(req.PromptTokens[reused:])
				slot.LastToken = argmax(logits)
			}
		} else {
			slot.LastToken = 1
		}
		slot.sess = sess
		if restored {
			slot.PrefixHits = 1
			slot.PrefixMatchedTokens = reused
		}
		// Store the boundary this prompt produced so a later turn can extend it.
		cb.retainPrefixLocked(slot)
	} else {
		if len(req.PromptTokens) > 0 {
			slot.LastToken = req.PromptTokens[len(req.PromptTokens)-1]
		} else {
			slot.LastToken = 42 + slot.Index*31
		}
	}

	cb.slots[index] = slot
	cb.sessionMap[req.SessionID] = slot
	return slot
}

// YieldSlot transitions a slot to SlotStateYieldedIO when a subagent executes tools,
// freeing compute iteration time without evicting its KV cache.
func (cb *ContinuousBatcher) YieldSlot(sessionID string) error {
	if sessionID == "" {
		return errors.New("continuous_batcher: empty session ID")
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return ErrBatcherClosed
	}

	slot, ok := cb.sessionMap[sessionID]
	if !ok {
		return ErrSessionNotFound
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	if slot.State != SlotStateActiveDecode {
		return ErrSlotNotActive
	}

	slot.State = SlotStateYieldedIO
	slot.YieldCount++
	slot.EvictionDuration = 0
	slot.BytesSwapped = 0
	slot.KVCacheStationary = true

	return nil
}

// ResumeSlot resumes a yielded subagent immediately when tool execution completes.
func (cb *ContinuousBatcher) ResumeSlot(sessionID string) error {
	if sessionID == "" {
		return errors.New("continuous_batcher: empty session ID")
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return ErrBatcherClosed
	}

	slot, ok := cb.sessionMap[sessionID]
	if !ok {
		return ErrSessionNotFound
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	if slot.State != SlotStateYieldedIO {
		return ErrSlotNotYielded
	}

	slot.State = SlotStateActiveDecode
	slot.ResumeCount++
	slot.ReprefillTokens = 0

	return nil
}

// Step performs a decode-only iteration-level step: it gathers resident decodable
// slots (stable-UID order), generates one token for each simultaneously, retires
// finished slots, and pulls queued requests into freed slots. It is exactly
// stepWithBudget(ctx, 0). The pre-existing Step observable behavior is preserved
// except for two intentional refinements: the decode batch is ordered by stable
// submission sequence, and it is filtered through slotDecodableLocked so a slot
// already at its target is not advanced again.
func (cb *ContinuousBatcher) Step(ctx context.Context) (*BatchStepResult, error) {
	return cb.stepWithBudget(ctx, 0)
}

// StepPhase is the budgeted scheduler step: a PREFILL step (up to PrefillBudget
// prompt tokens, FIFO) when a slot is prefilling and the decode-starvation bound
// allows it; otherwise a DECODE step over the resident decodable slots.
func (cb *ContinuousBatcher) StepPhase(ctx context.Context) (*BatchStepResult, error) {
	return cb.stepWithBudget(ctx, cb.cfg.PrefillBudget)
}

// DecodeResident returns the M2 decode-residency roster: slots in
// SlotStateActiveDecode that are still decodable, ordered by ascending stable
// submission sequence (tie-break SessionID). The order is deterministic and
// independent of Go map iteration or slot-index reuse.
func (cb *ContinuousBatcher) DecodeResident() []*Slot {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.decodeResidentLocked()
}

// DecodeResidentUIDs returns the stable submission-sequence ids of DecodeResident,
// in the same order. It is the ordering key a caller can pin deterministically.
func (cb *ContinuousBatcher) DecodeResidentUIDs() []uint64 {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	resident := cb.decodeResidentLocked()
	uids := make([]uint64, len(resident))
	for i, slot := range resident {
		uids[i] = slot.SubmissionSeq
	}
	return uids
}

// decodeResidentLocked gathers decodable SlotStateActiveDecode slots and sorts
// them into stable-UID order. Caller MUST hold cb.mu.
func (cb *ContinuousBatcher) decodeResidentLocked() []*Slot {
	resident := make([]*Slot, 0, len(cb.slots))
	for _, slot := range cb.slots {
		slot.mu.Lock()
		ok := slot.State == SlotStateActiveDecode && slotDecodableLocked(slot)
		slot.mu.Unlock()
		if ok {
			resident = append(resident, slot)
		}
	}
	sortSlotsByStableUID(resident)
	return resident
}

// slotDecodableLocked reports whether an active-decode slot still has work under
// the same completion rule Step uses: recurrent slots are bounded by
// ExecutionDepth, ordinary slots by TargetTokens. Caller holds slot.mu.
func slotDecodableLocked(slot *Slot) bool {
	if slot.ExecutionDepth > 1 {
		return slot.CurrentDepth < slot.ExecutionDepth
	}
	return len(slot.GeneratedTokens) < slot.TargetTokens
}

// sortSlotsByStableUID orders slots by SubmissionSeq ascending, tie-break
// SessionID ascending, so the decode batch is deterministic.
func sortSlotsByStableUID(slots []*Slot) {
	sort.Slice(slots, func(i, j int) bool {
		if slots[i].SubmissionSeq != slots[j].SubmissionSeq {
			return slots[i].SubmissionSeq < slots[j].SubmissionSeq
		}
		return slots[i].SessionID < slots[j].SessionID
	})
}

// stepWithBudget is the shared scheduler step. budget > 0 enables the PREFILL arm
// for SlotStatePrefilling slots; budget == 0 is the decode-only legacy path.
func (cb *ContinuousBatcher) stepWithBudget(ctx context.Context, budget int) (*BatchStepResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return nil, ErrBatcherClosed
	}

	stepStart := time.Now()

	// 1. Pull queued requests into any empty slots before gathering batches.
	// initSlot applies the budgeted-prefill rule for ChunkedPrefill requests.
	promotedIDs := cb.tryPromoteWaitingLocked()

	// Capture and reset once-only admission accounting now that this step's
	// promotions (which may reuse a cached boundary) are included.
	prefixHits := cb.pendingPrefixHits
	prefixReuseTokens := cb.pendingPrefixReuseTokens
	cb.pendingPrefixHits = 0
	cb.pendingPrefixReuseTokens = 0
	prefixQueriedTokens := cb.pendingPrefixQueriedTokens
	cb.pendingPrefixQueriedTokens = 0

	// 2. Gather prefilling slots and the resident decodable roster, and count
	// the other states in the same pass.
	var prefilling []*Slot
	resident := cb.decodeResidentLocked()
	yieldedCount := 0
	emptyCount := 0
	finishedCount := 0

	for _, slot := range cb.slots {
		slot.mu.Lock()
		st := slot.State
		slot.mu.Unlock()

		switch st {
		case SlotStatePrefilling:
			prefilling = append(prefilling, slot)
		case SlotStateYieldedIO:
			yieldedCount++
		case SlotStateFinished:
			finishedCount++
		case SlotStateEmpty:
			emptyCount++
		}
	}

	// 3. PREFILL arm, consuming up to budget prompt tokens FIFO. The refill gate
	// adapts TGI backends/v3/src/backend.rs:186-195@b4adbf2 (Apache-2.0).
	refillDue := len(resident) == 0 || cb.decodeSinceRefill >= cb.cfg.MaxWaitingDecodeSteps ||
		len(prefilling) >= int(cb.cfg.WaitingServedRatio*float64(len(resident)))
	if budget > 0 && len(prefilling) > 0 && refillDue {
		cb.decodeSinceRefill = 0
		sortSlotsByStableUID(prefilling)
		remaining := budget
		total := 0
		for _, slot := range prefilling {
			if remaining <= 0 {
				break
			}
			slot.mu.Lock()
			pending := len(slot.PendingPrompt) - slot.PrefillPos
			if pending <= 0 {
				cb.finishPrefillLocked(slot, nil)
				slot.mu.Unlock()
				continue
			}
			chunk := remaining
			if chunk > pending {
				chunk = pending
			}
			// Capture the chunk of prompt consumed THIS step BEFORE advancing
			// PrefillPos, so the model session's KV cache advances in lockstep
			// with PrefillPos and no already-consumed token is ever re-prefilled.
			// Upstream mini-sglang prefill.py:65-151 is a resumable chunked
			// prefill; the in-repo precedent is
			// internal/modelengine/nativesched_prefill.go:481-563.
			var logits []float32
			if slot.sess != nil {
				chunkIDs := slot.PendingPrompt[slot.PrefillPos : slot.PrefillPos+chunk]
				if slot.PrefillPos+chunk == len(slot.PendingPrompt) {
					// Final chunk: its last-token distribution seeds decode.
					logits = slot.sess.Prefill(chunkIDs)
				} else {
					// Intermediate chunk: grow KV only, discard logits.
					slot.sess.PrefillNoLogits(chunkIDs)
				}
			}
			slot.PrefillPos += chunk
			remaining -= chunk
			total += chunk
			if slot.PrefillPos >= len(slot.PendingPrompt) {
				cb.finishPrefillLocked(slot, logits)
			}
			slot.mu.Unlock()
		}

		cb.iteration++
		residentUIDs := make([]uint64, len(resident))
		for i, slot := range resident {
			residentUIDs[i] = slot.SubmissionSeq
		}
		return &BatchStepResult{
			Iteration:            cb.iteration,
			ActiveSlots:          len(resident),
			YieldedSlots:         yieldedCount,
			FinishedSlots:        finishedCount,
			EmptySlots:           emptyCount,
			TotalSlots:           len(cb.slots),
			GeneratedTokens:      make(map[string]int),
			OperationalIntensity: cb.OperationalIntensity(len(resident)),
			ArithmeticIntensity:  cb.OperationalIntensity(len(resident)),
			AggregateThroughput:  cb.AggregateThroughput(len(resident)),
			StepDuration:         time.Since(stepStart),
			PromotedSessionIDs:   promotedIDs,
			KVCacheBytesUsed:     cb.currentKVCacheBytesLocked(),
			SlotDepths:           make(map[string]int),
			Phase:                PhasePrefill,
			PrefillTokens:        total,
			DecodeResidentUIDs:   residentUIDs,
			PrefixHits:           prefixHits,
			PrefixReuseTokens:    prefixReuseTokens,
			PrefixQueriedTokens:  prefixQueriedTokens,
		}, nil
	}

	// 4. DECODE arm. If nothing is resident, mirror the legacy empty path.
	if len(resident) == 0 {
		cb.iteration++
		return &BatchStepResult{
			Iteration:           cb.iteration,
			YieldedSlots:        yieldedCount,
			FinishedSlots:       finishedCount,
			EmptySlots:          emptyCount,
			TotalSlots:          len(cb.slots),
			GeneratedTokens:     make(map[string]int),
			StepDuration:        time.Since(stepStart),
			PromotedSessionIDs:  promotedIDs,
			KVCacheBytesUsed:    cb.currentKVCacheBytesLocked(),
			SlotDepths:          make(map[string]int),
			Phase:               PhaseIdle,
			DecodeResidentUIDs:  []uint64{},
			PrefixHits:          prefixHits,
			PrefixReuseTokens:   prefixReuseTokens,
			PrefixQueriedTokens: prefixQueriedTokens,
		}, nil
	}

	cb.decodeSinceRefill++
	activeSlots := resident
	activeCount := len(activeSlots)
	tokensThisStep := make([]int, activeCount)
	generatedTokens := make(map[string]int, activeCount)
	slotDepths := make(map[string]int, activeCount)

	// 5. Forward pass over the ordered resident slots simultaneously.
	if cb.cfg.Model != nil {
		if activeCount == 1 {
			slot := activeSlots[0]
			logits := slot.sess.Step(slot.LastToken)
			tokensThisStep[0] = argmax(logits)
		} else {
			seqs := make([]*model.Session, activeCount)
			ids := make([]int, activeCount)
			for i, slot := range activeSlots {
				seqs[i] = slot.sess
				ids[i] = slot.LastToken
			}
			bs := &model.BatchSession{M: cb.cfg.Model, Seqs: seqs}
			batchLogits := bs.StepBatch(ids)
			for i, logits := range batchLogits {
				tokensThisStep[i] = argmax(logits)
			}
		}
	} else {
		// Pure scheduler mode: compute deterministic next token
		for i, slot := range activeSlots {
			seed := slot.LastToken
			if seed == 0 {
				seed = 42
			}
			nextToken := (seed*37+13+len(slot.GeneratedTokens))%32000 + 1
			tokensThisStep[i] = nextToken
		}
	}

	// 6. Deliver tokens and mark completion.
	var newlyFinished []*Slot
	for i, slot := range activeSlots {
		tok := tokensThisStep[i]
		slot.mu.Lock()
		slot.LastToken = tok
		slot.GeneratedTokens = append(slot.GeneratedTokens, tok)
		slot.CurrentDepth++
		slot.RecurrentPasses++
		cb.totalTokens++
		generatedTokens[slot.SessionID] = tok
		slotDepths[slot.SessionID] = slot.CurrentDepth

		select {
		case slot.tokenCh <- tok:
		default:
		}

		if !slotDecodableLocked(slot) {
			slot.State = SlotStateFinished
			slot.CompletedAt = time.Now()
			close(slot.doneCh)
			close(slot.tokenCh)
			newlyFinished = append(newlyFinished, slot)
		}
		slot.mu.Unlock()
	}

	// 7. Retire finished slots and pull queued requests into freed slots.
	var retiredIDs []string
	for _, slot := range newlyFinished {
		retiredIDs = append(retiredIDs, slot.SessionID)
		cb.completedMap[slot.SessionID] = slot
		delete(cb.sessionMap, slot.SessionID)

		idx := slot.Index
		cb.slots[idx] = &Slot{
			Index: idx,
			State: SlotStateEmpty,
		}
	}

	promotedAfterRetire := cb.tryPromoteWaitingLocked()
	promotedIDs = append(promotedIDs, promotedAfterRetire...)

	cb.iteration++

	opIntensity := cb.OperationalIntensity(activeCount)
	aggTPS := cb.AggregateThroughput(activeCount)

	residentUIDs := make([]uint64, len(resident))
	for i, slot := range resident {
		residentUIDs[i] = slot.SubmissionSeq
	}

	return &BatchStepResult{
		Iteration:            cb.iteration,
		ActiveSlots:          activeCount,
		YieldedSlots:         yieldedCount,
		FinishedSlots:        len(newlyFinished) + finishedCount,
		EmptySlots:           len(cb.slots) - (activeCount - len(newlyFinished) + yieldedCount),
		TotalSlots:           len(cb.slots),
		GeneratedTokens:      generatedTokens,
		TokensGenerated:      activeCount,
		OperationalIntensity: opIntensity,
		ArithmeticIntensity:  opIntensity,
		AggregateThroughput:  aggTPS,
		StepDuration:         time.Since(stepStart),
		StallDuration:        0, // zero compute stalls on yield/resume
		RetiredSessionIDs:    retiredIDs,
		PromotedSessionIDs:   promotedIDs,
		KVCacheBytesUsed:     cb.currentKVCacheBytesLocked(),
		SlotDepths:           slotDepths,
		Phase:                PhaseDecode,
		DecodeTokens:         activeCount,
		DecodeResidentUIDs:   residentUIDs,
		PrefixHits:           prefixHits,
		PrefixReuseTokens:    prefixReuseTokens,
		PrefixQueriedTokens:  prefixQueriedTokens,
	}, nil
}

// finishPrefillLocked completes a budgeted-prefill chunk: it promotes the slot to
// SlotStateActiveDecode and sets the token decode starts from. On the model path the
// caller passes the final chunk's logits and the argmax must be preserved (the prompt's
// last token is the teacher-forced input, not the next-token prediction); on the
// simulation path (sess == nil) the last prompt token is the seed exactly as before.
// Caller MUST hold cb.mu AND slot.mu.
func (cb *ContinuousBatcher) finishPrefillLocked(slot *Slot, logits []float32) {
	switch {
	case slot.sess != nil && len(logits) > 0:
		slot.LastToken = argmax(logits)
	case len(slot.PendingPrompt) > 0:
		slot.LastToken = slot.PendingPrompt[len(slot.PendingPrompt)-1]
	default:
		slot.LastToken = 42 + slot.Index*31
	}
	slot.State = SlotStateActiveDecode
	// The recurrent boundary is only valid once the whole prompt has been
	// folded, so retain it only when the slot has consumed its entire pending
	// prompt. Using slot.PromptTokens (the stored full prompt) as the key is
	// exact: the snapshot captures the fold over all of it.
	if slot.PrefillPos >= len(slot.PendingPrompt) {
		cb.retainPrefixLocked(slot)
	}
}

// retainPrefixLocked stores the recurrent boundary a fully-consumed prompt
// produced, so a later conversation turn can extend it. It is a no-op unless a
// cache is enabled, a real model session exists, and the slot holds a non-empty
// prompt. Ownership of the captured snapshot transfers to the cache; a refused
// store is recovered by evicting the stale entry and re-capturing once, because
// a stale held prefix (diverged prompt) would otherwise block all future
// retention for this key and force every later turn to full-prefill.
// Caller MUST hold cb.mu and slot.mu. The one eager-path call from initSlot runs
// before the slot is published to cb.slots/sessionMap, so no other goroutine can
// observe it.
func (cb *ContinuousBatcher) retainPrefixLocked(slot *Slot) {
	if cb.prefixCache == nil || cb.cfg.Model == nil || slot.sess == nil || len(slot.PromptTokens) == 0 {
		return
	}
	snap, err := slot.sess.PrefixSnapshot()
	if err != nil || snap == nil {
		return
	}
	if err := cb.prefixCache.Store(slot.SessionID, slot.PromptTokens, snap); err != nil {
		// Divergence liveness: the stale entry blocks retention for this key, so
		// evict it and store a fresh boundary once more.
		cb.prefixCache.Evict(slot.SessionID)
		if fresh, ferr := slot.sess.PrefixSnapshot(); ferr == nil && fresh != nil {
			_ = cb.prefixCache.Store(slot.SessionID, slot.PromptTokens, fresh)
		}
	}
}

// Cancel cancels a subagent session and frees its slot.
func (cb *ContinuousBatcher) Cancel(sessionID string) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return ErrBatcherClosed
	}

	slot, ok := cb.sessionMap[sessionID]
	if !ok {
		// Check waiting queue
		for i, q := range cb.waitingQueue {
			if q.SessionID == sessionID {
				cb.waitingQueue = append(cb.waitingQueue[:i], cb.waitingQueue[i+1:]...)
				return nil
			}
		}
		return ErrSessionNotFound
	}

	slot.mu.Lock()
	slot.State = SlotStateFinished
	slot.err = context.Canceled
	select {
	case <-slot.doneCh:
	default:
		close(slot.doneCh)
	}
	close(slot.tokenCh)
	slot.mu.Unlock()

	cb.completedMap[sessionID] = slot
	delete(cb.sessionMap, sessionID)

	idx := slot.Index
	cb.slots[idx] = &Slot{
		Index: idx,
		State: SlotStateEmpty,
	}
	cb.tryPromoteWaitingLocked()

	return nil
}

// RequiredKVCacheBytes calculates the required KV cache footprint for a request based on
// token dimensions, KV bytes per token, and recurrent execution depth:
// KV capacity required = (len(PromptTokens) + TargetTokens) * KVBytesPerToken * ExecutionDepth.
func (cb *ContinuousBatcher) RequiredKVCacheBytes(req *SubagentRequest) int64 {
	if req == nil {
		return 0
	}
	depth := int64(req.effectiveDepth())
	bpt := req.KVBytesPerToken
	if bpt <= 0 {
		bpt = cb.cfg.KVBytesPerToken
	}
	if bpt <= 0 {
		bpt = 1024
	}
	targetTokens := req.TargetTokens
	if targetTokens <= 0 {
		targetTokens = req.effectiveDepth()
	}
	tokens := int64(len(req.PromptTokens) + targetTokens)
	return tokens * bpt * depth
}

// CurrentKVCacheBytes returns the total KV cache memory footprint in bytes allocated across active and yielded slots.
func (cb *ContinuousBatcher) CurrentKVCacheBytes() int64 {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.currentKVCacheBytesLocked()
}

// MaxKVCacheBytes returns the configured maximum KV cache memory capacity in bytes (0 = unconstrained).
func (cb *ContinuousBatcher) MaxKVCacheBytes() int64 {
	return cb.cfg.MaxKVCacheBytes
}

func (cb *ContinuousBatcher) currentKVCacheBytesLocked() int64 {
	var total int64
	for _, s := range cb.slots {
		if s.State == SlotStateActiveDecode || s.State == SlotStateYieldedIO {
			total += s.KVCacheBytes
		}
	}
	return total
}

func (cb *ContinuousBatcher) tryPromoteWaitingLocked() []string {
	var promoted []string
	for len(cb.waitingQueue) > 0 {
		emptyIdx := -1
		for i, s := range cb.slots {
			if s.State == SlotStateEmpty {
				emptyIdx = i
				break
			}
		}
		if emptyIdx == -1 {
			break
		}

		req := cb.waitingQueue[0]
		reqBytes := cb.RequiredKVCacheBytes(req)
		if cb.cfg.MaxKVCacheBytes > 0 && cb.currentKVCacheBytesLocked()+reqBytes > cb.cfg.MaxKVCacheBytes {
			break
		}

		cb.waitingQueue = cb.waitingQueue[1:]
		cb.initSlot(emptyIdx, req)
		promoted = append(promoted, req.SessionID)
	}
	return promoted
}

// WaitingQueue returns a shallow copy of the pending subagent requests in the waiting queue.
func (cb *ContinuousBatcher) WaitingQueue() []*SubagentRequest {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	res := make([]*SubagentRequest, len(cb.waitingQueue))
	copy(res, cb.waitingQueue)
	return res
}

// Close closes the batcher and marks remaining active sessions finished.
func (cb *ContinuousBatcher) Close() error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return nil
	}
	cb.closed = true

	for _, slot := range cb.slots {
		slot.mu.Lock()
		if slot.State == SlotStateActiveDecode || slot.State == SlotStateYieldedIO {
			slot.err = ErrBatcherClosed
			select {
			case <-slot.doneCh:
			default:
				close(slot.doneCh)
			}
			close(slot.tokenCh)
		}
		slot.mu.Unlock()
	}

	if cb.prefixCache != nil {
		cb.prefixCache.Clear()
	}

	return nil
}

func argmax(v []float32) int {
	if len(v) == 0 {
		return 0
	}
	m, idx := v[0], 0
	for i, x := range v[1:] {
		if x > m {
			m, idx = x, i+1
		}
	}
	return idx
}
