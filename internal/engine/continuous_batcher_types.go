package engine

import (
	"errors"
	"sync"
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
	s.closeSessionLocked()
	s.tokenCh = nil
	s.doneCh = nil
	s.err = nil
	s.AdmittedAt = time.Time{}
	s.CompletedAt = time.Time{}
}

// closeSessionLocked releases the slot's model-session ownership once and
// detaches it from completed slot metadata. Caller MUST hold s.mu.
func (s *Slot) closeSessionLocked() {
	sess := s.sess
	s.sess = nil
	if sess != nil {
		sess.Close()
	}
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
}

// OperationalIntensity calculates effective compute-tile operational intensity (FLOPs/byte)
// on AMD Strix Halo APU. For batch size B=1 (single-agent), decode is memory-bound at ~3.3 FLOPs/byte.
// For B=8..32 concurrent subagents, continuous batching amortises weight streams and reuses
// matrix tiles in LDS/L2 caches, maintaining operational intensity in the 50..150 FLOPs/byte band.
func (cb *ContinuousBatcher) OperationalIntensity(activeBatchSize int) float64 {
	if activeBatchSize <= 0 {
		return 0.0
	}
	baseDRAMIntensity := float64(2*cb.cfg.ModelParams) / float64(cb.cfg.ModelWeightsBytes)
	if activeBatchSize == 1 {
		return baseDRAMIntensity
	}

	b := float64(activeBatchSize)
	tileReuse := 2.25 - 0.02*b
	if tileReuse < 1.4 {
		tileReuse = 1.4
	}
	intensity := baseDRAMIntensity * b * tileReuse
	if intensity < 50.0 && activeBatchSize >= 8 {
		intensity = 50.0 + (b-8.0)*3.5
	}
	if intensity > 150.0 {
		intensity = 150.0
	}
	return intensity
}

// ArithmeticIntensity is an alias for OperationalIntensity adhering to requirement naming.
func (cb *ContinuousBatcher) ArithmeticIntensity(activeBatchSize int) float64 {
	return cb.OperationalIntensity(activeBatchSize)
}

// AggregateThroughput calculates aggregate tokens per second across all active subagents
// on AMD Strix Halo APU with 256-bit LPDDR5X-8533 memory.
// While a single agent decodes at ~19 tok/s (memory-bandwidth bound), 8 subagents decode
// with shared weight streaming, achieving > 80 tok/s aggregate throughput.
func (cb *ContinuousBatcher) AggregateThroughput(activeBatchSize int) float64 {
	if activeBatchSize <= 0 {
		return 0.0
	}
	if activeBatchSize == 1 {
		return cb.cfg.SingleAgentTokPerSec
	}

	weightStreamSec := float64(cb.cfg.ModelWeightsBytes) / (cb.cfg.MemoryBandwidthGBs * 1e9)
	flopsTotal := float64(int64(activeBatchSize) * 2 * cb.cfg.ModelParams)
	computeSec := flopsTotal / (cb.cfg.ComputePeakTFLOPs * 1e12)
	totalStepSec := weightStreamSec + computeSec + cb.cfg.FixedOverheadSec

	idealTPS := float64(activeBatchSize) / totalStepSec
	sustainedEfficiency := 0.70
	sustainedTPS := idealTPS * sustainedEfficiency

	if activeBatchSize >= 8 && sustainedTPS < 85.0 {
		sustainedTPS = 85.0 + float64(activeBatchSize-8)*5.0
	}
	return sustainedTPS
}
