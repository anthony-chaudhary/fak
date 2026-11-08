package modelengine

// nativesched.go — native continuous batching for the in-kernel engine lifecycle.
//
// NativeScheduler admits many requests, then a SINGLE scheduler loop advances all live
// non-Q4_K lanes with one model.BatchSession StepBatch call per iteration: the shared
// weight-stream move that continuous batching exists to make. It dynamically admits new
// requests between decode steps, retires finished/cancelled lanes immediately, fans each
// token into that lane's own stream, and drops the lane's KV-bearing Session on terminal.
//
// Honest scope. This is the native in-kernel syscall scheduler, not a vLLM-class
// multi-tenant serving scheduler: paged-KV pressure relief, priority/fairness admission,
// and SLA p99 policy live in the gateway admission/preemption leaves that consume this
// seam. Resident Q4_K decode is preserved by falling back to per-lane Session.Step because
// BatchSession does not yet implement the Q4_K kernel.

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/radixkv"

	"github.com/anthony-chaudhary/fak/internal/refutil"
)

var errSchedClosed = errors.New("modelengine: native scheduler closed")

// prefillDrainPerInterval is the number of waiting lanes one prefill-interval
// window drains: exactly one release step per window, and a release step
// advances one prefill lane. A waiting queue deeper than that outruns the
// interval's drain, so the engine records itself capacity-bound and stops
// throttling (the vLLM prefill_capacity_bound escape, scheduler.py:1562-1564).
const prefillDrainPerInterval = 1

// ErrNativePromptExceedsKVCapacity reports a request whose prompt alone can
// never fit the configured native paged-KV budget. Callers may use errors.Is.
var ErrNativePromptExceedsKVCapacity = errors.New("modelengine: prompt exceeds native KV capacity")

// ErrNativeBatchSequenceNotActive reports a proposed decode batch whose
// sequence id is not a member of the scheduler's active set. Callers may use
// errors.Is; BatchSequenceError names the first offending sequence.
var ErrNativeBatchSequenceNotActive = errors.New("modelengine: batch sequence is not in the active set")

// BatchSequenceError identifies the first proposed-batch sequence id outside
// the scheduler's active set, carrying the offending admission UID
// (schedLane.seqNo). It matches ErrNativeBatchSequenceNotActive via errors.Is.
type BatchSequenceError struct {
	SeqNo int64
}

func (e *BatchSequenceError) Error() string {
	return fmt.Sprintf("modelengine: batch sequence %d is not in the active set", e.SeqNo)
}

// Unwrap matches every BatchSequenceError against ErrNativeBatchSequenceNotActive
// so callers can use errors.Is on the sentinel.
func (e *BatchSequenceError) Unwrap() error { return ErrNativeBatchSequenceNotActive }

type nativeAllocationStatus uint8

const (
	nativeAllocationOK nativeAllocationStatus = iota
	nativeAllocationLater
	nativeAllocationNever
)

type schedPrepare struct {
	prompt []int
	tok    NLTokenizer
	q4k    bool
}

type schedPrepareFunc func(context.Context, *abi.ToolCall, *model.Model) schedPrepare

func defaultSchedPrepare(ctx context.Context, c *abi.ToolCall, m *model.Model) schedPrepare {
	return schedPrepare{prompt: tokenize(c.Tool, refutil.Bytes(ctx, c.Args), m.Cfg.VocabSize)}
}

// NativeScheduler is the continuous-batching LifecycleEngine used by the registered
// in-kernel Engine and by tests that register it under a separate id.
type NativeScheduler struct {
	m       *model.Model
	prepare schedPrepareFunc
	now     func() time.Time
	// modelName is fixed by the owning Engine before the scheduler is published.
	// Empty preserves the Default identity fallback for standalone schedulers.
	modelName string

	cachePhaseLatency modelperfobs.CachePhaseLatencyRecorder

	mu sync.Mutex
	// waiting holds admitted-but-not-yet-running lanes (the WAITING queue); lanes
	// holds the RUNNING set the per-iteration StepBatch advances. Admit enqueues into
	// waiting; run() promotes waiting->lanes between steps, EDF when deadlines exist
	// and otherwise FIFO, up to maxRunning.
	waiting []*schedLane
	lanes   []*schedLane
	// maxRunning caps the running set; 0 = unbounded (every admitted lane runs at once,
	// the pre-queue behaviour). A positive cap is the BARE structural admission knob the
	// issue scopes ("the bare admit/evict loop and queues it sits on") — NOT a priority/
	// fairness/KV-budget policy, which is the sibling issue's job.
	maxRunning       int
	maxTokensPerStep int
	promotionPicker  PromotionPicker
	sessionAffinity  map[string]int
	// longPrefillTokenThreshold caps a lone long prefill's per-step advance
	// while other requests are queued (waiting or preempted); 0 disables it.
	// Guarded by mu; set before publication like maxTokensPerStep.
	longPrefillTokenThreshold int
	// maxObservedRunning is the high-water mark of the running set, written only by the
	// run goroutine under mu. It lets a witness assert the waiting queue actually gated
	// (peak == maxRunning) without racing on a live concurrency count.
	maxObservedRunning   int
	maxObservedKVBlocks  int
	sharedBatchSteps     uint64
	sharedPanels         uint64
	sharedMACs           uint64
	q4kGateUpOutputSlab  bool
	residentQ4K          bool
	metalQ4K             bool
	sessionProfiler      func(NativeSessionLifecycle) *model.PhaseProfiler
	captureQwenState     bool
	qwenStateReceipts    []model.Qwen35MetalStateIdentityReceipt
	qwenPrefillTokens    int
	packedPrefillEnabled bool                 // compile-selected serving lane; fixed before publication
	packedPrefillStats   PackedPrefillReceipt // guarded by mu
	qwenPrefillCap       *residentQ4KPrefillCapability
	closed               bool

	// preemption is disabled until MaxBlocks is set. When enabled it treats MaxBlocks as
	// the live paged-KV block budget and preempts running lanes at scheduler boundaries
	// when the running set exceeds that budget.
	preemption         NativePreemptionPolicy
	hostSwapLimitBytes int64 // protected by mu; zero retains legacy unbounded host swap
	preempted          []*schedLane
	seqNo              int64
	preemptRound       int64
	preemptStats       NativePreemptionStats

	// In-batch cold-prefix dedup (fak#1914). prefixFlights coalesces concurrent cold
	// admissions whose prompts share a prefix so only one prefill leader runs the shared
	// GEMM; twins adopt its KV. inBatchDedup is the opt-in switch (default false keeps
	// the historical per-lane prefill byte-for-byte). prefixFlightsMu guards lazy init
	// only; inBatchDedup and inBatchDedupStats are guarded by mu.
	// Prefix state belongs to this scheduler, including admissions that are
	// still acquiring their owner when Close returns. Never detach it at Close.
	prefixStateMu sync.RWMutex
	prefixState   *nativePrefixState

	// LoRA slot-cache currency, adapted from llama.cpp's slot reset: the server
	// compares the global adapter configuration and resets slots whose cached
	// prompt was produced under a previous one, because a prefix hit installed
	// from cache replays KV and logits that only the producing adapters justify.
	// loraAdapters is the last-applied global adapter list and loraGeneration
	// bumps once per genuine change; both are guarded by mu (not prefixStateMu).
	// A genuine change invalidates every recorded prefix hit so affected lanes
	// cold-prefill under the new adapters instead of consuming stale KV.
	loraAdapters   []string
	loraGeneration uint64

	prefixFlights     *radixkv.PrefixFlightGroup
	prefixFlightsMu   sync.Mutex
	inBatchDedup      bool
	inBatchDedupStats InBatchPrefixDedupStats
	// prefillFlightHook is an unexported test witness; nil in production.
	prefillFlightHook func()

	coupler *WorkerCoupler

	// fwd is the forward-pass slot (Continual Engine phase 0): every decode
	// step loads the live generation once, before its forward call, so the
	// whole step runs on one generation. Initialized with the default
	// (pre-slot) impl; a hot swap never changes this pointer.
	fwd *ForwardSlot

	// elastic is the FAK_NATIVE_KV_ELASTIC_BLOCKS block ledger; nil (default) is off.
	// Set once before the scheduler is published.
	elastic *nativeElasticLedger

	// Spec admission padding, adapted from vLLM's pad_spec_decode (source pin in
	// the commit body). With specPadK > 0, a waiting lane promoted into a
	// spec-carrying decode step reserves specPadK placeholder draft slots so its
	// first decode step rides the same fixed-shape panel as the resident
	// 1+specPadK lanes; a step whose token budget cannot carry the padded cost
	// holds the lane in waiting instead of admitting a short row off the
	// captured bucket. Zero (the default) keeps historical admission
	// byte-for-byte. Guarded by mu; written by SetSpecAdmissionPad, read only
	// by the executor.
	specPadK int
	// specAdmissionPads/specAdmissionPadSlots count pad_spec_decode reservations
	// made at admission time: one reservation per padded promotion, plus the
	// placeholder draft slots it reserved. Guarded by mu; written only by the
	// executor.
	specAdmissionPads     uint64
	specAdmissionPadSlots uint64

	// sessionDecorator decorates every lane session this scheduler mints (fresh
	// and restored) before use — e.g. binding the device backend a P/D decode
	// role resolved. Set once before the first Admit; nil keeps the historical
	// session shape byte-for-byte.
	sessionDecorator func(*model.Session)

	// skipDrafts* carries the vLLM skip_drafts/_drafts_unused adaptation: the
	// per-execution verdicts that tell the MTP self-speculation path to propose
	// K=0 drafts on steps whose drafts are SURE to be unused. Counters are
	// guarded by mu; the observer, when set, is invoked outside it, in
	// execution order.
	skipDraftsObserver       func(SkipDraftsDecision)
	skipDraftsSteps          uint64
	skipDraftsSkipped        uint64
	skipDraftsPrefillChunks  uint64
	skipDraftsTerminalTokens uint64

	wake    chan struct{} // buffered(1): Admit/Close nudge an idle loop
	started sync.Once
	stopped chan struct{}

	runStarted bool

	// executor serializes lane mutation between the ordinary scheduler goroutine and
	// Complete callers that donate their blocked drain to advancing scheduler work.
	// The goroutine remains the fallback whenever donation is disabled or no Complete
	// drain is waiting.
	executor          sync.Mutex
	drainDonation     bool
	blockedDrains     int
	donatedIterations uint64
	iteration         uint64

	// Local prefill cadence (vLLM prefill_schedule_interval port, ADAPT): hold
	// prefill to every Nth engine step so decode runs in longer uninterrupted
	// stretches. prefillInterval <= 1 keeps today's mixed-step order bit-for-bit.
	// lastPrefillStep is the executed-step index of the last prefill-carrying
	// step, armed at -prefillInterval so the first step is never gated;
	// executedSteps counts iterations that scheduled tokens; prefillCapacityBound
	// records a backed-up engine whose waiting queue outruns one interval's
	// drain, which stops throttling so the queue cannot grow without bound. All
	// four are written only by runIteration under mu.
	prefillInterval      int
	lastPrefillStep      int
	executedSteps        int
	prefillCapacityBound bool

	closeSession       func(*model.Session)
	observeNativeEvent func(nativeSchedulerEvent)
	beforeModelExecute func(nativeSchedulerEventKind, *schedLane)

	// recorder receives the per-iteration native scheduler boundary (cohort size,
	// decode step, prefill, queue depth) as fak_engine_* metrics. It defaults to
	// enginestep.Default; a zero-value NativeScheduler leaves it nil and every
	// Observe* call is a no-op (enginestep guards a nil receiver). It is unexported
	// so production callers cannot redirect the process recorder; in-package tests
	// inject a private recorder instead of mutating enginestep.Default.
	recorder *enginestep.Recorder
}

// NativeSessionLifecycle identifies whether a scheduler-created model session is
// the initial owner or a readmitted owner rebuilt around restored KV state.
type NativeSessionLifecycle string

const (
	NativeSessionFresh    NativeSessionLifecycle = "fresh"
	NativeSessionRestored NativeSessionLifecycle = "restored"
)

// SetMaxRunning bounds how many admitted lanes run concurrently; the rest wait in the
// waiting queue and are promoted by deadline, or FIFO without deadlines, as slots free. n<=0 means
// unbounded (the default). Set it before the first Admit; it is read by the run loop.
func (s *NativeScheduler) SetMaxRunning(n int) {
	s.mu.Lock()
	s.maxRunning = n
	s.mu.Unlock()
	s.signal()
}

// SetWorkerCoupler installs a custom or reconfigured worker coupler on this scheduler.
func (s *NativeScheduler) SetWorkerCoupler(c *WorkerCoupler) {
	s.mu.Lock()
	s.coupler = c
	s.mu.Unlock()
}

// WorkerCoupler returns the active worker coupler.
func (s *NativeScheduler) WorkerCoupler() *WorkerCoupler {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.coupler
}

// SetSessionDecorator installs the per-lane session decorator applied to every
// fresh and restored lane session before use. Set it before the first Admit;
// like the profiler hook it is read lock-free at mint time, so a post-boot
// write cannot silently change the shape of already-minted sessions. nil
// clears the decorator.
func (s *NativeScheduler) SetSessionDecorator(d func(*model.Session)) {
	s.mu.Lock()
	s.sessionDecorator = d
	s.mu.Unlock()
}

// WorkerCouplingStats returns telemetry for this scheduler's worker coupler.
func (s *NativeScheduler) WorkerCouplingStats() WorkerCouplingStats {
	coupler := s.WorkerCoupler()
	if coupler == nil {
		return WorkerCouplingStats{}
	}
	return coupler.Stats()
}

// WriteWorkerCouplingMetrics renders Prometheus-format metrics for the worker coupler.
func (s *NativeScheduler) WriteWorkerCouplingMetrics(b *strings.Builder) {
	coupler := s.WorkerCoupler()
	if coupler != nil {
		coupler.WriteMetrics(b)
	}
}

// SetQ4KGateUpOutputSlab selects the explicit session-owned slab for Q4_K lanes.
// Set it before Admit; applying it to every session constructor keeps preemption
// restore and ordinary admission behavior identical.
func (s *NativeScheduler) SetQ4KGateUpOutputSlab(enabled bool) {
	s.mu.Lock()
	s.q4kGateUpOutputSlab = enabled
	s.mu.Unlock()
}

// SetResidentQ4K selects the scheduler-owned resident Q4_K lane and optionally
// its fak-native Metal backend. Set it before Admit. Restored sessions inherit the
// same ownership; readmission never changes engine or backend.
func (s *NativeScheduler) SetResidentQ4K(metal bool) {
	s.mu.Lock()
	s.residentQ4K = true
	s.metalQ4K = metal
	s.mu.Unlock()
}

// SetSessionProfilerFactory attaches an existing model phase/Metal profiler to
// every fresh and restored session. It is an opt-in witness seam; nil is the
// allocation-free production default.
func (s *NativeScheduler) SetSessionProfilerFactory(factory func(NativeSessionLifecycle) *model.PhaseProfiler) {
	s.mu.Lock()
	s.sessionProfiler = factory
	s.mu.Unlock()
}

// EnableQwen35MetalStateIdentityCapture opts exact P32 Qwen Metal admissions
// into the model-owned state-digest receipt. Unsupported sessions fail Admit
// rather than silently omitting the requested witness.
func (s *NativeScheduler) EnableQwen35MetalStateIdentityCapture() {
	s.mu.Lock()
	s.captureQwenState = true
	s.mu.Unlock()
}

// Qwen35MetalStateIdentityReceipts returns independent copies captured after
// fresh P32 prefills and before any scheduler preemption.
func (s *NativeScheduler) Qwen35MetalStateIdentityReceipts() []model.Qwen35MetalStateIdentityReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]model.Qwen35MetalStateIdentityReceipt, len(s.qwenStateReceipts))
	for i, receipt := range s.qwenStateReceipts {
		out[i] = receipt
		out[i].States = append([]model.Qwen35MetalStateDigest(nil), receipt.States...)
	}
	return out
}

// MaxObservedRunning reports the peak running-set size the loop reached — the witness
// that a maxRunning cap actually gated admission (peak == cap), or that an uncapped
// scheduler co-batched every lane (peak == #admitted). Safe to read after draining.
func (s *NativeScheduler) MaxObservedRunning() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxObservedRunning
}

// ActiveLanes reports the live lane population: non-terminal lanes across the
// WAITING, RUNNING, and preempted sets. The P/D gateway router reads it as the
// peer load counter behind fewest-active-lanes selection; zero means no work
// is queued or running on this scheduler.
func (s *NativeScheduler) ActiveLanes() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, ln := range s.lanes {
		if !ln.terminal {
			n++
		}
	}
	for _, ln := range s.waiting {
		if !ln.terminal {
			n++
		}
	}
	for _, ln := range s.preempted {
		if !ln.terminal {
			n++
		}
	}
	return n
}

// MaxObservedKVBlocks reports the peak live paged-KV block estimate observed
// before pressure enforcement. It proves the ON arm crossed its configured
// budget rather than merely carrying a nonzero swap counter.
func (s *NativeScheduler) MaxObservedKVBlocks() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxObservedKVBlocks
}

// SharedWorkReceipt reports model work that actually crossed a shared BatchSession
// panel. Running-set size alone is admission evidence; nonzero Panels and MACs are
// the execution receipt that one set of weights served at least two live lanes.
type SharedWorkReceipt struct {
	Steps  uint64
	Panels uint64
	MACs   uint64
}

// CachePhaseLatencyReceipt reports fak-native KV-bearing prefill/decode latency
// with a bounded pipeline-phase label and a reconciling unlabeled total.
func (s *NativeScheduler) CachePhaseLatencyReceipt() modelperfobs.CachePhaseLatencyReceipt {
	return s.cachePhaseLatency.Receipt()
}

func (s *NativeScheduler) SharedWorkReceipt() SharedWorkReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()
	return SharedWorkReceipt{
		Steps:  s.sharedBatchSteps,
		Panels: s.sharedPanels,
		MACs:   s.sharedMACs,
	}
}

// SetLoRAAdapters applies the global LoRA adapter configuration to the
// scheduler's prefix-cache currency. A configuration that differs from the
// last-applied one bumps the LoRA generation and invalidates every recorded
// prefix hit, so a lane that would have consumed cached KV or logits produced
// under the previous adapters cold-prefills instead (llama.cpp slot reset:
// common/server.cpp compares the global adapter list per request and resets
// slots whose cached prompt predates the change). Re-applying the identical
// configuration is a no-op that preserves recorded hits. Adapter loading is
// untouched; the shared radix tree's entries are not yet generation-keyed, so
// only the recorded per-slot hits are invalidated here.
func (s *NativeScheduler) SetLoRAAdapters(adapters []string) {
	s.mu.Lock()
	changed := !slices.Equal(s.loraAdapters, adapters)
	if changed {
		s.loraAdapters = append([]string(nil), adapters...)
		s.loraGeneration++
	}
	s.mu.Unlock()
	if changed {
		s.invalidateRecordedPrefixHits()
	}
}

// LoRAGeneration reports the current LoRA configuration generation: bumped
// exactly once per genuine global adapter-config change and stable across
// identical re-application.
func (s *NativeScheduler) LoRAGeneration() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loraGeneration
}

// invalidateRecordedPrefixHits releases every recorded prefix-hit boundary
// lease and drops the per-session and per-lane hit records, so the next
// prefill iteration cold-prefills instead of consuming KV produced under a
// superseded LoRA configuration. Lane hits alias their holder's record, so
// each distinct record is released exactly once; this mirrors the per-session
// cleanup in clearPrefixLookup at scheduler scope.
func (s *NativeScheduler) invalidateRecordedPrefixHits() {
	st := s.getPrefixState()
	if st == nil {
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	released := make(map[*prefixHitInfo]struct{}, len(st.holderLookups)+len(st.laneLookups))
	for _, info := range st.holderLookups {
		if info == nil {
			continue
		}
		if _, seen := released[info]; seen {
			continue
		}
		released[info] = struct{}{}
		if info.boundary != nil && info.ownerTree != nil {
			info.ownerTree.Done(info.boundary)
			info.boundary = nil
		}
	}
	for _, info := range st.laneLookups {
		if info == nil {
			continue
		}
		if _, seen := released[info]; seen {
			continue
		}
		released[info] = struct{}{}
		if info.boundary != nil && info.ownerTree != nil {
			info.ownerTree.Done(info.boundary)
			info.boundary = nil
		}
	}
	st.holderLookups = make(map[*model.Session]*prefixHitInfo)
	st.laneLookups = make(map[*schedLane]*prefixHitInfo)
}

// SpecAdmissionPadReceipt reports pad_spec_decode admission activity: how many
// promotions reserved placeholder draft slots and how many slots were reserved
// in total. All-zero means no padded admission ever ran.
type SpecAdmissionPadReceipt struct {
	Pads  uint64
	Slots uint64
}

func (s *NativeScheduler) SpecAdmissionPadReceipt() SpecAdmissionPadReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()
	return SpecAdmissionPadReceipt{Pads: s.specAdmissionPads, Slots: s.specAdmissionPadSlots}
}

// SkipDraftsReason is the closed vocabulary for a skip=true verdict: why the
// drafts a drafter would stage after the step are SURE to be unused.
type SkipDraftsReason string

const (
	// SkipDraftsPrefillChunk marks a prefill execution that ran only
	// non-final chunks: every scheduled lane sampled nothing, so no next
	// decode step exists to verify drafts against.
	SkipDraftsPrefillChunk SkipDraftsReason = "prefill_chunk"
	// SkipDraftsTerminalToken marks a step after which every scheduled lane
	// left without a draft-consuming next step: each lane either retired or
	// staged its last output token (EOS observed, or the genTokens cap
	// reached even if every draft rejects). Stop tokens and strings cannot
	// be known in advance, so no other ground earns skip=true.
	SkipDraftsTerminalToken SkipDraftsReason = "terminal_token"
)

// SkipDraftsDecision is one completed model execution's skip-drafts verdict.
// Skip=true tells the MTP self-speculation path to propose K=0 drafts: the
// drafts it would stage after this execution are SURE to be unused, so the
// drafter forward and the draft KV writes it would make are skipped entirely.
// Acceptance accounting of previously staged drafts is unaffected.
type SkipDraftsDecision struct {
	Iteration uint64
	Skip      bool
	Reason    SkipDraftsReason // zero when Skip is false
}

// SkipDraftsReceipt counts skip-drafts verdicts since scheduler construction.
// Steps counts every prefill/decode phase that produced a verdict; Skipped
// counts the skip=true subset, split by the closed reason vocabulary.
type SkipDraftsReceipt struct {
	Steps          uint64
	Skipped        uint64
	PrefillChunks  uint64
	TerminalTokens uint64
}

// SetSkipDraftsObserver installs the per-execution verdict sink the MTP
// self-speculation path consumes to skip the drafter forward (K=0 drafts).
// Verdicts arrive in execution order, outside the scheduler mutex. Nil
// (default) keeps the scheduler allocation-free; in-package tests inject a
// witness instead.
func (s *NativeScheduler) SetSkipDraftsObserver(obs func(SkipDraftsDecision)) {
	s.mu.Lock()
	s.skipDraftsObserver = obs
	s.mu.Unlock()
}

// SkipDraftsReceipt reports the skip-drafts verdict ledger.
func (s *NativeScheduler) SkipDraftsReceipt() SkipDraftsReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()
	return SkipDraftsReceipt{
		Steps:          s.skipDraftsSteps,
		Skipped:        s.skipDraftsSkipped,
		PrefillChunks:  s.skipDraftsPrefillChunks,
		TerminalTokens: s.skipDraftsTerminalTokens,
	}
}

// laneDraftsUnusedLocked reports whether drafts proposed after the execution
// that just completed are SURE to be unused for this lane. Adaptation of
// vLLM's _drafts_unused predicate (vllm/v1/core/sched/scheduler.py:483-525,
// with the can_skip_drafts gate :311-315 @ae065918, Apache-2.0): drafts are
// verified in the lane's next decode step, and that step either never samples
// (a non-final prefill chunk, num_computed+num_new < num_tokens) or publishes
// the lane's last output token. Caller holds s.mu.
func (s *NativeScheduler) laneDraftsUnusedLocked(ln *schedLane, prefillPhase bool) (bool, SkipDraftsReason) {
	if ln.terminal {
		return true, SkipDraftsTerminalToken
	}
	if prefillPhase {
		// Still PREFILLING after its chunk: the cursor sits short of the
		// prompt tail, so the step sampled nothing. A lane that transitioned
		// to DECODE (final chunk, or a full prefix-cache hit that consumed
		// the whole prompt without a forward) keeps its draft opportunity in
		// the decode phase and reports not-unused here — conservative, never
		// a missed skip.
		return ln.state == schedLanePrefilling, SkipDraftsPrefillChunk
	}
	// Decode phase: the step's sampled token is the freshly staged deferred
	// token; done marks the last output token (EOS or the genTokens cap).
	return ln.deferredToken != nil && ln.deferredToken.done, SkipDraftsTerminalToken
}

// noteStepSkipDrafts evaluates and records the skip-drafts verdict for one
// completed model execution over its scheduled lane set. prefillPhase marks
// an execution that carried only prefill chunks; otherwise it is a decode
// step. The verdict is the AND over scheduled lanes of
// laneDraftsUnusedLocked — vLLM proposes K=0 drafts only when the predicate
// holds for EVERY scheduled request (scheduler.py:1528-1530 per-request OR
// into drafts_needed, :1693 skip_drafts=not drafts_needed @ae065918).
// Caller must not hold s.mu; the observer, when set, is invoked outside it.
func (s *NativeScheduler) noteStepSkipDrafts(iteration uint64, prefillPhase bool, lanes []*schedLane) {
	if len(lanes) == 0 {
		return
	}
	s.mu.Lock()
	skip := true
	var reason SkipDraftsReason
	for _, ln := range lanes {
		unused, r := s.laneDraftsUnusedLocked(ln, prefillPhase)
		if !unused {
			skip = false
			reason = "" // a keep-drafts verdict carries no reason
			break
		}
		if reason == "" {
			reason = r
		}
	}
	s.skipDraftsSteps++
	if skip {
		s.skipDraftsSkipped++
		switch reason {
		case SkipDraftsPrefillChunk:
			s.skipDraftsPrefillChunks++
		case SkipDraftsTerminalToken:
			s.skipDraftsTerminalTokens++
		}
	}
	obs := s.skipDraftsObserver
	s.mu.Unlock()
	if obs != nil {
		obs(SkipDraftsDecision{Iteration: iteration, Skip: skip, Reason: reason})
	}
}

// noteQwenPrefillSkipDrafts records the verdict for the qwen bounded-prefill
// route once advanceQwenPrefill returns. The chunk counts as executed only
// when the lane's cursor actually advanced (budget-starved steps yield
// unchanged; error paths finish the lane without moving it), so no verdict
// is minted for a step that ran no forward. Caller must not hold s.mu.
func (s *NativeScheduler) noteQwenPrefillSkipDrafts(iteration uint64, ln *schedLane, cursorBefore int) {
	s.mu.Lock()
	advanced := ln.promptCursor > cursorBefore
	s.mu.Unlock()
	if advanced {
		s.noteStepSkipDrafts(iteration, true, []*schedLane{ln})
	}
}

// notePackedPrefillSkipDrafts records the verdict for a packed-prefill
// execution over the lanes whose cursors advanced; declined batches move no
// cursor and mint no verdict. cursors parallels candidates. Caller must not
// hold s.mu.
func (s *NativeScheduler) notePackedPrefillSkipDrafts(iteration uint64, candidates []*schedLane, cursors []int) {
	s.mu.Lock()
	advanced := make([]*schedLane, 0, len(candidates))
	for i, ln := range candidates {
		if ln.promptCursor > cursors[i] {
			advanced = append(advanced, ln)
		}
	}
	s.mu.Unlock()
	s.noteStepSkipDrafts(iteration, true, advanced)
}

// NewNativeScheduler builds a scheduler over an already-constructed model (a real
// export or model.NewSynthetic). It is deliberately NOT auto-registered as a second
// kernel engine id; the registered "inkernel" Engine owns its own scheduler.
func NewNativeScheduler(m *model.Model) *NativeScheduler {
	return newNativeScheduler(m, nil)
}

func newNativeScheduler(m *model.Model, prepare schedPrepareFunc) *NativeScheduler {
	if prepare == nil {
		prepare = defaultSchedPrepare
	}
	s := &NativeScheduler{
		m:             m,
		prepare:       prepare,
		now:           time.Now,
		wake:          make(chan struct{}, 1),
		stopped:       make(chan struct{}),
		drainDonation: true,
		coupler:       NewDefaultWorkerCoupler(),
		fwd:           NewForwardSlot(DefaultForwardImpl()),
		closeSession:  func(sess *model.Session) { sess.Close() },
		recorder:      enginestep.Default,
	}
	if m != nil && (m.Q4KCount() > 0 || m.Q2Count() > 0) {
		s.qwenPrefillCap = &residentQ4KPrefillCapability{model: m}
		if m.Q2Count() > 0 {
			// A PQ2-only Bonsai checkpoint has no Q4_K weights, but its packed
			// matmuls require the same resident session kernel and Qwen hybrid
			// prefill route. Select that lane from model-owned residency even
			// when the caller did not set the historical Q4K intent flag.
			s.residentQ4K = true
		}
	}
	return s
}

// SetDrainDonation selects whether a blocked Complete drain may execute native
// scheduler iterations while it awaits its own tokens. Enabled by default. Disable
// it before the first Admit to retain the scheduler-goroutine-only oracle/fallback.
func (s *NativeScheduler) SetDrainDonation(enabled bool) {
	s.mu.Lock()
	s.drainDonation = enabled
	s.mu.Unlock()
	s.signal()
}

// DrainDonationReceipt is the execution witness for await-caller donation. Each
// iteration represents native scheduler work executed by a blocked Complete caller,
// not by a substitute engine or runtime.
type DrainDonationReceipt struct {
	Iterations uint64
}

func (s *NativeScheduler) DrainDonationReceipt() DrainDonationReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()
	return DrainDonationReceipt{Iterations: s.donatedIterations}
}

// Caps advertises the lifecycle seam (so a consumer negotiates streaming/cancel
// without a type assertion) plus the scheduler's own id token.
func (s *NativeScheduler) Caps() []abi.Capability {
	return []abi.Capability{"engine.native-sched", "engine.continuous-batching", abi.EngineLifecycleCap}
}

// WeightBearing declares that the native scheduler runs model-forwards.
func (s *NativeScheduler) WeightBearing() bool { return true }

// Admit registers one request. By default it preserves the historical synchronous
// prefill-before-enqueue path. A qualification-selected dense host packed lane or
// an explicitly enabled, supported Qwen Q4_K lane instead
// enters WAITING in PREFILLING state and lets runIteration advance one bounded prompt
// chunk before returning to decode work. A surviving lane's output remains independent
// of promotion timing because each lane owns its KV and decode state.
// Optional c.Meta["priority"] is a signed decimal integer (lower is more important),
// captured at admission. Missing, malformed, or out-of-range values default to 0.
// Priority is metadata only; it does not change promotion or preemption policy.
func (s *NativeScheduler) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	return s.AdmitWithHint(ctx, c, dispatchtick.WaveHint{})
}

func (s *NativeScheduler) AdmitWithHint(ctx context.Context, c *abi.ToolCall, hint dispatchtick.WaveHint) (abi.EngineRequest, error) {
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return nil, errSchedClosed
	}
	return s.admitPrepared(ctx, c, hint, s.prepare(ctx, c, s.m))
}

// AdmitTokenIDs admits one exact token sequence through the ordinary native
// scheduler lifecycle. It exists for modelbench and deterministic engine
// witnesses that must bind exact artifact token IDs instead of byte tokenization.
func (s *NativeScheduler) AdmitTokenIDs(ctx context.Context, name string, prompt []int) (abi.EngineRequest, error) {
	call := &abi.ToolCall{Tool: name}
	return s.admitPrepared(ctx, call, dispatchtick.WaveHint{}, schedPrepare{prompt: append([]int(nil), prompt...)})
}

// ValidateBatchSequences checks a caller-composed decode batch against the
// scheduler's active set BEFORE dispatch: every sequence id must belong to a
// non-terminal lane this scheduler admitted — running (lanes), waiting, or
// preempted. The first unknown sequence in input order is refused with a
// BatchSequenceError, and no slot, queue, counter, or session state changes on
// refusal. Known sequences pass in their given order; the check reorders
// nothing and assigns no slots.
//
// Port of the batch sequence-membership validation in ggml-org/llama.cpp
// llama_batch_allocr::init at pinned revision
// 25603ef89168a570ea4f34803ce36531c05fd357 (MIT), referenced at
// src/llama-batch.cpp:475 (PR #30386): a batch carrying a sequence outside the
// allocated set is refused before the allocator touches slot state instead of
// corrupting it. Concept adaptation, not copied code: the upstream guard
// protects per-token sequence-set bookkeeping; here the active set is the
// scheduler's live lane population keyed by the stable admission UID
// (schedLane.seqNo).
func (s *NativeScheduler) ValidateBatchSequences(seqNos []int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	active := make(map[int64]struct{}, len(s.lanes)+len(s.waiting)+len(s.preempted))
	for _, ln := range s.lanes {
		if !ln.terminal {
			active[ln.seqNo] = struct{}{}
		}
	}
	for _, ln := range s.waiting {
		if !ln.terminal {
			active[ln.seqNo] = struct{}{}
		}
	}
	for _, ln := range s.preempted {
		if !ln.terminal {
			active[ln.seqNo] = struct{}{}
		}
	}
	for _, seqNo := range seqNos {
		if _, ok := active[seqNo]; !ok {
			return &BatchSequenceError{SeqNo: seqNo}
		}
	}
	return nil
}

func (s *NativeScheduler) admitPrepared(ctx context.Context, c *abi.ToolCall, hint dispatchtick.WaveHint, prep schedPrepare) (abi.EngineRequest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errSchedClosed
	}
	s.mu.Unlock()

	prompt := prep.prompt
	if len(prompt) == 0 {
		prompt = []int{0}
	}
	q4k := prep.q4k || s.residentQ4K
	kvReuseHits, kvPinned, kvPinUntil := nativeKVBMHintsFromMeta(c.Meta)
	priority, err := strconv.Atoi(strings.TrimSpace(c.Meta["priority"]))
	if err != nil {
		priority = 0
	}

	cctx, cancel := context.WithCancel(ctx)
	ln := &schedLane{
		sched:       s,
		ctx:         cctx,
		cancel:      cancel,
		tool:        c.Tool,
		prompt:      append([]int(nil), prompt...),
		putCtx:      ctx,
		tok:         prep.tok,
		q4k:         q4k,
		prefillQ4K:  prep.q4k,
		kvReuseHits: kvReuseHits,
		kvPinned:    kvPinned,
		kvPinUntil:  kvPinUntil,
		tokens:      newNativeRequestOutbox(),
		done:        make(chan struct{}),
		hint:        hint,
		priority:    priority,
	}

	s.mu.Lock()
	if s.qwenPrefillTokens >= nativeQwenPrefillMinChunkTokens &&
		prep.q4k &&
		q4k &&
		s.qwenPrefillCap != nil &&
		s.qwenPrefillCap.model == s.m &&
		nativeQwenResidentAppendEligible(s.m.Cfg, len(prompt)) {
		ln.prefillChunkTokens = s.qwenPrefillTokens
	}
	if s.packedPrefillCandidateLocked(ln) {
		ln.packedPrefill = true
		ln.prefillChunkTokens = model.PackedPrefillMaxTokens
	}
	capacityGateEnabled := s.effectiveMaxRunningLocked() > 0 && s.preemptionEnabledLocked()
	if capacityGateEnabled {
		allocation := s.canMaterializeLaneLocked(ln)
		if allocation == nativeAllocationNever {
			s.mu.Unlock()
			cancel()
			return nil, ErrNativePromptExceedsKVCapacity
		}
		deferredAhead := len(s.preempted) > 0
		for _, waitingLane := range s.waiting {
			if !waitingLane.materializing && waitingLane.sess == nil {
				deferredAhead = true
			}
		}
		deferMaterialization := deferredAhead || allocation == nativeAllocationLater
		if deferMaterialization {
			if s.closed {
				s.mu.Unlock()
				cancel()
				return nil, errSchedClosed
			}
			if err := ctx.Err(); err != nil {
				s.mu.Unlock()
				cancel()
				return nil, err
			}
			s.seqNo++
			ln.seqNo = s.seqNo
			s.enqueueNewWaitingLocked(ln)
			materializationPending := false
			for _, waitingLane := range s.waiting {
				if waitingLane.materializing {
					materializationPending = true
					break
				}
			}
			s.mu.Unlock()
			if !materializationPending {
				s.started.Do(func() {
					s.mu.Lock()
					s.runStarted = true
					s.mu.Unlock()
					go s.run()
				})
			}
			s.signal()
			return ln, nil
		}
		ln.materializing = true
		s.seqNo++
		ln.seqNo = s.seqNo
		s.enqueueNewWaitingLocked(ln)
		s.mu.Unlock()

		candidate := *ln
		initializationErr := s.initializeLaneExecution(&candidate)
		s.mu.Lock()
		removeReservation := func() {
			for index, waitingLane := range s.waiting {
				if waitingLane == ln {
					s.waiting = append(s.waiting[:index], s.waiting[index+1:]...)
					return
				}
			}
		}
		switch {
		case initializationErr != nil:
			removeReservation()
		case s.closed:
			removeReservation()
			s.closeLaneSession(candidate.sess)
			initializationErr = errSchedClosed
		case ctx.Err() != nil:
			removeReservation()
			s.closeLaneSession(candidate.sess)
			initializationErr = ctx.Err()
		default:
			switch s.canMaterializeLaneLocked(ln) {
			case nativeAllocationNever:
				removeReservation()
				s.closeLaneSession(candidate.sess)
				initializationErr = ErrNativePromptExceedsKVCapacity
			case nativeAllocationLater:
				s.closeLaneSession(candidate.sess)
				ln.materializing = false
			default:
				ln.sess = candidate.sess
				ln.logits = candidate.logits
				ln.state = candidate.state
				ln.promptCursor = candidate.promptCursor
				ln.promptLen = candidate.promptLen
				ln.prefillChunkTokens = candidate.prefillChunkTokens
				ln.packedPrefill = candidate.packedPrefill
				ln.materializing = false
			}
		}
		waitingRemains := len(s.waiting) > 0
		s.mu.Unlock()
		if initializationErr != nil {
			if waitingRemains {
				s.started.Do(func() {
					s.mu.Lock()
					s.runStarted = true
					s.mu.Unlock()
					go s.run()
				})
				s.signal()
			}
			cancel()
			return nil, initializationErr
		}
		s.started.Do(func() {
			s.mu.Lock()
			s.runStarted = true
			s.mu.Unlock()
			go s.run()
		})
		s.signal()
		return ln, nil
	}
	s.mu.Unlock()

	if err := s.initializeLaneExecution(ln); err != nil {
		cancel()
		return nil, err
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		// A shutdown that began during synchronous prefill has already retired the
		// run loop; publishing this lane would orphan its request and KV-bearing
		// session. Reject it and release the freshly created session exactly once.
		cancel()
		s.closeLaneSession(ln.sess)
		return nil, errSchedClosed
	}
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		cancel()
		s.closeLaneSession(ln.sess)
		return nil, err
	}
	s.seqNo++
	ln.seqNo = s.seqNo
	s.enqueueNewWaitingLocked(ln)
	s.mu.Unlock()

	s.started.Do(func() {
		s.mu.Lock()
		s.runStarted = true
		s.mu.Unlock()
		go s.run()
	})
	s.signal()
	return ln, nil
}

// enqueueNewWaitingLocked publishes a fresh lane's first queue entry. The
// timestamp belongs to this transition, not request arrival or synchronous
// prefill before enqueue. Caller holds mu; requeues must not call this helper.
func (s *NativeScheduler) enqueueNewWaitingLocked(ln *schedLane) {
	if s.recorder != nil {
		ln.queueEnteredAt = s.now()
		ln.queueWaitPending = true
	}
	s.waiting = append(s.waiting, ln)
}

func (s *NativeScheduler) initializeLaneExecution(ln *schedLane) error {
	sess := s.newLaneSession(ln.q4k, NativeSessionFresh)
	if s.captureQwenState {
		if err := sess.EnableQwen35MetalStateIdentityReceipt(ln.prompt); err != nil {
			sess.Close()
			return err
		}
	}
	prefillChunkTokens := s.qwenPrefillChunkBudget(schedPrepare{prompt: ln.prompt, tok: ln.tok, q4k: ln.prefillQ4K}, sess, len(ln.prompt))
	if ln.packedPrefill {
		if sess.PackedPrefillEligible() {
			prefillChunkTokens = model.PackedPrefillMaxTokens
		} else {
			ln.packedPrefill = false
		}
	}
	state := schedLaneDecode
	promptCursor := len(ln.prompt)
	promptLen := len(ln.prompt)
	var logits []float32
	if prefillChunkTokens == 0 {
		if s.inBatchDedupEligible(sess, ln.prompt) {
			var err error
			logits, err = s.prefillCoalesced(ln.ctx, sess, ln.prompt)
			if err != nil {
				s.closeLaneSession(sess)
				return err
			}
		} else {
			logits = s.coldPrefillSync(sess, ln.prompt)
		}
		if s.captureQwenState {
			executed, err := sess.FinalizeQwen35MetalStateIdentityReceipt()
			if err != nil || !executed {
				sess.Close()
				if err == nil {
					err = errors.New("modelengine: Qwen Metal state identity was not executed")
				}
				return err
			}
			receipt, ok := sess.Qwen35MetalStateIdentityReceipt()
			if !ok {
				sess.Close()
				return errors.New("modelengine: Qwen Metal state identity receipt unavailable")
			}
			s.mu.Lock()
			s.qwenStateReceipts = append(s.qwenStateReceipts, receipt)
			s.mu.Unlock()
		}
	} else {
		state = schedLanePrefilling
		promptCursor = 0
		promptLen = 0
	}
	ln.sess = sess
	ln.logits = logits
	ln.state = state
	ln.promptCursor = promptCursor
	ln.promptLen = promptLen
	ln.prefillChunkTokens = prefillChunkTokens
	s.elastic.laneStarted(ln, len(ln.prompt))
	return nil
}

// Complete is the one-shot shim every LifecycleEngine offers so it also satisfies
// the bare EngineDriver: admit, drain the stream, return the assembled turn.
func (s *NativeScheduler) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	donate := s.beginBlockedDrain()
	if donate {
		defer s.endBlockedDrain()
	}
	req, err := s.Admit(ctx, c)
	if err != nil {
		return nil, err
	}
	if donate {
		s.drainWithDonation(req.Tokens())
	} else {
		for range req.Tokens() {
		}
	}
	res, err := req.Result()
	if err != nil {
		return nil, err
	}
	if res != nil && res.Call == nil {
		res.Call = c
	}
	return res, nil
}

// Close aborts every outstanding request and stops the scheduler. It cancels each
// live lane's context so the run loop unblocks — even a lane wedged on an undrained
// send — and exits once the lanes retire, so a non-draining consumer can no longer
// leak the loop. Idempotent. A host that wants in-flight requests to FINISH rather
// than abort must drain them before calling Close.
func (s *NativeScheduler) Close() {
	s.mu.Lock()
	s.closed = true
	for _, ln := range s.lanes {
		ln.cancel() // unblock a lane wedged on an undrained send so run() can exit
	}
	for _, ln := range s.waiting {
		ln.cancel() // a never-promoted waiting lane is aborted too, so run() can exit
	}
	for _, ln := range s.preempted {
		ln.cancel() // a swapped/recompute-held lane is aborted too
	}
	s.mu.Unlock()
	s.signal()
}

// CloseAndWait closes the scheduler and waits until its loop has retired every
// terminal lane. A scheduler that never admitted work is already stopped.
func (s *NativeScheduler) CloseAndWait(ctx context.Context) error {
	s.Close()
	s.mu.Lock()
	runStarted := s.runStarted
	s.mu.Unlock()
	if !runStarted {
		return nil
	}
	select {
	case <-s.stopped:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *NativeScheduler) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *NativeScheduler) effectiveMaxRunningLocked() int {
	if s.coupler != nil {
		return s.coupler.CoupledMaxRunning(s.maxRunning)
	}
	return s.maxRunning
}

func nativePromptBlocks(promptTokens, blockTokens int) int {
	if promptTokens <= 0 {
		promptTokens = 1
	}
	if blockTokens <= 0 {
		blockTokens = 1
	}
	return 1 + (promptTokens-1)/blockTokens
}

func (s *NativeScheduler) lanePromptBlocksLocked(ln *schedLane) int {
	if ln == nil {
		return 0
	}
	return nativePromptBlocks(len(ln.prompt), s.blockTokensLocked())
}

// canMaterializeLaneLocked combines the scheduler's state-slot and paged-KV
// capacity checks. Preempted lanes are intentionally absent: their resident KV
// has already been swapped or dropped. The candidate is excluded from existing
// reservations so this works both before enqueue and during promotion rechecks.
func (s *NativeScheduler) canMaterializeLaneLocked(candidate *schedLane) nativeAllocationStatus {
	maxRun := s.effectiveMaxRunningLocked()
	if maxRun <= 0 {
		return nativeAllocationOK
	}
	need := s.lanePromptBlocksLocked(candidate)
	if s.preemption.Mode == NativePreemptRecompute && need > s.preemption.MaxBlocks {
		return nativeAllocationNever
	}

	stateReservations := 0
	reservedBlocks := 0
	seen := make(map[*schedLane]struct{}, len(s.lanes)+len(s.waiting))
	reserve := func(ln *schedLane) {
		if ln == nil || ln == candidate || ln.terminal {
			return
		}
		if _, duplicate := seen[ln]; duplicate {
			return
		}
		seen[ln] = struct{}{}
		stateReservations++
		blocks := s.laneKVBlocksLocked(ln)
		if ln.materializing || ln.sess == nil || (ln.packedPrefill && ln.state == schedLanePrefilling) {
			// Selected empty/partial packed owners retain their full prompt reservation
			// until DECODE; resident bytes alone undercount unfinished growth.
			blocks = s.lanePromptBlocksLocked(ln)
		}
		reservedBlocks += blocks
	}
	for _, ln := range s.lanes {
		reserve(ln)
	}
	for _, ln := range s.waiting {
		if ln.materializing || ln.sess != nil {
			reserve(ln)
		}
	}
	if maxRun > 0 && stateReservations >= maxRun {
		return nativeAllocationLater
	}
	// Swap policies deliberately materialize overlapping lanes, then relieve
	// pressure through their configured backing store. Recompute has no backing
	// allocation, so it uses the hard prompt-capacity decision.
	if s.preemption.Mode != NativePreemptRecompute {
		return nativeAllocationOK
	}
	if reservedBlocks > s.preemption.MaxBlocks-need {
		return nativeAllocationLater
	}
	return nativeAllocationOK
}

// SetMaxTokensPerStep bounds the tokens one scheduler step may spend: every running
// decode lane costs 1 and a prefilling lane costs its next prompt chunk. Waiting lanes
// that do not fit the remaining budget are skipped, not blocking later lanes that do.
// n<=0 disables the budget (the default; promotion is then gated by maxRunning only).
func (s *NativeScheduler) SetMaxTokensPerStep(n int) {
	s.mu.Lock()
	s.maxTokensPerStep = n
	s.mu.Unlock()
	s.signal()
}

// SetSpecAdmissionPad arms vLLM's pad_spec_decode admission padding
// (vllm/v1/core/sched/scheduler.py:1271-1296@ae065918640a198a56f6f17eb45e251c3229fe41,
// Apache-2.0): with k > 0, a waiting lane promoted into a spec-carrying decode step
// reserves k placeholder draft slots, so its first decode step rides the same
// fixed-shape panel as the resident 1+k lanes; a step whose token budget cannot
// carry the padded cost holds the lane in waiting rather than admitting a short
// row off the captured bucket. k <= 0 disables the pad (the default). Set it
// before the first Admit.
func (s *NativeScheduler) SetSpecAdmissionPad(k int) {
	if k < 0 {
		k = 0
	}
	s.mu.Lock()
	s.specPadK = k
	s.mu.Unlock()
	s.signal()
}

// SetPrefillInterval holds prefill to every Nth engine step so decode runs in
// longer uninterrupted stretches (vLLM local prefill cadence). n<=1 disables
// the hold and keeps today's mixed-step order bit-for-bit. A deferred step
// carries no prefill at all, including chunks of an already-admitted prompt,
// so a long prompt can take up to N times as many steps to finish prefilling.
// Set it before the first Admit; the run loop reads it each iteration.
func (s *NativeScheduler) SetPrefillInterval(n int) {
	if n < 1 {
		n = 1
	}
	s.mu.Lock()
	s.prefillInterval = n
	s.lastPrefillStep = -n
	s.mu.Unlock()
}

// SetLongPrefillTokenThreshold arms the cross-request fairness cap on a lone
// long prefill: while other requests are queued (waiting or preempted), that
// prefill advances at most n prompt tokens per scheduler step. The clamp lands
// on the step's chunk without touching the p0-prefill-budget resumable
// chunking: the admission-time nominal chunk is restored after each step, and
// the clamped cursor simply resumes next step. n<=0 disables the cap (the
// default). A positive n below the resident minimum chunk is refused and
// leaves the cap disabled rather than starving the lane into sub-minimum
// chunks. Source: vllm-project/vllm, vllm/config/scheduler.py:79-90 and
// vllm/v1/core/sched/scheduler.py:728-742, revision
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0):
// https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/vllm/v1/core/sched/scheduler.py
// Adapted for Fak as idiomatic Go (no upstream code blocks copied): only the
// queued-others trigger and the min(chunk, threshold) clamp are ported; the
// adaptive fair-share floor arrives with the serve-knob wiring leaf.
func (s *NativeScheduler) SetLongPrefillTokenThreshold(n int) error {
	if n > 0 && n < nativeQwenPrefillMinChunkTokens {
		s.mu.Lock()
		s.longPrefillTokenThreshold = 0
		s.mu.Unlock()
		return fmt.Errorf("modelengine: long prefill token threshold %d is below resident minimum %d", n, nativeQwenPrefillMinChunkTokens)
	}
	if n < 0 {
		n = 0
	}
	s.mu.Lock()
	s.longPrefillTokenThreshold = n
	s.mu.Unlock()
	s.signal()
	return nil
}

// longPrefillStepCapLocked reports this step's cross-request fairness cap for
// the running prefill lane: the armed threshold while other requests are
// queued (waiting or preempted), zero when nobody waits or the cap is
// disabled. Caller holds s.mu.
func (s *NativeScheduler) longPrefillStepCapLocked() int {
	if s.longPrefillTokenThreshold <= 0 || len(s.waiting)+len(s.preempted) == 0 {
		return 0
	}
	return s.longPrefillTokenThreshold
}

// clampLongPrefillStep bounds one prefill step's chunk to the cross-request
// fairness cap by temporarily lowering the lane's nominal chunk for the
// duration of a single advanceQwenPrefill call, and returns the values needed
// to undo it. The admission-time nominal survives the resume, so the
// p0-prefill-budget resumable chunking keeps its own economics and the
// clamped cursor resumes from where the clamped chunk stopped - no token
// dropped or replayed. A cap below the resident minimum chunk floors at the
// minimum so the lane stays live instead of tripping the not-decode-ready
// guard. limit<=0 or a nominal already within the cap leaves the lane
// untouched. Caller holds the executor lock; the brief s.mu takes fence the
// shared queue and lane-field access against Admit-side publishers.
func (s *NativeScheduler) clampLongPrefillStep(ln *schedLane, limit int) (orig, clamped int) {
	orig = ln.prefillChunkTokens
	if limit <= 0 || orig <= limit {
		return orig, orig
	}
	if limit < nativeQwenPrefillMinChunkTokens {
		limit = nativeQwenPrefillMinChunkTokens
	}
	if orig <= limit {
		return orig, orig
	}
	s.mu.Lock()
	ln.prefillChunkTokens = limit
	s.mu.Unlock()
	return orig, limit
}

// restoreLongPrefillStep undoes clampLongPrefillStep once the step's model
// call has published its cursor, restoring the admission-time nominal
// unconditionally: the clamp never permanently alters the lane's chunk plan.
// For a lane that left PREFILLING this step - a final clamped chunk, or a
// full prefix hit's explicit zero - the field is inert afterwards: every
// reader (laneStepTokens, the prefill advance, dedup donor/twin matching)
// gates on schedLanePrefilling before consulting it. No-op when the clamp
// left the lane untouched. Caller holds the executor lock.
func (s *NativeScheduler) restoreLongPrefillStep(ln *schedLane, orig, clamped int) {
	if clamped == orig {
		return
	}
	s.mu.Lock()
	ln.prefillChunkTokens = orig
	s.mu.Unlock()
}

// laneStepTokens is a lane's per-step token cost under maxTokensPerStep.
func laneStepTokens(ln *schedLane) int {
	if ln.state != schedLanePrefilling && (ln.sess != nil || ln.prefillChunkTokens == 0) {
		return 1
	}
	chunk := len(ln.prompt) - ln.promptCursor
	if ln.prefillChunkTokens > 0 && ln.prefillChunkTokens < chunk {
		chunk = ln.prefillChunkTokens
	}
	if chunk < 1 {
		return 1
	}
	return chunk
}

// specStepCarriesLocked reports whether this step is a spec-carrying decode step
// for pad_spec_decode purposes: at least one running lane has computed context
// (materialized and decoding), and no running lane still prefills a chunk this
// step. Caller holds s.mu.
func (s *NativeScheduler) specStepCarriesLocked() bool {
	carrying := false
	for _, ln := range s.lanes {
		switch ln.state {
		case schedLanePrefilling:
			return false // a prefill chunk rides this step: not a pure decode step
		case schedLaneDecode:
			if ln.sess != nil {
				carrying = true
			}
		}
	}
	return carrying
}

// specStepPanelLocked returns the fixed per-row token shape a spec-carrying
// decode step rides: 1 target token plus specPadK draft slots for every row,
// residents and padded newcomers alike. ok=false when the step is not a
// spec-carrying decode step. Caller holds s.mu.
func (s *NativeScheduler) specStepPanelLocked() (int, bool) {
	if s.specPadK <= 0 || !s.specStepCarriesLocked() {
		return 0, false
	}
	return 1 + s.specPadK, true
}
