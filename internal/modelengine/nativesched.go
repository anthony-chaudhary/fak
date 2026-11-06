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
	"sort"
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

// promoteWaitingLocked moves waiting lanes into the running set in deadline order
// (FIFO without deadlines), up to
// maxRun and, when set, within maxTokensPerStep (vLLM v1 scheduler token_budget,
// vllm/v1/core/sched/scheduler.py:569@975dca5, Apache-2.0). An empty running set always
// admits its first live candidate so an oversized first chunk cannot stall the loop.
func (s *NativeScheduler) promoteWaitingLocked(maxRun int) (queueWaits []time.Duration) {
	for _, ln := range s.waiting {
		if ln.materializing {
			return
		}
	}
	deadlineOrdered := s.orderWaitingDeadlinesLocked()
	budget := 0
	if s.maxTokensPerStep > 0 {
		budget = s.maxTokensPerStep
		for _, ln := range s.lanes {
			budget -= laneStepTokens(ln)
		}
	}
	kept := s.waiting[:0]
	capacityBlocked := false
	for _, ln := range s.waiting {
		if ln.ctx.Err() != nil {
			ln.finish(nil, ln.ctx.Err())
			continue
		}
		if maxRun > 0 && s.preemptionEnabledLocked() {
			// EDF can put a deferred lane ahead of the waiting session whose
			// reservation blocks it. Let that owner pass through the same capacity
			// checks and drain; deferred waiters must not overtake the blocked one.
			if capacityBlocked && (!deadlineOrdered || ln.sess == nil) {
				kept = append(kept, ln)
				continue
			}
			switch s.canMaterializeLaneLocked(ln) {
			case nativeAllocationNever:
				ln.finish(nil, ErrNativePromptExceedsKVCapacity)
				continue
			case nativeAllocationLater:
				capacityBlocked = true
				kept = append(kept, ln)
				continue
			case nativeAllocationOK:
				// ready now: fall through to the remaining capacity checks below
			}
		}
		if maxRun > 0 && len(s.lanes) >= maxRun {
			kept = append(kept, ln)
			continue
		}
		cost := laneStepTokens(ln)
		pad := 0
		if s.specPadK > 0 && cost == 1 && s.specStepCarriesLocked() {
			// pad_spec_decode admission: reserve K placeholder draft slots so the
			// newcomer's first decode step rides the same fixed-shape panel as the
			// resident 1+K lanes instead of forcing the step off the captured bucket.
			pad = s.specPadK
			cost += pad
		}
		if s.maxTokensPerStep > 0 {
			if cost > budget && len(s.lanes) > 0 {
				// Skip this lane, not break the pass: a padded row that does not fit
				// stays in waiting while later lanes that do fit are still admitted.
				kept = append(kept, ln)
				continue
			}
			budget -= cost
		}
		s.lanes = append(s.lanes, ln)
		if pad > 0 {
			ln.specAdmissionPad = pad
			s.specAdmissionPads++
			s.specAdmissionPadSlots += uint64(pad)
		}
		// First scheduling only, following vLLM v1 metrics/stats.py at
		// 0c63ce9f0e0b5b2782f0f30cb491069dac158bb2 (Apache-2.0).
		// This is a Go concept adaptation, not copied upstream code. A later
		// materialization requeue or preemption does not restart the interval.
		if ln.queueWaitPending {
			ln.queueWaitPending = false
			if s.recorder != nil {
				d := s.now().Sub(ln.queueEnteredAt)
				// time.Sub saturates instead of wrapping; omit an unrepresentable
				// interval or a regressing injected clock rather than invent zero.
				if d >= 0 && d < time.Duration(1<<63-1) {
					queueWaits = append(queueWaits, d)
				}
			}
		}
	}
	s.waiting = kept
	return queueWaits
}

// run is the single scheduler loop. Each iteration recomputes the running set: it
// compacts retired lanes out, then promotes waiting lanes into the freed slots (EDF,
// or FIFO without deadlines,
// up to maxRunning) — so the per-step batch geometry tracks admissions and completions
// between every decode step. All lane mutation (terminal/logits/gen) happens HERE, on
// this one goroutine, so those fields need no lock; only the shared waiting/lanes
// slices (appended by Admit) and the closed flag are mutex-guarded.
func (s *NativeScheduler) run() {
	defer close(s.stopped)
	for {
		s.mu.Lock()
		donorOwnsWork := s.drainDonation && s.blockedDrains > 0
		closed := s.closed
		idle := len(s.lanes) == 0 && len(s.waiting) == 0 && len(s.preempted) == 0
		s.mu.Unlock()
		if donorOwnsWork {
			if closed && idle {
				return
			}
			<-s.wake
			continue
		}

		s.executor.Lock()
		_, idle, closed = s.runIteration(false)
		s.executor.Unlock()
		if idle {
			if closed {
				return
			}
			<-s.wake
		}
	}
}

func (s *NativeScheduler) beginBlockedDrain() bool {
	s.mu.Lock()
	if !s.drainDonation || s.closed {
		s.mu.Unlock()
		return false
	}
	s.blockedDrains++
	s.mu.Unlock()
	s.signal()
	return true
}

func (s *NativeScheduler) endBlockedDrain() {
	s.mu.Lock()
	s.blockedDrains--
	s.mu.Unlock()
	s.signal()
}

func (s *NativeScheduler) drainWithDonation(tokens <-chan abi.EngineToken) {
	for {
		select {
		case _, ok := <-tokens:
			if !ok {
				return
			}
			continue
		default:
		}

		if s.executor.TryLock() {
			didWork, _, _ := s.runIteration(true)
			s.executor.Unlock()
			if didWork {
				continue
			}
		}

		s.mu.Lock()
		s.blockedDrains--
		s.mu.Unlock()
		s.signal()

		_, ok := <-tokens

		s.mu.Lock()
		s.blockedDrains++
		s.mu.Unlock()

		if !ok {
			return
		}
	}
}

// runIteration advances one native scheduler boundary. executor must be held.
// donated limits the solo fast path to one token so the donating caller can return
// to draining its bounded token channel before executing another iteration.
func (s *NativeScheduler) runIteration(donated bool) (didWork, idle, closed bool) {
	s.mu.Lock()
	// 1. Drop finished/cancelled lanes from the running set, freeing their slots.
	live := s.lanes[:0]
	for _, ln := range s.lanes {
		if !ln.terminal {
			live = append(live, ln)
		}
	}
	s.lanes = live
	maxRun := s.effectiveMaxRunningLocked()
	// 2. Retire cancelled preempted lanes, then readmit older preempted lanes before
	// promoting fresh waiting work so a preempted victim cannot starve behind arrivals.
	s.dropCanceledPreemptedLocked()
	// An Admit may be materializing a reserved waiting lane without the executor.
	// Do not restore a preempted lane until that reservation has either published
	// its session or disappeared; readmit's resident-block count cannot see it yet.
	waitingReservation := false
	if maxRun > 0 && s.preemptionEnabledLocked() && s.preemption.Mode == NativePreemptRecompute {
		for _, ln := range s.waiting {
			if ln.materializing || ln.sess != nil {
				waitingReservation = true
				break
			}
		}
	}
	if !waitingReservation {
		s.readmitPreemptedLocked()
	}
	// 3. Promote waiting lanes by deadline (FIFO without deadlines), up to maxRunning. A lane
	// cancelled while it was still waiting is retired here rather than promoted.
	waitingMaterialization := false
	for _, ln := range s.waiting {
		if ln.materializing {
			waitingMaterialization = true
			break
		}
	}
	if s.promotionPicker != nil && len(s.waiting) > 1 && !waitingMaterialization {
		slots := len(s.waiting)
		if maxRun > 0 {
			slots = maxRun - len(s.lanes)
		}
		c := make([]PromotionCandidate, len(s.waiting))
		for i, ln := range s.waiting {
			c[i] = PromotionCandidate{Index: i, Hint: ln.hint}
		}
		order := s.promotionPicker(c, slots, s.sessionAffinity)
		reordered := make([]*schedLane, 0, len(s.waiting))
		used := map[int]bool{}
		for _, i := range order {
			if i >= 0 && i < len(s.waiting) && !used[i] {
				reordered = append(reordered, s.waiting[i])
				used[i] = true
			}
		}
		for i, ln := range s.waiting {
			if !used[i] {
				reordered = append(reordered, ln)
			}
		}
		s.waiting = reordered
	}
	queueWaits := s.promoteWaitingLocked(maxRun)
	// Every return below releases mu. Record after this iteration so recorder
	// callbacks never join the scheduler's mutex ordering, including the early
	// materialization/error returns. Durations were fixed at promotion.
	defer func() {
		for _, d := range queueWaits {
			s.recorder.ObservePhase(enginestep.PhaseSchedulerWait, d)
		}
	}()
	var pendingMaterialization *schedLane
	for _, ln := range s.lanes {
		if !ln.terminal && ln.sess == nil {
			pendingMaterialization = ln
			break
		}
	}
	if pendingMaterialization != nil {
		if maxRun > 0 && s.preemptionEnabledLocked() {
			switch s.canMaterializeLaneLocked(pendingMaterialization) {
			case nativeAllocationNever:
				pendingMaterialization.finish(nil, ErrNativePromptExceedsKVCapacity)
				closed = s.closed
				s.mu.Unlock()
				return true, false, closed
			case nativeAllocationOK:
				// materializable now: proceed to lane initialization below
			case nativeAllocationLater:
				for i, ln := range s.lanes {
					if ln == pendingMaterialization {
						s.lanes = append(s.lanes[:i], s.lanes[i+1:]...)
						break
					}
				}
				s.waiting = append([]*schedLane{pendingMaterialization}, s.waiting...)
				closed = s.closed
				s.mu.Unlock()
				return true, false, closed
			}
		}
		closed = s.closed
		s.mu.Unlock()
		if err := pendingMaterialization.ctx.Err(); err != nil {
			s.mu.Lock()
			pendingMaterialization.finish(nil, err)
			closed = s.closed
			s.mu.Unlock()
			return true, false, closed
		}
		candidate := *pendingMaterialization
		initializationErr := s.initializeLaneExecution(&candidate)
		s.mu.Lock()
		switch {
		case initializationErr != nil:
			pendingMaterialization.finish(nil, initializationErr)
		case s.closed:
			s.closeLaneSession(candidate.sess)
			pendingMaterialization.finish(nil, errSchedClosed)
		case pendingMaterialization.ctx.Err() != nil:
			s.closeLaneSession(candidate.sess)
			pendingMaterialization.finish(nil, pendingMaterialization.ctx.Err())
		default:
			pendingMaterialization.sess = candidate.sess
			pendingMaterialization.logits = candidate.logits
			pendingMaterialization.state = candidate.state
			pendingMaterialization.promptCursor = candidate.promptCursor
			pendingMaterialization.promptLen = candidate.promptLen
			pendingMaterialization.prefillChunkTokens = candidate.prefillChunkTokens
			pendingMaterialization.packedPrefill = candidate.packedPrefill
		}
		closed = s.closed
		s.mu.Unlock()
		return true, false, closed
	}
	if used := s.usedKVBlocksLocked(); used > s.maxObservedKVBlocks {
		s.maxObservedKVBlocks = used
	}
	if promoted := len(s.lanes); promoted > s.maxObservedRunning {
		s.maxObservedRunning = promoted
	}
	// Keep enforcing the live KV-block budget while protecting partially-prefilled
	// lanes, whose prompt cursor and recurrent state cannot yet use decode's
	// swap/recompute restore contract.
	s.enforcePreemptionPreservingPrefillLocked()
	running := len(s.lanes)
	var solo *schedLane
	var prefill *schedLane
	var active []*schedLane
	for _, ln := range s.lanes {
		if ln.state == schedLanePrefilling {
			prefill = ln
			break
		}
	}
	if prefill == nil {
		if running == 1 && len(s.waiting) == 0 && len(s.preempted) == 0 {
			solo = s.lanes[0]
		} else if running > 0 {
			active = make([]*schedLane, running)
			copy(active, s.lanes)
		}
	} else {
		active = make([]*schedLane, 0, running-1)
		for _, ln := range s.lanes {
			if ln.state == schedLaneDecode {
				active = append(active, ln)
			}
		}
	}
	// Prefill cadence gate (vLLM local prefill interval, scheduler.py:748-769):
	// defer this iteration's prefill when fewer than prefillInterval executed
	// steps have passed since the last prefill-carrying step and at least one
	// running lane is not a prefill chunk. A capacity-bound engine stops
	// throttling so a backed-up waiting queue cannot grow without bound. The
	// captured cursor/state let the stamp below tell a step that carried
	// prefill from one that yielded it unchanged.
	var prefillStartCursor int
	var prefillStartState schedLaneState
	if prefill != nil {
		prefillStartCursor = prefill.promptCursor
		prefillStartState = prefill.state
	}
	deferPrefill := prefill != nil && s.prefillInterval > 1 && !s.prefillCapacityBound &&
		(s.executedSteps+1-s.lastPrefillStep) < s.prefillInterval && len(active) > 0

	// Skip-drafts verdict inputs (vLLM skip_drafts adaptation): the cursors
	// the prefilling lanes hold before this iteration's prefill execution, so
	// the post-execution verdict counts only lanes whose chunk advanced.
	prefillCursorBefore := -1
	var packedCandidates []*schedLane
	var packedCandidateCursors []int
	if prefill != nil {
		if prefill.packedPrefill {
			for _, ln := range s.lanes {
				if ln.packedPrefill && ln.state == schedLanePrefilling {
					packedCandidates = append(packedCandidates, ln)
					packedCandidateCursors = append(packedCandidateCursors, ln.promptCursor)
				}
			}
		} else {
			prefillCursorBefore = prefill.promptCursor
		}
	}
	closed = s.closed
	waitingDepth := len(s.waiting)
	idle = len(s.lanes) == 0 && len(s.waiting) == 0 && len(s.preempted) == 0
	// Cross-request fairness cap (vLLM long_prefill_token_threshold): while
	// other requests are queued, a lone long prefill advances at most the
	// armed threshold this step. Packed prefill keeps its own packing
	// economics and stays uncapped.
	longPrefillLimit := 0
	if prefill != nil && !prefill.packedPrefill {
		longPrefillLimit = s.longPrefillStepCapLocked()
	}
	s.mu.Unlock()

	s.recorder.SetQueueDepth(waitingDepth)

	if idle {
		return false, idle, closed
	}
	s.iteration++
	iteration := s.iteration
	if prefill != nil && !deferPrefill {
		prefillStarted := s.now()
		var packedReady []*schedLane
		if prefill.packedPrefill {
			packedReady = s.advancePackedPrefill(iteration)
			s.notePackedPrefillSkipDrafts(iteration, packedCandidates, packedCandidateCursors)
		} else {
			origChunk, clampedChunk := s.clampLongPrefillStep(prefill, longPrefillLimit)
			s.advanceQwenPrefill(prefill, iteration)
			s.restoreLongPrefillStep(prefill, origChunk, clampedChunk)
			s.noteQwenPrefillSkipDrafts(iteration, prefill, prefillCursorBefore)
		}
		// PhasePrefill only: ObservePrefillChunk counts prompt tokens forwarded by
		// prefill, and advanceQwenPrefill may resolve a KV prefix hit that skips
		// tokens before the real chunk with no separate signal here. Reporting the
		// cursor delta would bill cache hits as prefill; the prefix-match family is
		// the honest owner of those tokens.
		s.recorder.ObservePhase(enginestep.PhasePrefill, s.now().Sub(prefillStarted))
		if prefill.packedPrefill {
			active = append(active, packedReady...)
		} else if !prefill.terminal && prefill.state == schedLaneDecode {
			active = append(active, prefill)
		}
		if len(active) > 0 {
			s.recorder.ObserveCohort(len(active))
			stepStarted := s.now()
			s.stepOnce(active, iteration)
			s.recorder.ObserveDecodeStep(enginestep.PathBatched, len(active), s.now().Sub(stepStarted))
		}
	} else if solo != nil {
		s.recorder.ObserveCohort(1)
		stepStarted := s.now()
		if donated {
			s.stepOnce([]*schedLane{solo}, iteration)
			s.recorder.ObserveDecodeStep(enginestep.PathSerial, 1, s.now().Sub(stepStarted))
		} else {
			s.stepSolo(solo, iteration)
			s.recorder.ObserveDecodeStep(enginestep.PathSerial, 1, s.now().Sub(stepStarted))
		}
	} else {
		s.recorder.ObserveCohort(len(active))
		stepStarted := s.now()
		s.stepOnce(active, iteration)
		s.recorder.ObserveDecodeStep(enginestep.PathBatched, len(active), s.now().Sub(stepStarted))
	}
	// Cadence stamping (vLLM scheduler.py:1562-1577): count only steps that
	// actually scheduled tokens, restart the interval from any step that
	// carried prefill (new admission or running chunk), and record the capacity
	// escape on every release step so a backed-up engine stops throttling.
	s.mu.Lock()
	carried := prefill != nil && !deferPrefill &&
		(prefill.promptCursor != prefillStartCursor || prefill.state != prefillStartState)
	if solo != nil || len(active) > 0 || carried {
		s.executedSteps++
	}
	if carried {
		s.lastPrefillStep = s.executedSteps
	}
	if !deferPrefill {
		s.prefillCapacityBound = len(s.waiting) > prefillDrainPerInterval
	}
	s.mu.Unlock()
	if donated {
		s.mu.Lock()
		s.donatedIterations++
		s.mu.Unlock()
	}
	return true, false, closed
}

// nativeDeferredToken separates sampling from externally visible publication.
// A token sampled from logits produced by forward N is retained until the next
// scheduler boundary owns its forward, then retired exactly once.
type nativeDeferredToken struct {
	id        int
	done      bool
	accounted bool
}

func (ln *schedLane) stageDeferredToken() bool {
	s := ln.sched
	s.mu.Lock()
	defer s.mu.Unlock()
	if ln.terminal || ln.deferredToken != nil {
		return !ln.terminal
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		return false
	}
	if ln.state != schedLaneDecode || len(ln.logits) == 0 || ln.sess == nil || ln.sess == ln.statsSess {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		return false
	}
	// pad_spec_decode placeholder resolution: the lane's first decode forward
	// consumes the admission-reserved draft slots; rejected placeholders publish
	// nothing (upstream resolves them to -1 in update_draft_token_ids_in_output).
	if ln.specAdmissionPad > 0 {
		ln.specAdmissionPad = 0
	}
	next := argmax(ln.logits)
	ln.deferredToken = &nativeDeferredToken{
		id:   next,
		done: ln.sess.M.Cfg.IsEOS(next) || ln.emitted+1 >= genTokens,
	}
	return true
}

func (ln *schedLane) deferredForForward() (int, bool) {
	s := ln.sched
	s.mu.Lock()
	defer s.mu.Unlock()
	if ln.terminal || ln.deferredToken == nil || ln.deferredToken.done {
		return 0, false
	}
	return ln.deferredToken.id, true
}

// retireDeferredToken is the synchronize/postprocess/release boundary. It is
// the only operation that publishes a staged token or advances generated state.
func (ln *schedLane) retireDeferredToken(iteration uint64) bool {
	s := ln.sched
	s.mu.Lock()
	if ln.terminal || ln.deferredToken == nil {
		s.mu.Unlock()
		return false
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		s.mu.Unlock()
		return false
	}
	pending := *ln.deferredToken
	ln.deferredToken = nil
	s.mu.Unlock()

	var publishStarted time.Time
	if s.recorder != nil {
		publishStarted = s.now()
	}
	publishCanceled := false
	select {
	case ln.tokens <- abi.EngineToken{ID: pending.id}:
	case <-ln.ctx.Done():
		publishCanceled = true
	}
	// This is the completed select's wall time, including already-ready sends
	// and cancellation winners, not model compute or pure blocked time.
	if s.recorder != nil {
		elapsed := s.now().Sub(publishStarted)
		if elapsed >= 0 && elapsed < time.Duration(1<<63-1) {
			s.recorder.ObservePhase(enginestep.PhaseTokenPublish, elapsed)
		}
	}
	if publishCanceled {
		s.mu.Lock()
		ln.finish(nil, ln.ctx.Err())
		s.mu.Unlock()
		return false
	}

	s.mu.Lock()
	ln.gen = append(ln.gen, pending.id)
	if !pending.accounted {
		ln.emitted++
	}
	state := ln.state
	var res *abi.Result
	if pending.done {
		modelName := s.modelName
		if modelName == "" {
			modelName = Default.ModelName()
		}
		res = assembleSyscallResult(ln.putCtx, ln.tool, EngineID, modelName, ln.promptLen, append([]int(nil), ln.gen...), ln.tok)
	}
	s.mu.Unlock()

	s.observeEvent(nativeSchedulerEvent{Iteration: iteration, Kind: nativeSchedulerEventDecode, Lane: ln, State: state, Token: pending.id})
	if pending.done {
		s.mu.Lock()
		ln.finish(res, nil)
		s.mu.Unlock()
		return false
	}
	return true
}

// stepSolo advances one lane without rebuilding the scheduler batch between every token.
// It returns to run() whenever another Admit/Close signal arrives, preserving in-flight
// batch addition while keeping uncontended B=1 latency off the shared-batch bookkeeping path.
func (s *NativeScheduler) stepSolo(ln *schedLane, iteration uint64) {
	if ln.deferredToken == nil {
		if !ln.stageDeferredToken() {
			return
		}
	}
	for {
		next, forward := ln.deferredForForward()
		if !forward {
			ln.retireDeferredToken(iteration)
			return
		}
		sess := s.takeDecodeSession(ln)
		if sess == nil {
			return
		}
		s.beforeModelExecution(nativeSchedulerEventDecode, ln)
		decodeStarted := s.now()
		// Load the live forward generation once, BEFORE the forward call, so
		// this whole step runs on one generation (in-flight continuity).
		gen := s.fwd.Load()
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return gen.Impl.Step(sess, next)
		})
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, s.now().Sub(decodeStarted))
		if !s.publishDecodeSession(ln, sess, logits) {
			return
		}
		if !ln.retireDeferredToken(iteration) || !ln.stageDeferredToken() {
			return
		}
		s.noteStepSkipDrafts(iteration, false, []*schedLane{ln})
		select {
		case <-s.wake:
			return
		default:
		}
	}
}

// stepOnce stages a token per active lane, advances it with ONE shared StepBatch,
// then publishes that prior token and stages its successor. A terminal staged token
// is retired before the batch so it never enters StepBatch's id panel.
//
// Batch composition is ordered by the stable request UID (schedLane.seqNo), never by
// mutable scheduler slice position: promotion appends to the running set, preemption
// removes from the middle, and readmission appends an OLDER victim behind a lane
// admitted after it, so slice order is not a function of the UID. If two ranks of a
// tensor-parallel deployment observed different promotion or preemption history, they
// would assign different requests to the same batch slot, MoE expert and route
// selection would diverge across ranks, and a batched decode step would disagree with
// itself. Ordering by the immutable admission identity makes every rank derive the same
// slot assignment from the same running set, so churn cannot silently change it.
//
// Scope and honesty. seqNo is the scheduler-local monotonic admission counter, so this
// invariant is "equal admission order implies equal batch order" — it removes churn as a
// source of divergence and makes the assignment reproducible, but it does not by itself
// join ranks that admit in different orders (there is no distributed multi-rank
// admission plane in this scheduler today). Port of the stable-UID decode ordering
// convention from sgl-project/mini-sglang at pinned revision
// 9a91cfafe754aa85daee49998176275667eb58f2 (MIT), referenced at decode.py:32-35; the
// upstream file is not vendored. Slot position only, and a correctness fix rather than a
// speed one: the per-request token stream, KV accounting and the existing seqNo
// preemption ordering are untouched, no latency, throughput or token-rate gain is
// claimed, and StepBatch already guarantees per-user logits are bit-identical to a
// serial Step, so ordering cannot change any lane's output.
func (s *NativeScheduler) stepOnce(active []*schedLane, iteration uint64) {
	sort.SliceStable(active, func(i, j int) bool { return active[i].seqNo < active[j].seqNo })
	cont := make([]*schedLane, 0, len(active))
	ids := make([]int, 0, len(active))
	for _, ln := range active {
		if ln.deferredToken == nil {
			if !ln.stageDeferredToken() {
				continue
			}
		}
		next, forward := ln.deferredForForward()
		if !forward {
			ln.retireDeferredToken(iteration)
			continue
		}
		cont = append(cont, ln)
		ids = append(ids, next)
	}
	if len(cont) == 0 {
		return
	}
	execLanes := make([]*schedLane, 0, len(cont))
	execIDs := make([]int, 0, len(cont))
	seqs := make([]*model.Session, 0, len(cont))
	for i, ln := range cont {
		if sess := s.takeDecodeSession(ln); sess != nil {
			execLanes = append(execLanes, ln)
			execIDs = append(execIDs, ids[i])
			seqs = append(seqs, sess)
		}
	}
	if len(execLanes) == 0 {
		return
	}
	if len(execLanes) == 1 {
		s.beforeModelExecution(nativeSchedulerEventDecode, execLanes[0])
		decodeStarted := s.now()
		// Load the live forward generation once, BEFORE the forward call, so
		// this whole step runs on one generation (in-flight continuity).
		gen := s.fwd.Load()
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return gen.Impl.Step(seqs[0], execIDs[0])
		})
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, s.now().Sub(decodeStarted))
		if s.publishDecodeSession(execLanes[0], seqs[0], logits) {
			execLanes[0].retireDeferredToken(iteration)
			execLanes[0].stageDeferredToken()
		}
		s.noteStepSkipDrafts(iteration, false, active)
		return
	}
	// The shared, weight-stream-amortised decode step: ONE StepBatch over every
	// still-running lane's own Session (each owns its KV). This is the exact
	// continuous-batching primitive a real native scheduler is built on.
	bs := &model.BatchSession{M: s.m, Seqs: seqs}
	for _, ln := range execLanes {
		s.beforeModelExecution(nativeSchedulerEventDecode, ln)
	}
	decodeStarted := s.now()
	// Load the live forward generation once, BEFORE the forward call, so this
	// whole batch step runs on one generation (in-flight continuity).
	gen := s.fwd.Load()
	out := RunWithOp(s.coupler, OpBatch, func() [][]float32 {
		return gen.Impl.StepBatch(bs, execIDs)
	})
	s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, s.now().Sub(decodeStarted))
	if panels := bs.LastStepSharedPanels(); panels > 0 {
		s.mu.Lock()
		s.sharedBatchSteps++
		s.sharedPanels += uint64(panels)
		s.sharedMACs += uint64(bs.LastStepMACs())
		s.mu.Unlock()
	}
	for i, ln := range execLanes {
		if s.publishDecodeSession(ln, seqs[i], copyF32(out[i])) {
			ln.retireDeferredToken(iteration)
			ln.stageDeferredToken()
		}
	}
	s.noteStepSkipDrafts(iteration, false, active)
}

// decodeBatchBucket selects the fixed-shape decode graph bucket that a live
// batch of n rows replays through: the smallest captured bucket >= n, the
// pad-to-captured-size rule from sgl-project/mini-sglang at pinned revision
// 9a91cfafe754aa85daee49998176275667eb58f2 (MIT), referenced at
// python/minisgl/engine/graph.py:16-156; the upstream file is not vendored.
// Buckets must be sorted unique positive sizes (strictly increasing); any
// other set, n < 1, or n above the largest captured size refuses with ok=false
// rather than mis-selecting a shape.
func decodeBatchBucket(n int, buckets []int) (int, bool) {
	if n < 1 || len(buckets) == 0 {
		return 0, false
	}
	for i, b := range buckets {
		if b < 1 || (i > 0 && b <= buckets[i-1]) {
			return 0, false
		}
	}
	for _, b := range buckets {
		if b >= n {
			return b, true
		}
	}
	return 0, false
}

// padDecodeRows extends a live decode batch row list to a fixed captured bucket
// shape by appending pad rows AFTER every live row, mirroring mini-sglang's
// dummy-request padding (same pin as decodeBatchBucket). Live rows keep their
// original order — the stable-UID order stepOnce establishes — and are never
// truncated or reordered; paddedSize below the live count refuses instead of
// shrinking a fixed-shape replay.
func padDecodeRows[T any](live []T, paddedSize int, pad T) ([]T, bool) {
	if paddedSize < len(live) {
		return nil, false
	}
	out := make([]T, paddedSize)
	copy(out, live)
	for i := len(live); i < paddedSize; i++ {
		out[i] = pad
	}
	return out, true
}

// takeDecodeSession replaces the live model session with a non-nil immutable
// stats shell while Step/StepBatch mutates Cache outside s.mu. Preemption runs
// only on executor boundaries, so it can never select the shell as a victim.
func (s *NativeScheduler) takeDecodeSession(ln *schedLane) *model.Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.runningDecodeLaneLocked(ln) {
		return nil
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		return nil
	}
	if ln.deferredToken != nil && !ln.deferredToken.accounted {
		ln.deferredToken.accounted = true
		ln.emitted++
	}
	return ln.takeSessionForModelLocked()
}

// publishDecodeSession restores the real session and publishes the logits only
// after the cache mutation has completed. Cancellation restores ownership first,
// then finish closes the real session exactly once under s.mu.
func (s *NativeScheduler) publishDecodeSession(ln *schedLane, sess *model.Session, logits []float32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.runningDecodeLaneLocked(ln) || !ln.restoreSessionFromModelLocked(sess) {
		if ln != nil && ln.inflightSess == sess {
			ln.inflightSess = nil
		}
		s.closeLaneSession(sess)
		return false
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		return false
	}
	if len(logits) == 0 {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		return false
	}
	ln.logits = logits
	return true
}

func (s *NativeScheduler) runningDecodeLaneLocked(ln *schedLane) bool {
	if ln == nil || ln.terminal || ln.state != schedLaneDecode {
		return false
	}
	for _, running := range s.lanes {
		if running == ln {
			return true
		}
	}
	return false
}

func (s *NativeScheduler) beforeModelExecution(kind nativeSchedulerEventKind, ln *schedLane) {
	if s != nil && s.beforeModelExecute != nil {
		s.beforeModelExecute(kind, ln)
	}
}

// schedLane is one admitted request's state + its EngineRequest handle.
// PromotionPicker is an opt-in WAITING-queue order when no live deadline exists.
// Nil preserves FIFO; deadline-bearing work uses EDF before promotion.
type PromotionPicker func([]PromotionCandidate, int, map[string]int) []int

type PromotionCandidate struct {
	Index int
	Hint  dispatchtick.WaveHint
}

// SetSessionAffinity records the worker holding an agent's warm state. The wave
// picker may prefer it; capacity is still filled by work stealing.
func (s *NativeScheduler) SetSessionAffinity(agent string, worker int) {
	s.mu.Lock()
	if s.sessionAffinity == nil {
		s.sessionAffinity = make(map[string]int)
	}
	s.sessionAffinity[agent] = worker
	s.mu.Unlock()
	s.signal()
}

func (s *NativeScheduler) SetPromotionPicker(p PromotionPicker) {
	s.mu.Lock()
	s.promotionPicker = p
	s.mu.Unlock()
	s.signal()
}

// WavePromotionPicker co-promotes a ready-set, prefers session affinity, and fills
// spare slots by work stealing. The input hints were authored by dispatchtick.
func WavePromotionPicker(c []PromotionCandidate, slots int, affinity map[string]int) []int {
	if slots <= 0 || len(c) == 0 {
		return nil
	}
	type score struct {
		wave                          string
		count, first, steps, affinity int
	}
	groups := map[string]*score{}
	for _, x := range c {
		if x.Hint.Wave == "" {
			continue
		}
		g := groups[x.Hint.Wave]
		if g == nil {
			g = &score{wave: x.Hint.Wave, first: x.Index, steps: x.Hint.StepsToExecution}
			groups[x.Hint.Wave] = g
		}
		g.count++
		if x.Hint.StepsToExecution < g.steps {
			g.steps = x.Hint.StepsToExecution
		}
		if w, ok := affinity[x.Hint.Agent]; ok && w == x.Hint.Worker {
			g.affinity++
		}
	}
	var best *score
	for _, g := range groups {
		if best == nil || (g.count <= slots && best.count > slots) || ((g.count <= slots) == (best.count <= slots) && (g.affinity > best.affinity || (g.affinity == best.affinity && (g.count > best.count || (g.count == best.count && (g.steps < best.steps || (g.steps == best.steps && g.first < best.first))))))) {
			best = g
		}
	}
	chosen := make([]int, 0, slots)
	seen := map[int]bool{}
	if best != nil {
		for _, x := range c {
			if x.Hint.Wave == best.wave && len(chosen) < slots {
				chosen = append(chosen, x.Index)
				seen[x.Index] = true
			}
		}
	}
	for _, x := range c {
		if len(chosen) >= slots {
			break
		}
		if !seen[x.Index] {
			chosen = append(chosen, x.Index)
			seen[x.Index] = true
		}
	}
	return chosen
}

type schedLane struct {
	sched  *NativeScheduler // back-pointer so Cancel can wake the run loop
	ctx    context.Context
	cancel context.CancelFunc

	// Model execution is serialized by executor. Fields consumed by live stats
	// (sess, promptLen, emitted, terminal) and completed prefill/decode publication
	// are additionally synchronized by sched.mu.
	sess               *model.Session
	statsSess          *model.Session // immutable non-nil shell exposed while inflightSess mutates
	inflightSess       *model.Session // real session owned by the executor during model execution
	logits             []float32
	gen                []int
	emitted            int
	tool               string
	prompt             []int
	promptLen          int
	putCtx             context.Context
	tok                NLTokenizer
	q4k                bool
	prefillQ4K         bool
	state              schedLaneState
	promptCursor       int
	prefillChunkTokens int
	packedPrefill      bool
	kvReuseHits        int
	kvPinned           bool
	kvPinUntil         time.Time
	terminal           bool
	hint               dispatchtick.WaveHint
	seqNo              int64
	// First WAITING residence only; both fields are guarded by sched.mu.
	queueEnteredAt   time.Time
	queueWaitPending bool
	// Immutable submit metadata, also visible through nativeSchedulerEvent.Lane.
	// Lower values are more important, following vLLM's request-priority convention
	// (vllm/v1/core/sched/scheduler.py:735@975dca5, Apache-2.0).
	priority int

	// specAdmissionPad is the number of placeholder draft slots reserved for this
	// lane's first decode step when it was promoted into a spec-carrying decode
	// step (pad_spec_decode admission). Consumed by the first decode staging; the
	// placeholders publish no tokens. Guarded by sched.mu.
	specAdmissionPad int

	// Preemption state. A preempted lane is removed from the running set without closing
	// its token stream; readmit restores sess/logits and the stream resumes.
	preemptMode   NativePreemptionMode
	preemptRound  int64
	hostKV        []byte
	gpuDirectKV   *model.Qwen38GPUDirectDescriptor
	savedLogits   []float32
	materializing bool
	deferredToken *nativeDeferredToken

	tokens chan abi.EngineToken
	done   chan struct{}

	// terminal outputs: written once by finish() before close(done); read only
	// after <-done (close is the happens-before edge).
	res       *abi.Result
	err       error
	reclaimed bool
}

func (ln *schedLane) Tokens() <-chan abi.EngineToken { return ln.tokens }

func (ln *schedLane) Result() (*abi.Result, error) {
	<-ln.done
	return ln.res, ln.err
}

func (ln *schedLane) Cancel() {
	ln.cancel()
	if ln.sched != nil {
		ln.sched.signal() // wake the loop so the cancel is observed promptly, not only on its next step
	}
}

// Reclaimed reports whether the lane released its KV-bearing session (slot
// reclaim). True once terminal, on done AND on cancellation. Blocks until terminal.
func (ln *schedLane) Reclaimed() bool {
	<-ln.done
	return ln.reclaimed
}

// finish records the terminal state once, closes and drops the session (KV reclaim),
// and closes the stream + done edges. Called only by the serialized scheduler executor.
func (ln *schedLane) finish(res *abi.Result, err error) {
	if ln.terminal {
		return
	}
	ln.res, ln.err, ln.reclaimed = res, err, true
	ln.hostKV = nil
	ln.savedLogits = nil
	ln.terminal = true
	published, inflight, shell := ln.sess, ln.inflightSess, ln.statsSess
	ln.sess, ln.inflightSess, ln.statsSess = nil, nil, nil
	ln.closeRealSession(published, shell)
	if inflight != published {
		ln.closeRealSession(inflight, shell)
	}
	if ln.gpuDirectKV != nil {
		if ln.sched != nil && ln.sched.preemption.GPUDirectSwapper != nil {
			ln.sched.preemption.GPUDirectSwapper.FreeDescriptor(ln.gpuDirectKV)
		}
		ln.gpuDirectKV = nil
	}
	if ln.sched != nil {
		ln.sched.elastic.laneFinished(ln)
	}
	ln.cancel() // release the derived context
	close(ln.tokens)
	close(ln.done)
}

func (ln *schedLane) closeRealSession(sess, shell *model.Session) {
	if sess == nil || sess == shell {
		return
	}
	if ln.sched != nil {
		ln.sched.closeLaneSession(sess)
	} else {
		sess.Close()
	}
}

// takeSessionForModelLocked transfers the real session to executor-local
// ownership and leaves laneKVBlocksLocked an immutable, Cache-free view. The
// shell keeps the lane present in Running while promptLen+emitted supplies the
// synchronized live-token count.
func (ln *schedLane) takeSessionForModelLocked() *model.Session {
	if ln == nil || ln.sess == nil || ln.sess == ln.statsSess || ln.inflightSess != nil {
		return nil
	}
	real := ln.sess
	if ln.statsSess == nil {
		ln.statsSess = &model.Session{M: real.M, Quant: real.Quant, Q4K: real.Q4K}
	}
	ln.inflightSess = real
	ln.sess = ln.statsSess
	return real
}

func (ln *schedLane) restoreSessionFromModelLocked(real *model.Session) bool {
	if ln == nil || real == nil || ln.inflightSess != real || ln.statsSess == nil || ln.sess != ln.statsSess {
		return false
	}
	ln.inflightSess = nil
	ln.sess = real
	return true
}

func copyF32(v []float32) []float32 { return append([]float32(nil), v...) }

// NativeScheduler is a LifecycleEngine and each lane satisfies EngineRequest —
// the same interface the in-kernel per-request engine and the external adapter
// implement. That this compiles is the cross-shape contract the issue requires.
var (
	_ abi.LifecycleEngine = (*NativeScheduler)(nil)
	_ abi.EngineRequest   = (*schedLane)(nil)
)
