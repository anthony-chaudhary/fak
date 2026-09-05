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
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"

	"github.com/anthony-chaudhary/fak/internal/refutil"
)

var errSchedClosed = errors.New("modelengine: native scheduler closed")

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

	cachePhaseLatency modelperfobs.CachePhaseLatencyRecorder

	mu sync.Mutex
	// waiting holds admitted-but-not-yet-running lanes (the WAITING queue); lanes
	// holds the RUNNING set the per-iteration StepBatch advances. Admit enqueues into
	// waiting; run() promotes waiting->lanes between steps, FIFO, up to maxRunning.
	waiting []*schedLane
	lanes   []*schedLane
	// maxRunning caps the running set; 0 = unbounded (every admitted lane runs at once,
	// the pre-queue behaviour). A positive cap is the BARE structural admission knob the
	// issue scopes ("the bare admit/evict loop and queues it sits on") — NOT a priority/
	// fairness/KV-budget policy, which is the sibling issue's job.
	maxRunning      int
	promotionPicker PromotionPicker
	sessionAffinity map[string]int
	// maxObservedRunning is the high-water mark of the running set, written only by the
	// run goroutine under mu. It lets a witness assert the waiting queue actually gated
	// (peak == maxRunning) without racing on a live concurrency count.
	maxObservedRunning  int
	maxObservedKVBlocks int
	sharedBatchSteps    uint64
	sharedPanels        uint64
	sharedMACs          uint64
	q4kGateUpOutputSlab bool
	residentQ4K         bool
	metalQ4K            bool
	sessionProfiler     func(NativeSessionLifecycle) *model.PhaseProfiler
	captureQwenState    bool
	qwenStateReceipts   []model.Qwen35MetalStateIdentityReceipt
	qwenPrefillTokens   int
	qwenPrefillCap      *residentQ4KPrefillCapability
	closed              bool

	// preemption is disabled until MaxBlocks is set. When enabled it treats MaxBlocks as
	// the live paged-KV block budget and preempts running lanes at scheduler boundaries
	// when the running set exceeds that budget.
	preemption   NativePreemptionPolicy
	preempted    []*schedLane
	seqNo        int64
	preemptRound int64
	preemptStats NativePreemptionStats

	coupler *WorkerCoupler

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

	closeSession       func(*model.Session)
	observeNativeEvent func(nativeSchedulerEvent)
	beforeModelExecute func(nativeSchedulerEventKind, *schedLane)
}

// NativeSessionLifecycle identifies whether a scheduler-created model session is
// the initial owner or a readmitted owner rebuilt around restored KV state.
type NativeSessionLifecycle string

const (
	NativeSessionFresh    NativeSessionLifecycle = "fresh"
	NativeSessionRestored NativeSessionLifecycle = "restored"
)

// SetMaxRunning bounds how many admitted lanes run concurrently; the rest wait in the
// waiting queue and are promoted FIFO as running slots free between steps. n<=0 means
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
		closeSession:  func(sess *model.Session) { sess.Close() },
	}
	if m != nil && m.Q4KCount() > 0 {
		s.qwenPrefillCap = &residentQ4KPrefillCapability{model: m}
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
// prefill-before-enqueue path. An explicitly enabled, supported Qwen Q4_K lane instead
// enters WAITING in PREFILLING state and lets runIteration advance one bounded prompt
// chunk before returning to decode work. A surviving lane's output remains independent
// of promotion timing because each lane owns its KV and decode state.
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

func (s *NativeScheduler) admitPrepared(ctx context.Context, c *abi.ToolCall, hint dispatchtick.WaveHint, prep schedPrepare) (abi.EngineRequest, error) {
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
	sess := s.newLaneSession(q4k, NativeSessionFresh)
	if s.captureQwenState {
		if err := sess.EnableQwen35MetalStateIdentityReceipt(prompt); err != nil {
			sess.Close()
			return nil, err
		}
	}
	prefillChunkTokens := s.qwenPrefillChunkBudget(prep, sess, len(prompt))
	state := schedLaneDecode
	promptCursor := len(prompt)
	promptLen := len(prompt)
	var logits []float32
	if prefillChunkTokens == 0 {
		prefillStarted := s.now()
		logits = RunWithOp(s.coupler, OpPrefill, func() []float32 {
			return sess.Prefill(prompt)
		})
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhasePrefill, s.now().Sub(prefillStarted))
		if s.captureQwenState {
			executed, err := sess.FinalizeQwen35MetalStateIdentityReceipt()
			if err != nil || !executed {
				sess.Close()
				if err == nil {
					err = errors.New("modelengine: Qwen Metal state identity was not executed")
				}
				return nil, err
			}
			receipt, ok := sess.Qwen35MetalStateIdentityReceipt()
			if !ok {
				sess.Close()
				return nil, errors.New("modelengine: Qwen Metal state identity receipt unavailable")
			}
			s.mu.Lock()
			s.qwenStateReceipts = append(s.qwenStateReceipts, receipt)
			s.mu.Unlock()
		}
	} else {
		state = schedLanePrefilling
		promptCursor = 0
		// promptLen is also the scheduler's resident-KV accounting input. Do not
		// reserve the not-yet-executed tail of an asynchronously-prefilled prompt.
		promptLen = 0
	}
	kvReuseHits, kvPinned, kvPinUntil := nativeKVBMHintsFromMeta(c.Meta)

	cctx, cancel := context.WithCancel(ctx)
	ln := &schedLane{
		sched:              s,
		ctx:                cctx,
		cancel:             cancel,
		sess:               sess,
		logits:             logits,
		tool:               c.Tool,
		prompt:             append([]int(nil), prompt...),
		promptLen:          promptLen,
		putCtx:             ctx,
		tok:                prep.tok,
		q4k:                q4k,
		state:              state,
		promptCursor:       promptCursor,
		prefillChunkTokens: prefillChunkTokens,
		kvReuseHits:        kvReuseHits,
		kvPinned:           kvPinned,
		kvPinUntil:         kvPinUntil,
		tokens:             make(chan abi.EngineToken, 1),
		done:               make(chan struct{}),
		hint:               hint,
	}

	s.mu.Lock()
	s.seqNo++
	ln.seqNo = s.seqNo
	s.waiting = append(s.waiting, ln)
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

// run is the single scheduler loop. Each iteration recomputes the running set: it
// compacts retired lanes out, then promotes waiting lanes into the freed slots (FIFO,
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
	// 2. Retire cancelled preempted lanes, then readmit older preempted lanes before
	// promoting fresh waiting work so a preempted victim cannot starve behind arrivals.
	s.dropCanceledPreemptedLocked()
	s.readmitPreemptedLocked()
	// 3. Promote waiting lanes into the running set, FIFO, up to maxRunning. A lane
	// cancelled while it was still waiting is retired here rather than promoted.
	maxRun := s.effectiveMaxRunningLocked()
	if s.promotionPicker != nil && len(s.waiting) > 1 {
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
	kept := s.waiting[:0]
	for _, ln := range s.waiting {
		if ln.ctx.Err() != nil {
			ln.finish(nil, ln.ctx.Err())
			continue
		}
		if maxRun > 0 && len(s.lanes) >= maxRun {
			kept = append(kept, ln)
			continue
		}
		s.lanes = append(s.lanes, ln)
	}
	s.waiting = kept
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
	closed = s.closed
	idle = len(s.lanes) == 0 && len(s.waiting) == 0 && len(s.preempted) == 0
	s.mu.Unlock()

	if idle {
		return false, idle, closed
	}
	s.iteration++
	iteration := s.iteration
	if prefill != nil {
		s.advanceQwenPrefill(prefill, iteration)
		if !prefill.terminal && prefill.state == schedLaneDecode {
			active = append(active, prefill)
		}
		if len(active) > 0 {
			s.stepOnce(active, iteration)
		}
	} else if solo != nil {
		if donated {
			s.stepOnce([]*schedLane{solo}, iteration)
		} else {
			s.stepSolo(solo, iteration)
		}
	} else {
		s.stepOnce(active, iteration)
	}
	if donated {
		s.mu.Lock()
		s.donatedIterations++
		s.mu.Unlock()
	}
	return true, false, closed
}

// emitToken advances the lane by one token: it argmaxes the current logits, delivers
// the token on the stream, and records it. It finishes the lane (KV reclaim + stream
// close) on cancellation — before or during delivery — or once the lane hits EOS or the
// generation budget. It returns the emitted token id and whether the lane is still
// running: ok==false means the lane was finished and the caller must not touch it again.
// The solo and shared-batch step paths differ only in what they do with a still-running
// lane, so this is the copy-identical per-lane emit both of them share.
func (ln *schedLane) emitToken(iteration uint64) (next int, ok bool) {
	s := ln.sched
	s.mu.Lock()
	if err := ln.ctx.Err(); err != nil { // cancelled between steps
		ln.finish(nil, err)
		s.mu.Unlock()
		return 0, false
	}
	if ln.state != schedLaneDecode || len(ln.logits) == 0 || ln.sess == nil || ln.sess == ln.statsSess {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		s.mu.Unlock()
		return 0, false
	}
	next = argmax(ln.logits)
	eos := ln.sess.M.Cfg.IsEOS(next)
	s.mu.Unlock()

	select {
	case ln.tokens <- abi.EngineToken{ID: next}:
	case <-ln.ctx.Done(): // cancelled while delivering
		s.mu.Lock()
		ln.finish(nil, ln.ctx.Err())
		s.mu.Unlock()
		return 0, false
	}

	s.mu.Lock()
	ln.gen = append(ln.gen, next)
	ln.emitted++
	done := eos || ln.emitted >= genTokens
	var res *abi.Result
	if done {
		res = assembleResult(ln.putCtx, ln.tool, ln.promptLen, append([]int(nil), ln.gen...), ln.tok)
	}
	state := ln.state
	s.mu.Unlock()

	s.observeEvent(nativeSchedulerEvent{
		Iteration: iteration,
		Kind:      nativeSchedulerEventDecode,
		Lane:      ln,
		State:     state,
		Token:     next,
	})
	if done {
		s.mu.Lock()
		ln.finish(res, nil)
		s.mu.Unlock()
		return 0, false
	}
	return next, true
}

// stepSolo advances one lane without rebuilding the scheduler batch between every token.
// It returns to run() whenever another Admit/Close signal arrives, preserving in-flight
// batch addition while keeping uncontended B=1 latency off the shared-batch bookkeeping path.
func (s *NativeScheduler) stepSolo(ln *schedLane, iteration uint64) {
	for {
		next, ok := ln.emitToken(iteration)
		if !ok {
			return
		}
		sess := s.takeDecodeSession(ln)
		if sess == nil {
			return
		}
		s.beforeModelExecution(nativeSchedulerEventDecode, ln)
		decodeStarted := s.now()
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return sess.Step(next)
		})
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, s.now().Sub(decodeStarted))
		if !s.publishDecodeSession(ln, sess, logits) {
			return
		}
		select {
		case <-s.wake:
			return
		default:
		}
	}
}

// stepOnce emits one token per active lane, then advances every lane that is still
// running with ONE shared StepBatch. A lane is retired (cancelled or done) before
// the batch so it never enters StepBatch's id panel.
func (s *NativeScheduler) stepOnce(active []*schedLane, iteration uint64) {
	cont := make([]*schedLane, 0, len(active))
	ids := make([]int, 0, len(active))
	for _, ln := range active {
		next, ok := ln.emitToken(iteration)
		if !ok {
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
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return seqs[0].Step(execIDs[0])
		})
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, s.now().Sub(decodeStarted))
		s.publishDecodeSession(execLanes[0], seqs[0], logits)
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
	out := RunWithOp(s.coupler, OpBatch, func() [][]float32 {
		return bs.StepBatch(execIDs)
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
		s.publishDecodeSession(ln, seqs[i], copyF32(out[i]))
	}
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
// PromotionPicker is an opt-in WAITING-queue order. Nil preserves FIFO.
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
	state              schedLaneState
	promptCursor       int
	prefillChunkTokens int
	kvReuseHits        int
	kvPinned           bool
	kvPinUntil         time.Time
	terminal           bool
	hint               dispatchtick.WaveHint
	seqNo              int64

	// Preemption state. A preempted lane is removed from the running set without closing
	// its token stream; readmit restores sess/logits and the stream resumes.
	preemptMode  NativePreemptionMode
	preemptRound int64
	hostKV       []byte
	gpuDirectKV  *model.Qwen38GPUDirectDescriptor
	savedLogits  []float32

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
