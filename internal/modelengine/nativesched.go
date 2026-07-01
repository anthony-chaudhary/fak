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
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var errSchedClosed = errors.New("modelengine: native scheduler closed")

type schedPrepare struct {
	prompt []int
	tok    NLTokenizer
	q4k    bool
}

type schedPrepareFunc func(context.Context, *abi.ToolCall, *model.Model) schedPrepare

func defaultSchedPrepare(ctx context.Context, c *abi.ToolCall, m *model.Model) schedPrepare {
	return schedPrepare{prompt: tokenize(c.Tool, refBytes(ctx, c.Args), m.Cfg.VocabSize)}
}

// NativeScheduler is the continuous-batching LifecycleEngine used by the registered
// in-kernel Engine and by tests that register it under a separate id.
type NativeScheduler struct {
	m       *model.Model
	prepare schedPrepareFunc

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
	maxRunning int
	// maxObservedRunning is the high-water mark of the running set, written only by the
	// run goroutine under mu. It lets a witness assert the waiting queue actually gated
	// (peak == maxRunning) without racing on a live concurrency count.
	maxObservedRunning int
	closed             bool

	// preemption is disabled until MaxBlocks is set. When enabled it treats MaxBlocks as
	// the live paged-KV block budget and preempts running lanes at scheduler boundaries
	// when the running set exceeds that budget.
	preemption   NativePreemptionPolicy
	preempted    []*schedLane
	seqNo        int64
	preemptRound int64
	preemptStats NativePreemptionStats

	wake    chan struct{} // buffered(1): Admit/Close nudge an idle loop
	started sync.Once
}

// SetMaxRunning bounds how many admitted lanes run concurrently; the rest wait in the
// waiting queue and are promoted FIFO as running slots free between steps. n<=0 means
// unbounded (the default). Set it before the first Admit; it is read by the run loop.
func (s *NativeScheduler) SetMaxRunning(n int) {
	s.mu.Lock()
	s.maxRunning = n
	s.mu.Unlock()
}

// MaxObservedRunning reports the peak running-set size the loop reached — the witness
// that a maxRunning cap actually gated admission (peak == cap), or that an uncapped
// scheduler co-batched every lane (peak == #admitted). Safe to read after draining.
func (s *NativeScheduler) MaxObservedRunning() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxObservedRunning
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
	return &NativeScheduler{m: m, prepare: prepare, wake: make(chan struct{}, 1)}
}

// Caps advertises the lifecycle seam (so a consumer negotiates streaming/cancel
// without a type assertion) plus the scheduler's own id token.
func (s *NativeScheduler) Caps() []abi.Capability {
	return []abi.Capability{"engine.native-sched", "engine.continuous-batching", abi.EngineLifecycleCap}
}

// WeightBearing declares that the native scheduler runs model-forwards.
func (s *NativeScheduler) WeightBearing() bool { return true }

// Admit registers one request: it prefills the prompt synchronously (so the lane
// enters the batch with its first logits ready), enqueues the lane on the WAITING
// queue, and nudges the scheduler loop. The loop promotes it into the running set
// (subject to maxRunning) between steps; decoding then proceeds in the shared
// StepBatch loop. A surviving lane's output is independent of when it is promoted —
// each lane owns its KV and StepBatch is bit-exact regardless of co-batch membership.
func (s *NativeScheduler) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errSchedClosed
	}
	s.mu.Unlock()

	prep := s.prepare(ctx, c, s.m)
	prompt := prep.prompt
	if len(prompt) == 0 {
		prompt = []int{0}
	}
	sess := s.m.NewSession()
	if prep.q4k {
		// Resident-Q4_K preload: engage the Q4_K decode kernel. Multi-lane Q4_K
		// falls back to serial Step in stepOnce because BatchSession does not yet
		// implement q4kw dispatch.
		sess.Quant = true
		sess.Q4K = true
	}
	logits := sess.Prefill(prompt)

	cctx, cancel := context.WithCancel(ctx)
	ln := &schedLane{
		sched:     s,
		ctx:       cctx,
		cancel:    cancel,
		sess:      sess,
		logits:    logits,
		tool:      c.Tool,
		prompt:    append([]int(nil), prompt...),
		promptLen: len(prompt),
		putCtx:    ctx,
		tok:       prep.tok,
		q4k:       prep.q4k,
		tokens:    make(chan abi.EngineToken, 1),
		done:      make(chan struct{}),
	}

	s.mu.Lock()
	s.seqNo++
	ln.seqNo = s.seqNo
	s.waiting = append(s.waiting, ln)
	s.mu.Unlock()

	s.started.Do(func() { go s.run() })
	s.signal()
	return ln, nil
}

// Complete is the one-shot shim every LifecycleEngine offers so it also satisfies
// the bare EngineDriver: admit, drain the stream, return the assembled turn.
func (s *NativeScheduler) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	req, err := s.Admit(ctx, c)
	if err != nil {
		return nil, err
	}
	for range req.Tokens() {
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

func (s *NativeScheduler) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// run is the single scheduler loop. Each iteration recomputes the running set: it
// compacts retired lanes out, then promotes waiting lanes into the freed slots (FIFO,
// up to maxRunning) — so the per-step batch geometry tracks admissions and completions
// between every decode step. All lane mutation (terminal/logits/gen) happens HERE, on
// this one goroutine, so those fields need no lock; only the shared waiting/lanes
// slices (appended by Admit) and the closed flag are mutex-guarded.
func (s *NativeScheduler) run() {
	for {
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
		kept := s.waiting[:0]
		for _, ln := range s.waiting {
			if ln.ctx.Err() != nil {
				ln.finish(nil, ln.ctx.Err())
				continue
			}
			if s.maxRunning > 0 && len(s.lanes) >= s.maxRunning {
				kept = append(kept, ln)
				continue
			}
			s.lanes = append(s.lanes, ln)
		}
		s.waiting = kept
		s.enforcePreemptionLocked()
		running := len(s.lanes)
		if running > s.maxObservedRunning {
			s.maxObservedRunning = running
		}
		var solo *schedLane
		var active []*schedLane
		if running == 1 && len(s.waiting) == 0 && len(s.preempted) == 0 {
			solo = s.lanes[0]
		} else if running > 0 {
			active = make([]*schedLane, running)
			copy(active, s.lanes)
		}
		closed := s.closed
		idle := len(s.lanes) == 0 && len(s.waiting) == 0 && len(s.preempted) == 0
		s.mu.Unlock()

		if idle {
			if closed {
				return
			}
			<-s.wake
			continue
		}
		if solo != nil {
			s.stepSolo(solo)
			continue
		}
		s.stepOnce(active)
	}
}

// emitToken advances the lane by one token: it argmaxes the current logits, delivers
// the token on the stream, and records it. It finishes the lane (KV reclaim + stream
// close) on cancellation — before or during delivery — or once the lane hits EOS or the
// generation budget. It returns the emitted token id and whether the lane is still
// running: ok==false means the lane was finished and the caller must not touch it again.
// The solo and shared-batch step paths differ only in what they do with a still-running
// lane, so this is the copy-identical per-lane emit both of them share.
func (ln *schedLane) emitToken() (next int, ok bool) {
	if ln.ctx.Err() != nil { // cancelled between steps
		ln.finish(nil, ln.ctx.Err())
		return 0, false
	}
	next = argmax(ln.logits)
	select {
	case ln.tokens <- abi.EngineToken{ID: next}:
	case <-ln.ctx.Done(): // cancelled while delivering
		ln.finish(nil, ln.ctx.Err())
		return 0, false
	}
	ln.gen = append(ln.gen, next)
	ln.emitted++
	if ln.sess.M.Cfg.IsEOS(next) || ln.emitted >= genTokens {
		ln.finish(assembleResult(ln.putCtx, ln.tool, ln.promptLen, ln.gen, ln.tok), nil)
		return 0, false
	}
	return next, true
}

// stepSolo advances one lane without rebuilding the scheduler batch between every token.
// It returns to run() whenever another Admit/Close signal arrives, preserving in-flight
// batch addition while keeping uncontended B=1 latency off the shared-batch bookkeeping path.
func (s *NativeScheduler) stepSolo(ln *schedLane) {
	for {
		next, ok := ln.emitToken()
		if !ok {
			return
		}
		ln.logits = ln.sess.Step(next)
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
func (s *NativeScheduler) stepOnce(active []*schedLane) {
	cont := make([]*schedLane, 0, len(active))
	ids := make([]int, 0, len(active))
	for _, ln := range active {
		next, ok := ln.emitToken()
		if !ok {
			continue
		}
		cont = append(cont, ln)
		ids = append(ids, next)
	}
	if len(cont) == 0 {
		return
	}
	if len(cont) == 1 || anyQ4K(cont) {
		for i, ln := range cont {
			ln.logits = ln.sess.Step(ids[i])
		}
		return
	}
	// The shared, weight-stream-amortised decode step: ONE StepBatch over every
	// still-running lane's own Session (each owns its KV). This is the exact
	// continuous-batching primitive a real native scheduler is built on.
	seqs := make([]*model.Session, len(cont))
	for i, ln := range cont {
		seqs[i] = ln.sess
	}
	bs := &model.BatchSession{M: s.m, Seqs: seqs}
	out := bs.StepBatch(ids)
	for i, ln := range cont {
		ln.logits = copyF32(out[i])
	}
}

func anyQ4K(lanes []*schedLane) bool {
	for _, ln := range lanes {
		if ln.q4k {
			return true
		}
	}
	return false
}

// schedLane is one admitted request's state + its EngineRequest handle.
type schedLane struct {
	sched  *NativeScheduler // back-pointer so Cancel can wake the run loop
	ctx    context.Context
	cancel context.CancelFunc

	// loop-private decode state (touched only by the run goroutine).
	sess      *model.Session
	logits    []float32
	gen       []int
	emitted   int
	tool      string
	prompt    []int
	promptLen int
	putCtx    context.Context
	tok       NLTokenizer
	q4k       bool
	terminal  bool
	seqNo     int64

	// Preemption state. A preempted lane is removed from the running set without closing
	// its token stream; readmit restores sess/logits and the stream resumes.
	preemptMode  NativePreemptionMode
	preemptRound int64
	hostKV       []byte
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

// finish records the terminal state once, drops the session (KV reclaim), and
// closes the stream + done edges. Called only from the run goroutine.
func (ln *schedLane) finish(res *abi.Result, err error) {
	ln.res, ln.err, ln.reclaimed = res, err, true
	ln.terminal = true
	ln.sess = nil // reclaim the KV-bearing session
	ln.cancel()   // release the derived context
	close(ln.tokens)
	close(ln.done)
}

func copyF32(v []float32) []float32 { return append([]float32(nil), v...) }

// NativeScheduler is a LifecycleEngine and each lane satisfies EngineRequest —
// the same interface the in-kernel per-request engine and the external adapter
// implement. That this compiles is the cross-shape contract the issue requires.
var (
	_ abi.LifecycleEngine = (*NativeScheduler)(nil)
	_ abi.EngineRequest   = (*schedLane)(nil)
)
