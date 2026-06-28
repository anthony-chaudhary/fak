package modelengine

// nativesched.go — the NATIVE continuous-batching consumer of the lifecycle seam.
//
// WHY IT EXISTS. The engine-seam issue's #1 design risk is shaping the lifecycle
// against ONLY the external-adapter consumer (a per-request async stream), then
// discovering the native continuous-batching scheduler cannot fit the same
// interface — forcing a breaking rework once the native engine lands. The defense
// is to land a native-scheduler STUB that compiles against the SAME unchanged
// abi.LifecycleEngine the in-kernel per-request engine (lifecycle.go) and the
// external adapter (internal/engine) implement, BEFORE either real consumer is
// built. This file is that stub.
//
// WHAT IT PROVES (and only that). NativeScheduler admits many requests, then a
// SINGLE scheduler loop advances ALL admitted lanes with ONE model.BatchSession
// StepBatch call per iteration — the shared-weight-stream continuous-batching move
// — fanning each lane's produced token into that lane's own stream, and freeing a
// lane's KV-bearing session the instant its request is cancelled. That is the
// admit -> shared-step -> per-lane-stream -> reclaim shape, expressed through the
// frozen interface, which is the cross-shape review the seam exists to pass.
//
// WHAT IT IS NOT (explicit non-goals from the issue). It carries the BARE waiting/
// running queue the issue scopes (Admit enqueues; the loop promotes waiting->running
// between steps, FIFO, under a structural maxRunning cap), but NOT the production KV-
// governance scheduler: no paged KV, no preemption, no priority/fairness/KV-budget
// admission, no ragged-lane MAC accounting, no throughput claim. The maxRunning knob
// is a plain capacity gate, not an admission POLICY — priority/fairness/budget is the
// sibling issue's job. The loop rebuilds a transient BatchSession wrapper each tick
// over the live lanes' own *Session objects (each keeps its own KV), which is correct
// but not allocation-tuned. A lane whose consumer neither drains nor cancels will
// back-pressure the loop (head-of-line) — acceptable here, called out so no one
// mistakes it for the real scheduler. Paged KV, preemption, and the priority/fairness
// admission policy are the sibling issues that CONSUME this same contract.

import (
	"context"
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var errSchedClosed = errors.New("modelengine: native scheduler closed")

// NativeScheduler is the continuous-batching-shaped LifecycleEngine stub.
type NativeScheduler struct {
	m *model.Model

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
// export or model.NewSynthetic). It is deliberately NOT auto-registered as a kernel
// engine id: it is a shape proof a test drives directly, not the default dispatch
// path (auto-registering it would also perturb abi.Engine("")'s lowest-id pick).
func NewNativeScheduler(m *model.Model) *NativeScheduler {
	return &NativeScheduler{m: m, wake: make(chan struct{}, 1)}
}

// Caps advertises the lifecycle seam (so a consumer negotiates streaming/cancel
// without a type assertion) plus the scheduler's own id token.
func (s *NativeScheduler) Caps() []abi.Capability {
	return []abi.Capability{"engine.native-sched", abi.EngineLifecycleCap}
}

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

	prompt := tokenize(c.Tool, refBytes(ctx, c.Args), s.m.Cfg.VocabSize)
	sess := s.m.NewSession()
	logits := sess.Prefill(prompt)

	cctx, cancel := context.WithCancel(ctx)
	ln := &schedLane{
		sched:     s,
		ctx:       cctx,
		cancel:    cancel,
		sess:      sess,
		logits:    copyF32(logits),
		tool:      c.Tool,
		promptLen: len(prompt),
		putCtx:    ctx,
		tokens:    make(chan abi.EngineToken),
		done:      make(chan struct{}),
	}

	s.mu.Lock()
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
		// 2. Promote waiting lanes into the running set, FIFO, up to maxRunning. A lane
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
		active := make([]*schedLane, len(s.lanes))
		copy(active, s.lanes)
		if len(active) > s.maxObservedRunning {
			s.maxObservedRunning = len(active)
		}
		closed := s.closed
		idle := len(s.lanes) == 0 && len(s.waiting) == 0
		s.mu.Unlock()

		if idle {
			if closed {
				return
			}
			<-s.wake
			continue
		}
		s.stepOnce(active)
	}
}

// stepOnce emits one token per active lane, then advances every lane that is still
// running with ONE shared StepBatch. A lane is retired (cancelled or done) before
// the batch so it never enters StepBatch's id panel.
func (s *NativeScheduler) stepOnce(active []*schedLane) {
	cont := make([]*schedLane, 0, len(active))
	ids := make([]int, 0, len(active))
	for _, ln := range active {
		if ln.ctx.Err() != nil { // cancelled between steps
			ln.finish(nil, ln.ctx.Err())
			continue
		}
		next := argmax(ln.logits)
		select {
		case ln.tokens <- abi.EngineToken{ID: next}:
		case <-ln.ctx.Done(): // cancelled while delivering
			ln.finish(nil, ln.ctx.Err())
			continue
		}
		ln.gen = append(ln.gen, next)
		ln.emitted++
		if ln.sess.M.Cfg.IsEOS(next) || ln.emitted >= genTokens {
			ln.finish(assembleResult(ln.putCtx, ln.tool, ln.promptLen, ln.gen), nil)
			continue
		}
		cont = append(cont, ln)
		ids = append(ids, next)
	}
	if len(cont) == 0 {
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
	promptLen int
	putCtx    context.Context
	terminal  bool

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
