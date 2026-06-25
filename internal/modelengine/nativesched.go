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
// WHAT IT IS NOT (explicit non-goals from the issue). This is a SHAPE proof, not
// the production continuous-batching scheduler: no paged KV, no preemption, no
// admission control / fairness, no ragged-lane MAC accounting, no throughput
// claim. The loop rebuilds a transient BatchSession wrapper each tick over the
// live lanes' own *Session objects (each keeps its own KV), which is correct but
// not allocation-tuned. A lane whose consumer neither drains nor cancels will
// back-pressure the loop (head-of-line) — acceptable for a stub, called out so no
// one mistakes it for the real scheduler. The real scheduler (per-lane queues,
// paged KV, preemption) is the sibling issue that CONSUMES this same contract.

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

	mu     sync.Mutex
	lanes  []*schedLane
	closed bool

	wake    chan struct{} // buffered(1): Admit/Close nudge an idle loop
	started sync.Once
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
// enters the batch with its first logits ready), appends the lane, and nudges the
// scheduler loop. Decoding then proceeds in the shared StepBatch loop.
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
	s.lanes = append(s.lanes, ln)
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

// Close stops the scheduler once its live lanes drain. In-flight requests still
// complete (or must be cancelled); a fully-idle loop exits promptly. Idempotent.
func (s *NativeScheduler) Close() {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.signal()
}

func (s *NativeScheduler) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// run is the single scheduler loop: compact retired lanes, snapshot the live set,
// and advance them all with one StepBatch. All lane mutation (terminal/logits/gen)
// happens HERE, on this one goroutine, so those fields need no lock; only the
// shared lanes slice (appended by Admit) and the closed flag are mutex-guarded.
func (s *NativeScheduler) run() {
	for {
		s.mu.Lock()
		live := s.lanes[:0]
		for _, ln := range s.lanes {
			if !ln.terminal {
				live = append(live, ln)
			}
		}
		s.lanes = live
		active := make([]*schedLane, len(s.lanes))
		copy(active, s.lanes)
		closed := s.closed
		s.mu.Unlock()

		if len(active) == 0 {
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

func (ln *schedLane) Cancel() { ln.cancel() }

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
