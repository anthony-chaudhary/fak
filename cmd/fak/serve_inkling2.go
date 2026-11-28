package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	capmatrix "github.com/anthony-chaudhary/fak/internal/capabilitymatrix"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// serveInkling2LoopArch is the GGUF general.architecture of the looped family and
// the model_type its served config carries.
const serveInkling2LoopArch = "inkling2_loop"

// serveInkling2LoopDefaultModelID is advertised when --model is left at its default.
const serveInkling2LoopDefaultModelID = "inkling2"

// serveIsInkling2GGUF reports an inkling2-family GGUF (inkling2_loop or
// inkling2, any shard of a split set) from headers only: a directory, a missing
// file or an unreadable header is not the family.
func serveIsInkling2GGUF(path string) bool {
	if path == "" {
		return false
	}
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return false
	}
	arch, err := serveInkling2Arch(path)
	return err == nil && arch != ""
}

// serveInkling2LoopBackendName names backend for the looped-forward gate; nil is
// the host CPU forward.
func serveInkling2LoopBackendName(backend compute.Backend, metal bool) string {
	switch {
	case backend != nil:
		return backend.Name()
	case metal:
		return "metal"
	}
	return "cpu"
}

// serveRefuseInkling2LoopGeneric is the refusal the generic *model.Model loaders
// return for a looped GGUF: they cannot hold that forward on any backend.
func serveRefuseInkling2LoopGeneric(backend compute.Backend) error {
	if err := fakmodel.RefuseInkling2LoopBackend(serveInkling2LoopBackendName(backend, false)); err != nil {
		return err
	}
	_, err := fakmodel.ClassifyForwardPath(fakmodel.Config{ModelType: serveInkling2LoopArch}, nil)
	return err
}

// inkling2LoopSessionOpener yields a session at position 0.
type inkling2LoopSessionOpener func() (ggufload.Inkling2LoopSession, error)

// inkling2Rewinder mirrors ggufload.Inkling2LoopRewinder: a session that keeps its
// cache across requests by checkpointing and restoring its history state.
type inkling2Rewinder interface {
	Pos() int
	Checkpoint(horizon int) error
	Restore() error
	Reset() error
}

// inkling2PrefixReuseEnabled reads FAK_INKLING2_PREFIX_REUSE; reuse is on unless
// it is 0, false or off.
func inkling2PrefixReuseEnabled() bool { return inkling2PrefixReuseFrom(os.LookupEnv) }

func inkling2PrefixReuseFrom(lookup func(string) (string, bool)) bool {
	v, _ := lookup("FAK_INKLING2_PREFIX_REUSE")
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "false", "off":
		return false
	}
	return true
}

type inkling2PrefixReuseMode uint8

const (
	inkling2PrefixReuseOn inkling2PrefixReuseMode = iota
	inkling2PrefixReuseOff
)

func (m inkling2PrefixReuseMode) disabled() bool { return m != inkling2PrefixReuseOn }

func inkling2PrefixReuseModeFrom(lookup func(string) (string, bool)) (inkling2PrefixReuseMode, error) {
	if !inkling2PrefixReuseFrom(lookup) {
		return inkling2PrefixReuseOff, nil
	}
	_, err := ggufload.Inkling2KVDTypeFrom(lookup)
	if err != nil {
		return inkling2PrefixReuseOff, err
	}
	return inkling2PrefixReuseOn, nil
}

var errInkling2LoopEngineClosed = errors.New("inkling2_loop engine: closed")

type inkling2LoopRequest struct {
	ctx      context.Context
	prompt   []int
	maxNew   int
	stops    map[int]bool
	sample   func([]float32) (int, error)
	draft    agent.Inkling2LoopDrafter
	emit     func(int) bool
	dropOnly bool
	res      agent.Inkling2LoopGeneration
	err      error
	done     chan struct{}
	// received is the engine Generate entry, not HTTP receipt, so first_emit
	// includes engine queue wait; the span ends when the first emit callback returns.
	received time.Time
}

// inkling2LoopEngine adapts the looped forward to agent.Inkling2LoopGenerator.
// One worker goroutine, locked to its OS thread because a resident session's CUDA
// graph and context are bound to the thread that captured them, owns the session
// and serves requests one at a time. When the session is an inkling2Rewinder and
// reuse is on, sessions persist across requests as prefix-cache slots, and a prompt
// that extends a slot's checkpointed prefix prefills only its suffix; otherwise each
// request runs on a fresh session.
type inkling2LoopEngine struct {
	open        inkling2LoopSessionOpener
	prefixReuse inkling2PrefixReuseMode
	// slots bounds the resident sessions; capBytes bounds what they hold together
	// once there is more than one. Zero values keep one slot.
	slots    int
	capBytes int64

	life            sync.Mutex
	materialization inkling2PrefixMaterialization
	closed          bool
	reqs            chan *inkling2LoopRequest
	ctrl            chan func()
	quit            chan struct{}
	done            chan struct{}

	// owned by the worker goroutine
	cache    []*inkling2PrefixSlot
	cur      *inkling2PrefixSlot // the slot the request in flight runs on
	clock    uint64
	closeErr error
	// closeSpec releases any shared speculative draft weights owned by the engine.
	closeSpec      func() error
	prefixHitWatch inkling2PrefixHitWatch
}

var (
	_ agent.Inkling2LoopGenerator      = (*inkling2LoopEngine)(nil)
	_ agent.Inkling2LoopLogitGenerator = (*inkling2LoopEngine)(nil)
	_ agent.Inkling2LoopDraftGenerator = (*inkling2LoopEngine)(nil)
)

// SerializesGPUAccess lets the planner hand concurrent turns to the worker, which
// queues them, so a caller that gives up while queued returns at once.
func (*inkling2LoopEngine) SerializesGPUAccess() bool { return true }

func newInkling2LoopEngineFrom(open inkling2LoopSessionOpener) *inkling2LoopEngine {
	slots, capBytes := inkling2PrefixCacheFromEnv(os.LookupEnv)
	mode := inkling2PrefixReuseOn
	if !inkling2PrefixReuseEnabled() {
		mode = inkling2PrefixReuseOff
	}
	return &inkling2LoopEngine{open: open, prefixReuse: mode, slots: slots, capBytes: capBytes}
}

func newInkling2LoopEngine(m *ggufload.Inkling2LoopModel, be compute.Backend) *inkling2LoopEngine {
	return newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		return ggufload.NewInkling2LoopSession(m, be)
	})
}

// worker starts the serving goroutine on first use; ok is false once closed.
func (e *inkling2LoopEngine) worker() (reqs chan *inkling2LoopRequest, quit chan struct{}, ok bool) {
	e.life.Lock()
	defer e.life.Unlock()
	if e.closed {
		return nil, nil, false
	}
	if e.reqs == nil {
		e.reqs, e.ctrl, e.quit, e.done = make(chan *inkling2LoopRequest), make(chan func()), make(chan struct{}), make(chan struct{})
		go e.run()
	}
	return e.reqs, e.quit, true
}

// Generate hands the request to the worker and waits for it; a caller whose ctx
// ends while queued returns without touching the session.
func (e *inkling2LoopEngine) Generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	return e.generate(ctx, prompt, maxNew, stops, nil, nil, emit)
}

// GenerateLogits runs non-greedy host-side sampling using the session's LastLogits
// on each prefill and decode step.
func (e *inkling2LoopEngine) GenerateLogits(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	return e.generate(ctx, prompt, maxNew, stops, sample, nil, emit)
}

// GenerateDraft runs greedy or sampled decode accelerated by deterministic
// forced-token jump-forward spans committed via batched Prefill.
func (e *inkling2LoopEngine) GenerateDraft(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), draft agent.Inkling2LoopDrafter, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	return e.generate(ctx, prompt, maxNew, stops, sample, draft, emit)
}

func (e *inkling2LoopEngine) generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), draft agent.Inkling2LoopDrafter, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	if err := ctx.Err(); err != nil {
		return agent.Inkling2LoopGeneration{}, err
	}
	reqs, quit, ok := e.worker()
	if !ok {
		return agent.Inkling2LoopGeneration{}, errInkling2LoopEngineClosed
	}
	r := &inkling2LoopRequest{ctx: ctx, prompt: prompt, maxNew: maxNew, stops: stops, sample: sample, draft: draft, emit: emit, done: make(chan struct{}), received: time.Now()}
	select {
	case reqs <- r:
	case <-ctx.Done():
		return agent.Inkling2LoopGeneration{}, ctx.Err()
	case <-quit:
		return agent.Inkling2LoopGeneration{}, errInkling2LoopEngineClosed
	}
	<-r.done
	if r.err == nil {
		if alert := e.prefixHitWatch.observe(len(prompt), r.res.CachedPromptTokens, !e.prefixReuse.disabled()); alert != "" {
			log.Printf("fak serve: %s", alert)
		}
	}
	return r.res, r.err
}

// Close finishes the in-flight request, refuses queued ones and closes the
// session on the worker's thread.
func (e *inkling2LoopEngine) Close() error {
	e.life.Lock()
	if e.closed {
		e.life.Unlock()
		return nil
	}
	e.closed = true
	quit, done := e.quit, e.done
	e.life.Unlock()
	if quit == nil {
		if e.closeSpec != nil {
			e.closeErr = e.closeSpec()
			e.closeSpec = nil
		}
		return e.closeErr
	}
	close(quit)
	<-done
	return e.closeErr
}

func (e *inkling2LoopEngine) run() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(e.done)
	for {
		select {
		case r := <-e.reqs:
			e.serveRecovering(r)
		case fn := <-e.ctrl:
			fn()
		case <-e.quit:
			err := e.dropAllSlots()
			if e.closeSpec != nil {
				err = errors.Join(err, e.closeSpec())
				e.closeSpec = nil
			}
			e.closeErr = err
			return
		}
	}
}

// serveRecovering turns a panic in the caller's emit callback into that caller's
// error: it runs on the shared worker, where net/http's per-handler recovery no
// longer applies, so an unrecovered panic would end the whole serve.
func (e *inkling2LoopEngine) serveRecovering(r *inkling2LoopRequest) {
	defer close(r.done)
	if r.dropOnly {
		r.err = e.dropAllSlots()
		return
	}
	defer func() {
		if p := recover(); p != nil {
			_ = e.dropCurrent()
			r.res, r.err = agent.Inkling2LoopGeneration{}, fmt.Errorf("inkling2_loop engine: request panicked: %v", p)
		}
	}()
	r.res, r.err = e.serve(r)
}

// dropCachedSlots closes any resident scalar prefix slots on the engine's locked OS
// worker thread so a subsequent concurrent cohort or sampled fallback starts with clean
// device memory and graph slots without racing the scalar thread.
func (e *inkling2LoopEngine) dropCachedSlots() error {
	if e == nil {
		return nil
	}
	e.life.Lock()
	if e.closed || e.reqs == nil {
		e.life.Unlock()
		return nil
	}
	reqs, quit := e.reqs, e.quit
	e.life.Unlock()
	req := &inkling2LoopRequest{dropOnly: true, done: make(chan struct{})}
	select {
	case reqs <- req:
	case <-quit:
		return nil
	}
	select {
	case <-req.done:
		return req.err
	case <-quit:
		return nil
	}
}

// acquire positions a slot's session for prompt and reports how many leading prompt
// tokens its cache already holds; rw is nil when the request runs without reuse. A
// hit is the slot whose checkpoint is the longest proper prefix of prompt under an
// equal match key (materialization + tenant salt); a miss runs on a new slot while
// the cache may grow, else on the slot victim picks.
func (e *inkling2LoopEngine) acquire(ctx context.Context, prompt []int) (rw inkling2Rewinder, reused int, err error) {
	e.cur = nil
	e.clock++
	st := e.prefixState()
	now := st.nowTime()
	e.parkIdleSlots()
	want := e.matchKeyFor(ctx)
	conv := inkling2ConversationFromContext(ctx)
	if !e.prefixReuse.disabled() {
		sl := e.prefixHit(prompt, want)
		parked, parkedIdx := e.parkedHit(prompt, want)
		if sl != nil && (parked == nil || len(sl.ckpt) >= len(parked.ckpt)) {
			e.cur, sl.used, sl.lastUsedTime, sl.hasCkpt = sl, e.clock, now, false
			sl.key, sl.convKey, sl.pinned = want, conv.key, conv.pinned
			rw := sl.sess.(inkling2Rewinder)
			switch err := rw.Restore(); {
			case err == nil:
				if rw.Pos() == len(sl.ckpt) {
					return rw, len(sl.ckpt), nil
				}
			case !errors.Is(err, ggufload.ErrInkling2NoCheckpoint):
				return nil, 0, err
			}
			return e.resetCurrent(ctx, prompt, rw)
		}
		if parked != nil {
			p := e.takeParked(parkedIdx)
			var target *inkling2PrefixSlot
			if e.prefixCanGrow() {
				if s, oerr := e.open(); oerr == nil {
					target = &inkling2PrefixSlot{sess: s}
					e.cache = append(e.cache, target)
				}
			}
			if target == nil {
				if v := e.prefixVictim(conv.key); v != nil {
					e.parkSlot(v)
					v.hasCkpt = false
					target = v
				} else {
					s, oerr := e.open()
					if oerr != nil {
						return nil, 0, oerr
					}
					target = &inkling2PrefixSlot{sess: s}
					e.cache = append(e.cache, target)
				}
			}
			e.cur, target.used, target.lastUsedTime = target, e.clock, now
			target.key, target.convKey, target.pinned = want, conv.key, conv.pinned
			if rerr := e.restoreParkedIntoSlot(target, p, want); rerr == nil {
				st.hits++
				st.parkRestores++
				st.lastHitTime = now
				return target.sess.(inkling2Rewinder), len(p.ckpt), nil
			} else {
				st.parkRestoreFallbacks++
				fmt.Fprintf(st.writer(), "inkling2.prefix parked restore failed (%v); falling back to cold prefill\n", rerr)
				if rw, ok := target.sess.(inkling2Rewinder); ok {
					return e.resetCurrent(ctx, prompt, rw)
				}
				return nil, 0, nil
			}
		}
	}
	var sl *inkling2PrefixSlot
	fresh := e.prefixCanGrow()
	if fresh {
		s, err := e.open()
		switch {
		case err == nil:
			sl = &inkling2PrefixSlot{sess: s}
			e.cache = append(e.cache, sl)
		case len(e.cache) == 0:
			return nil, 0, err
		default:
			// a slot the device cannot open caps the cache at the slots it holds
			e.slots, fresh = len(e.cache), false
		}
	}
	if !fresh {
		sl = e.prefixVictim(conv.key)
		e.parkSlot(sl)
		sl.hasCkpt = false
	}
	e.cur, sl.used, sl.lastUsedTime = sl, e.clock, now
	sl.key, sl.convKey, sl.pinned = want, conv.key, conv.pinned
	if e.prefixReuse.disabled() {
		return nil, 0, nil
	}
	rw, ok := sl.sess.(inkling2Rewinder)
	switch {
	case !ok:
		return nil, 0, nil
	case fresh:
		return rw, 0, nil
	}
	return e.resetCurrent(ctx, prompt, rw)
}

// resetCurrent rewinds the current slot's session to position 0.
func (e *inkling2LoopEngine) resetCurrent(ctx context.Context, prompt []int, rw inkling2Rewinder) (inkling2Rewinder, int, error) {
	err := rw.Reset()
	if errors.Is(err, ggufload.ErrInkling2CheckpointUnavailable) {
		// a fresh session starts at position 0 without the reset's zero fill
		if err = e.dropCurrent(); err == nil {
			return e.acquire(ctx, prompt)
		}
	}
	if err != nil {
		return nil, 0, err
	}
	return rw, 0, nil
}

// serve prefills prompt[:n-1] past the reused prefix, checkpoints there, then
// prefills the last token alone. Reuse off keeps the same split so both modes run
// identical kernels; a session that cannot rewind prefills the prompt whole. Decode
// never extends the checkpoint, so no cache row written by decode is reused. A
// session fault or a context cancelled during prefill drops that slot alone.
func (e *inkling2LoopEngine) serve(r *inkling2LoopRequest) (res agent.Inkling2LoopGeneration, err error) {
	if err := r.ctx.Err(); err != nil {
		return res, err
	}
	prompt, n := r.prompt, len(r.prompt)
	if n == 0 {
		return res, errors.New("inkling2_loop engine: empty prompt")
	}
	start := time.Now()
	rw, reused, err := e.acquire(r.ctx, prompt)
	if err != nil {
		_ = e.dropCurrent()
		return res, err
	}
	fault := true
	defer func() {
		switch {
		case fault:
			_ = e.dropCurrent()
		case rw == nil:
			if cerr := e.dropCurrent(); err == nil {
				err = cerr
			}
		default:
			e.prefixTrim()
			e.cur = nil
		}
	}()
	sl := e.cur
	s := sl.sess
	_, split := s.(inkling2Rewinder)
	last := n - 1
	if !split {
		last = 0
	}
	if last > reused {
		if _, err = s.Prefill(r.ctx, prompt[reused:last]); err != nil {
			return res, err
		}
		if cerr := r.ctx.Err(); cerr != nil {
			return res, cerr
		}
	}
	if rw != nil {
		// the last prompt token and up to maxNew-1 decode steps run past it
		switch err = rw.Checkpoint(max(r.maxNew, 1)); {
		case err == nil:
			sl.ckpt, sl.hasCkpt = append(sl.ckpt[:0], prompt[:n-1]...), true
			sl.boundary = fakmodel.Inkling2SnapshotBoundary{Kind: fakmodel.Inkling2SnapshotBoundaryTurnEnd, Pos: len(sl.ckpt)}
		case errors.Is(err, ggufload.ErrInkling2CheckpointUnavailable):
			err = nil
		default:
			return res, err
		}
	}
	var initPlanned []int
	var initTailStop bool
	if r.draft != nil {
		initPlanned, initTailStop = sliceForcedTokens(r.draft.NextForced(0, false), r.maxNew, r.stops)
	}
	needFirstPred := len(initPlanned) < r.maxNew && !initTailStop
	prefillBatch := prompt[last:]
	if needFirstPred && len(initPlanned) > 0 {
		prefillBatch = make([]int, len(prompt[last:])+len(initPlanned))
		copy(prefillBatch, prompt[last:])
		copy(prefillBatch[len(prompt[last:]):], initPlanned)
	}
	next, err := s.Prefill(r.ctx, prefillBatch)
	if err != nil {
		return res, err
	}
	if cerr := r.ctx.Err(); cerr != nil {
		fault = false
		return res, cerr
	}
	ls, hasLogits := s.(ggufload.Inkling2LoopLogitSession)
	if r.sample != nil && !hasLogits {
		fmt.Fprintln(os.Stderr, "fak serve: inkling2_loop: non-greedy sampling requested on greedy-only session; falling back to greedy")
	}
	var maskFn func([]float32)
	if masker, ok := r.draft.(interface{ MaskLogits([]float32) }); ok {
		maskFn = masker.MaskLogits
	}
	if needFirstPred && hasLogits && (r.sample != nil || maskFn != nil) {
		if next, err = selectInkling2NextToken(ls, r.sample, maskFn); err != nil {
			return res, err
		}
	}
	sp, useSpec := s.(ggufload.Inkling2LoopSpecSession)
	useSpec = useSpec && r.sample == nil && r.draft == nil && sp.SpecMaxDraft() > 0
	if useSpec {
		if err = sp.SpecStart(prompt); err != nil {
			return res, err
		}
		if r.maxNew >= sp.SpecMaxDraft()+2 {
			if err = sp.SpecWarm(next); err != nil {
				return res, err
			}
		}
	}
	res.PrefillSeconds = time.Since(start).Seconds()
	res.CachedPromptTokens, res.Width = reused, 1
	if r, ok := s.(interface{ Reused() int }); ok {
		res.CachedPromptTokens = r.Reused()
	}
	start = time.Now()
	defer func() { res.DecodeSeconds = time.Since(start).Seconds() }()
	if useSpec {
		if fault, err = inkling2ServeSpecDecode(r, s, sp, next, &res); err != nil {
			return res, err
		}
		return res, nil
	}
	if len(initPlanned) > 0 {
		_, emitStopped := emitForcedTokens(r, &res, initPlanned)
		if emitStopped || initTailStop {
			res.Stopped = true
		}
	} else if initTailStop {
		res.Stopped = true
	}
	for !res.Stopped && res.Generated < r.maxNew {
		if cerr := r.ctx.Err(); cerr != nil {
			fault = false
			return res, cerr
		}
		if r.stops[next] {
			res.Stopped = true
			break
		}
		res.Generated++
		if !r.emit(next) {
			if cerr := r.ctx.Err(); cerr != nil {
				fault = false
				return res, cerr
			}
			res.Stopped = true
			break
		}
		// Stamped after emit returns: first_byte_sent is the callback's return, not the socket write.
		if res.Generated == 1 && !r.received.IsZero() {
			res.FirstEmitSeconds = time.Since(r.received).Seconds()
		}
		if res.Generated == r.maxNew {
			break
		}
		if cerr := r.ctx.Err(); cerr != nil {
			fault = false
			return res, cerr
		}
		var forced []int
		if r.draft != nil {
			forced = r.draft.NextForced(next, true)
		}
		if len(forced) > 0 {
			planned, tailStop := sliceForcedTokens(forced, r.maxNew-res.Generated, r.stops)
			emitted, emitStopped := emitForcedTokens(r, &res, planned)
			if emitStopped || tailStop {
				res.Stopped = true
				break
			}
			if res.Generated == r.maxNew {
				break
			}
			if cerr := r.ctx.Err(); cerr != nil {
				fault = false
				return res, cerr
			}
			if len(emitted) == 0 {
				if next, err = s.Step(next); err != nil {
					return res, err
				}
			} else if fv, ok := s.(ggufload.Inkling2LoopForcedVerifySession); ok {
				if next, _, err = fv.CommitForcedSpan(r.ctx, next, emitted); err != nil {
					return res, err
				}
			} else {
				batch := make([]int, 1+len(emitted))
				batch[0] = next
				copy(batch[1:], emitted)
				if next, err = s.Prefill(r.ctx, batch); err != nil {
					return res, err
				}
			}
		} else {
			if next, err = s.Step(next); err != nil {
				return res, err
			}
		}
		if cerr := r.ctx.Err(); cerr != nil {
			fault = false
			return res, cerr
		}
		if hasLogits && (r.sample != nil || maskFn != nil) {
			if next, err = selectInkling2NextToken(ls, r.sample, maskFn); err != nil {
				return res, err
			}
		}
	}
	fault = false
	return res, nil
}

// inkling2ServeSpecDecode runs greedy speculative decode on a session that
// exposes batched verify/draft hooks, preserving stops, emit abort, maxNew, and
// checkpoint horizon bounds (k <= rem-1 keeps s.pos <= ck.pos+maxNew).
func inkling2ServeSpecDecode(r *inkling2LoopRequest, s ggufload.Inkling2LoopSession, sp ggufload.Inkling2LoopSpecSession, next int, res *agent.Inkling2LoopGeneration) (bool, error) {
	if r.maxNew <= 0 {
		return false, nil
	}
	specHist := make([]int, 0, min(r.maxNew, 256))
	depth := sp.SpecMaxDraft()
	plainVerify := sp.SpecPlainUsesVerify()
	for res.Generated < r.maxNew {
		if cerr := r.ctx.Err(); cerr != nil {
			return false, cerr
		}
		if r.stops[next] {
			res.Stopped = true
			return false, nil
		}
		res.Generated++
		if !r.emit(next) {
			if cerr := r.ctx.Err(); cerr != nil {
				return false, cerr
			}
			res.Stopped = true
			return false, nil
		}
		// Stamped after emit returns: first_byte_sent is the callback's return, not the socket write.
		if res.Generated == 1 && !r.received.IsZero() {
			res.FirstEmitSeconds = time.Since(r.received).Seconds()
		}
		if res.Generated == r.maxNew {
			return false, nil
		}
		if cerr := r.ctx.Err(); cerr != nil {
			return false, cerr
		}
		specHist = append(specHist, next)
		rem := r.maxNew - res.Generated
		k := min(depth, rem-1)
		var props []int
		if k > 0 {
			props = sp.SpecDraft(specHist, k)
			if len(props) > k {
				props = props[:k]
			}
		}
		if props == nil && !plainVerify {
			prev := next
			t0 := time.Now()
			var err error
			if next, err = s.Step(next); err != nil {
				return true, err
			}
			sp.SpecObserveCost(1, time.Since(t0))
			if err = sp.SpecObserve([]int{prev}, 1); err != nil {
				return true, err
			}
			continue
		}
		fed := make([]int, 1+len(props))
		fed[0] = next
		copy(fed[1:], props)
		t0 := time.Now()
		g, err := sp.SpecVerify(fed)
		if err != nil {
			return true, err
		}
		if len(g) < len(fed) {
			return true, fmt.Errorf("inkling2 spec: verify returned %d tokens for %d fed", len(g), len(fed))
		}
		a := 0
		for a < len(props) && props[a] == g[a] {
			a++
		}
		if err = sp.SpecCommitVerify(len(fed), a+1); err != nil {
			return true, err
		}
		sp.SpecObserveCost(len(fed), time.Since(t0))
		if err = sp.SpecObserve(fed, a+1); err != nil {
			return true, err
		}
		for _, tok := range props[:a] {
			if r.stops[tok] {
				res.Stopped = true
				return false, nil
			}
			res.Generated++
			if !r.emit(tok) {
				if cerr := r.ctx.Err(); cerr != nil {
					return false, cerr
				}
				res.Stopped = true
				return false, nil
			}
			if res.Generated == r.maxNew {
				return false, nil
			}
			specHist = append(specHist, tok)
		}
		next = g[a]
	}
	return false, nil
}

// selectInkling2NextToken reads the session's last logits, optionally applies
// grammar masking, and picks the next token via sample or greedy argmax.
func selectInkling2NextToken(ls ggufload.Inkling2LoopLogitSession, sample func([]float32) (int, error), mask func([]float32)) (int, error) {
	logits, err := ls.LastLogits()
	if err != nil {
		return 0, err
	}
	if len(logits) == 0 {
		return 0, fmt.Errorf("inkling2_loop: empty logits")
	}
	if mask != nil {
		mask(logits)
	}
	if sample != nil {
		return sample(logits)
	}
	return fakmodel.ArgmaxF32(logits), nil
}

// sliceForcedTokens bounds forced tokens to limit and stops before any stop ID.
func sliceForcedTokens(forced []int, limit int, stops map[int]bool) (planned []int, tailStop bool) {
	if limit <= 0 || len(forced) == 0 {
		return nil, false
	}
	for i, id := range forced {
		if stops[id] {
			return forced[:i], true
		}
		if i+1 == limit {
			return forced[:i+1], false
		}
	}
	return forced, false
}

// emitForcedTokens emits planned tokens through r.emit while updating res
// generation counters identically to the standard decode loop.
func emitForcedTokens(r *inkling2LoopRequest, res *agent.Inkling2LoopGeneration, planned []int) (emitted []int, stopped bool) {
	emitted = make([]int, 0, len(planned))
	for _, id := range planned {
		res.Generated++
		if !r.emit(id) {
			return emitted, true
		}
		if res.Generated == 1 && !r.received.IsZero() {
			res.FirstEmitSeconds = time.Since(r.received).Seconds()
		}
		emitted = append(emitted, id)
	}
	return emitted, false
}

func inkling2HasPrefix(s, prefix []int) bool {
	if len(prefix) > len(s) {
		return false
	}
	for i, v := range prefix {
		if s[i] != v {
			return false
		}
	}
	return true
}

// serveInkling2EnvelopeLookup supplies solved envelope defaults for unset
// prefill-chunk and batch-row keys while preserving every explicit override.
func serveInkling2EnvelopeLookup(lookup func(string) (string, bool), env ggufload.Inkling2ServeEnvelope) func(string) (string, bool) {
	return func(key string) (string, bool) {
		if lookup != nil {
			if v, ok := lookup(key); ok && strings.TrimSpace(v) != "" {
				return v, true
			}
		}
		switch key {
		case "FAK_INKLING2_PREFILL_CHUNK":
			if env.DefaultChunk > 0 {
				return strconv.Itoa(env.DefaultChunk), true
			}
		case "FAK_INKLING2_SERVE_BATCH_ROWS":
			if env.TargetConcurrency > 0 {
				return strconv.Itoa(env.TargetConcurrency), true
			}
		}
		return "", false
	}
}

// newServeInkling2Generator selects the serving engine and names it at startup:
// the cohort engine on a batch-eligible load when enabled, or the serialized engine.
func newServeInkling2Generator(m *ggufload.Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) (agent.Inkling2LoopGenerator, string, error) {
	return newServeInkling2GeneratorWithContext(m, be, ggufload.Inkling2LoopResidentContext, lookup)
}

func newServeInkling2GeneratorWithContext(m *ggufload.Inkling2LoopModel, be compute.Backend, contextTokens int, lookup func(string) (string, bool)) (agent.Inkling2LoopGenerator, string, error) {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	spec, err := ggufload.OpenInkling2ServeSpec(m, be, lookup)
	if err != nil {
		return nil, "", err
	}
	scalar := newInkling2LoopEngineFrom(spec.OpenSession)
	scalar.closeSpec = spec.Close
	specSuffix := ""
	if spec.Active() {
		specSuffix = "; " + spec.Label()
	}
	mode, err := inkling2PrefixReuseModeFrom(lookup)
	if err != nil {
		_ = scalar.Close()
		return nil, "", err
	}
	batchEligible := ggufload.Inkling2LoopBatchEligible(m, be)
	cudaPrefill := batchEligible && inkling2CUDAPrefillReady(be, lookup)
	if cudaPrefill {
		lookup = inkling2CUDAPrefillDefaults(lookup)
	}
	for _, w := range inkling2ConfigSanityWarnings(lookup, batchEligible && inkling2ServeBatchEnabled(lookup), mode) {
		log.Printf("fak serve: %s", w)
	}
	scalar.prefixReuse = mode
	scalar.slots, scalar.capBytes = inkling2PrefixCacheFromEnv(lookup)
	kvDtype, _ := ggufload.Inkling2KVDTypeFrom(lookup)
	scalar.setPrefixMaterialization(inkling2PrefixMaterialization{
		PositionConfig:    inkling2PositionConfig(m, contextTokens, fmt.Sprint(kvDtype)),
		CheckpointVersion: inkling2CheckpointFormatVersion,
	})
	if !inkling2ServeBatchEnabled(lookup) {
		return scalar, "greedy decode" + specSuffix + inkling2PrefixSlotsNote(scalar), nil
	}
	adm, err := inkling2AdmissionFromEnv(lookup)
	if err != nil {
		_ = scalar.Close()
		return nil, "", err
	}
	if !batchEligible {
		if v, ok := lookup("FAK_INKLING2_SERVE_BATCH"); ok && strings.TrimSpace(v) != "" {
			return scalar, "greedy decode" + specSuffix + "; FAK_INKLING2_SERVE_BATCH ignored: load is not batch eligible" + inkling2PrefixSlotsNote(scalar), nil
		}
		return scalar, "greedy decode" + specSuffix + inkling2PrefixSlotsNote(scalar), nil
	}
	prefix, err := inkling2BatchPrefixNote(be, lookup)
	if err != nil {
		_ = scalar.Close()
		return nil, "", err
	}
	suffix := inkling2PrefixSlotsNote(scalar)
	if cudaPrefill {
		kv, _ := lookup("FAK_INKLING2_BATCH_KV")
		if strings.TrimSpace(kv) == "" {
			kv = "f32"
		}
		suffix += fmt.Sprintf("; CUDA concurrent prefill policy (%s), batch KV=%s", inkling2CUDAPrefillKernelLabel(be, lookup), kv)
	}
	batchEng := newInkling2LoopBatchEngine(scalar.Generate, m, be, adm, contextTokens, lookup)
	batchEng.scalarGenerateLogits = scalar.GenerateLogits
	batchEng.scalarGenerateDraft = scalar.GenerateDraft
	batchEng.scalarOpen = scalar.open
	batchEng.scalarClose = scalar.Close
	batchEng.ledger = scalar.prefixState().ledger
	batchEng.evictIdleSlots = scalar.evictIdleSlotsForCohort
	batchEng.shedIdleSlots = scalar.ShedIdleSlots
	batchEng.slotMemoryStatus = scalar.SlotMemoryStatus
	batchEng.specActive = spec.Active()
	if spec.Active() {
		batchEng.scalarDropSlots = scalar.dropCachedSlots
	}
	return batchEng,
		fmt.Sprintf("greedy decode%s; cohorts up to %d, admission gap=%s wait=%s%s%s", specSuffix, adm.max, adm.gap, adm.maxWait, prefix, suffix), nil
}

// newServeInkling2LoopPlanner binds an engine to the chat surface with a config
// for which IsInkling2Loop holds, so prompts render through the shared in-kernel
// Inkling-2 template.
func newServeInkling2LoopPlanner(gen agent.Inkling2LoopGenerator, tok *tokenizer.Tokenizer, modelID string, contextTokens int) (*agent.Inkling2LoopPlanner, error) {
	cfg := fakmodel.Config{ModelType: serveInkling2LoopArch, VocabSize: tok.Vocab(), MaxPositionEmbeddings: contextTokens}
	return agent.NewInkling2LoopPlanner(gen, tok, modelID, cfg, contextTokens)
}

// inkling2MetalServeRefusal is the typed refusal of the Apple-Silicon serve
// route for an inkling2-family GGUF the native target cannot bind.
type inkling2MetalServeRefusal struct {
	Path   string
	Reason string
	Err    error
}

func (e *inkling2MetalServeRefusal) Error() string {
	return fmt.Sprintf("fak serve: Inkling 2 on Metal refused for %s: %s", e.Path, e.Reason)
}

func (e *inkling2MetalServeRefusal) Unwrap() error { return e.Err }

// serveInkling2MetalRoute reports whether path is an inkling2-family GGUF the
// Metal session forward was selected for, with its admission verdict.
func serveInkling2MetalRoute(path string, backend compute.Backend, metal bool) (bool, error) {
	routed, _, err := serveInkling2MetalPathOf(path, backend, metal)
	return routed, err
}

// serveInkling2MetalPathOf is serveInkling2MetalRoute naming the Metal path selected.
func serveInkling2MetalPathOf(path string, backend compute.Backend, metal bool) (bool, inkling2MetalPath, error) {
	if backend != nil || !metal || !serveIsInkling2GGUF(path) {
		return false, 0, nil
	}
	kind, err := serveInkling2MetalSelected(path, os.Getenv(inkling2MetalTargetEnv))
	if err != nil {
		return true, kind, err
	}
	shards := len(serveInkling2ShardSet(path))
	return true, kind, serveAdmitInkling2Resident(path, shards)
}

// serveInkling2MetalWeights is the admission plan of the resident Metal path at
// the served context, or nil off the Metal route: the resident weights with
// every prefix slot the engine may hold.
func serveInkling2MetalWeights(path string, backend compute.Backend, metal bool) (*localadmission.MemoryPlan, error) {
	routed, _, err := serveInkling2MetalPathOf(path, backend, metal)
	if !routed || err != nil {
		return nil, err
	}
	sz, err := serveInkling2ResidentSizingOf(path, os.LookupEnv)
	if err != nil {
		return nil, err
	}
	return &sz.Plan, nil
}

// warmInkling2Session runs one throwaway prefill of a turn opener on a session
// from open and returns it to the model; a ctx that ends first opens nothing.
func warmInkling2Session(ctx context.Context, open inkling2LoopSessionOpener, tok *tokenizer.Tokenizer) (err error) {
	ids, err := tok.Encode("<|message_user|><|content_text|>")
	if err != nil || len(ids) == 0 {
		return fmt.Errorf("encode warm-up prompt: %v", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s, err := open()
	if err != nil {
		return err
	}
	defer func() {
		if cerr := s.Close(); err == nil {
			err = cerr
		}
	}()
	_, err = s.Prefill(ctx, ids)
	return err
}

// resolveServeInkling2ContextFlags wires --native-context-tokens and --fit into
// Inkling-2 context sizing: an explicit --native-context-tokens > 0 pins the
// context and takes precedence over FAK_INKLING2_CTX; --fit auto with no explicit
// --native-context-tokens clears any ambient FAK_INKLING2_CTX so the fit planner
// derives the context from free device memory.
func resolveServeInkling2ContextFlags(sf *serveFlags, lookup func(string) (string, bool)) (requestedCtx int, fitAuto bool, envLookup func(string) (string, bool)) {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	if sf != nil {
		if sf.nativeContextTokens != nil && *sf.nativeContextTokens > 0 {
			requestedCtx = *sf.nativeContextTokens
		}
		if sf.fit != nil && strings.EqualFold(strings.TrimSpace(*sf.fit), "auto") {
			fitAuto = true
		}
	}
	switch {
	case requestedCtx > 0:
		ctxStr := strconv.Itoa(requestedCtx)
		envLookup = func(k string) (string, bool) {
			if k == ggufload.Inkling2ResidentCtxEnv {
				return ctxStr, true
			}
			return lookup(k)
		}
	case fitAuto:
		envLookup = func(k string) (string, bool) {
			if k == ggufload.Inkling2ResidentCtxEnv {
				return "", false
			}
			return lookup(k)
		}
	default:
		envLookup = lookup
	}
	return requestedCtx, fitAuto, envLookup
}

// loadInkling2LoopModel serves an inkling2-family GGUF on the dedicated looped
// forward and reports whether it claimed the load. A backend without that forward
// is refused here, before the generic loaders mis-read the family. On CUDA the
// weights bind under the free-VRAM budget; a session runs the resident forward
// when every expert fits and the host-driven forward over cold experts otherwise.
// On Apple Silicon the native Metal target holds the one resident model.
func (rt *serveRuntime) loadInkling2LoopModel(sf *serveFlags) bool {
	path := *sf.ggufPath
	if !serveIsInkling2GGUF(path) {
		return false
	}
	if err := fakmodel.RefuseInkling2LoopBackend(serveInkling2LoopBackendName(rt.chatBackend, rt.useMetal)); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	metal, _, err := serveInkling2MetalPathOf(path, rt.chatBackend, rt.useMetal)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	requestedCtx, fitAuto, envLookup := resolveServeInkling2ContextFlags(sf, os.LookupEnv)
	contextTokens := ggufload.Inkling2LoopResidentContext
	var (
		gen       agent.Inkling2LoopGenerator
		decode    string
		closer    func() error
		placement string
		solvedEnv ggufload.Inkling2ServeEnvelope
	)
	tLoad := time.Now()
	switch {
	case metal:
		sz, err := serveInkling2ResidentSizingOf(path, envLookup)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(2)
		}
		m, be, err := openInkling2MetalResident(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 resident Metal load:", err)
			os.Exit(2)
		}
		contextTokens = sz.Ctx
		if requestedCtx > 0 || fitAuto {
			ggufload.Inkling2ResidentCtxPin(contextTokens)
		}
		if gen, decode, err = newServeInkling2GeneratorWithContext(m, be, contextTokens, serveInkling2ResidentLookup(envLookup, sz)); err != nil {
			ggufload.Inkling2ResidentCtxPin(0)
			_ = m.Close()
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(2)
		}
		closer = func() error {
			ggufload.Inkling2ResidentCtxPin(0)
			var cerr error
			if c, ok := gen.(io.Closer); ok {
				cerr = c.Close()
			}
			return errors.Join(cerr, m.Close())
		}
		placement = inkling2ServeResidentPlacement(m.Arch, m.HP.Layers, m.HP.Hidden, m.HP.Passes, sz)
	default:
		l, err := loadServeInkling2(path, rt.chatBackend, os.Stderr)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 load:", err)
			os.Exit(2)
		}
		m := l.Model
		served := *sf.model
		if served == "mock" || served == "" {
			served = serveInkling2LoopDefaultModelID
		}
		capmatrix.RegisterCheckpointTrainedLoops(served, m.HP.Passes)
		env := ggufload.ResolveInkling2ServeEnvelopeFor(m, rt.chatBackend, requestedCtx, envLookup)
		solvedEnv = env
		contextTokens = env.MaxCtx
		if fitAuto {
			fmt.Fprintf(os.Stderr, "fak serve --fit auto (inkling2_loop): resolved context=%d\n", contextTokens)
		}
		switch {
		case requestedCtx > 0:
			ggufload.Inkling2ResidentCtxPin(contextTokens)
		case fitAuto:
			ggufload.Inkling2ResidentCtxPin(ggufload.Inkling2ResidentCtxAutoMode)
		default:
			ggufload.Inkling2ResidentCtxPin(0)
		}
		genLookup := serveInkling2EnvelopeLookup(envLookup, env)
		if gen, decode, err = newServeInkling2GeneratorWithContext(m, rt.chatBackend, contextTokens, genLookup); err != nil {
			ggufload.Inkling2ResidentCtxPin(0)
			_ = m.Close()
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(2)
		}
		// the engine's resident session is freed before the weights it reads
		closer = func() error {
			ggufload.Inkling2ResidentCtxPin(0)
			var cerr error
			if c, ok := gen.(io.Closer); ok {
				cerr = c.Close()
			}
			return errors.Join(cerr, m.Close())
		}
		placement = fmt.Sprintf("%s looped forward on %s: layers=%d hidden=%d passes=%d; %s; envelope(max_ctx=%d kv_dtype=%s default_chunk=%d min_chunk=%d target_concurrency=%d long_ctx_c1_threshold=%d)",
			m.Arch, rt.chatBackend.Name(), m.HP.Layers, m.HP.Hidden, m.HP.Passes, l.Placement,
			env.MaxCtx, env.KVDtype, env.DefaultChunk, env.MinChunk, env.TargetConcurrency, env.LongCtxC1Threshold)
	}
	loadDur := time.Since(tLoad)
	// teardown is reachable from both the admission release and serve shutdown
	var once sync.Once
	var closeErr error
	closeModel := func() error {
		once.Do(func() { closeErr = closer() })
		return closeErr
	}
	// only the first split shard carries the tokenizer
	tok, ok := resolveServeTokenizer(*sf.tokPath, serveInkling2ShardSet(path)[0], rt.addStartupMessage)
	if !ok {
		_ = closeModel()
		fmt.Fprintln(os.Stderr, "fak serve: inkling2_loop needs the GGUF's embedded tokenizer or --tokenizer")
		os.Exit(2)
	}
	if scalar, ok := gen.(*inkling2LoopEngine); ok {
		scalar.life.Lock()
		mat := scalar.materialization
		scalar.life.Unlock()
		mat.ModelFile = inkling2ModelFileIdentity(path)
		mat.TokenizerID = inkling2TokenizerIdentity(tok)
		scalar.setPrefixMaterialization(mat)
	}
	if *sf.model == "mock" || *sf.model == "" {
		*sf.model = serveInkling2LoopDefaultModelID
	}
	planner, err := newServeInkling2LoopPlanner(gen, tok, *sf.model, contextTokens)
	if err != nil {
		_ = closeModel()
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	if metal {
		// The first prefill compiles the target graph and pages the weights in,
		// which outlasts a request's first-token window; pay it before readiness.
		var warmOpen inkling2LoopSessionOpener
		switch g := gen.(type) {
		case *inkling2LoopEngine:
			warmOpen = g.open
		case *inkling2LoopBatchEngine:
			warmOpen = g.scalarOpen
		}
		if warmOpen != nil {
			tWarm := time.Now()
			// no request ctx exists yet; an interrupt still ends the process at once
			if err := warmInkling2Session(context.Background(), warmOpen, tok); err != nil {
				_ = closeModel()
				fmt.Fprintln(os.Stderr, "fak serve: inkling2 Metal warm-up:", err)
				os.Exit(2)
			}
			placement += fmt.Sprintf("; warm-up %.1fs", time.Since(tWarm).Seconds())
			loadDur = time.Since(tLoad)
		}
	}
	rt.inkling2Planner, rt.inkling2Close, rt.inkling2Metal, rt.inkling2MaxPromptTokens = planner, closeModel, metal, contextTokens
	rt.inkling2ServeStatus = newInkling2ServeStatusProvider(*sf.model, solvedEnv, gen)
	rt.inkling2SpecDecode = decode
	rt.startupPhases = append(rt.startupPhases, gateway.StartupPhase{Name: "model-load", Dur: loadDur})
	rt.addStartupMessage(newServeStartupMessage("serve", "inkling2-loop", "info",
		fmt.Sprintf("%s; context=%d model=%q (%s)", placement, contextTokens, *sf.model, decode)))
	return true
}

func newInkling2ServeStatusProvider(modelID string, env ggufload.Inkling2ServeEnvelope, gen agent.Inkling2LoopGenerator) func() gateway.ServeStatusSnapshot {
	batchEng, _ := gen.(*inkling2LoopBatchEngine)
	return func() gateway.ServeStatusSnapshot {
		snap := gateway.ServeStatusSnapshot{
			Schema:             "fak.serve.status.v1",
			Model:              modelID,
			MaxCtx:             env.MaxCtx,
			KVDtype:            env.KVDtype.String(),
			DefaultChunk:       env.DefaultChunk,
			MinChunk:           env.MinChunk,
			TargetConcurrency:  env.TargetConcurrency,
			LongCtxC1Threshold: env.LongCtxC1Threshold,
		}
		if batchEng != nil {
			stats := batchEng.AutotuneStats()
			snap.ChunkStepdownsTotal = stats.ChunkStepdownsTotal
			snap.C1LongCtxFallbacksTotal = stats.C1LongCtxFallbacksTotal
			snap.LastCohortWidth = stats.LastCohortWidth
			snap.LastChunkSize = stats.LastChunkSize
		}
		return snap
	}
}
