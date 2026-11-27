package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
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
	inkling2PrefixReuseOffCohort
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
	ctx    context.Context
	prompt []int
	maxNew int
	stops  map[int]bool
	emit   func(int) bool
	res    agent.Inkling2LoopGeneration
	err    error
	done   chan struct{}
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

	life   sync.Mutex
	closed bool
	reqs   chan *inkling2LoopRequest
	quit   chan struct{}
	done   chan struct{}

	// owned by the worker goroutine
	cache    []*inkling2PrefixSlot
	cur      *inkling2PrefixSlot // the slot the request in flight runs on
	clock    uint64
	closeErr error
}

var _ agent.Inkling2LoopGenerator = (*inkling2LoopEngine)(nil)

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
		e.reqs, e.quit, e.done = make(chan *inkling2LoopRequest), make(chan struct{}), make(chan struct{})
		go e.run()
	}
	return e.reqs, e.quit, true
}

// Generate hands the request to the worker and waits for it; a caller whose ctx
// ends while queued returns without touching the session.
func (e *inkling2LoopEngine) Generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	if err := ctx.Err(); err != nil {
		return agent.Inkling2LoopGeneration{}, err
	}
	reqs, quit, ok := e.worker()
	if !ok {
		return agent.Inkling2LoopGeneration{}, errInkling2LoopEngineClosed
	}
	r := &inkling2LoopRequest{ctx: ctx, prompt: prompt, maxNew: maxNew, stops: stops, emit: emit, done: make(chan struct{}), received: time.Now()}
	select {
	case reqs <- r:
	case <-ctx.Done():
		return agent.Inkling2LoopGeneration{}, ctx.Err()
	case <-quit:
		return agent.Inkling2LoopGeneration{}, errInkling2LoopEngineClosed
	}
	<-r.done
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
		return nil
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
		case <-e.quit:
			e.closeErr = e.dropAllSlots()
			return
		}
	}
}

// serveRecovering turns a panic in the caller's emit callback into that caller's
// error: it runs on the shared worker, where net/http's per-handler recovery no
// longer applies, so an unrecovered panic would end the whole serve.
func (e *inkling2LoopEngine) serveRecovering(r *inkling2LoopRequest) {
	defer close(r.done)
	defer func() {
		if p := recover(); p != nil {
			_ = e.dropCurrent()
			r.res, r.err = agent.Inkling2LoopGeneration{}, fmt.Errorf("inkling2_loop engine: request panicked: %v", p)
		}
	}()
	r.res, r.err = e.serve(r)
}

// acquire positions a slot's session for prompt and reports how many leading prompt
// tokens its cache already holds; rw is nil when the request runs without reuse. A
// hit is the slot whose checkpoint is the longest proper prefix of prompt; a miss
// runs on a new slot while the cache may grow, else on the slot victim picks.
func (e *inkling2LoopEngine) acquire(prompt []int) (rw inkling2Rewinder, reused int, err error) {
	e.cur = nil
	e.clock++
	if !e.prefixReuse.disabled() {
		if sl := e.prefixHit(prompt); sl != nil {
			e.cur, sl.used, sl.hasCkpt = sl, e.clock, false
			rw := sl.sess.(inkling2Rewinder)
			switch err := rw.Restore(); {
			case err == nil:
				if rw.Pos() == len(sl.ckpt) {
					return rw, len(sl.ckpt), nil
				}
			case !errors.Is(err, ggufload.ErrInkling2NoCheckpoint):
				return nil, 0, err
			}
			return e.resetCurrent(prompt, rw)
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
		sl = e.prefixVictim()
		sl.hasCkpt = false
	}
	e.cur, sl.used = sl, e.clock
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
	return e.resetCurrent(prompt, rw)
}

// resetCurrent rewinds the current slot's session to position 0.
func (e *inkling2LoopEngine) resetCurrent(prompt []int, rw inkling2Rewinder) (inkling2Rewinder, int, error) {
	err := rw.Reset()
	if errors.Is(err, ggufload.ErrInkling2CheckpointUnavailable) {
		// a fresh session starts at position 0 without the reset's zero fill
		if err = e.dropCurrent(); err == nil {
			return e.acquire(prompt)
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
	rw, reused, err := e.acquire(prompt)
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
	}
	if rw != nil {
		// the last prompt token and up to maxNew-1 decode steps run past it
		switch err = rw.Checkpoint(max(r.maxNew, 1)); {
		case err == nil:
			sl.ckpt, sl.hasCkpt = append(sl.ckpt[:0], prompt[:n-1]...), true
		case errors.Is(err, ggufload.ErrInkling2CheckpointUnavailable):
			err = nil
		default:
			return res, err
		}
	}
	next, err := s.Prefill(r.ctx, prompt[last:])
	if err != nil {
		return res, err
	}
	res.PrefillSeconds = time.Since(start).Seconds()
	res.CachedPromptTokens, res.Width = reused, 1
	if r, ok := s.(interface{ Reused() int }); ok {
		res.CachedPromptTokens = r.Reused()
	}
	start = time.Now()
	defer func() { res.DecodeSeconds = time.Since(start).Seconds() }()
	for res.Generated < r.maxNew {
		if r.stops[next] {
			res.Stopped = true
			break
		}
		res.Generated++
		if !r.emit(next) {
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
		if next, err = s.Step(next); err != nil {
			return res, err
		}
	}
	fault = false
	return res, nil
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

// newServeInkling2Generator selects the serving engine and names it at startup.
func newServeInkling2Generator(m *ggufload.Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) (agent.Inkling2LoopGenerator, string, error) {
	return newServeInkling2GeneratorWithContext(m, be, ggufload.Inkling2LoopResidentContext, lookup)
}

func newServeInkling2GeneratorWithContext(m *ggufload.Inkling2LoopModel, be compute.Backend, contextTokens int, lookup func(string) (string, bool)) (agent.Inkling2LoopGenerator, string, error) {
	scalar := newInkling2LoopEngine(m, be)
	mode, err := inkling2PrefixReuseModeFrom(lookup)
	if err != nil {
		return nil, "", err
	}
	scalar.prefixReuse = mode
	scalar.slots, scalar.capBytes = inkling2PrefixCacheFromEnv(lookup)
	batchEligible := ggufload.Inkling2LoopBatchEligible(m, be)
	cudaPrefill := batchEligible && inkling2CUDAPrefillReady(be, lookup)
	if cudaPrefill {
		lookup = inkling2CUDAPrefillDefaults(lookup)
	}
	if !inkling2ServeBatchEnabled(lookup) {
		return scalar, "greedy decode" + inkling2PrefixSlotsNote(scalar), nil
	}
	adm, err := inkling2AdmissionFromEnv(lookup)
	if err != nil {
		return nil, "", err
	}
	if !batchEligible {
		return scalar, "greedy decode; FAK_INKLING2_SERVE_BATCH ignored: load is not batch eligible" + inkling2PrefixSlotsNote(scalar), nil
	}
	prefix, err := inkling2BatchPrefixNote(be, lookup)
	if err != nil {
		return nil, "", err
	}
	ignored := inkling2ScalarBehindCohorts(scalar)
	if cudaPrefill {
		kv, _ := lookup("FAK_INKLING2_BATCH_KV")
		if strings.TrimSpace(kv) == "" {
			kv = "f32"
		}
		ignored += fmt.Sprintf("; CUDA concurrent prefill policy, batch KV=%s", kv)
	}
	return newInkling2LoopBatchEngine(scalar.Generate, m, be, adm, contextTokens, lookup),
		fmt.Sprintf("greedy decode; cohorts up to %d, admission gap=%s wait=%s%s%s", adm.max, adm.gap, adm.maxWait, prefix, ignored), nil
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
	contextTokens := ggufload.Inkling2LoopResidentContext
	var (
		engine    *inkling2LoopEngine
		gen       agent.Inkling2LoopGenerator
		decode    string
		closer    func() error
		placement string
	)
	tLoad := time.Now()
	switch {
	case metal:
		sz, err := serveInkling2ResidentSizingOf(path, os.LookupEnv)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(2)
		}
		m, be, err := openInkling2MetalResident(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 resident Metal load:", err)
			os.Exit(2)
		}
		engine = newInkling2LoopEngine(m, be)
		contextTokens = sz.Ctx
		if gen, decode, err = newServeInkling2GeneratorWithContext(m, be, contextTokens, serveInkling2ResidentLookup(os.LookupEnv, sz)); err != nil {
			_ = m.Close()
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(2)
		}
		closer = func() error {
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
		requestedCtx := 0
		if sf.nativeContextTokens != nil {
			requestedCtx = *sf.nativeContextTokens
		}
		contextTokens = ggufload.Inkling2ResolveServeContext(m, rt.chatBackend, requestedCtx, os.LookupEnv)
		if contextTokens > 0 {
			ggufload.Inkling2ResidentCtxPin(contextTokens)
		}
		if gen, decode, err = newServeInkling2GeneratorWithContext(m, rt.chatBackend, contextTokens, os.LookupEnv); err != nil {
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
		placement = fmt.Sprintf("%s looped forward on %s: layers=%d hidden=%d passes=%d; %s",
			m.Arch, rt.chatBackend.Name(), m.HP.Layers, m.HP.Hidden, m.HP.Passes, l.Placement)
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
		tWarm := time.Now()
		// no request ctx exists yet; an interrupt still ends the process at once
		if err := warmInkling2Session(context.Background(), engine.open, tok); err != nil {
			_ = closeModel()
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 Metal warm-up:", err)
			os.Exit(2)
		}
		placement += fmt.Sprintf("; warm-up %.1fs", time.Since(tWarm).Seconds())
		loadDur = time.Since(tLoad)
	}
	rt.inkling2Planner, rt.inkling2Close, rt.inkling2Metal, rt.inkling2MaxPromptTokens = planner, closeModel, metal, contextTokens
	rt.startupPhases = append(rt.startupPhases, gateway.StartupPhase{Name: "model-load", Dur: loadDur})
	rt.addStartupMessage(newServeStartupMessage("serve", "inkling2-loop", "info",
		fmt.Sprintf("%s; context=%d model=%q (%s)", placement, contextTokens, *sf.model, decode)))
	return true
}
