package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/inklingmetal"
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

// inkling2LoopSessionOpener yields a fresh per-request decode session.
type inkling2LoopSessionOpener func() (ggufload.Inkling2LoopSession, error)

// inkling2LoopEngine adapts the looped forward to agent.Inkling2LoopGenerator.
type inkling2LoopEngine struct {
	open inkling2LoopSessionOpener
}

var _ agent.Inkling2LoopGenerator = (*inkling2LoopEngine)(nil)

func newInkling2LoopEngine(m *ggufload.Inkling2LoopModel, be compute.Backend) *inkling2LoopEngine {
	return &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		return ggufload.NewInkling2LoopSession(m, be)
	}}
}

func (e *inkling2LoopEngine) Generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(int) bool) (res agent.Inkling2LoopGeneration, err error) {
	if err := ctx.Err(); err != nil {
		return res, err
	}
	// a resident session's CUDA graph is bound to the thread that captured it
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s, err := e.open()
	if err != nil {
		return res, err
	}
	defer func() {
		if cerr := s.Close(); err == nil {
			err = cerr
		}
	}()
	start := time.Now()
	next, err := s.Prefill(prompt)
	if err != nil {
		return res, err
	}
	res.PrefillSeconds = time.Since(start).Seconds()
	start = time.Now()
	defer func() { res.DecodeSeconds = time.Since(start).Seconds() }()
	for res.Generated < maxNew {
		if stops[next] {
			res.Stopped = true
			return res, nil
		}
		res.Generated++
		if !emit(next) {
			res.Stopped = true
			return res, nil
		}
		if res.Generated == maxNew {
			break
		}
		if err := ctx.Err(); err != nil {
			return res, err
		}
		if next, err = s.Step(next); err != nil {
			return res, err
		}
	}
	return res, nil
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

// serveAdmitInkling2Metal decides whether the native Metal target can serve the
// checkpoint. available is inklingmetal.Available(): false on a build without
// darwin/arm64 cgo and on a host without a supported Apple GPU.
func serveAdmitInkling2Metal(path, arch string, shards int, available bool) error {
	switch {
	case !available:
		return &inkling2MetalServeRefusal{Path: path, Err: inklingmetal.ErrUnavailable,
			Reason: inklingmetal.ErrUnavailable.Error() + "; build on darwin/arm64 with cgo enabled"}
	case shards > 1:
		return &inkling2MetalServeRefusal{Path: path,
			Reason: fmt.Sprintf("the Metal path binds a single-file GGUF and this one is split into %d shards", shards)}
	case arch != serveInkling2LoopArch:
		return &inkling2MetalServeRefusal{Path: path,
			Reason: fmt.Sprintf("the native Metal target runs %s checkpoints, not %s", serveInkling2LoopArch, arch)}
	}
	return nil
}

// serveInkling2MetalRoute reports whether path is an inkling2-family GGUF the
// Metal session forward was selected for, with its admission verdict.
func serveInkling2MetalRoute(path string, backend compute.Backend, metal bool) (bool, error) {
	if backend != nil || !metal || !serveIsInkling2GGUF(path) {
		return false, nil
	}
	arch, err := serveInkling2Arch(path)
	if err != nil {
		return true, err
	}
	return true, serveAdmitInkling2Metal(path, arch, len(serveInkling2ShardSet(path)), inklingmetal.Available())
}

// serveInkling2MetalWeights is the admission plan of the target-only native
// model with one session at the served context, or nil off the Metal route.
func serveInkling2MetalWeights(path string, backend compute.Backend, metal bool) (*localadmission.MemoryPlan, error) {
	if routed, err := serveInkling2MetalRoute(path, backend, metal); !routed || err != nil {
		return nil, err
	}
	planned, err := inklingmetal.EstimateWithDraft(path, "", ggufload.Inkling2LoopResidentContext)
	if err != nil {
		return nil, err
	}
	if planned.Bytes() > math.MaxInt64 {
		return nil, &inkling2MetalServeRefusal{Path: path, Reason: "the Metal plan exceeds the address space"}
	}
	plan := localadmission.MemoryPlan{SteadyBytes: int64(planned.Bytes())}
	plan.StartupPeakBytes, _ = metalServeStartupPeakBytes(serveLoadArmResidentQ4K, plan.SteadyBytes, 0, false)
	return &plan, nil
}

// inkling2MetalTarget is the part of a native session one greedy stream drives.
type inkling2MetalTarget interface {
	PrefillID(ids []int32) (int32, error)
	BeginBlockIDs(ids []int32) ([]int32, error)
	Commit(keep int) error
	Close() error
}

// inkling2MetalSession adapts a native target session to the per-request
// Prefill/Step contract, one target submission per token.
type inkling2MetalSession struct{ s inkling2MetalTarget }

var _ ggufload.Inkling2LoopSession = inkling2MetalSession{}

func inkling2MetalToken(id int) (int32, error) {
	if id < 0 || id >= inklingmetal.Vocabulary {
		return 0, fmt.Errorf("Inkling 2: token %d outside vocabulary", id)
	}
	return int32(id), nil
}

func (m inkling2MetalSession) Prefill(prompt []int) (int, error) {
	if len(prompt) == 0 {
		return 0, fmt.Errorf("inkling2_loop: empty prompt")
	}
	ids := make([]int32, len(prompt))
	for i, id := range prompt {
		var err error
		if ids[i], err = inkling2MetalToken(id); err != nil {
			return 0, err
		}
	}
	next, err := m.s.PrefillID(ids)
	return int(next), err
}

func (m inkling2MetalSession) Step(token int) (int, error) {
	id, err := inkling2MetalToken(token)
	if err != nil {
		return 0, err
	}
	next, err := m.s.BeginBlockIDs([]int32{id})
	if err != nil {
		return 0, err
	}
	if len(next) != 1 {
		return 0, fmt.Errorf("Inkling 2: Metal target returned %d rows for one token", len(next))
	}
	return int(next[0]), m.s.Commit(1)
}

func (m inkling2MetalSession) Close() error { return m.s.Close() }

// newInkling2MetalEngine decodes on the resident native target. Each request
// resets the model's one idle session, sized for the whole served context.
func newInkling2MetalEngine(m *inklingmetal.Model, contextTokens int) *inkling2LoopEngine {
	return &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		s, err := m.NewSession(contextTokens)
		if err != nil {
			return nil, err
		}
		return inkling2MetalSession{s: s}, nil
	}}
}

// warmInkling2Session runs one throwaway prefill of a turn opener on a session
// from open and returns it to the model.
func warmInkling2Session(open inkling2LoopSessionOpener, tok *tokenizer.Tokenizer) (err error) {
	ids, err := tok.Encode("<|message_user|><|content_text|>")
	if err != nil || len(ids) == 0 {
		return fmt.Errorf("encode warm-up prompt: %v", err)
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
	_, err = s.Prefill(ids)
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
	metal, err := serveInkling2MetalRoute(path, rt.chatBackend, rt.useMetal)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	contextTokens := ggufload.Inkling2LoopResidentContext
	var (
		engine    *inkling2LoopEngine
		closer    func() error
		placement string
	)
	tLoad := time.Now()
	if metal {
		target, err := inklingmetal.OpenWithDraft(path, "")
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 Metal load:", err)
			os.Exit(2)
		}
		engine, closer = newInkling2MetalEngine(target, contextTokens), target.Close
		placement = fmt.Sprintf("%s native target on metal: one resident model, single-stream decode", serveInkling2LoopArch)
	} else {
		l, err := loadServeInkling2(path, rt.chatBackend, os.Stderr)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 load:", err)
			os.Exit(2)
		}
		m := l.Model
		engine, closer = newInkling2LoopEngine(m, rt.chatBackend), m.Close
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
	planner, err := newServeInkling2LoopPlanner(engine, tok, *sf.model, contextTokens)
	if err != nil {
		_ = closeModel()
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	if metal {
		// The first prefill compiles the target graph and pages the weights in,
		// which outlasts a request's first-token window; pay it before readiness.
		tWarm := time.Now()
		if err := warmInkling2Session(engine.open, tok); err != nil {
			_ = closeModel()
			fmt.Fprintln(os.Stderr, "fak serve: inkling2 Metal warm-up:", err)
			os.Exit(2)
		}
		placement += fmt.Sprintf("; warm-up %.1fs", time.Since(tWarm).Seconds())
		loadDur = time.Since(tLoad)
	}
	rt.inkling2Planner, rt.inkling2Close, rt.inkling2Metal = planner, closeModel, metal
	rt.startupPhases = append(rt.startupPhases, gateway.StartupPhase{Name: "model-load", Dur: loadDur})
	rt.addStartupMessage(newServeStartupMessage("serve", "inkling2-loop", "info",
		fmt.Sprintf("%s; context=%d model=%q (greedy decode)", placement, contextTokens, *sf.model)))
	return true
}
