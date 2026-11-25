package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// serveInkling2LoopArch is the GGUF general.architecture of the looped family and
// the model_type its served config carries.
const serveInkling2LoopArch = "inkling2_loop"

// serveInkling2LoopDefaultModelID is advertised when --model is left at its default.
const serveInkling2LoopDefaultModelID = "inkling2"

// serveIsInkling2LoopGGUF reads only the GGUF header: a directory, a missing
// file or an unreadable header is not the looped family.
func serveIsInkling2LoopGGUF(path string) bool {
	if path == "" {
		return false
	}
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return false
	}
	f, err := ggufload.Open(path)
	if err != nil {
		return false
	}
	arch, _ := f.String("general.architecture")
	return arch == serveInkling2LoopArch
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

// loadInkling2LoopModel serves an inkling2_loop GGUF on the dedicated looped
// forward and reports whether it claimed the load. A backend without that forward
// is refused here, before the generic loaders mis-read the family.
func (rt *serveRuntime) loadInkling2LoopModel(sf *serveFlags) bool {
	path := *sf.ggufPath
	if !serveIsInkling2LoopGGUF(path) {
		return false
	}
	if err := fakmodel.RefuseInkling2LoopBackend(serveInkling2LoopBackendName(rt.chatBackend, rt.useMetal)); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	tLoad := time.Now()
	m, err := ggufload.OpenInkling2LoopDevice(path, rt.chatBackend)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: inkling2_loop load:", err)
		os.Exit(2)
	}
	loadDur := time.Since(tLoad)
	tok, ok := resolveServeTokenizer(*sf.tokPath, path, rt.addStartupMessage)
	if !ok {
		_ = m.Close()
		fmt.Fprintln(os.Stderr, "fak serve: inkling2_loop needs the GGUF's embedded tokenizer or --tokenizer")
		os.Exit(2)
	}
	if *sf.model == "mock" || *sf.model == "" {
		*sf.model = serveInkling2LoopDefaultModelID
	}
	planner, err := newServeInkling2LoopPlanner(newInkling2LoopEngine(m, rt.chatBackend), tok, *sf.model, ggufload.Inkling2LoopResidentContext)
	if err != nil {
		_ = m.Close()
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	rt.inkling2Planner, rt.inkling2Model = planner, m
	rt.startupPhases = append(rt.startupPhases, gateway.StartupPhase{Name: "model-load", Dur: loadDur})
	rt.addStartupMessage(newServeStartupMessage("serve", "inkling2-loop", "info",
		fmt.Sprintf("inkling2_loop looped forward on %s: layers=%d hidden=%d passes=%d context=%d model=%q (greedy decode)",
			rt.chatBackend.Name(), m.HP.Layers, m.HP.Hidden, m.HP.Passes, ggufload.Inkling2LoopResidentContext, *sf.model)))
	return true
}
