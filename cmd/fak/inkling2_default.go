package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

// turnkeyInkling2Servable is nil when the serving surface admits Inkling-2 on
// the Apple GPU. It is the one gate between the Mac default and a model the
// server would refuse.
var turnkeyInkling2Servable = func() error {
	return fakmodel.RefuseInkling2LoopBackend("metal")
}

// turnkeyMacDefaultTierFor is the tier `fak up` serves by default on the given
// platform, or false to stay on the standard ladder.
func turnkeyMacDefaultTierFor(goos, goarch string, memoryBytes uint64) (macfit.ModelTier, bool) {
	return macfit.MacDefaultTier(goos, goarch, memoryBytes, turnkeyInkling2Servable() == nil)
}

// runDefaultModelFor is the model a bare `fak run` uses on the given platform.
// Only Apple Silicon has one: Inkling-2-Mini when it fits, otherwise the ladder
// tier for the host's memory. `fak run` loads in-process, so it does not wait
// on the serving gate.
func runDefaultModelFor(goos, goarch string, memoryBytes uint64) (string, bool) {
	if goos != "darwin" || goarch != "arm64" {
		return "", false
	}
	if _, ok := macfit.MacDefaultTier(goos, goarch, memoryBytes, true); ok {
		return modelreg.MacDefaultAlias, true
	}
	plan, err := macfit.ConfigureTurnkeyWithOptions(memoryBytes, macfit.TurnkeyOptions{StaticOnly: true})
	if err != nil {
		return "", false
	}
	return resolveTurnkeyModelRef(plan.Tier.ModelID), true
}

// runDefaultModel is a seam so argument parsing is testable off the host.
var runDefaultModel = func() (string, bool) {
	memoryBytes, err := macfit.DetectUnifiedMemory()
	if err != nil || memoryBytes == 0 {
		return "", false
	}
	return runDefaultModelFor(runtime.GOOS, runtime.GOARCH, memoryBytes)
}

// runArgIsPrompt reports a first positional argument that is prompt text rather
// than a model: no alias, URI or path contains whitespace unless the file exists.
func runArgIsPrompt(arg string) bool {
	if !strings.ContainsAny(arg, " \t\n") {
		return false
	}
	_, err := os.Stat(arg)
	return err != nil
}

// fetchModelArtifact resolves an hf:// target to a local file, reusing an
// existing local copy of the Mac default and bringing the companions the
// artifact expects beside it. Companions are optional accelerators, so a failed
// companion fetch is reported and the artifact still loads.
func fetchModelArtifact(ctx context.Context, uri string, progress io.Writer) (string, error) {
	if local, ok := modelreg.LocalMacDefaultArtifact(uri); ok {
		if err := modelreg.VerifyLocalArtifact(uri, local); err != nil {
			return "", err
		}
		fmt.Fprintf(progress, "using local copy: %s\n", local)
		return local, nil
	}
	resolved, err := modelreg.FetchVerified(ctx, hfhub.NewClient(), uri, progress)
	if err != nil {
		return "", err
	}
	if err := modelreg.FetchCompanions(ctx, uri, resolved, progress); err != nil {
		fmt.Fprintf(progress, "warning: %v; continuing without it\n", err)
	}
	return resolved, nil
}

// turnkeyInkling2Planner serves the in-process Inkling-2 runner one request at
// a time and reports the context its admission planned for.
type turnkeyInkling2Planner struct {
	agent.Planner
	mu     sync.Mutex
	window int
}

func (p *turnkeyInkling2Planner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.Planner.Complete(ctx, messages, tools, opts...)
}

func (p *turnkeyInkling2Planner) ContextWindow() int { return p.window }

// openTurnkeyInkling2 loads an Inkling-2 GGUF for `fak up`, refusing whenever
// the serving gate does.
func openTurnkeyInkling2(path, name string) (agent.Planner, func(), error) {
	if err := turnkeyInkling2Servable(); err != nil {
		return nil, nil, err
	}
	planner, release, err := openInkling2Run(path, name, "", false, nativeControlConfig{})
	if err != nil {
		return nil, nil, err
	}
	if planner == nil {
		return nil, nil, fmt.Errorf("%s is not an Inkling-2 GGUF", path)
	}
	return &turnkeyInkling2Planner{Planner: planner, window: gateway.DefaultAdmissionPolicy().TokenBudget}, release, nil
}
