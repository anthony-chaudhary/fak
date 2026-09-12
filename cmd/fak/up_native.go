package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// turnkeyNativeResources owns the native objects whose lifetime must extend through
// HTTP shutdown. Close is called after the server has drained its requests: native
// weight handles and streamed checkpoint mappings are released before the admission
// reservation that protected their residency.
type turnkeyNativeResources struct {
	Planner     *agent.InKernelPlanner
	Model       *fakmodel.Model
	LoadProfile *gateway.ModelLoadProfile
	Startup     turnkeyNativeStartup

	closeModel       func() error
	releaseAdmission func()
	closeOnce        sync.Once
	closeErr         error
}

type turnkeyNativeStartup struct {
	MetalLive           bool                     `json:"metal_live"`
	MetalCompiled       bool                     `json:"metal_compiled"`
	HostTotalBefore     int64                    `json:"host_total_before_bytes,omitempty"`
	HostAvailableBefore int64                    `json:"host_available_before_bytes,omitempty"`
	HostTotalAfter      int64                    `json:"host_total_after_bytes,omitempty"`
	HostAvailableAfter  int64                    `json:"host_available_after_bytes,omitempty"`
	LoadMode            string                   `json:"load_mode,omitempty"`
	LoadSeconds         float64                  `json:"load_seconds,omitempty"`
	LoadBytes           int64                    `json:"load_bytes,omitempty"`
	LoadTensors         int                      `json:"load_tensors,omitempty"`
	LoadBottleneck      string                   `json:"load_bottleneck,omitempty"`
	Resident            *fakmodel.ResidentReport `json:"resident,omitempty"`
	MetalLiveQ6Weights  int                      `json:"metal_live_q6_weights"`
	MetalLiveQ8Weights  int                      `json:"metal_live_q8_weights"`
}

func (r *turnkeyNativeResources) Close() error {
	if r == nil {
		return nil
	}
	r.closeOnce.Do(func() {
		if r.closeModel != nil {
			r.closeErr = r.closeModel()
		}
		if r.releaseAdmission != nil {
			r.releaseAdmission()
		}
	})
	return r.closeErr
}

type turnkeyNativeLoadDeps struct {
	resolveBackend func() (compute.Backend, error)
	resolveMetal   func() (serveMetalDecision, error)
	admitAndLoad   func(bool, string, func()) (func(), error)
	refusePeak     func(string) error
	loadModel      func(string, compute.Backend, int) (*fakmodel.Model, bool, *gateway.ModelLoadProfile)
	loadTokenizer  func(string) (*tokenizer.Tokenizer, bool)
	newPlanner     func(*fakmodel.Model, *tokenizer.Tokenizer, string, bool, compute.Backend, bool, int) *agent.InKernelPlanner
	hostMemory     func() (int64, int64, bool)
	metalResidency func() (int, int)
}

func defaultTurnkeyNativeLoadDeps() turnkeyNativeLoadDeps {
	return turnkeyNativeLoadDeps{
		resolveBackend: func() (compute.Backend, error) { return resolveServeChatBackend("") },
		resolveMetal:   func() (serveMetalDecision, error) { return resolveServeMetalDecision(false, false, "") },
		refusePeak:     refuseOversubscribedMetalGGUF,
		admitAndLoad: func(metal bool, path string, load func()) (func(), error) {
			return loadLocalLauncherModelWithMetalLease(metal, path, gpulease.Options{}, load)
		},
		loadModel: func(path string, backend compute.Backend, contextTokens int) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
			m, q4k, profile, _ := loadServeInKernelModel(path, backend, false, contextTokens, nil, 1, nil)
			return m, q4k, profile
		},
		loadTokenizer:  func(path string) (*tokenizer.Tokenizer, bool) { return resolveServeTokenizer("", path) },
		newPlanner:     newTurnkeyInKernelPlanner,
		hostMemory:     compute.HostSystemMemoryInfo,
		metalResidency: func() (int, int) { return metalgemm.LiveQ6KWeights(), metalgemm.LiveQ8Weights() },
	}
}

// loadTurnkeyNativeResources performs the native-only portion of fak up startup.
// modelPath is already registry-resolved and fetched by the caller.
func loadTurnkeyNativeResources(ctx context.Context, modelPath, modelID string, contextTokens int) (*turnkeyNativeResources, error) {
	return loadTurnkeyNativeResourcesWith(ctx, modelPath, modelID, contextTokens, defaultTurnkeyNativeLoadDeps())
}

func loadTurnkeyNativeResourcesWith(_ context.Context, modelPath, modelID string, contextTokens int, deps turnkeyNativeLoadDeps) (*turnkeyNativeResources, error) {
	var startup turnkeyNativeStartup
	if deps.hostMemory != nil {
		startup.HostTotalBefore, startup.HostAvailableBefore, _ = deps.hostMemory()
	}
	backend, err := deps.resolveBackend()
	if err != nil {
		return nil, fmt.Errorf("backend: %w", err)
	}
	metalDecision, err := deps.resolveMetal()
	if err != nil {
		return nil, err
	}
	metal := metalDecision.live
	if metal {
		if err := deps.refusePeak(modelPath); err != nil {
			return nil, err
		}
	}

	var model *fakmodel.Model
	var q4k bool
	var profile *gateway.ModelLoadProfile
	release, err := deps.admitAndLoad(metal, modelPath, func() {
		model, q4k, profile = deps.loadModel(modelPath, backend, contextTokens)
	})
	if err != nil {
		return nil, err
	}
	if model == nil {
		release()
		return nil, fmt.Errorf("failed to load %q into the in-kernel engine", modelPath)
	}
	if deps.hostMemory != nil {
		startup.HostTotalAfter, startup.HostAvailableAfter, _ = deps.hostMemory()
	}
	if profile != nil {
		startup.LoadMode = profile.Mode
		startup.LoadSeconds = profile.TotalSeconds
		startup.LoadBytes = profile.Bytes
		startup.LoadTensors = profile.Tensors
		startup.LoadBottleneck = profile.Bottleneck
	}
	startup.Resident = model.ResidentReport()
	if deps.metalResidency != nil {
		startup.MetalLiveQ6Weights, startup.MetalLiveQ8Weights = deps.metalResidency()
	}
	startup.MetalLive = metalDecision.live
	startup.MetalCompiled = metalgemm.Compiled()
	tok, ok := deps.loadTokenizer(modelPath)
	if !ok || tok == nil {
		_ = model.CloseWeights()
		release()
		return nil, fmt.Errorf("%q has no usable tokenizer; pass a GGUF with an embedded tokenizer", modelPath)
	}

	return &turnkeyNativeResources{
		Planner:          deps.newPlanner(model, tok, modelID, q4k, backend, metal, contextTokens),
		Model:            model,
		LoadProfile:      profile,
		Startup:          startup,
		closeModel:       model.CloseWeights,
		releaseAdmission: release,
	}, nil
}
