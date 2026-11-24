package main

import (
	"errors"
	"fmt"
	"os"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveAutoGPULayersContextFloor is the context the split is chosen against when the operator did
// not pin one: the largest band that only fits the 512-token sizer floor would starve every turn.
const serveAutoGPULayersContextFloor = 4096

// errServeAutoGPULayersNoFit marks the refusal when no partial placement fits the device either.
var errServeAutoGPULayersNoFit = errors.New("AUTO_GPU_LAYERS_NO_FIT: no --gpu-layers split fits the device")

// serveAutoGPULayersInputs carries the serve facts that decide whether a dense layer split is a
// legal placement for this boot, plus test seams for the host probe and environment.
type serveAutoGPULayersInputs struct {
	CPUOffloadExperts bool
	UseMetal          bool
	RequireDevice     bool
	Ranks             int
	RequestedContext  int
	Path              string
	Getenv            func(string) string
	// HostAvail reports available host bytes; nil reads the live host probe.
	HostAvail func() (int64, bool)
	// HostPeak overrides the header estimate of the load's host peak; nil estimates it.
	HostPeak func() (ggufload.HostLoadPeak, error)
}

func (in serveAutoGPULayersInputs) hostAvail() (int64, bool) {
	if in.HostAvail != nil {
		return in.HostAvail()
	}
	total, free, known := compute.HostSystemMemoryInfo()
	return free, known && total > 0
}

// serveAutoGPULayersEligible reports whether a dense contiguous layer split can serve this boot:
// a single-rank dense model on a discrete, capacity-probed CUDA device, on a k-quant arm whose
// session admits GPULayers > 0. MoE checkpoints keep the activated-expert ring, and device-only
// serves cannot run a host tail.
func serveAutoGPULayersEligible(ws *ggufload.WeightSource, be compute.Backend, in serveAutoGPULayersInputs) (fakmodel.Config, bool) {
	if ws == nil || ws.File == nil || be == nil || in.CPUOffloadExperts || in.UseMetal || in.RequireDevice || in.Ranks > 1 {
		return fakmodel.Config{}, false
	}
	c := be.Caps()
	if !c.DeviceMemory || !c.CapacityProbe || be.Name() != "cuda" || ggufload.BackendSharesHostRAM(be) {
		return fakmodel.Config{}, false
	}
	cfg, err := ws.File.Config()
	if err != nil || cfg.IsMoE() || cfg.NumLayers < 2 {
		return fakmodel.Config{}, false
	}
	arm := resolveDeviceServeLoadArm(ws, be, false, false)
	if arm != serveLoadArmResidentQ4K && arm != serveLoadArmQuantProfileQ8 {
		return fakmodel.Config{}, false
	}
	// Packed Q2_K and tied Q6_K token tables still refuse any layer split in the session guard.
	opts := serveQ4KFitOptions(in.Path, ws, be, arm, serveDeviceFitBudget(be))
	if ggufload.ApplyQ4KLoadOptions(opts).Q2KEmbeddingResident || ws.ResolvesTiedQ6KEmbedding(cfg, opts...) {
		return fakmodel.Config{}, false
	}
	return cfg, true
}

// serveDevicePlanFitsAtLayers judges the load's own device plan with layers [n, NumLayers) on host.
func serveDevicePlanFitsAtLayers(ws *ggufload.WeightSource, be compute.Backend, path string, contextTokens, n int) error {
	fit := serveDeviceFitBudget(be)
	fit.DenseGPULayers = n
	plan, err := serveGGUFDeviceDemands(ws, path, be, false, contextTokens, &fit)
	_, err = refuseIfTooBigOnDevice(plan, err, be, &fit)
	return err
}

// serveSplitFitsContext reports whether layers [0, n) on device leave room for contextTokens of KV:
// the split weights the context sizer will see, plus exactly that window, under the load's device
// admission. The load's own plan sizes KV against the unsplit weights, so it cannot answer this.
func serveSplitFitsContext(ws *ggufload.WeightSource, be compute.Backend, cfg fakmodel.Config, path string, contextTokens, n int) error {
	weights, fit, err := serveNativeContextSizingInputs(ws, be, false, false, 1, n, path)
	if err != nil {
		return err
	}
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	if got, _ := compute.AutoSizeContextPlan(csc, weights, fit.avail(), contextTokens); got < contextTokens {
		return &compute.FitError{Verdict: compute.FitTooBig, Want: weights.DeviceTotal(), Avail: fit.avail(), Demands: weights, Scope: compute.MemoryScopeDevice}
	}
	_, err = refuseIfTooBigOnDevice(append(weights, csc.PerContextMemoryPlan(contextTokens)...), nil, be, &fit)
	return err
}

// serveSplitLargestContext is the largest KV window layers [0, n) on device leave, or 0.
func serveSplitLargestContext(ws *ggufload.WeightSource, be compute.Backend, cfg fakmodel.Config, path string, n int) int {
	weights, fit, err := serveNativeContextSizingInputs(ws, be, false, false, 1, n, path)
	if err != nil {
		return 0
	}
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	// An explicit full window is clamped to the largest that fits (#13036).
	tokens, _ := compute.AutoSizeContextPlan(csc, weights, fit.avail(), csc.MaxContext)
	return tokens
}

// serveLargestFittingSplit is the largest n in [1, NumLayers) whose split fits contextTokens, or 0.
func serveLargestFittingSplit(ws *ggufload.WeightSource, be compute.Backend, cfg fakmodel.Config, path string, contextTokens int) int {
	firstMiss := 1 + sort.Search(cfg.NumLayers-1, func(i int) bool {
		return serveSplitFitsContext(ws, be, cfg, path, contextTokens, i+1) != nil
	})
	return firstMiss - 1
}

// serveAutoGPULayers is the fit-with-offload default for a serve with no --gpu-layers: when the
// full device plan is FitTooBig it returns the LARGEST contiguous device band [0, N) whose device
// plan fits and whose host load peak fits, so an oversize model serves split instead of refusing.
// It returns (0, nil) when the full plan fits or the boot cannot run a split, and an error only
// when neither pool can hold any split. The device charge never falls as N grows (each layer moves
// from the host share to the device), so the band edge is found like the context sizer's.
func serveAutoGPULayers(ws *ggufload.WeightSource, be compute.Backend, in serveAutoGPULayersInputs) (int, error) {
	cfg, ok := serveAutoGPULayersEligible(ws, be, in)
	if !ok {
		return 0, nil
	}
	maxContext := cfg.ContextSizeConfigWithPrecision(serveKVPrecision()).MaxContext
	ctx := in.RequestedContext
	if ctx <= 0 {
		ctx = serveAutoGPULayersContextFloor
	}
	if maxContext > 0 && ctx > maxContext {
		ctx = maxContext
	}
	// The trigger is the load's own verdict: a model whose weights fit keeps full placement.
	fullErr := serveDevicePlanFitsAtLayers(ws, be, in.Path, ctx, 0)
	if !serveFullPlanFitTooBig(fullErr) {
		return 0, nil
	}
	if err := serveAutoGPULayersHostAdmission(ws, be, in); err != nil {
		return 0, fmt.Errorf("%w (full device plan: %v)", err, fullErr)
	}
	n := serveLargestFittingSplit(ws, be, cfg, in.Path, ctx)
	if n < 1 && ctx > compute.MinAutoContextTokens {
		// No split holds the window. Unpinned, serve the sizer's floor with the most layers. An
		// explicit window clamps like full placement does (#13036): keep the largest window the
		// lightest split holds, then the most layers at it; the context sizer logs the clamp.
		window := compute.MinAutoContextTokens
		if in.RequestedContext > 0 {
			window = max(serveSplitLargestContext(ws, be, cfg, in.Path, 1), compute.MinAutoContextTokens)
			window = min(window, ctx)
		}
		n = serveLargestFittingSplit(ws, be, cfg, in.Path, window)
	}
	if n < 1 {
		return 0, fmt.Errorf("%w at %d context tokens: %w", errServeAutoGPULayersNoFit, ctx, fullErr)
	}
	return n, nil
}

// serveAutoGPULayersHostAdmission applies the load's own host-peak decision to the split. The
// header estimate charges every layer's staging transit whatever the split, so one check covers
// every N; a refusal here is the HOST_LOAD_PEAK_TOO_BIG the load would raise after the search.
func serveAutoGPULayersHostAdmission(ws *ggufload.WeightSource, be compute.Backend, in serveAutoGPULayersInputs) error {
	getenv := in.Getenv
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	if serveHostPeakAdmissionDisabled(getenv) {
		return nil
	}
	arm := resolveDeviceServeLoadArm(ws, be, false, false)
	opts := serveQ4KFitOptions(in.Path, ws, be, arm, serveDeviceFitBudget(be))
	var peak ggufload.HostLoadPeak
	var estErr error
	if in.HostPeak != nil {
		peak, estErr = in.HostPeak()
	} else {
		peak, estErr = estimateServeNativeHostLoadPeakFromWeights(ws, opts)
	}
	avail, known := in.hostAvail()
	_, err := serveNativeHostLoadPeakDecision(peak, estErr, avail, known, getenv, opts...)
	return err
}

// resolveAutoGPULayers applies the fit-with-offload default before context sizing, so the
// context, the fit-commit budget and the session all see the chosen split. An explicit
// --gpu-layers (either alias) is the operator's placement and is never moved.
func (rt *serveRuntime) resolveAutoGPULayers(sf *serveFlags, ws *ggufload.WeightSource, ranks, requestedContext int, path string) error {
	if sf == nil || sf.nativeGPULayers == nil || *sf.nativeGPULayers != 0 ||
		sf.isExplicitFlag("gpu-layers") || sf.isExplicitFlag("native-gpu-layers") {
		return nil
	}
	n, err := serveAutoGPULayers(ws, rt.chatBackend, serveAutoGPULayersInputs{
		CPUOffloadExperts: sf.cpuOffloadExperts != nil && *sf.cpuOffloadExperts,
		UseMetal:          rt.useMetal,
		RequireDevice:     rt.requireDeviceExecution,
		Ranks:             ranks,
		RequestedContext:  requestedContext,
		Path:              path,
		Getenv:            os.Getenv,
	})
	if err != nil {
		return fmt.Errorf("fak serve: fit-with-offload: %w", err)
	}
	if n <= 0 {
		return nil
	}
	*sf.nativeGPULayers = n
	numLayers := 0
	if cfg, cfgErr := ws.File.Config(); cfgErr == nil {
		numLayers = cfg.NumLayers
	}
	msg := fmt.Sprintf("full device plan exceeds the device budget; fit-with-offload chose --gpu-layers %d (layers [0,%d) on device, [%d,%d) on host)", n, n, n, numLayers)
	fmt.Fprintln(os.Stderr, "fak serve:", msg)
	rt.addStartupMessage(newServeStartupMessage("serve", "auto-gpu-layers", "info", msg))
	return nil
}
