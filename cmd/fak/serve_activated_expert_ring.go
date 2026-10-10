package main

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serve_activated_expert_ring.go — the guard-admitted device placement for a routed-MoE checkpoint
// whose full device-resident plan is FitTooBig on a device-only (Strix Halo) serve (fak#13668).
//
// The only fitting route used to be the streamed --cpu-offload-experts arm, which the Halo guard
// refuses because it runs host expert GEMMs. This placement keeps the dense base device-resident,
// leaves the routed band on disk behind the R5 checkpoint tier with zero host retention
// (stream-through), and declares a bounded DEVICE ring on that tier, so a faulted expert is staged
// into device memory and executed there. It is selected without --cpu-offload-experts, so
// validateServeHaloCPUOffload is untouched.
//
// The ring is sized from ggufload.ActivatedExpertFit — the same header arithmetic
// FitActivatedExpertsOnDevice admits on — but judged against the serve's own device fit snapshot.
// FitActivatedExpertsOnDevice itself is not called: it charges the un-ringed band host-resident, which
// a host-probing backend (Vulkan on Halo) would refuse, while this placement never holds it in RAM.

const (
	serveActivatedRingBaseDetail = "gguf-device-dense-base"
	serveActivatedRingDetail     = "gguf-device-activated-expert-ring"
)

// errServeActivatedExpertRingUnstageable names why a FitTooBig MoE checkpoint gets no ring placement.
var errServeActivatedExpertRingUnstageable = errors.New("activated-expert device ring unavailable: checkpoint has no stageable routed-expert slab")

type serveActivatedExpertRing struct {
	Fit       ggufload.ActivatedExpertFit
	RingBytes int64
	Plan      compute.MemoryPlan
}

func serveFullPlanFitTooBig(err error) bool {
	var fe *compute.FitError
	return errors.As(err, &fe) && fe.Verdict == compute.FitTooBig
}

// serveActivatedExpertRingWeightPlan sizes the ring to one whole token's activated experts when the
// budget allows, never below one layer's activated set and never above the room the budget leaves.
// Filling the whole room would leave the KV cache nothing.
func serveActivatedExpertRingWeightPlan(f ggufload.ActivatedExpertFit) (compute.MemoryPlan, int64) {
	ring := min(f.ActivatedTokenBytes, f.RingBytes)
	ring = max(ring, f.ActivatedLayerBytes)
	plan := make(compute.MemoryPlan, 0, 2)
	if f.DeviceBaseBytes > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryWeights, Bytes: f.DeviceBaseBytes, Detail: serveActivatedRingBaseDetail, Scope: compute.MemoryScopeDevice})
	}
	if ring > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryWeights, Bytes: ring, Detail: serveActivatedRingDetail, Scope: compute.MemoryScopeDevice})
	}
	return plan, ring
}

// serveActivatedExpertRingPlacement returns the ring placement when fullErr is the full device plan's
// FitTooBig on a device-only serve. Any other case returns fullErr unchanged, so a non-Halo serve and
// a fitting checkpoint keep their historical arm.
func serveActivatedExpertRingPlacement(ws *ggufload.WeightSource, be compute.Backend, requireDevice bool, fullErr error, contextBudgetTokens int, fit serveFitBudget) (serveActivatedExpertRing, bool, error) {
	if !requireDevice || be == nil || ws == nil || !serveFullPlanFitTooBig(fullErr) {
		return serveActivatedExpertRing{}, false, fullErr
	}
	if !serveStreamedExpertsCapable(ws) {
		return serveActivatedExpertRing{}, false, fmt.Errorf("%w; %w", fullErr, errServeActivatedExpertRingUnstageable)
	}
	f, ok, err := ws.ActivatedExpertFitFor(fit.avail())
	if err != nil {
		return serveActivatedExpertRing{}, false, err
	}
	if !ok {
		return serveActivatedExpertRing{}, false, fmt.Errorf("%w; %w", fullErr, errServeActivatedExpertRingUnstageable)
	}
	weights, ring := serveActivatedExpertRingWeightPlan(f)
	plan := appendServeGGUFDevicePlan(ws, be, weights, contextBudgetTokens, fit)
	plan, err = refuseIfTooBigOnDevice(plan, nil, be, &fit)
	if err != nil {
		// Admission can fail before the caller logs the selected placement. Keep the
		// candidate byte split visible without returning an admitted ring or load options.
		return serveActivatedExpertRing{}, false, fmt.Errorf("activated-expert device ring refused (dense base=%d bytes, ring=%d bytes): %w", f.DeviceBaseBytes, ring, err)
	}
	return serveActivatedExpertRing{Fit: f, RingBytes: ring, Plan: plan}, true, nil
}

func serveActivatedExpertRingPathPlacement(ggufPath string, be compute.Backend, requireDevice bool, fullErr error, contextBudgetTokens int, override *serveFitBudget) (serveActivatedExpertRing, bool, error) {
	if !requireDevice || be == nil || !serveFullPlanFitTooBig(fullErr) {
		return serveActivatedExpertRing{}, false, fullErr
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return serveActivatedExpertRing{}, false, fullErr
	}
	defer ws.Close()
	return serveActivatedExpertRingPlacement(ws, be, requireDevice, fullErr, contextBudgetTokens, serveDeviceFitBudgetFromReported(be, override))
}

// serveActivatedExpertRingLoadOptions streams every routed expert through the checkpoint tier with
// zero host retention and bounds the tier's device ring to the admitted size. The dense base is
// also left on disk as checkpoint ranges (fak#13668): it is uploaded to device memory one tensor at
// a time instead of being staged whole in host RAM first, which swapped a 31 GiB-MemTotal carve-out
// box loading a 63 GiB base. The load path replaces the stream-through working set declared here
// with serveActivatedExpertRingDenseOption's sized bound.
func serveActivatedExpertRingLoadOptions(ring serveActivatedExpertRing) []ggufload.Q4KLoadOption {
	return []ggufload.Q4KLoadOption{ggufload.WithStreamedExperts(0), ggufload.WithStreamedExpertDeviceRing(ring.RingBytes), ggufload.WithStreamedDenseQ4KWorkingSet(0)}
}

// serveActivatedExpertRingDenseOption sizes the ring route's dense host working set from the
// headroom left after its estimated fixed host peak. This is planning only: the actual positive
// bound still reaches host-peak admission, which refuses an unqualified streamed staging peak.
// An unmeasurable host or estimate keeps the historical host-fit bound; admission remains separate.
func serveActivatedExpertRingDenseOption(ggufPath string, opts []ggufload.Q4KLoadOption, fallback serveFitBudget, getenv func(string) string) ggufload.Q4KLoadOption {
	return serveActivatedExpertRingDenseOptionForHost(ggufPath, opts, fallback, getenv, compute.HostSystemMemoryInfo)
}

func serveActivatedExpertRingDenseOptionForHost(ggufPath string, opts []ggufload.Q4KLoadOption, fallback serveFitBudget, getenv func(string) string, hostMemoryInfo func() (int64, int64, bool)) ggufload.Q4KLoadOption {
	fixed, err := estimateServeActivatedExpertRingFixedHostPeak(ggufPath, opts)
	_, free, known := hostMemoryInfo()
	if err != nil || !known || free <= 0 {
		return ggufload.WithStreamedDenseQ4KWorkingSet(serveStreamedDenseQ4KWorkingSetBound(fallback))
	}
	return ggufload.WithStreamedDenseQ4KWorkingSet(ggufload.StreamedDenseWorkingSetFromHeadroom(fixed, free, serveHostPeakMarginBytes(getenv), serveCPUOffloadStreamedResidentMargin))
}

// The zero-bound probe asks only for the estimator's fixed terms. It is never a load option or
// admission decision: estimateServeNativeHostLoadPeak correctly rejects zero on the actual route.
func estimateServeActivatedExpertRingFixedHostPeak(ggufPath string, opts []ggufload.Q4KLoadOption) (ggufload.HostLoadPeak, error) {
	probe := append(append([]ggufload.Q4KLoadOption(nil), opts...), ggufload.WithStreamedDenseQ4KWorkingSet(0))
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return ggufload.HostLoadPeak{}, err
	}
	defer ws.Close()
	return ws.EstimateStreamedExpertHostLoadPeak(probe...)
}
