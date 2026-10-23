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
	// Sizing is the --expert-ring-bytes choice and its inputs; Mode is token by default.
	Sizing serveExpertRingSizing
}

func serveFullPlanFitTooBig(err error) bool {
	var fe *compute.FitError
	return errors.As(err, &fe) && fe.Verdict == compute.FitTooBig
}

// serveActivatedExpertRingWeightPlan sizes the ring to one whole token's activated experts when the
// budget allows, never below one layer's activated set and never above the room the budget leaves.
// Filling the whole room would leave the KV cache nothing. Host-gathered tables (the V4.1 Engram
// rows) are never device-resident; they ride as a host-scoped page reserve so a shared-pool check
// still counts them.
func serveActivatedExpertRingWeightPlan(f ggufload.ActivatedExpertFit) (compute.MemoryPlan, int64) {
	ring := min(f.ActivatedTokenBytes, f.RingBytes)
	ring = max(ring, f.ActivatedLayerBytes)
	plan := make(compute.MemoryPlan, 0, 3)
	if f.DeviceBaseBytes > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryWeights, Bytes: f.DeviceBaseBytes, Detail: serveActivatedRingBaseDetail, Scope: compute.MemoryScopeDevice})
	}
	if ring > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryWeights, Bytes: ring, Detail: serveActivatedRingDetail, Scope: compute.MemoryScopeDevice})
	}
	if d, ok := f.HostGatheredDemand(); ok {
		plan = append(plan, d)
	}
	return plan, ring
}

// serveActivatedExpertRingSplit returns every plan row except the device ring row.
func serveActivatedExpertRingSplit(plan compute.MemoryPlan) compute.MemoryPlan {
	out := make(compute.MemoryPlan, 0, len(plan))
	for _, d := range plan {
		if d.Detail == serveActivatedRingDetail && d.DeviceScoped() {
			continue
		}
		out = append(out, d)
	}
	return out
}

// serveActivatedExpertRingResize sets the device ring row to ring bytes; every other row is kept.
func serveActivatedExpertRingResize(plan compute.MemoryPlan, ring int64) compute.MemoryPlan {
	out := append(compute.MemoryPlan(nil), plan...)
	for i := range out {
		if out[i].Detail == serveActivatedRingDetail && out[i].DeviceScoped() {
			out[i].Bytes = ring
		}
	}
	return out
}

// serveActivatedExpertRingPlacement returns the ring placement when fullErr is the full device plan's
// FitTooBig on a device-only serve. Any other case returns fullErr unchanged, so a non-Halo serve and
// a fitting checkpoint keep their historical arm.
func serveActivatedExpertRingPlacement(ws *ggufload.WeightSource, be compute.Backend, requireDevice bool, fullErr error, contextBudgetTokens int, fit serveFitBudget) (serveActivatedExpertRing, bool, error) {
	return serveActivatedExpertRingPlacementSized(ws, be, requireDevice, fullErr, contextBudgetTokens, fit, serveExpertRingSize{Mode: serveExpertRingToken}, serveExpertRingHost{})
}

// serveActivatedExpertRingPlacementSized is the placement with an operator ring size. The token
// ring is admitted first, so KV for the target context is reserved before a larger ring takes the
// remaining room; a non-token size is then judged against the device bound and the host bound.
func serveActivatedExpertRingPlacementSized(ws *ggufload.WeightSource, be compute.Backend, requireDevice bool, fullErr error, contextBudgetTokens int, fit serveFitBudget, size serveExpertRingSize, host serveExpertRingHost) (serveActivatedExpertRing, bool, error) {
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
	sizing, err := sizeServeExpertRing(size, f, ring, serveActivatedExpertRingSplit(plan), fit.avail(), host)
	if err != nil {
		return serveActivatedExpertRing{}, false, err
	}
	if sizing.RingBytes != ring {
		ring = sizing.RingBytes
		plan = serveActivatedExpertRingResize(plan, ring)
	}
	plan, ring, err = serveAdmitActivatedExpertRing(plan, ring, &sizing, be, fit)
	if err != nil {
		// Admission can fail before the caller logs the selected placement. Keep the
		// candidate byte split visible without returning an admitted ring or load options.
		err = fmt.Errorf("activated-expert device ring refused (dense base=%d bytes, ring=%d bytes): %w", f.DeviceBaseBytes, ring, err)
		if sizing.Mode == serveExpertRingAuto || sizing.Mode == serveExpertRingExplicit {
			err = fmt.Errorf("%w: %w", ErrServeExpertRingTooBig, err)
		}
		return serveActivatedExpertRing{}, false, err
	}
	return serveActivatedExpertRing{Fit: f, RingBytes: ring, Plan: plan, Sizing: sizing}, true, nil
}

// serveAdmitActivatedExpertRingAttempts bounds how often an auto ring is trimmed to the
// load-time guard's own shortfall before the refusal stands.
const serveAdmitActivatedExpertRingAttempts = 4

// serveAdmitActivatedExpertRing runs the load-time device and shared-pool guard on the plan. An
// auto ring that a guard refuses is trimmed by the guard's reported shortfall and retried, so the
// guard (live device budget, physical pool) stays the authority over the size the bounds chose;
// token and explicit rings are judged as given.
func serveAdmitActivatedExpertRing(plan compute.MemoryPlan, ring int64, sizing *serveExpertRingSizing, be compute.Backend, fit serveFitBudget) (compute.MemoryPlan, int64, error) {
	for attempt := 0; ; attempt++ {
		admitted, err := refuseIfTooBigOnDevice(plan, nil, be, &fit)
		if err == nil {
			return admitted, ring, nil
		}
		var fe *compute.FitError
		if sizing.Mode != serveExpertRingAuto || attempt+1 >= serveAdmitActivatedExpertRingAttempts || !errors.As(err, &fe) || fe.Want <= fe.Avail {
			return nil, ring, err
		}
		trimmed := ring - (fe.Want - fe.Avail)
		if trimmed < sizing.Floor {
			return nil, ring, err
		}
		sizing.GuardTrimBytes += ring - trimmed
		ring = trimmed
		sizing.RingBytes = ring
		sizing.Binding = serveExpertRingBindingGuard
		plan = serveActivatedExpertRingResize(plan, ring)
	}
}

// serveActivatedExpertRingPathPlacement resolves the --expert-ring-bytes size from getenv only
// once the ring route applies, so a serve that never takes it never reads the knob.
func serveActivatedExpertRingPathPlacement(ggufPath string, be compute.Backend, requireDevice bool, fullErr error, contextBudgetTokens int, override *serveFitBudget, getenv func(string) string, baseOpts []ggufload.Q4KLoadOption) (serveActivatedExpertRing, bool, error) {
	if !requireDevice || be == nil || !serveFullPlanFitTooBig(fullErr) {
		return serveActivatedExpertRing{}, false, fullErr
	}
	size, err := resolveServeExpertRingSize("", getenv)
	if err != nil {
		return serveActivatedExpertRing{}, false, err
	}
	ws, openErr := ggufload.OpenWeights(ggufPath)
	if openErr != nil {
		return serveActivatedExpertRing{}, false, fullErr
	}
	defer ws.Close()
	var host serveExpertRingHost
	if size.Mode != serveExpertRingToken && size.Mode != "" {
		host = serveExpertRingHostProbe(ggufPath, be, baseOpts, compute.HostSystemMemoryInfo)
	}
	return serveActivatedExpertRingPlacementSized(ws, be, requireDevice, fullErr, contextBudgetTokens, serveDeviceFitBudgetFromReported(be, override), size, host)
}

// serveExpertRingHostProbe reads the host snapshot a non-token ring size is judged against:
// MemAvailable at load, the route's estimated anonymous host peak, and whether the device ring
// draws on that same pool (integrated tier with a carve-out <= 4 GiB or an unreadable one).
func serveExpertRingHostProbe(ggufPath string, be compute.Backend, baseOpts []ggufload.Q4KLoadOption, hostMemoryInfo func() (int64, int64, bool)) serveExpertRingHost {
	var h serveExpertRingHost
	h.Total, h.Avail, h.Known = compute.HostMemoryInfo(be)
	if !h.Known {
		h.Total, h.Avail, h.Known = hostMemoryInfo()
	}
	h.Carveout, h.CarveoutKnown = compute.DedicatedVRAMCarveoutInfo(be)
	h.SharedPool = serveExpertRingSharedPool(ggufload.BackendSharesHostRAM(be), h.Carveout, h.CarveoutKnown)
	opts := append(append([]ggufload.Q4KLoadOption(nil), baseOpts...), ggufload.WithStreamedExperts(0), ggufload.WithStreamedExpertDeviceRing(1), ggufload.WithStreamedDenseQ4KWorkingSet(0))
	if peak, err := estimateServeActivatedExpertRingFixedHostPeak(ggufPath, opts); err == nil {
		h.RSSAnonPeak = peak.PeakAnonBytes
	}
	return h
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

// serveExpertRingSharedPool reports whether a backend's device ring is judged against host RAM.
func serveExpertRingSharedPool(sharesHostRAM bool, carveout int64, carveoutKnown bool) bool {
	return sharesHostRAM && (!carveoutKnown || carveout <= serveExpertRingUMACarveoutMax)
}

// serveExpertRingContextWeights is the weight plan every ring size resolves the native
// context against. A device-only serve whose full resident weights exceed the device budget loads
// the ring placement, so sizing the context against the full weights collapses it to the minimum
// window; the placement's token plan (dense base + token ring + host page reserve) is what KV is
// actually reserved next to before the ring takes the rest. Token mode uses the same
// bounded charge: retaining the full checkpoint here would clamp its context before
// the loader selects the already bounded token ring (fak#13863).
func serveExpertRingContextWeights(ws *ggufload.WeightSource, be compute.Backend, requireDevice bool, size serveExpertRingSize, weights compute.MemoryPlan, fit serveFitBudget) (compute.MemoryPlan, bool) {
	if !requireDevice || be == nil || ws == nil {
		return weights, false
	}
	avail := fit.avail()
	if avail <= 0 || weights.DeviceTotal() <= avail || !serveStreamedExpertsCapable(ws) {
		return weights, false
	}
	f, ok, err := ws.ActivatedExpertFitFor(avail)
	if err != nil || !ok {
		return weights, false
	}
	ringWeights, _ := serveActivatedExpertRingWeightPlan(f)
	return ringWeights, true
}
