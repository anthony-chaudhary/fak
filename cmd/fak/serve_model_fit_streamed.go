package main

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveStreamedCPUOffloadPlan is serveGGUFCPUOffloadMemoryPlan under the bounded-resident
// NVMe-streamed expert policy (fak#13121). It builds the FULL-charge plan first and measures its
// HOST-scoped subset against the host budget: when the full routed charge fits (or the host is
// unprobeable), streaming is not needed and the caller keeps the resident arm byte-for-byte -- the
// preserved case for small-expert and non-MoE artifacts. Only when the full charge cannot fit AND
// the checkpoint tier can stage the routed slabs does it return the streamed plan with streamed=true,
// so the load arm threads WithStreamedExperts and the routed experts fault from the staged shards.
//
// The streamed plan charges the device dense side IDENTICALLY (the same per-tensor classification),
// so the device fit check and the resident arm cannot disagree about what stays on the device.
func serveStreamedCPUOffloadPlan(ws *ggufload.WeightSource, be compute.Backend, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, bool, error) {
	return serveStreamedCPUOffloadPlanForPool(ws, be, ranks, contextBudgetTokens, fit, false)
}

// serveStreamedCPUOffloadPlanForPool is serveStreamedCPUOffloadPlan with the pool topology made
// explicit. sharedPool reports whether the device-scoped dense charge draws from the SAME physical
// DRAM pool the host-resident expert set is judged against -- an integrated/APU tier
// (ggufload.BackendSharesHostRAM).
//
// On a DISCRETE device the two pools are independent: the device dense side lives in VRAM and must
// NOT be double-counted against host RAM, so streaming is selected when the HOST-scoped routed set
// alone exceeds host avail (the historical #13121 trigger), unchanged.
//
// On a SHARED pool the device dense side is ALSO physical DRAM (staged through host RAM and, on the
// integrated tier, resident in the same unified heap), so the quantity that must fit is the plan's
// GRAND total. The witnessed strix3 refusal is exactly this: weights=63.09 GiB device-scoped +
// offload(host)=43.86 GiB routed, HostTotal 43.86 <= avail 48.72 so the resident arm was kept, yet
// the grand total 107.08 GiB cannot fit 62.4 GiB MemTotal -> typed FitTooBig before the forward.
// Selecting streaming on the grand total admits the same checkpoint with a bounded resident expert
// set instead of refusing. The device dense side is still charged IDENTICALLY by the streamed plan,
// so the device fit check and the resident arm cannot disagree about what stays on the device.
func serveStreamedCPUOffloadPlanForPool(ws *ggufload.WeightSource, be compute.Backend, ranks, contextBudgetTokens int, fit serveFitBudget, sharedPool bool) (compute.MemoryPlan, bool, error) {
	return serveStreamedCPUOffloadPlanForAperture(ws, be, ranks, contextBudgetTokens, fit, sharedPool, false)
}

// serveStreamedCPUOffloadPlanForAperture is serveStreamedCPUOffloadPlanForPool with the device
// APERTURE made explicit. splitAperture reports whether the device-scoped dense side has its OWN
// device-local capacity DISTINCT from the host system window it is judged against (a split
// VRAM/system carve-out, e.g. strix3's 84.28 GiB device-local heap vs its 62.4 GiB MemTotal).
//
// On a shared pool WITHOUT a split aperture (a single unified heap: the device dense side and the
// host expert set genuinely co-reside in one physical pool) the whole device transit must be
// subtracted from the resident-expert bound -- that is the #13175 fix and it is preserved
// byte-for-byte. On a SPLIT aperture the device dense side is DESTINED for the device window; it
// merely TRANSITS host RAM while staging (read -> transcode -> upload -> free, the fak#13171
// lesson), so it does not permanently occupy the host window the resident-expert set is judged
// against. Subtracting the whole transit there collapses remaining = avail - transit to a negative
// number and forces stream-through (bound 0), which is not a genuine capacity floor: the witnessed
// strix3 refusal (63.09 GiB dense transit vs 48.57 GiB host avail -> bound 0 -> dense-only
// FitTooBig) is exactly that self-referential collapse, not a wall.
func serveStreamedCPUOffloadPlanForAperture(ws *ggufload.WeightSource, be compute.Backend, ranks, contextBudgetTokens int, fit serveFitBudget, sharedPool, splitAperture bool) (compute.MemoryPlan, bool, error) {
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, ranks, contextBudgetTokens, fit)
	if err != nil {
		return nil, false, err
	}
	if ws == nil || fit.Base <= 0 {
		return plan, false, nil
	}
	// Discrete: judge the HOST-scoped subset (the device dense side is independent VRAM and must not
	// be double-counted against host RAM). Shared pool: judge the GRAND total, because the device
	// dense side consumes the same physical DRAM the host expert set is bound against.
	if sharedPool {
		if plan.Total() <= fit.avail() {
			return plan, false, nil
		}
	} else if plan.HostTotal() <= fit.avail() {
		return plan, false, nil
	}
	if !serveStreamedExpertsCapable(ws) {
		return plan, false, nil
	}
	bound := serveCPUOffloadStreamedResidentBoundForAperture(fit, plan.DeviceTotal(), sharedPool, splitAperture)
	// fak#13215: when the device --cpu-offload-experts arm declares a bounded streamed-dense working
	// set (serveCPUOffloadBoundedDenseOptions, gated on the SAME FAK_STREAM_Q4K knob), the streamed
	// plan must fold BOTH bounds -- otherwise the historical EstimateCPUOffloadExpertsStreamedMemoryPlan
	// (denseResident = -1) charges the whole 63.22 GiB V4.1 device dense transit and the staging guard
	// refuses FitTooBig against the host window. The estimator is the ONE combined kernel; with no
	// declared bounded dense option (the default, every non-streamed serve) this is byte-for-byte
	// EstimateCPUOffloadExpertsStreamedMemoryPlan.
	var streamed compute.MemoryPlan
	if denseBound, ok := serveBoundedDenseWorkingSetBound(serveCPUOffloadBoundedDenseOptions(be, fit)); ok {
		// fak#13249: the bounded dense working set must be sized from the budget REMAINING after the
		// resident expert bound, not from the whole host budget. Deriving both independently at
		// (1-margin)*avail admitted their SUM (up to ~1.8*avail) and the kernel OOM-killed the serve
		// mid-staging -- the [HW-WITNESSED] strix3 run. Min keeps a declared bound from inflating past
		// the remainder the expert fold left.
		if combined := serveCombinedStreamedDenseResidentBound(fit, bound); combined < denseBound {
			denseBound = combined
		}
		streamed, err = ws.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(ranks, bound, denseBound)
	} else {
		streamed, err = ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(bound)
	}
	if err != nil {
		return nil, false, err
	}
	return appendServeGGUFDevicePlan(ws, be, streamed, contextBudgetTokens, fit), true, nil
}

// serveCPUOffloadStreamedResidentBoundForPool is the ONE derivation of the bounded host-resident
// expert working set, keyed on the pool topology so the load arm and the sizing path share it.
// On a DISCRETE device (sharedPool=false) the device dense side is independent VRAM, so the bound is
// the whole headroom-adjusted host budget less the resident margin (serveCPUOffloadStreamedResidentBound,
// the #13121/#13140 behaviour, unchanged).
// On a SHARED pool (sharedPool=true, an integrated/APU tier) the device-scoped dense charge and the
// host-resident expert set draw from ONE physical DRAM pool, so the bound is sized from the budget
// that REMAINS after the device dense transit -- otherwise the streamed plan re-charges the whole
// pool (device transit + full resident bound) against the same budget and ties/overruns it.
// deviceTransit is the resident plan's device-scoped dense total (0 for the host-only arm, where the
// historical whole-avail bound is kept).
func serveCPUOffloadStreamedResidentBoundForPool(fit serveFitBudget, deviceTransit int64, sharedPool bool) int64 {
	if !sharedPool {
		return serveCPUOffloadStreamedResidentBound(fit)
	}
	return serveCPUOffloadSharedPoolResidentBound(fit, deviceTransit)
}

// serveCPUOffloadStreamedResidentBoundForAperture is the ONE derivation of the bounded host-resident
// expert working set with the device APERTURE made explicit. It preserves
// serveCPUOffloadStreamedResidentBoundForPool byte-for-byte EXCEPT on a shared pool whose device
// side has its OWN aperture distinct from the host window (splitAperture=true): there the device
// dense transit is a TRANSIENT host staging cost (fak#13171), not a permanent co-resident claim on
// the host window, so it is NOT subtracted from the resident-expert budget. Subtracting it would
// make (avail - transit) negative on a box whose dense side is larger than the host window
// (strix3: 63.09 GiB transit vs 48.57 GiB host avail) and force a stream-through bound of zero --
// a self-referential refusal masquerading as a capacity floor, which is the witnessed #13175 rung.
func serveCPUOffloadStreamedResidentBoundForAperture(fit serveFitBudget, deviceTransit int64, sharedPool, splitAperture bool) int64 {
	if sharedPool && splitAperture {
		return serveCPUOffloadStreamedResidentBound(fit)
	}
	return serveCPUOffloadStreamedResidentBoundForPool(fit, deviceTransit, sharedPool)
}

// serveCPUOffloadSharedPoolResidentBound is the bounded host-resident expert working set for a
// SHARED-pool (integrated/APU) tier, where the device-scoped dense charge and the host-resident
// expert set draw from ONE physical DRAM pool. It sizes that set from the SAME host budget the
// bound is judged against (fit.avail()) MINUS the device-scoped dense bytes that must transit that
// pool first, then applies the fixed resident margin so the streamed plan's host total lands
// STRICTLY below the budget that judges it (the #13140 property, preserved). A device-scoped
// transit already at or above the budget yields zero (stream-through) -- the honest floor, because a
// pool with no room left cannot be promised residency. An unprobeable host also yields zero.
func serveCPUOffloadSharedPoolResidentBound(fit serveFitBudget, deviceTransit int64) int64 {
	avail := fit.avail()
	if avail <= 0 {
		return 0
	}
	remaining := avail - deviceTransit
	if remaining <= 0 {
		return 0
	}
	bound := int64(float64(remaining) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= remaining {
		// A razor-thin remainder must still land strictly below it, and never become negative.
		bound = remaining - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}

// serveCombinedStreamedDenseResidentBound is the bounded host-resident DENSE working set for the
// COMBINED streamed-expert + bounded-dense arm (fak#13209/#13215), sized from the budget that
// REMAINS after the resident EXPERT working set instead of from the whole host budget.
//
// Both working sets are simultaneously resident in host RAM: the streamed routed-expert set
// (gguf-host-expert-offload-streamed) and the bounded dense staging working set
// (gguf-host-dense-streamed). Deriving each independently as (1-margin)*avail admits their SUM at up
// to ~1.8*avail -- the [HW-WITNESSED] strix3 OOM: MemTotal 62.4 GiB, host budget 48.65 GiB, expert
// bound 43.788 GiB AND dense bound 43.788 GiB in one serve, kernel-OOM-killed at 57.8G peak / 28.1G
// swap before the forward. Sizing the dense bound from the post-expert remainder keeps the plan's
// host total strictly below the budget that judges it (the #13140 property, now over the SUM).
//
// expertBound is the SAME serveCPUOffloadStreamedResidentBoundForAperture value the streamed fold
// threads (one measurement, so the two cannot disagree). A non-positive remainder yields zero
// (stream-through) -- the honest floor, never negative.
func serveCombinedStreamedDenseResidentBound(fit serveFitBudget, expertBound int64) int64 {
	avail := fit.avail()
	if avail <= 0 {
		return 0
	}
	if expertBound < 0 {
		expertBound = 0
	}
	remaining := avail - expertBound
	if remaining <= 0 {
		return 0
	}
	bound := int64(float64(remaining) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= remaining {
		// A razor-thin remainder must still land strictly below it, and never become negative.
		bound = remaining - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}

// serveStreamedCPUOffloadPathDecision is the path-form of serveStreamedCPUOffloadPlan: the LOAD
// arm's WithStreamedExperts threading and the SIZING path's plan are decided by ONE measurement
// over one opened checkpoint, so they cannot disagree. An unopenable path, a non-MoE artifact, or a
// routed set that already fits the host budget all return (false, 0, nil) -- the resident arm.
// sharedPool is threaded straight to serveStreamedCPUOffloadPlanForPool so the load arm selects
// streaming on the GRAND total on an integrated/APU tier (fak#13171/#13172 follow-on).
func serveStreamedCPUOffloadPathDecision(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int, fit serveFitBudget, sharedPool bool) (bool, int64, error) {
	streamed := false
	bound := int64(0)
	splitAperture := serveSplitAperture(be)
	_, err := withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		plan, isStreamed, perr := serveStreamedCPUOffloadPlanForAperture(ws, be, ranks, contextBudgetTokens, fit, sharedPool, splitAperture)
		if perr != nil {
			return nil, perr
		}
		if isStreamed {
			// Derive the bound from the SAME resident plan the sizing path used, so the load arm's
			// WithStreamedExperts working set and the sizing plan cannot disagree. On a shared pool the
			// device dense transit has already consumed part of the pool, exactly as the plan charged it.
			streamed, bound = true, serveCPUOffloadStreamedResidentBoundForAperture(fit, plan.DeviceTotal(), sharedPool, splitAperture)
		}
		return nil, nil
	})
	if err != nil {
		return false, 0, err
	}
	return streamed, bound, nil
}

// fitServeStreamedCPUOffloadPathOnHost is the device-less counterpart of
// fitServeStreamedCPUOffloadPathOnDevice. It judges the bounded-resident streamed plan against the
// host budget that SELECTED it, so the derivation and the admission share one snapshot.
func fitServeStreamedCPUOffloadPathOnHost(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget) error {
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return err
	}
	defer ws.Close()
	plan, _, err := serveStreamedCPUOffloadPlanForPool(ws, nil, ranks, contextBudgetTokens, fit, false)
	if err != nil {
		return err
	}
	return refuseHostPlanAgainstFit(plan, fit)
}

// fitServeStreamedCPUOffloadPathOnDevice is fitAndPlanServeGGUFCPUOffloadPathOnDevice under the
// bounded-resident streamed policy: when the full routed charge does not fit the host budget and the
// tier can stage the slabs, judge the DEVICE side against the streamed plan (whose host-scoped
// demand is the bounded resident set) and return that plan. Otherwise it falls back to the resident
// plan unchanged, so a fitting artifact keeps the historical arm.
//
// hostFit is the SAME host-fit snapshot serveStreamedCPUOffloadPathDecision used to SELECT this
// arm (the load path's one measurement), threaded in rather than re-probed, so the streamed
// decision and the sizing plan cannot disagree about what is host-resident. override remains the
// DEVICE fit override: device admission stays serveDeviceFitBudgetFromReported(be, override).
func fitServeStreamedCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int, hostFit serveFitBudget, override *serveFitBudget) (compute.MemoryPlan, bool, error) {
	fit := hostFit
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return nil, false, err
	}
	defer ws.Close()
	plan, streamed, err := serveStreamedCPUOffloadPlanForAperture(ws, be, ranks, contextBudgetTokens, fit, ggufload.BackendSharesHostRAM(be), serveSplitAperture(be))
	if err != nil {
		return nil, false, err
	}
	dev := serveDeviceFitBudgetFromReported(be, override)
	admitted, rerr := refuseIfTooBigOnDevice(plan, nil, be, &dev)
	return admitted, streamed, rerr
}
