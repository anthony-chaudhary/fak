package main

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveKVPrecision is the realized KV storage tier the fit estimator charges, read
// from FAK_UP_KV_PRECISION (the same seam the serve --kv-precision flag publishes to).
// Unset or "f32" yields the exact F32 tier; a q8 token yields the denser mixed tier so
// the admission math matches the engine's residency. Unknown tokens fall back to F32
// (fail-open) Ã¢â‚¬â€ an invalid flag is refused at serve-flag validation, not here.
func serveKVPrecision() compute.KVPrecision {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_UP_KV_PRECISION"))) {
	case "q8", "q8_0", "8":
		return compute.KVPrecisionQ8
	default:
		return compute.KVPrecisionF32
	}
}

const serveGGUFDeviceHeadroom = 0.15

// refuseEPPlanIfUnfit fails the serve closed when `--expert-parallel N>1` cannot fit the model
// resident across N GPUs. It partitions the loaded model's resident weights into the replicated
// remainder and the routed experts (model.MoEResidentWeightBytes), builds the BUSIEST rank's
// per-card plan (compute.ExpertParallelPerRankPlan: replicated + largest expert band), and checks
// it against the device backend's PER-GPU capacity with the same headroom the load-time fit uses.
//
// It is FAIL-OPEN by construction (the contract every capacity check here keeps): a non-MoE model,
// a model whose weights cannot be accounted, ranks<=1, a nil backend, or a backend whose capacity is
// unknown (cpu-ref, a non-probing device) all return nil. So it can ONLY turn a KNOWN per-card
// overflow (e.g. a 434 GiB model at N=4 Ã¢â€°Ë† 118 GiB/card on 80 GiB GPUs) into a clean pre-serve
// refusal Ã¢â‚¬â€ instead of an OOM that surfaces minutes in, when rank r uploads its expert band to GPU r.
func refuseEPPlanIfUnfit(m *fakmodel.Model, be compute.Backend, ranks, contextBudgetTokens int) error {
	if m == nil || be == nil || ranks <= 1 {
		return nil
	}
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok {
		return nil // nothing accounted (non-MoE / unloaded) -> fail open
	}
	// KV is a per-rank cost: pure EP replicates attention, so each rank holds the full KV for the
	// context it serves. Size it from the model geometry at the context budget Ã¢â‚¬â€ the SAME KV the
	// load-time fit plan sizes from contextBudgetTokens Ã¢â‚¬â€ so the per-card check is weights + KV, not
	// weights alone (matching the established serve fit pattern). 0 budget leaves a weights-only plan.
	var extra compute.MemoryPlan
	if contextBudgetTokens > 0 {
		extra = compute.EstimateKVStoreMemoryPlan(compute.KVConfig{
			NumLayers:  m.Cfg.NumLayers,
			NumKVHeads: m.Cfg.NumKVHeads,
			HeadDim:    m.Cfg.HeadDim,
			RopeTheta:  m.Cfg.RopeTheta,
			Precision:  serveKVPrecision(),
		}, contextBudgetTokens)
	}
	plan := compute.ExpertParallelPerRankPlan(replicated, expert, m.Cfg.NumExperts, ranks, extra)
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

// serveGGUFHostHeadroom reserves a fraction of the process host's allocatable RAM (MemAvailable)
// for the pure-CPU reference serve path's costs NOT in the header estimate: the resident-Q4K
// struct overshoot over the raw-payload estimate (~458 GiB resident vs ~433 GiB on-disk on
// GLM-5.2 UD-Q4_K_M, #974), gateway and KV init, and MemAvailable jitter as clean page cache is
// evicted during the multi-minute load. Matched to serveGGUFDeviceHeadroom for parity with the
// device fit plan, and comfortably above the observed ~6% resident overshoot.
const serveGGUFHostHeadroom = 0.15

// serveFitBudget is the memory ceiling the #1046 context auto-sizer derives the largest fitting
// context against: the raw budget base (a backend's device free-or-total, or the host's
// MemAvailable) and the headroom fraction the matching load-time fit check reserves. A
// non-positive Base means the ceiling is unprobeable (the cpu-ref floor, a device that cannot
// report capacity) Ã¢â‚¬â€ avail() then yields FreeUnknown and the auto-sizer falls open to the model's
// full declared window, exactly as before #1046.
type serveFitBudget struct {
	Base     int64
	Headroom float64
}

type serveNativeContextResolution struct {
	RequestedTokens     int    `json:"requested_tokens"`
	ModelDeclaredTokens int    `json:"model_declared_tokens"`
	ResolvedTokens      int    `json:"resolved_tokens"`
	Source              string `json:"source"`
}

func validateServeNativeContextTokens(tokens int) error {
	if tokens < 0 {
		return fmt.Errorf("--native-context-tokens must be 0 (auto) or positive (got %d)", tokens)
	}
	return nil
}

// resolveServeNativeContext is the header-only authority for the native model
// window. weights and fit are the selected load arm's exact sizing inputs, so
// auto mode returns the same token count later used to build the load plan.
func resolveServeNativeContext(ws *ggufload.WeightSource, weights compute.MemoryPlan, fit serveFitBudget, requested int) (serveNativeContextResolution, compute.MemoryPlan, error) {
	resolution := serveNativeContextResolution{RequestedTokens: requested, Source: "auto"}
	if err := validateServeNativeContextTokens(requested); err != nil {
		return resolution, nil, err
	}
	if requested > 0 {
		resolution.Source = "explicit"
		resolution.ResolvedTokens = requested
	}
	if ws == nil {
		return resolution, nil, nil
	}
	cfg, err := ws.File.Config()
	if err != nil {
		return resolution, nil, err
	}
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	resolution.ModelDeclaredTokens = csc.MaxContext
	if requested > 0 && csc.MaxContext > 0 && requested > csc.MaxContext {
		return resolution, nil, fmt.Errorf("--native-context-tokens %d exceeds model-declared context window %d", requested, csc.MaxContext)
	}
	override := -1
	if requested > 0 {
		override = requested
	}
	resolution.ResolvedTokens, _ = compute.AutoSizeContextPlan(csc, weights, fit.avail(), override)
	return resolution, csc.PerContextMemoryPlan(resolution.ResolvedTokens), nil
}

// avail is the headroom-adjusted budget passed to compute.AutoSizeContextPlan Ã¢â‚¬â€ byte-identical to
// the budget the matching RefuseMemoryPlanIfTooBig* check computes (same compute.BudgetAfterHeadroom
// formula), so a context derived against it provably passes that check. An unknown base yields
// FreeUnknown so the sizer fails open to the full window.
func (b serveFitBudget) avail() int64 {
	if b.Base <= 0 {
		return compute.FreeUnknown
	}
	return compute.BudgetAfterHeadroom(b.Base, b.Headroom)
}

// serveDeviceFitBudget reads the device memory ceiling a device serve arm's fit check uses
// (DeviceCeilingInfo: the STABLE device-local heap capacity, with headroom reserved against
// THAT). A device-destined plan must be sized against the capacity it lands in, not the volatile
// VK_EXT_memory_budget headroom: the live budget swings with VRAM pressure, so the same 63.22 GiB
// dense transit is refused at one reading and admitted - then kernel-OOM-killed mid-staging - at
// another (fak#13186). When the backend cannot report a stable ceiling (cpu-ref, a non-probing
// device, nil) this falls back byte-for-byte to serveFitBudgetBase(total, free, known) on the
// volatile DeviceMemoryInfo reading. Unknown capacity -> a zero base -> the auto-sizer keeps the
// full window.
func serveDeviceFitBudget(be compute.Backend) serveFitBudget {
	if ceiling, ok := compute.DeviceCeilingInfo(be); ok && ceiling > 0 {
		return serveFitBudget{Base: ceiling, Headroom: serveGGUFDeviceHeadroom}
	}
	total, free, known := compute.DeviceMemoryInfo(be)
	return serveFitBudget{Base: serveFitBudgetBase(total, free, known), Headroom: serveGGUFDeviceHeadroom}
}

// serveSplitAperture reports whether a device backend has its OWN aperture DISTINCT from the
// process host window it is judged against -- a split VRAM/system carve-out (strix3: an 84.28 GiB
// device-local heap beside a 62.4 GiB MemTotal). It composes with the ESTABLISHED split-aperture
// notion in ggufload.RefuseUnifiedHostResidencyIfTooBigForReportedAperture: the two windows are
// split when both are KNOWN and their totals DIFFER (a genuinely unified heap reports one window
// under both names). It deliberately reuses that test rather than inventing a ratio so the serve
// sizing path and the unified-host-residency admission cannot disagree about what "split" means.
// A backend that reports no device capacity, or a host that reports no total, is NOT split -- fail
// open, exactly as every other capacity rung here.
func serveSplitAperture(be compute.Backend) bool {
	devTotal, _, devKnown := compute.DeviceMemoryInfo(be)
	if !devKnown || devTotal <= 0 {
		return false
	}
	hostTotal, _, hostKnown := compute.HostSystemMemoryInfo()
	if !hostKnown || hostTotal <= 0 {
		return false
	}
	return devTotal != hostTotal
}

// serveHostFitBudget reads the process host's allocatable RAM the pure-CPU serve arm's fit check
// uses (HostSystemMemoryInfo// serveHostFitBudget reads the process host's allocatable RAM the pure-CPU serve arm's fit check
// uses (HostSystemMemoryInfo Ã¢â€ â€™ Linux MemAvailable). Unknown Ã¢â€ â€™ a zero base Ã¢â€ â€™ the full window.
func serveHostFitBudget() serveFitBudget {
	total, free, known := compute.HostSystemMemoryInfo()
	return serveFitBudget{Base: serveFitBudgetBase(total, free, known), Headroom: serveGGUFHostHeadroom}
}

// serveFitBudgetBase collapses a (total, free, known) capacity report into the raw budget base
// the fit check would size against: free when known, the total ceiling when free is unprobeable
// (parity with fitsWithinReportedMemory), and 0 when capacity is unknown.
func serveFitBudgetBase(total, free int64, known bool) int64 {
	if !known || total <= 0 {
		return 0
	}
	if free < 0 { // FreeUnknown -> the total ceiling, conservatively
		return total
	}
	return free
}

// serveHostFitBudgetFromReported pins host sizing and host admission to ONE measured snapshot.
// An injected override (the resolve pass's snapshot, threaded through the load) wins verbatim, so
// the two stages cannot disagree; otherwise the caller's (total, free, known) is used instead of a
// second live HostSystemMemoryInfo probe. A zero Base stays "unprobeable" and fails open upstream.
func serveHostFitBudgetFromReported(total, free int64, known bool, override *serveFitBudget) serveFitBudget {
	if override != nil {
		return *override
	}
	return serveFitBudget{Base: serveFitBudgetBase(total, free, known), Headroom: serveGGUFHostHeadroom}
}

// serveDeviceFitBudgetFromReported is the device counterpart: an injected override wins, else the
// live device probe. It exists so the device arm's sizing and admission share one snapshot.
func serveDeviceFitBudgetFromReported(be compute.Backend, override *serveFitBudget) serveFitBudget {
	if override != nil {
		return *override
	}
	return serveDeviceFitBudget(be)
}

// refuseHostPlanAgainstFit judges a pure-CPU host plan against the SAME measured budget it was
// sized against. Base<=0 means the ceiling was unprobeable -> fail open, exactly as a live probe.
func refuseHostPlanAgainstFit(plan compute.MemoryPlan, fit serveFitBudget) error {
	if fit.Base <= 0 {
		return nil
	}
	return compute.RefuseMemoryPlanIfTooBigForReportedHost(plan, fit.Base, fit.Base, true, fit.Headroom)
}

// refuseDevicePlanAgainstFit is the device arm's counterpart of refuseHostPlanAgainstFit. Base<=0
// leaves device capacity unknown. Host-scoped demands still consult the reported-device refusal.
func refuseDevicePlanAgainstFit(be compute.Backend, plan compute.MemoryPlan, fit serveFitBudget) (compute.MemoryPlan, error) {
	return plan, compute.RefuseMemoryPlanIfTooBigForReportedDevice(be, plan, fit.Base, fit.Base, fit.Base > 0, fit.Headroom)
}

// serveDeviceStagingHostCharge is the TRANSIENT host-resident charge a device serve materializes
// while STAGING its device-scoped weights (fak#13171). A device-scoped weight is not born in VRAM:
// the loader reads its bytes into host RAM, dequantizes/transcodes, uploads it to the device, and
// frees the host copy Ã¢â‚¬â€ so the device-scoped dense total is ALSO a transient host demand, on top of
// the already-host-scoped routed-expert pool. RefuseHostScopedPlanIfTooBigForHost judges ONLY
// plan.HostTotal(), so a 63.09 GiB device dense charge sails past it and the process is SIGKILLed
// by the Linux OOM-killer mid-staging (the witnessed strix3 kill: weights=63.092GiB against
// MemTotal 62.4 GiB, anon-rss 37.1 GiB before the gateway bound :8084). This is the charge the
// guard below bounds against real host RAM.
func serveDeviceStagingHostCharge(plan compute.MemoryPlan) int64 {
	return plan.DeviceTotal()
}

// serveDenseBoundedStreamStagingDetail is the memory-plan row Detail the bounded streamed-dense
// loader attaches to a dense working-set row that the loader RETAINS as a bounded host working set
// rather than materializing whole (ggufload.EstimateQ4KLoadMemoryPlan's streamed fold). When the
// device --cpu-offload-experts arm threads the bounded streamed-dense route (fak#13209), the
// eligible dense k-quant staging transit is bounded to that resident working set, so the staging
// charge must reflect it instead of the full plan.DeviceTotal().
const serveDenseBoundedStreamStagingDetail = "gguf-host-dense-streamed"

// serveDenseStreamedStagingChargeDetail is the row Detail the staging guard stamps on the bounded
// transit row, so a physical receipt / test can tell the bounded charge from the whole-device-total
// charge (fak#13209).
const serveDenseStreamedStagingChargeDetail = "gguf-device-staging-host-transit:bounded-streamed-dense"

// serveBoundableDeviceDenseTotal is the device-scoped dense charge the bounded streamed-dense route
// could actually cover: the plan's device-scoped DENSE weight total (Class MemoryWeights). It is the
// upper bound (before the working-set cap) of what the streamed route bounds, and the bulk of the
// staging transit -- the moat the #13171 guard exists to bound.
//
// KV/scratch and any non-Weight class are EXCLUDED, because the streamed-dense route only covers the
// eligible dense k-quant matmul weights the loader turns into lazy checkpoint range descriptors; the
// runtime/KV/scratch device demands are materialized separately and are not what the dense working
// set bounds. Classifying by Class (not by Detail) keeps this robust to Detail churn: a plan that
// scopes device dense weights under any Detail still contributes its Weight bytes here.
func serveBoundableDeviceDenseTotal(plan compute.MemoryPlan) int64 {
	var total int64
	const maxInt64 = int64(^uint64(0) >> 1)
	for _, d := range plan {
		if d.Bytes <= 0 || d.Class != compute.MemoryWeights || !d.DeviceScoped() {
			continue
		}
		if total > maxInt64-d.Bytes {
			return maxInt64
		}
		total += d.Bytes
	}
	return total
}

// serveDeviceDenseStreamedBounded reports whether the plan carries a bounded streamed-dense
// working-set row -- i.e. the loader that produced it (or will produce it) retains the eligible
// dense k-quant side as a bounded host working set (ggufload.WithStreamedDenseQ4KWorkingSet)
// instead of materializing the full device dense side during staging (fak#13209). The row is the
// one ggufload's streamed fold appends; its presence is the observable evidence that the bounded
// route is active.
func serveDeviceDenseStreamedBounded(plan compute.MemoryPlan) bool {
	for _, d := range plan {
		if d.Detail == serveDenseBoundedStreamStagingDetail && d.Bytes >= 0 {
			return true
		}
	}
	return false
}

// serveDeviceDenseStagingBoundCharge is the BOUNDED dense staging transit (fak#13209): the eligible
// device dense side (the only device-scoped DENSE weight charge the streamed route covers) charged
// at the DECLARED host working set rather than the full on-disk dense total. It returns
// (bounded, ok):
//
//   - ok=false when the plan does NOT show the bounded streamed-dense route (no working-set row) --
//     the caller keeps the historical whole DeviceTotal() charge byte-for-byte, preserving the
//     fak#13171 kernel-OOM protection for the non-streamed shape;
//   - ok=true when the route is active: the charge is min(boundableDeviceDense, declaredWorkingSet),
//     so the guard judges the bounded transit. Every other device-scoped demand that is NOT the
//     bounded dense side (KV/scratch, and any weight class the streamed route does not cover) is
//     ADDED back whole, so the fail-closed floor is preserved: a plan whose non-streamed device
//     remainder cannot be staged still refuses typed.
//
// bound <= 0 (stream-through) yields a zero dense term: the loader retains no dense working set, so
// the dense transit does not consume host RAM. That is the honest floor, not a silent deletion of
// the guard -- the non-dense device remainder is still charged.
func serveDeviceDenseStagingBoundCharge(plan compute.MemoryPlan) (int64, bool) {
	if !serveDeviceDenseStreamedBounded(plan) {
		return 0, false
	}
	if plan.DeviceTotal() <= 0 {
		return 0, true
	}
	bound := serveStreamedDenseQ4KWorkingSetBound(serveFitBudget{})
	// Derive the declared working set from the plan's OWN working-set row, so the charge and the
	// loader's retained set cannot disagree about the bound (one measurement).
	for _, d := range plan {
		if d.Detail == serveDenseBoundedStreamStagingDetail {
			bound = d.Bytes
			break
		}
	}
	dense := serveBoundableDeviceDenseTotal(plan)
	resident := dense
	if bound < resident {
		resident = bound
	}
	if resident < 0 {
		resident = 0
	}
	nonDense := plan.DeviceTotal() - dense
	if nonDense < 0 {
		nonDense = 0
	}
	return resident + nonDense, true
}

// serveDeviceStagingHostPlan is the plan the staging guard judges: ONE host-scoped row carrying the
// device-scoped staging transit, so the established reported-host refusal measures it against host
// RAM without re-charging the resident expert pool (compute.RefuseHostScopedPlanIfTooBigForHost
// already judges those host-scoped rows on the same arm). An empty plan yields nil (nothing to
// stage -> the guard is inert).
//
// fak#13209: when the plan carries the bounded streamed-dense working-set row, the transit row is
// the BOUNDED charge (serveDeviceDenseStagingBoundCharge) rather than the whole plan.DeviceTotal():
// the device-destined eligible dense side is faulted as bounded checkpoint ranges and uploaded, so
// it is never materialized as one ~63 GiB host anon buffer. The fail-closed floor is preserved: the
// non-streamed device remainder is still charged whole, so a transit that cannot be placed even in
// bounded form still refuses typed.
func serveDeviceStagingHostPlan(plan compute.MemoryPlan) compute.MemoryPlan {
	staging := serveDeviceStagingHostCharge(plan)
	detail := "gguf-device-staging-host-transit"
	if bounded, ok := serveDeviceDenseStagingBoundCharge(plan); ok {
		staging = bounded
		detail = serveDenseStreamedStagingChargeDetail
	}
	if staging <= 0 {
		return nil
	}
	return compute.MemoryPlan{{
		Class:  compute.MemoryScratchpad,
		Scope:  compute.MemoryScopeHost,
		Bytes:  staging,
		Detail: detail,
	}}
}

// refuseDeviceStagingAgainstHostFit judges the device cpu-offload arm's transient host staging
// charge (fak#13171) against the SAME host budget snapshot the streamed arm decision was sized from
// (serveStreamedHostFit). Base<=0 means the host ceiling was unprobeable -> fail open, exactly as
// every other capacity rung here. On overflow it returns the typed *compute.FitError the reported-
// host refusal builds, which NAMES the demand and the shortfall instead of letting the kernel OOM
// decide. A device serve whose device dense staging fits is unchanged.
//
// This is the SINGLE-WINDOW form: the device staging transit is judged against the host window
// alone. Callers that hold a distinct device aperture (a split VRAM/system box) use
// refuseDeviceStagingAgainstReportedAperture, which judges the transit against the DEVICE window
// it is destined for (fak#13177) while preserving this single-window judgement when no distinct
// device window is known.
func refuseDeviceStagingAgainstHostFit(plan compute.MemoryPlan, fit serveFitBudget) error {
	if fit.Base <= 0 {
		return nil
	}
	staging := serveDeviceStagingHostPlan(plan)
	return compute.RefuseMemoryPlanIfTooBigForReportedHost(staging, fit.Base, fit.Base, true, fit.Headroom)
}

// refuseDeviceStagingAgainstReportedAperture is the SPLIT-AWARE form of the staging guard
// (fak#13177). On a split-aperture integrated box (a Strix Halo with a 64 GiB VRAM carve-out) the
// device-scoped dense side is DESTINED for the VRAM window: it transits host RAM only transiently
// during staging, but its residency is the device aperture. Judging it against the system window
// alone (62.43 GiB MemTotal) refuses a 63.22 GiB dense side that comfortably fits the 64 GiB VRAM
// window - the exact [HW-WITNESSED] strix3 refusal that blocked the first physical V4.1 token.
//
// When a device window is known AND distinct from the host window, the transit's RESIDENCY is
// judged against that device window; otherwise this delegates byte-for-byte to the single-window
// refuseDeviceStagingAgainstHostFit, so the fak#13171 kernel-OOM protection is preserved exactly
// (an unknown or coincident device window keeps today's host-RAM judgement). The genuine
// host-scoped expert pool is judged by the pre-existing host guard on the same arm, not here.
//
// fak#13186 (the rung this leaf closes): the device-window admission alone is NOT sufficient,
// because the transit is not born in VRAM - the loader reads each device-destined tensor into HOST
// RAM, transcodes, uploads, and frees, so the device dense total is ALSO a transient host
// allocation. The [HW-WITNESSED] strix3 rung is exactly this: the 63.22 GiB device transit fits the
// 64 GiB VRAM window (so the split form ADMITTED it) yet is materialized as a ~63 GiB host anon
// buffer against a ~48.5 GiB host budget, and the kernel OOM-killer kills the serve via
// filemap_fault before any device transfer (`staging-host-charge=63.223GiB host-budget=48.542GiB`
// reported, then proceeded). So the split form now requires BOTH windows: the device window for
// RESIDENCY and the host window for the staging TRANSIT. When the transit does not fit the host
// window it returns the typed host-scoped *compute.FitError naming the shortfall - the honest
// fail-closed replacement for the kernel OOM - while a transit that fits both windows is admitted
// unchanged. A genuinely bounded/streamed host->device staging path that keeps peak host anon-RSS
// under MemAvailable would supersede this check; until it is threaded, refusing is correct.
func refuseDeviceStagingAgainstReportedAperture(plan compute.MemoryPlan, fit serveFitBudget, deviceTotal, deviceFree int64, deviceKnown bool) error {
	if fit.Base <= 0 {
		return nil
	}
	staging := serveDeviceStagingHostPlan(plan)
	if staging == nil {
		return nil
	}
	if !deviceKnown || deviceTotal <= 0 || deviceTotal == fit.Base {
		return refuseDeviceStagingAgainstHostFit(plan, fit)
	}
	// The transit is DEVICE-destined, so its RESIDENCY must be judged as a device-scoped demand:
	// the reported device refusal sums plan.DeviceTotal(), and a host-scoped row would make that
	// sum zero and admit any transit. Re-scope the synthesized row to the device aperture for the
	// residency check.
	deviceStaging := make(compute.MemoryPlan, len(staging))
	copy(deviceStaging, staging)
	for i := range deviceStaging {
		deviceStaging[i].Scope = compute.MemoryScopeDevice
	}
	if err := compute.RefuseMemoryPlanIfTooBigForReportedDevice(nil, deviceStaging, deviceTotal, deviceFree, true, fit.Headroom); err != nil {
		return err
	}
	// fak#13186: the residency fits the device window, but the staging TRANSIT still materializes
	// in host RAM, so it must ALSO fit the host window. Reuse the single-window host judgement on
	// the original host-scoped transit row: a device-destined transit that cannot be staged through
	// host RAM is refused typed here instead of being kernel-OOM-killed mid-staging.
	return refuseDeviceStagingAgainstHostFit(plan, fit)
}

// logServeDeviceCPUOffloadArmStaging is the ONE pre-staging startup line the device
// --cpu-offload-experts arm emits BEFORE loadResidentQ4KDevice (fak#13171). The prior physical
// receipt had to INFER streamed=false from an ABSENT "serving-expert-residency" message: that
// message rides loadMessages, which the gateway prints only AFTER the load, so a serve SIGKILLed
// during staging never emitted it and the arm that actually ran was invisible. This prints to
// stderr at STAGING TIME (the logServeAutoSizedContext convention) and names the arm decision
// (streamed vs resident), the bounded/charged host bytes, and the host budget, so the next
// physical run witnesses which arm was taken without inference.
func logServeDeviceCPUOffloadArmStaging(backend compute.Backend, streamed bool, streamedBound int64, plan compute.MemoryPlan, fit serveFitBudget) {
	arm := "resident"
	charged := serveDeviceStagingHostCharge(plan) + plan.HostTotal()
	if streamed {
		arm = fmt.Sprintf("streamed(resident-bound=%s)", bytesText(uint64(max(streamedBound, 0))))
		charged = int64(max(streamedBound, 0)) + plan.DeviceTotal()
	}
	backendName := "none"
	if backend != nil {
		backendName = backend.Name()
	}
	fmt.Fprintf(os.Stderr,
		"fak: device --cpu-offload-experts arm=%s backend=%s staging-host-charge=%s host-budget=%s (bounded before staging; device dense bytes transit host RAM, fak#13171)\n",
		arm, backendName, bytesText(uint64(max(charged, 0))), bytesText(uint64(max(fit.avail(), 0))))
}

type deviceWeightBudgetBackend interface {
	DeviceWeightBudget() (bytes int64, enabled bool)
}

// applyDeviceWeightBudget splits immutable weight demand across the explicit
// device-local cap and host-visible Vulkan storage. Runtime/KV/scratch demands
// remain device-scoped. This is planning only: the backend's allocator is the
// source of truth for each actual placement.
func applyDeviceWeightBudget(plan compute.MemoryPlan, be compute.Backend) compute.MemoryPlan {
	budgeter, ok := be.(deviceWeightBudgetBackend)
	if !ok {
		return plan
	}
	budget, enabled := budgeter.DeviceWeightBudget()
	if !enabled || budget <= 0 {
		return plan
	}
	remaining := budget
	out := make(compute.MemoryPlan, 0, len(plan)+1)
	for _, demand := range plan {
		if demand.Class != compute.MemoryWeights || !demand.DeviceScoped() || demand.Bytes <= 0 {
			out = append(out, demand)
			continue
		}
		deviceBytes := demand.Bytes
		if deviceBytes > remaining {
			deviceBytes = remaining
		}
		if deviceBytes > 0 {
			device := demand
			device.Bytes = deviceBytes
			device.Scope = compute.MemoryScopeDevice
			device.Detail += ":device-local-budget"
			out = append(out, device)
			remaining -= deviceBytes
		}
		if spill := demand.Bytes - deviceBytes; spill > 0 {
			host := demand
			host.Bytes = spill
			host.Scope = compute.MemoryScopeHost
			host.Detail += ":host-visible-offload"
			out = append(out, host)
		}
	}
	return out
}
func fitServeGGUFOnDevice(ws *ggufload.WeightSource, be compute.Backend, f32Resident bool, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens, serveDeviceFitBudget(be))
	if err != nil {
		return err
	}
	plan = applyDeviceWeightBudget(plan, be)
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func fitServeGGUFCPUOffloadOnDevice(ws *ggufload.WeightSource, be compute.Backend, ranks, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, ranks, contextBudgetTokens, serveDeviceFitBudget(be))
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func resolveHostServeLoadArm(ws *ggufload.WeightSource, f32Resident, cpuOffloadExperts bool) serveLoadArm {
	if f32Resident {
		return serveLoadArmF32
	}
	if ws == nil {
		return serveLoadArmQuantProfileQ8
	}
	if cpuOffloadExperts {
		if ok, err := serveArtifactCPUOffloadExperts(ws); ok && err == nil {
			return serveLoadArmCPUOffloadExperts
		}
	}
	quant := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if (quant.Q4KResident || quant.Recipe == "UD-Q2_K_XL") && (serveDeviceResidentQ4K(nil) || (os.Getenv("FAK_Q4K") != "" && os.Getenv("FAK_Q4K") != "0")) {
		return serveLoadArmResidentQ4K
	}
	return serveLoadArmQuantProfileQ8
}

func resolveDeviceServeLoadArm(ws *ggufload.WeightSource, be compute.Backend, f32Resident, cpuOffloadExperts bool) serveLoadArm {
	if f32Resident {
		return serveLoadArmF32
	}
	if be == nil {
		return resolveHostServeLoadArm(ws, false, cpuOffloadExperts)
	}
	if ws == nil {
		return serveLoadArmResidentQ4K
	}
	if cpuOffloadExperts {
		if ok, err := serveArtifactCPUOffloadExperts(ws); ok && err == nil {
			return serveLoadArmCPUOffloadExperts
		}
	}
	quant := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if serveArtifactResidentQ4K(be, quant) {
		return serveLoadArmResidentQ4K
	}
	if be.Caps().UploadDtype {
		return serveLoadArmQuantProfileQ8
	}
	return serveLoadArmF32
}

// serveNativeContextSizingInputs returns the header-derived weight plan and
// memory ceiling for the load arm this process will actually take. Production
// callers pass the GGUF path so packed-embedding qualification matches loading;
// callers without an artifact path retain default embedding storage.
func serveNativeContextSizingInputs(ws *ggufload.WeightSource, be compute.Backend, cpuOffloadExperts, useMetal bool, ranks int, ggufPath ...string) (compute.MemoryPlan, serveFitBudget, error) {
	if ws == nil {
		return nil, serveFitBudget{}, nil
	}
	if be != nil && cpuOffloadExperts {
		if ok, err := serveArtifactCPUOffloadExperts(ws); ok && err == nil {
			// fak#13209: thread the bounded streamed-dense option so the arm's estimate charges the
			// eligible dense side at the working set, not the full on-disk dense total. The option
			// list is the SAME serveQ4KFitOptions list the path-based load arm uses, so the sizing
			// path and the loader carry one declaration.
			//
			// fak#13251: this sizing arm MUST fold the COMBINED streamed-expert remainder -- the same
			// rule the load arm applies (serve_load_helpers.go). Without it the context-sizing arm
			// derived its bounded dense working set from the whole host budget, so the auto-sized
			// context's view of the dense working set disagreed with the load guard: the same class of
			// over-admission fak#13249 removed from the load path. serveCPUOffloadSizingDenseBound is the
			// ONE derivation -- post-expert remainder on the combined arm, the historical whole-budget
			// bound on every other arm -- so the context plan and the load guard cannot disagree.
			opts := serveQ4KFitOptions("", ws, be, serveLoadArmCPUOffloadExperts, serveDeviceFitBudget(be))
			var weights compute.MemoryPlan
			var perr error
			if bound, bounded := serveBoundedDenseWorkingSetBound(opts); bounded {
				bound = serveCPUOffloadSizingDenseBound(ws, be, serveDeviceFitBudget(be), bound)
				weights, perr = ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(max(ranks, 1), bound)
			} else {
				weights, perr = ws.EstimateCPUOffloadExpertsExpertParallelMemoryPlan(max(ranks, 1))
			}
			return weights, serveDeviceFitBudget(be), perr
		}
	}
	path := ""
	if len(ggufPath) > 0 {
		path = ggufPath[0]
	}
	arm := resolveServeNativeContextLoadArm(ws, be, cpuOffloadExperts, useMetal)
	if be != nil {
		if ranks > 1 && arm == serveLoadArmResidentQ4K {
			weights, err := ws.EstimateExpertParallelLoadMemoryPlan(ranks)
			return weights, serveExpertParallelDeviceFitBudget(be), err
		}
		// ONE fit snapshot: the same serveDeviceFitBudget(be) the caller receives is what the
		// streamed-dense option list derives its bounded working set from (fak#13205).
		fit := serveDeviceFitBudget(be)
		weights, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, be, arm, fit)...)
		return applyDeviceWeightBudget(weights, be), fit, err
	}
	// ONE fit snapshot: the same serveHostFitBudget() the caller receives (fak#13205).
	fit := serveHostFitBudget()
	weights, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, nil, arm, fit)...)
	return weights, fit, err
}

func resolveServeNativeContextLoadArm(ws *ggufload.WeightSource, be compute.Backend, cpuOffloadExperts, useMetal bool) serveLoadArm {
	if be != nil {
		return resolveDeviceServeLoadArm(ws, be, false, cpuOffloadExperts)
	}
	if useMetal {
		return resolveMetalServeLoadArm(ws)
	}
	return resolveHostServeLoadArm(ws, false, cpuOffloadExperts)
}

func serveGGUFMemoryPlanForArm(ws *ggufload.WeightSource, arm serveLoadArm, contextBudgetTokens int, fit serveFitBudget, q4kOpts ...ggufload.Q4KLoadOption) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	weights, err := serveGGUFWeightMemoryPlanForArm(ws, arm, q4kOpts...)
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, weights, contextBudgetTokens, fit), nil
}

func serveGGUFWeightMemoryPlanForArm(ws *ggufload.WeightSource, arm serveLoadArm, q4kOpts ...ggufload.Q4KLoadOption) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	switch arm {
	case serveLoadArmF32:
		return ws.EstimateF32LoadMemoryPlan()
	case serveLoadArmQuantProfileQ8:
		return ws.EstimateQ8LoadMemoryPlan()
	case serveLoadArmCPUOffloadExperts:
		// arm-selected cpu-offload: the routed experts are host-scoped. rank-local
		// sharding is handled upstream (serveNativeContextSizingInputs) before this
		// unsharded single-rank arm is reached, so charge the full routed set here.
		//
		// fak#13209: probe the q4kOpts this call site received so a declared bounded
		// streamed-dense working set reaches the estimator. Before this, the arm IGNORED
		// q4kOpts and the bounded dense row was DEAD on the device --cpu-offload-experts arm.
		// The unsharded single-rank arm plans exactly as before when no bound is declared.
		if bound, ok := serveBoundedDenseWorkingSetBound(q4kOpts); ok {
			return ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(1, bound)
		}
		return ws.EstimateCPUOffloadExpertsMemoryPlan()
	case serveLoadArmResidentQ4K:
		plan, err := ws.EstimateQ4KLoadMemoryPlan(q4kOpts...)
		if errors.Is(err, ggufload.ErrQ4KLoadEstimateUnsupported) {
			// Unqualified routes retain their prior payload-based admission
			// policy. This is not a transformed-storage estimate; keep the
			// raw plan's provenance and the caller's historical peak bounds.
			return ws.EstimateLoadMemoryPlan()
		}
		if err != nil {
			return nil, fmt.Errorf("resident-Q4K weight admission: %w", err)
		}
		return plan, nil
	default:
		return nil, fmt.Errorf("unknown native weight load arm %q", arm)
	}
}

// serveQ4KFitOptions uses the existing loader selectors for the known path and
// backend. Pathless sizing uses default embedding storage; the path-based fit
// still runs before allocation. Neither plan includes transient load peaks.
//
// fit is the SAME measured budget the calling sizing path judges its plan against; the streamed
// dense route derives its bounded host working set from it (serveStreamedDenseQ4KWorkingSetBound)
// so the option list the loader receives charges the working set the fit guard already admitted,
// never the full on-disk dense side (fak#13205). A caller with no budget in scope passes
// serveFitBudget{} (unprobeable -> bound 0 -> stream-through), exactly as the expert precedent.
func serveQ4KFitOptions(path string, ws *ggufload.WeightSource, be compute.Backend, arm serveLoadArm, fit serveFitBudget) []ggufload.Q4KLoadOption {
	// fak#13209: the device --cpu-offload-experts arm also carries the bounded streamed-dense
	// working set, so the arm's DENSE side is judged against the bound rather than the full
	// on-disk dense total (the ~63.22 GiB V4.1 dense side charged as one host anon buffer is the
	// fak#13171 kernel OOM). This is the ONE option list the sizing path and the load path share,
	// so estimate and load cannot disagree. The arm's dense/router/attention weights are still
	// device-scoped; only the eligible dense k-quant staging transit is bounded. It needs no
	// WeightSource (the derivation is budget-only), so it is available on the path-form call sites.
	if arm == serveLoadArmCPUOffloadExperts {
		return serveCPUOffloadBoundedDenseOptions(be, fit)
	}
	if arm != serveLoadArmResidentQ4K || ws == nil {
		return nil
	}
	opts := serveResidentQ4KLoadOptions(be, path, true, ggufload.ClassifyTensorQuant(ws.File.Tensors))
	if os.Getenv("FAK_STREAM_Q4K") == "1" || os.Getenv("FAK_METAL_STREAM_Q4K") == "1" {
		// The dense working set is HOST-resident, so derive its bound from the true host
		// budget -- serveStreamedHostFit, exactly the fak#13142 rule the expert precedent
		// uses -- not the device aperture: a device-scale bound judged against real host RAM
		// is the bug that rule exists to prevent. The load path derives from the same rule
		// over the same fit, so estimate and load carry a byte-identical budget (fak#13205).
		opts = append(opts, ggufload.WithStreamedDenseQ4KWorkingSet(serveStreamedDenseQ4KWorkingSetBound(serveStreamedHostFit(be, &fit))))
	}
	return opts
}

// serveCPUOffloadBoundedDenseOptions is the ONE derivation of the device --cpu-offload-experts
// arm's bounded streamed-dense option list (fak#13209). It is gated on the SAME
// FAK_STREAM_Q4K/FAK_METAL_STREAM_Q4K knobs as the resident-Q4K streamed-dense route, and derives
// the bound from the HOST budget via serveStreamedHostFit -- the dense working set is HOST-resident,
// so it must not be sized from the device aperture (the fak#13142/#13205 rule). With the knob unset
// the list is empty, so every non-streamed device cpu-offload serve is byte-identical. The
// eligibility of the artifact's dense side is applied INSIDE the estimator and the loader via the
// same denseBoundedEligible predicate, so an ineligible dense side simply produces no bounded row
// and keeps the full charge (fail-closed) -- no pre-check is needed here.
func serveCPUOffloadBoundedDenseOptions(be compute.Backend, fit serveFitBudget) []ggufload.Q4KLoadOption {
	if os.Getenv("FAK_STREAM_Q4K") != "1" && os.Getenv("FAK_METAL_STREAM_Q4K") != "1" {
		return nil
	}
	return []ggufload.Q4KLoadOption{
		ggufload.WithStreamedDenseQ4KWorkingSet(serveStreamedDenseQ4KWorkingSetBound(serveStreamedHostFit(be, &fit))),
	}
}

// serveCPUOffloadSizingDenseBound folds the COMBINED streamed-expert remainder into the
// context-sizing arm's bounded dense working set, so the native-context auto-sizer sees the SAME
// combined host working set the load guard and the loader see (fak#13251).
//
// serveCPUOffloadBoundedDenseOptions derives the dense working set from the whole host budget
// (serveStreamedDenseQ4KWorkingSetBound(serveStreamedHostFit(...))). On the COMBINED
// streamed-expert + bounded-dense arm the streamed routed-expert set and the bounded dense set are
// BOTH simultaneously host-resident, and fak#13249 sized the dense bound from the budget REMAINING
// after the resident expert bound (serveCombinedStreamedDenseResidentBound) on the load path and the
// streamed sizing path. This context-sizing arm was the one caller left on the un-combined
// whole-budget bound, so the auto-sized context could claim residency the post-expert host headroom
// does not have -- the same over-admission fak#13249 removed from the load path.
//
// declared is the bound serveCPUOffloadBoundedDenseOptions already returned; the result is never
// larger than it (min, exactly as the fak#13249 call sites). When the streamed-expert policy is NOT
// selected (the full routed charge fits host, or the host is unprobeable) the declared bound is
// returned byte-for-byte, so every non-combined arm is unchanged.
func serveCPUOffloadSizingDenseBound(ws *ggufload.WeightSource, be compute.Backend, fit serveFitBudget, declared int64) int64 {
	if declared <= 0 || ws == nil {
		return declared
	}
	// The SAME host-fit rule serveCPUOffloadBoundedDenseOptions uses (one measurement): the injected
	// fit is taken verbatim on the device-less arm, the true host budget is probed when a device
	// backend is present, exactly as the load path does.
	hostFit := serveStreamedHostFit(be, &fit)
	if hostFit.avail() <= 0 {
		return declared
	}
	sharedPool := ggufload.BackendSharesHostRAM(be)
	splitAperture := serveSplitAperture(be)
	plan, streamed, err := serveStreamedCPUOffloadPlanForAperture(ws, be, 1, 0, hostFit, sharedPool, splitAperture)
	if err != nil || !streamed {
		return declared
	}
	expertBound := serveCPUOffloadStreamedResidentBoundForAperture(hostFit, plan.DeviceTotal(), sharedPool, splitAperture)
	if combined := serveCombinedStreamedDenseResidentBound(hostFit, expertBound); combined < declared {
		return combined
	}
	return declared
}

func serveGGUFMemoryPlan(ws *ggufload.WeightSource, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	arm := serveLoadArmResidentQ4K
	if f32Resident {
		arm = serveLoadArmF32
	}
	return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions("", ws, nil, arm, fit)...)
}

// serveGGUFCPUOffloadMemoryPlan plans the --cpu-offload-experts split: dense/router/attention
// weights device-scoped, routed and shared experts host-scoped.
//
// ranks is how many expert-parallel ranks this process's weights are split across Ã¢â‚¬â€ 1 for every
// unsharded serve, which plans exactly as it always has. Above 1 the rank has been handed a band
// and admits only experts [Lo,Hi) into the host expert pool (the loader's WithExpertShard seam),
// so the routed set must be charged one band and not in full: charging every rank the whole set
// overstated host demand ~ranks-fold and made RefuseHostScopedPlanIfTooBigForHost refuse a serve
// that fits Ã¢â‚¬â€ before the authoritative rank-local gate (refuseEPPlanIfUnfit, #2997) could run at
// all (#4952).
// serveGGUFCPUOffloadMemoryPlan is the CPU-offload sizing entry point. The variadic q4kOpts
// carries the SAME option list the load arm will thread, so when it declares a BOUNDED streamed-
// dense host working set (ggufload.WithStreamedDenseQ4KWorkingSet, fak#13209) the estimator charges
// the eligible dense side at that bound instead of the full on-disk dense total -- the estimate and
// the load cannot disagree because they read one declaration. Omitting q4kOpts (every historical
// caller) keeps EstimateCPUOffloadExpertsExpertParallelMemoryPlan byte-for-byte.
//
// THREADING CHOICE (fak#13209): a variadic parameter was chosen over a new argument on every
// CPU-offload helper because it is the least invasive form -- the streamed-expert selection path
// (serveStreamedCPUOffloadPlanForPool) and the host-fit probes call this WITHOUT an option list and
// stay byte-identical, while only the device arm's sizing + load call sites forward the bound.
// ggufload.ApplyQ4KLoadOptions probes the list without a config, so a non-bounded list is inert.
func serveGGUFCPUOffloadMemoryPlan(ws *ggufload.WeightSource, ranks, contextBudgetTokens int, fit serveFitBudget, q4kOpts ...ggufload.Q4KLoadOption) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	var plan compute.MemoryPlan
	var err error
	if bound, ok := serveBoundedDenseWorkingSetBound(q4kOpts); ok {
		plan, err = ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(ranks, bound)
	} else {
		plan, err = ws.EstimateCPUOffloadExpertsExpertParallelMemoryPlan(ranks)
	}
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
}

// serveBoundedDenseWorkingSetBound reports the declared bounded streamed-dense host working set an
// option list carries, if any. It reads the loader's own option-application surface
// (ggufload.ApplyQ4KLoadOptions) so the estimate-side probe and the load-side dispatch cannot drift.
// The bool is false when the list does not declare the bounded form (or is empty), in which case the
// caller keeps the historical full-charge estimator byte-for-byte. A declared zero IS bounded
// (stream-through), matching ggufload's StreamedDenseBounded semantics.
func serveBoundedDenseWorkingSetBound(opts []ggufload.Q4KLoadOption) (int64, bool) {
	if len(opts) == 0 {
		return 0, false
	}
	eff := ggufload.ApplyQ4KLoadOptions(opts)
	if !eff.StreamedDenseBounded {
		return 0, false
	}
	return eff.StreamedDenseBytes, true
}

func appendServeGGUFDevicePlan(ws *ggufload.WeightSource, plan compute.MemoryPlan, contextBudgetTokens int, fit serveFitBudget) compute.MemoryPlan {
	cfg, err := ws.File.Config()
	if err != nil {
		return plan
	}
	// Delegate to the single context auto-sizer (#1049) so the serve boot path sizes its
	// KV+scratch plan exactly as the in-kernel per-request planner does. #1046: pass the real
	// (headroom-adjusted) memory ceiling so that when no native context override is set the sizer
	// derives the LARGEST context that fits this box Ã¢â‚¬â€ instead of sizing against the full
	// MaxPositionEmbeddings window and refusing Ã¢â‚¬â€ and log the derived size for the operator.
	avail := fit.avail()
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	tokens, ctxPlan := compute.AutoSizeContextPlan(csc, plan, avail, serveContextTokenOverride(contextBudgetTokens))
	logServeAutoSizedContext(csc, plan, fit, avail, contextBudgetTokens, tokens)
	return append(plan, ctxPlan...)
}

// logServeAutoSizedContext prints the #1046 one-line auto-size record when the boot path DERIVED a
// context (no --native-context-tokens override, and a probeable memory ceiling) that is smaller than the
// model's full declared window Ã¢â‚¬â€ the case the operator needs to see, because the full window would
// have overflowed the box and refused. It is silent when an explicit budget was given, when the
// ceiling is unprobeable (the full window is kept, unchanged), or when the full window already fits
// (nothing was shrunk).
func logServeAutoSizedContext(csc compute.ContextSizeConfig, weights compute.MemoryPlan, fit serveFitBudget, avail int64, contextBudgetTokens, tokens int) {
	if contextBudgetTokens > 0 || avail <= 0 || csc.MaxContext <= 0 || tokens >= csc.MaxContext {
		return
	}
	kv := compute.EstimateKVStoreBytes(csc.KV, tokens)
	headroom := fit.Base - avail
	if headroom < 0 {
		headroom = 0
	}
	fmt.Fprintf(os.Stderr,
		"fak: auto-sized context to %d tokens (kv=%s, weights=%s, headroom=%s) Ã¢â‚¬â€ --native-context-tokens=0 selected auto sizing; the model's full %d-token window would overflow the %s fit budget\n",
		tokens, bytesText(uint64(max(kv, 0))), bytesText(uint64(max(weights.DeviceTotal(), 0))),
		bytesText(uint64(headroom)), csc.MaxContext, bytesText(uint64(max(avail, 0))))
}

// serveContextTokenOverride maps the serve flag convention (0 = unset, fall back to the
// model's full window) to the auto-sizer's override convention (<0 = unset, >=0 = explicit).
func serveContextTokenOverride(contextBudgetTokens int) int {
	if contextBudgetTokens > 0 {
		return contextBudgetTokens
	}
	return -1
}

// fitServeGGUFPathOnHost is the pure-CPU reference-path memory-fit pre-flight (#974). The CPU
// serve path (loadServeInKernelModel's FAK_Q4K and default cases) copies every super-block to
// ANONYMOUS host RAM with NO HAL backend to refuse via RefuseMemoryPlanIfTooBig, so without this
// it loads until the host OOM-wedges. It sizes the resident weights + KV + scratch off the GGUF
// HEADER ALONE (no tensor read Ã¢â‚¬â€ same EstimateLoadMemoryPlan proxy the device lean path uses) and
// refuses with a typed FitTooBig naming the shortfall when the plan exceeds MemAvailable less
// headroom Ã¢â‚¬â€ parity with the device path's fit plan. Fail-open: a platform that cannot report
// host memory loads exactly as before.
func fitServeGGUFPathOnHost(ggufPath string, f32Resident bool, contextBudgetTokens int, fit *serveFitBudget) error {
	total, free, known := compute.HostSystemMemoryInfo()
	return fitServeGGUFPathOnReportedHost(ggufPath, f32Resident, contextBudgetTokens, total, free, known, fit)
}

// fitServeGGUFPathOnHostForArm checks a host/unified-memory load whose runtime
// arm has already been selected. Metal uses this after choosing resident Q4_K,
// so its admission plan cannot silently fall back to the host Q8 estimate.
func fitServeGGUFPathOnHostForArm(ggufPath string, arm serveLoadArm, contextBudgetTokens int, fit *serveFitBudget) error {
	total, free, known := compute.HostSystemMemoryInfo()
	return fitServeGGUFPathOnReportedHostForArm(ggufPath, arm, contextBudgetTokens, total, free, known, fit)
}

func fitServeGGUFPathOnReportedHostForArm(ggufPath string, arm serveLoadArm, contextBudgetTokens int, total, free int64, known bool, override *serveFitBudget) error {
	if ggufPath == "" {
		return nil
	}
	fit := serveHostFitBudgetFromReported(total, free, known, override)
	plan, err := withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions(ggufPath, ws, nil, arm, fit)...)
	})
	if err != nil {
		return err
	}
	return refuseHostPlanAgainstFit(plan, fit)
}

func fitServeGGUFPathOnReportedHost(ggufPath string, f32Resident bool, contextBudgetTokens int, total, free int64, known bool, override *serveFitBudget) error {
	if ggufPath == "" {
		return nil
	}
	fit := serveHostFitBudgetFromReported(total, free, known, override)
	plan, err := withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		arm := resolveHostServeLoadArm(ws, f32Resident, false)
		return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions(ggufPath, ws, nil, arm, fit)...)
	})
	if err != nil {
		return err
	}
	return refuseHostPlanAgainstFit(plan, fit)
}

// refuseIfTooBigOnDevice applies the device-headroom refusal to a freshly-built plan Ã¢â‚¬â€
// the err-check + nil-backend passthrough + RefuseMemoryPlanIfTooBig tail the two
// fitAndPlanÃ¢â‚¬Â¦OnDevice helpers share.
func refuseIfTooBigOnDevice(plan compute.MemoryPlan, err error, be compute.Backend, override *serveFitBudget) (compute.MemoryPlan, error) {
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	var admitErr error
	if override != nil {
		plan, admitErr = refuseDevicePlanAgainstFit(be, plan, *override)
	} else {
		admitErr = compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
	}
	if admitErr != nil {
		return plan, admitErr
	}
	// #13172: on an INTEGRATED device (a Strix Halo APU Vulkan tier) the device-scoped weights
	// and the host-resident expert/staging charge draw from ONE physical DRAM pool, but the
	// device admission above judges only the device-scoped subset against a device probe that
	// reports the UNIFIED heap (~84 GiB on a 62.4 GiB Halo), not physical RAM. A plan whose
	// device dense side alone exceeds MemTotal then passes device admission and host admission,
	// and the kernel OOM-kills the serve during staging, before the forward. Bound the plan's
	// TOTAL simultaneous physical footprint against host RAM here so it is a typed fail-closed
	// refusal naming the shortfall. Inert on a discrete device or an unprobeable host (the
	// fail-open contract), so those loads are byte-for-byte unchanged. The sibling #13171 guard
	// (refuseDeviceStagingAgainstHostFit) bounds the --cpu-offload-experts arm's transient
	// staging transit; this bound is the loader-side total that covers the device arms that
	// guard does not, on the shared-pool tier.
	if unifiedErr := ggufload.RefuseUnifiedHostResidencyIfTooBig(plan, be, serveGGUFDeviceHeadroom); unifiedErr != nil {
		return plan, unifiedErr
	}
	return plan, nil
}

// withGGUFWeights opens the GGUF weights at ggufPath (an empty path plans nothing) and runs
// plan against them, closing the source after Ã¢â‚¬â€ the open+defer-close prelude the
// serveGGUFÃ¢â‚¬Â¦PathMemoryPlan helpers share.
func withGGUFWeights(ggufPath string, plan func(*ggufload.WeightSource) (compute.MemoryPlan, error)) (compute.MemoryPlan, error) {
	if ggufPath == "" {
		return nil, nil
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return plan(ws)
}

func fitAndPlanServeGGUFPathOnDevice(ggufPath string, be compute.Backend, f32Resident bool, contextBudgetTokens int, override *serveFitBudget) (compute.MemoryPlan, error) {
	plan, err := withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		arm := resolveDeviceServeLoadArm(ws, be, f32Resident, false)
		// ONE fit snapshot: the same device budget judges the plan and derives the
		// streamed-dense working set (fak#13205).
		fit := serveDeviceFitBudgetFromReported(be, override)
		return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions(ggufPath, ws, be, arm, fit)...)
	})
	if err == nil {
		plan = applyDeviceWeightBudget(plan, be)
	}
	return refuseIfTooBigOnDevice(plan, err, be, override)
}

// fitAndPlanServeGGUFCPUOffloadPathOnDevice keeps serveDeviceFitBudget's generic device headroom
// even for a sharded rank, rather than the tighter EP load-time one: on this arm the routed
// experts are host-resident, so the device side is the dense remainder plus KV Ã¢â‚¬â€ not the tight
// resident-EP case 0.05 exists for. ranks changes which routed bytes are charged, never the
// headroom.
func fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int, override *serveFitBudget) (compute.MemoryPlan, error) {
	// ONE fit snapshot (fak#13209): the same device budget the plan is judged against also derives
	// the bounded streamed-dense option list, so the estimate the fit gate reads is the bound the
	// loader will thread. serveQ4KFitOptions derives the bound from the HOST budget
	// (serveStreamedHostFit), exactly as the resident-Q4K route does -- the dense working set is
	// HOST-resident and must not be sized from the device aperture.
	devFit := serveDeviceFitBudgetFromReported(be, override)
	opts := serveQ4KFitOptions(ggufPath, nil, be, serveLoadArmCPUOffloadExperts, devFit)
	plan, err := serveGGUFCPUOffloadPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, devFit, opts...)
	return refuseIfTooBigOnDevice(plan, err, be, override)
}

func serveGGUFPathMemoryPlan(ggufPath string, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens, fit)
	})
}

func serveGGUFCPUOffloadPathMemoryPlan(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget, q4kOpts ...ggufload.Q4KLoadOption) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFCPUOffloadMemoryPlan(ws, ranks, contextBudgetTokens, fit, q4kOpts...)
	})
}

// serveStreamedExpertsCapable reports whether this artifact's routed-expert slabs are servable one
// stride at a time by the R5 checkpoint tier (ggufload.FusedExpertTensors is non-empty). It is the
// artifact-side half of the streamed policy decision: only a checkpoint whose expert encodings the
// tier can stage (Q2_K/Q4_K/Q5_K/Q6_K today) has a working fault path, so an unstageable artifact
// never takes the streamed arm and keeps the resident policy byte-for-byte.
func serveStreamedExpertsCapable(ws *ggufload.WeightSource) bool {
	if ws == nil {
		return false
	}
	shards, err := ws.FusedExpertTensors()
	return err == nil && len(shards) > 0
}

// serveCPUOffloadStreamedResidentMargin is the fraction of the headroom-adjusted host budget the
// bounded-resident streamed expert bound DECLINES to charge, so the streamed plan's host total lands
// STRICTLY below the budget that judges it. Without a margin the bound IS fit.avail(), so
// fitServeStreamedCPUOffloadPathOnHost compares the budget against itself: any non-expert host row
// (KV/scratch/activation) or the int64 truncation in compute.BudgetAfterHeadroom
// (int64(float64(budget)*(1-headroom))) pushes the plan total over by a few bytes and fails closed.
// That is the physical strix3 refusal (plan needs 48.01 GiB, host has 47.99 GiB, FitTooBig) with no
// real wall, on the exact host class the streamed arm exists to serve (fak#13140). 0.10 also leaves
// real slack for the resident-page jitter the host arm's own headroom exists to absorb; it is
// deliberately conservative rather than a razor-thin epsilon, and still a genuine bounded working
// set (10% of the budget), never zero.
const serveCPUOffloadStreamedResidentMargin = 0.10

// serveCPUOffloadStreamedResidentBound derives the bounded host-resident expert working set for the
// streamed policy from the SAME measured fit budget the resident arm is judged against: the
// headroom-adjusted host budget (fit.avail()) with serveCPUOffloadStreamedResidentMargin declared
// BELOW it, so the streamed plan's host total is strictly less than the budget it is judged against
// (fak#13140). An unprobeable host yields zero (stream-through) -- the honest floor, because an
// unmeasurable host must not be promised residency it may not have.
func serveCPUOffloadStreamedResidentBound(fit serveFitBudget) int64 {
	avail := fit.avail()
	if avail <= 0 {
		return 0
	}
	bound := int64(float64(avail) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= avail {
		// A razor-thin budget must still land strictly below avail, and never become negative.
		bound = avail - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}

// serveStreamedDenseQ4KWorkingSetBound derives the bounded host-resident DENSE working set for the
// streamed dense route (fak#13205) from the SAME measured fit budget the calling sizing path judges
// its plan against. The device-scoped dense transit is NOT born in VRAM: staging materializes it in
// host RAM (read -> dequant/transcode -> device upload -> free), so the fit guard must judge the
// bounded working set the loader actually RETAINS (ggufload.WithStreamedDenseQ4KWorkingSet, landed
// fak#13194) rather than the full on-disk dense side -- otherwise the ~63.22 GiB V4.1 dense side is
// ALSO charged as a transient host demand and the kernel OOM-kills the serve at staging
// (`staging-host-charge=63.223GiB host-budget=48.542GiB`, then OOM via filemap_fault; fak#13171).
//
// The arithmetic is identical to the bounded streamed-EXPERT precedent
// (serveCPUOffloadStreamedResidentBound, fak#13121/#13140): the headroom-adjusted host budget
// (fit.avail()) less serveCPUOffloadStreamedResidentMargin, clamped so the bound lands strictly
// below avail and never goes negative. An unprobeable host yields zero (stream-through) -- the
// honest floor, because an unmeasurable host must not be promised residency it may not have. The
// SAME value must flow to both the estimate and the load option list (one measurement).
func serveStreamedDenseQ4KWorkingSetBound(fit serveFitBudget) int64 {
	avail := fit.avail()
	if avail <= 0 {
		return 0
	}
	bound := int64(float64(avail) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= avail {
		// A razor-thin budget must still land strictly below avail, and never become negative.
		bound = avail - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}

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
	return appendServeGGUFDevicePlan(ws, streamed, contextBudgetTokens, fit), true, nil
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
