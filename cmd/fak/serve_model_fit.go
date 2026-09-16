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
// (fail-open) — an invalid flag is refused at serve-flag validation, not here.
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
// overflow (e.g. a 434 GiB model at N=4 ≈ 118 GiB/card on 80 GiB GPUs) into a clean pre-serve
// refusal — instead of an OOM that surfaces minutes in, when rank r uploads its expert band to GPU r.
func refuseEPPlanIfUnfit(m *fakmodel.Model, be compute.Backend, ranks, contextBudgetTokens int) error {
	if m == nil || be == nil || ranks <= 1 {
		return nil
	}
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok {
		return nil // nothing accounted (non-MoE / unloaded) -> fail open
	}
	// KV is a per-rank cost: pure EP replicates attention, so each rank holds the full KV for the
	// context it serves. Size it from the model geometry at the context budget — the SAME KV the
	// load-time fit plan sizes from contextBudgetTokens — so the per-card check is weights + KV, not
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
// report capacity) — avail() then yields FreeUnknown and the auto-sizer falls open to the model's
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

// avail is the headroom-adjusted budget passed to compute.AutoSizeContextPlan — byte-identical to
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
// (DeviceMemoryInfo: free, or the total ceiling when free is unprobeable). Unknown capacity → a
// zero base → the auto-sizer keeps the full window.
func serveDeviceFitBudget(be compute.Backend) serveFitBudget {
	total, free, known := compute.DeviceMemoryInfo(be)
	return serveFitBudget{Base: serveFitBudgetBase(total, free, known), Headroom: serveGGUFDeviceHeadroom}
}

// serveHostFitBudget reads the process host's allocatable RAM the pure-CPU serve arm's fit check
// uses (HostSystemMemoryInfo → Linux MemAvailable). Unknown → a zero base → the full window.
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
			weights, perr := ws.EstimateCPUOffloadExpertsExpertParallelMemoryPlan(max(ranks, 1))
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
		weights, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, be, arm)...)
		return applyDeviceWeightBudget(weights, be), serveDeviceFitBudget(be), err
	}
	weights, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, nil, arm)...)
	return weights, serveHostFitBudget(), err
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
func serveQ4KFitOptions(path string, ws *ggufload.WeightSource, be compute.Backend, arm serveLoadArm) []ggufload.Q4KLoadOption {
	if arm != serveLoadArmResidentQ4K || ws == nil {
		return nil
	}
	opts := serveResidentQ4KLoadOptions(be, path, true, ggufload.ClassifyTensorQuant(ws.File.Tensors))
	if os.Getenv("FAK_STREAM_Q4K") == "1" || os.Getenv("FAK_METAL_STREAM_Q4K") == "1" {
		opts = append(opts, ggufload.WithStreamedDenseQ4K(true))
	}
	return opts
}

func serveGGUFMemoryPlan(ws *ggufload.WeightSource, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	arm := serveLoadArmResidentQ4K
	if f32Resident {
		arm = serveLoadArmF32
	}
	return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions("", ws, nil, arm)...)
}

// serveGGUFCPUOffloadMemoryPlan plans the --cpu-offload-experts split: dense/router/attention
// weights device-scoped, routed and shared experts host-scoped.
//
// ranks is how many expert-parallel ranks this process's weights are split across — 1 for every
// unsharded serve, which plans exactly as it always has. Above 1 the rank has been handed a band
// and admits only experts [Lo,Hi) into the host expert pool (the loader's WithExpertShard seam),
// so the routed set must be charged one band and not in full: charging every rank the whole set
// overstated host demand ~ranks-fold and made RefuseHostScopedPlanIfTooBigForHost refuse a serve
// that fits — before the authoritative rank-local gate (refuseEPPlanIfUnfit, #2997) could run at
// all (#4952).
func serveGGUFCPUOffloadMemoryPlan(ws *ggufload.WeightSource, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	plan, err := ws.EstimateCPUOffloadExpertsExpertParallelMemoryPlan(ranks)
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
}

func appendServeGGUFDevicePlan(ws *ggufload.WeightSource, plan compute.MemoryPlan, contextBudgetTokens int, fit serveFitBudget) compute.MemoryPlan {
	cfg, err := ws.File.Config()
	if err != nil {
		return plan
	}
	// Delegate to the single context auto-sizer (#1049) so the serve boot path sizes its
	// KV+scratch plan exactly as the in-kernel per-request planner does. #1046: pass the real
	// (headroom-adjusted) memory ceiling so that when no native context override is set the sizer
	// derives the LARGEST context that fits this box — instead of sizing against the full
	// MaxPositionEmbeddings window and refusing — and log the derived size for the operator.
	avail := fit.avail()
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	tokens, ctxPlan := compute.AutoSizeContextPlan(csc, plan, avail, serveContextTokenOverride(contextBudgetTokens))
	logServeAutoSizedContext(csc, plan, fit, avail, contextBudgetTokens, tokens)
	return append(plan, ctxPlan...)
}

// logServeAutoSizedContext prints the #1046 one-line auto-size record when the boot path DERIVED a
// context (no --native-context-tokens override, and a probeable memory ceiling) that is smaller than the
// model's full declared window — the case the operator needs to see, because the full window would
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
		"fak: auto-sized context to %d tokens (kv=%s, weights=%s, headroom=%s) — --native-context-tokens=0 selected auto sizing; the model's full %d-token window would overflow the %s fit budget\n",
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
// HEADER ALONE (no tensor read — same EstimateLoadMemoryPlan proxy the device lean path uses) and
// refuses with a typed FitTooBig naming the shortfall when the plan exceeds MemAvailable less
// headroom — parity with the device path's fit plan. Fail-open: a platform that cannot report
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
		return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions(ggufPath, ws, nil, arm)...)
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
		return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, fit, serveQ4KFitOptions(ggufPath, ws, nil, arm)...)
	})
	if err != nil {
		return err
	}
	return refuseHostPlanAgainstFit(plan, fit)
}

// refuseIfTooBigOnDevice applies the device-headroom refusal to a freshly-built plan —
// the err-check + nil-backend passthrough + RefuseMemoryPlanIfTooBig tail the two
// fitAndPlan…OnDevice helpers share.
func refuseIfTooBigOnDevice(plan compute.MemoryPlan, err error, be compute.Backend, override *serveFitBudget) (compute.MemoryPlan, error) {
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	if override != nil {
		return refuseDevicePlanAgainstFit(be, plan, *override)
	}
	return plan, compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

// withGGUFWeights opens the GGUF weights at ggufPath (an empty path plans nothing) and runs
// plan against them, closing the source after — the open+defer-close prelude the
// serveGGUF…PathMemoryPlan helpers share.
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
		return serveGGUFMemoryPlanForArm(ws, arm, contextBudgetTokens, serveDeviceFitBudgetFromReported(be, override), serveQ4KFitOptions(ggufPath, ws, be, arm)...)
	})
	if err == nil {
		plan = applyDeviceWeightBudget(plan, be)
	}
	return refuseIfTooBigOnDevice(plan, err, be, override)
}

// fitAndPlanServeGGUFCPUOffloadPathOnDevice keeps serveDeviceFitBudget's generic device headroom
// even for a sharded rank, rather than the tighter EP load-time one: on this arm the routed
// experts are host-resident, so the device side is the dense remainder plus KV — not the tight
// resident-EP case 0.05 exists for. ranks changes which routed bytes are charged, never the
// headroom.
func fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int, override *serveFitBudget) (compute.MemoryPlan, error) {
	plan, err := serveGGUFCPUOffloadPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, serveDeviceFitBudgetFromReported(be, override))
	return refuseIfTooBigOnDevice(plan, err, be, override)
}

func serveGGUFPathMemoryPlan(ggufPath string, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens, fit)
	})
}

func serveGGUFCPUOffloadPathMemoryPlan(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFCPUOffloadMemoryPlan(ws, ranks, contextBudgetTokens, fit)
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
func serveStreamedCPUOffloadPlan(ws *ggufload.WeightSource, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, bool, error) {
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, ranks, contextBudgetTokens, fit)
	if err != nil {
		return nil, false, err
	}
	if ws == nil || fit.Base <= 0 {
		return plan, false, nil
	}
	// Judge the HOST-scoped subset, not the grand total: on a device serve the dense weights live in
	// VRAM and must not be double-counted against host RAM here.
	if plan.HostTotal() <= fit.avail() {
		return plan, false, nil
	}
	if !serveStreamedExpertsCapable(ws) {
		return plan, false, nil
	}
	bound := serveCPUOffloadStreamedResidentBound(fit)
	streamed, err := ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(bound)
	if err != nil {
		return nil, false, err
	}
	return appendServeGGUFDevicePlan(ws, streamed, contextBudgetTokens, fit), true, nil
}

// serveStreamedCPUOffloadPathDecision is the path-form of serveStreamedCPUOffloadPlan: the LOAD
// arm's WithStreamedExperts threading and the SIZING path's plan are decided by ONE measurement
// over one opened checkpoint, so they cannot disagree. An unopenable path, a non-MoE artifact, or a
// routed set that already fits the host budget all return (false, 0, nil) -- the resident arm.
func serveStreamedCPUOffloadPathDecision(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget) (bool, int64, error) {
	streamed := false
	bound := int64(0)
	_, err := withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		_, isStreamed, perr := serveStreamedCPUOffloadPlan(ws, ranks, contextBudgetTokens, fit)
		if perr != nil {
			return nil, perr
		}
		if isStreamed {
			streamed, bound = true, serveCPUOffloadStreamedResidentBound(fit)
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
	plan, _, err := serveStreamedCPUOffloadPlan(ws, ranks, contextBudgetTokens, fit)
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
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, ranks, contextBudgetTokens, fit)
	if err != nil {
		return nil, false, err
	}
	dev := serveDeviceFitBudgetFromReported(be, override)
	admitted, rerr := refuseIfTooBigOnDevice(plan, nil, be, &dev)
	return admitted, streamed, rerr
}
