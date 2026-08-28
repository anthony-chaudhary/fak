package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

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

func serveGGUFMemoryPlan(ws *ggufload.WeightSource, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	var plan compute.MemoryPlan
	if f32Resident {
		weights, err := ws.EstimateF32LoadMemoryPlan()
		if err != nil {
			return nil, err
		}
		plan = append(plan, weights...)
	} else {
		weights, err := ws.EstimateLoadMemoryPlan()
		if err != nil {
			return nil, err
		}
		plan = append(plan, weights...)
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
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
	// (headroom-adjusted) memory ceiling so that when no --context-budget-tokens is set the sizer
	// derives the LARGEST context that fits this box — instead of sizing against the full
	// MaxPositionEmbeddings window and refusing — and log the derived size for the operator.
	csc := cfg.ContextSizeConfig()
	avail := fit.avail()
	tokens, ctxPlan := compute.AutoSizeContextPlan(csc, plan, avail, serveContextTokenOverride(contextBudgetTokens))
	logServeAutoSizedContext(csc, plan, fit, avail, contextBudgetTokens, tokens)
	return append(plan, ctxPlan...)
}

// logServeAutoSizedContext prints the #1046 one-line auto-size record when the boot path DERIVED a
// context (no --context-budget-tokens, and a probeable memory ceiling) that is smaller than the
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
		"fak: auto-sized context to %d tokens (kv=%s, weights=%s, headroom=%s) — no --context-budget-tokens set; the model's full %d-token window would overflow the %s fit budget\n",
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
func fitServeGGUFPathOnHost(ggufPath string, f32Resident bool, contextBudgetTokens int) error {
	if ggufPath == "" {
		return nil
	}
	plan, err := serveGGUFPathMemoryPlan(ggufPath, f32Resident, contextBudgetTokens, serveHostFitBudget())
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBigForHost(plan, serveGGUFHostHeadroom)
}

// refuseIfTooBigOnDevice applies the device-headroom refusal to a freshly-built plan —
// the err-check + nil-backend passthrough + RefuseMemoryPlanIfTooBig tail the two
// fitAndPlan…OnDevice helpers share.
func refuseIfTooBigOnDevice(plan compute.MemoryPlan, err error, be compute.Backend) (compute.MemoryPlan, error) {
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
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

func fitAndPlanServeGGUFPathOnDevice(ggufPath string, be compute.Backend, f32Resident bool, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFPathMemoryPlan(ggufPath, f32Resident, contextBudgetTokens, serveDeviceFitBudget(be))
	if err == nil {
		plan = applyDeviceWeightBudget(plan, be)
	}
	return refuseIfTooBigOnDevice(plan, err, be)
}

// fitAndPlanServeGGUFCPUOffloadPathOnDevice keeps serveDeviceFitBudget's generic device headroom
// even for a sharded rank, rather than the tighter EP load-time one: on this arm the routed
// experts are host-resident, so the device side is the dense remainder plus KV — not the tight
// resident-EP case 0.05 exists for. ranks changes which routed bytes are charged, never the
// headroom.
func fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFCPUOffloadPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, serveDeviceFitBudget(be))
	return refuseIfTooBigOnDevice(plan, err, be)
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
