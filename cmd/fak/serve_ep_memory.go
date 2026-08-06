package main

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

const serveGGUFExpertParallelDeviceHeadroom = 0.05

func serveExpertParallelDeviceFitBudget(be compute.Backend) serveFitBudget {
	fit := serveDeviceFitBudget(be)
	fit.Headroom = serveGGUFExpertParallelDeviceHeadroom
	return fit
}

func fitServeGGUFExpertParallelOnDevice(ws *ggufload.WeightSource, be compute.Backend, ranks, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFExpertParallelMemoryPlan(ws, ranks, contextBudgetTokens, serveExpertParallelDeviceFitBudget(be))
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFExpertParallelDeviceHeadroom)
}

func fitAndPlanServeGGUFExpertParallelPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFExpertParallelPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, serveExpertParallelDeviceFitBudget(be))
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	return plan, compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFExpertParallelDeviceHeadroom)
}

func serveGGUFExpertParallelPathMemoryPlan(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFExpertParallelMemoryPlan(ws, ranks, contextBudgetTokens, fit)
	})
}

func serveGGUFExpertParallelMemoryPlan(ws *ggufload.WeightSource, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	plan, err := ws.EstimateExpertParallelLoadMemoryPlan(ranks)
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
}

// The --cpu-offload-experts arm of a SHARDED expert-parallel serve (#4952).
//
// That arm is explicitly sanctioned for a sharded rank — loadServeInKernelModel's shard guard
// names --cpu-offload-experts as one of the two resident-Q4K arms carrying the WithExpertShard
// seam — and the loader honours the band: the rank admits only experts [Lo,Hi) into the host
// expert pool. Its fit check did not. It planned through the whole-model
// EstimateCPUOffloadExpertsMemoryPlan, charging every rank the FULL routed-expert set, so
// RefuseHostScopedPlanIfTooBigForHost over-refused by ~ranks-fold on a serve that fits — and it
// fired BEFORE the authoritative rank-local gate (refuseEPPlanIfUnfit, #2997) could be consulted.
// These siblings plan the same host/device split against the busiest rank's band instead.
//
// Headroom stays serveDeviceFitBudget's generic device headroom rather than the tighter EP
// load-time one: on this arm the routed experts are host-resident, so the device side is the
// dense/attention/router remainder plus KV — not the tight resident-EP case that 0.05 exists for.
// The only thing #4952 changes is which routed-expert bytes are charged, never the headroom.
func fitAndPlanServeGGUFCPUOffloadExpertParallelPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFCPUOffloadExpertParallelPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, serveDeviceFitBudget(be))
	return refuseIfTooBigOnDevice(plan, err, be)
}

func serveGGUFCPUOffloadExpertParallelPathMemoryPlan(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFCPUOffloadExpertParallelMemoryPlan(ws, ranks, contextBudgetTokens, fit)
	})
}

// serveGGUFCPUOffloadExpertParallelMemoryPlan is serveGGUFCPUOffloadMemoryPlan for a sharded rank.
// ranks <= 1 delegates to the unsharded plan verbatim, so a non-EP serve — every serve that ran
// before #4952 — plans byte-for-byte as it did.
func serveGGUFCPUOffloadExpertParallelMemoryPlan(ws *ggufload.WeightSource, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	if ranks <= 1 {
		return serveGGUFCPUOffloadMemoryPlan(ws, contextBudgetTokens, fit)
	}
	plan, err := ws.EstimateCPUOffloadExpertsExpertParallelMemoryPlan(ranks)
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
}
