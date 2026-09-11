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

func fitAndPlanServeGGUFExpertParallelPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int, override *serveFitBudget) (compute.MemoryPlan, error) {
	fit := serveExpertParallelDeviceFitBudget(be)
	if override != nil {
		fit = *override
	}
	plan, err := serveGGUFExpertParallelPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, fit)
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	if override != nil {
		// Judge against the SAME device snapshot the rank plan was sized against; a later live
		// probe can read slightly less free memory and turn a fitting context into a false refusal.
		return refuseDevicePlanAgainstFit(be, plan, fit)
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
