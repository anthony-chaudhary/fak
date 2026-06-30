package main

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func fitServeGGUFExpertParallelOnDevice(ws *ggufload.WeightSource, be compute.Backend, ranks, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFExpertParallelMemoryPlan(ws, ranks, contextBudgetTokens, serveDeviceFitBudget(be))
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
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

func fitAndPlanServeGGUFExpertParallelPathOnDevice(ggufPath string, be compute.Backend, ranks, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFExpertParallelPathMemoryPlan(ggufPath, ranks, contextBudgetTokens, serveDeviceFitBudget(be))
	return refuseIfTooBigOnDevice(plan, err, be)
}

func serveGGUFExpertParallelPathMemoryPlan(ggufPath string, ranks, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFExpertParallelMemoryPlan(ws, ranks, contextBudgetTokens, fit)
	})
}
