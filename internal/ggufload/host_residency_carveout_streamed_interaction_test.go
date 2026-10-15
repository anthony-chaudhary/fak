package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak#13761 (TICKET-03): the two memory-accounting commits that landed under the 2026-10-09
// verification waiver were each witnessed alone, but their INTERACTION was not. 15c81f74897
// (streamed-expert host-peak estimator / bounded streamed dense plan) and b7f2b35029e (a
// dedicated VRAM carve-out counted in the unified physical-pool bound) meet at one seam:
// UnifiedHostResidencyPlan's streamed output carries a host-scoped MemoryOffload row, and the
// carve-out-aware judge withholds the carve-out credit from any plan that has such a row
// (planHasHostOffload, the fak#13280 guard: a streamed plan's device bytes cannot be trusted to
// stay out of system RAM). Every existing carve-out test judges a hand-built plan; every
// estimator test stops at the plan. Nothing pins the estimator's ACTUAL output against the
// carve-out judge, so a change to either side's row shape could silently re-enable the credit
// on a streamed plan. These two cases are that witness.
//
// fak-test:runtime fast est=5ms lane=default
func TestStreamedDensePlanWithholdsCarveoutCreditOnSplitAperture(t *testing.T) {
	const (
		bound        = int64(moeDenseEligibleBytes) // one full eligible dense side
		deviceWindow = 4 << 20
		memTotal     = 200 << 10
		carveout     = 4 << 20
		headroom     = 0.15
	)
	ws := moeStreamedTestWeightSource(t, moeDenseEligibleBytes)
	opts := []Q4KLoadOption{WithStreamedDenseQ4KWorkingSet(bound)}

	plan, err := ws.UnifiedHostResidencyPlan(opts...)
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet): %v", err)
	}
	// The streamed estimator's output must carry the host-scoped offload row the carve-out guard
	// keys on. If this premise ever breaks, the interaction has changed and the assertions below
	// stop being meaningful.
	if !planHasHostOffload(plan) {
		t.Fatalf("streamed plan %+v has no host-scoped MemoryOffload row; the carve-out interaction is unwitnessed", plan)
	}
	// The plan is small enough to fit a pool of MemTotal+carveout, so a refusal below can only be
	// the withheld credit, never raw size.
	if want, budget := plan.Total(), compute.BudgetAfterHeadroom(memTotal+carveout, headroom); want > budget {
		t.Fatalf("fixture too big: plan total %d exceeds the carve-out-credited budget %d", want, budget)
	}

	// Strix3-shaped split aperture: a distinct device VRAM window over the same physical DRAM,
	// MemTotal far below the plan's grand total, plus a large firmware carve-out.
	be := carveoutBackend{splitApertureBackend(deviceWindow, memTotal), carveout, true}

	// The streamed route must NOT receive the carve-out credit, so the plan is judged against
	// MemTotal alone and refuses typed (the fak#13280 guard on the estimator's real output).
	err = ws.RefuseUnifiedHostResidencyForLoadOptions(be, headroom, opts...)
	requireUnifiedFitTooBig(t, err)

	// The contrast proves the refusal is the host-offload gate, not the aperture: an equal-total
	// device-resident plan with NO host offload row is admitted under the SAME backend, because
	// the carve-out credit applies to it.
	deviceOnly := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: plan.Total(), Detail: "dense-base+ring"},
	}
	if err := RefuseUnifiedHostResidencyIfTooBig(deviceOnly, be, headroom); err != nil {
		t.Fatalf("a device-resident plan of equal total was refused though the carve-out covers it: %v", err)
	}
}
