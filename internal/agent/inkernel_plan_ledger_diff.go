package agent

import (
	"log"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// MemoryPlanDiffReporter is implemented by local planners that compare their
// predicted device memory plan with the backend's tagged allocation ledger.
// ok is false until a diff has been observed. Observability only.
type MemoryPlanDiffReporter interface {
	MemoryPlanDiff() (compute.MemoryPlanLedgerDiff, bool)
}

// MemoryPlanDiffPhaseFirstPrefill is the phase label for the diff taken once,
// right after the serve's first completed prefill.
const MemoryPlanDiffPhaseFirstPrefill = "first_prefill"

// planLedgerDiffState holds the once-per-serve first-prefill diff.
type planLedgerDiffState struct {
	planDiffMu    sync.Mutex
	planDiffTaken bool
	planDiff      *compute.MemoryPlanLedgerDiff
}

// MemoryPlanDiff reports the first-prefill planner-vs-ledger diff.
func (p *InKernelPlanner) MemoryPlanDiff() (compute.MemoryPlanLedgerDiff, bool) {
	if p == nil {
		return compute.MemoryPlanLedgerDiff{}, false
	}
	p.planDiffMu.Lock()
	defer p.planDiffMu.Unlock()
	if p.planDiff == nil {
		return compute.MemoryPlanLedgerDiff{}, false
	}
	return *p.planDiff.Clone(), true
}

// observeFirstPrefillPlanLedgerDiff takes the diff once, after the first
// prefill, when the backend carries an allocation ledger. The predicted plan is
// the request plan for (promptTokens, maxNew) with resident weights always
// included, so weights are compared even when admission skipped them.
func (p *InKernelPlanner) observeFirstPrefillPlanLedgerDiff(promptTokens, maxNew int) {
	if p == nil || p.backend == nil {
		return
	}
	p.planDiffMu.Lock()
	if p.planDiffTaken {
		p.planDiffMu.Unlock()
		return
	}
	p.planDiffTaken = true
	p.planDiffMu.Unlock()
	snap, ok := compute.DeviceAllocLedgerOf(p.backend)
	if !ok {
		return
	}
	plan, _ := p.requestMemoryPlanWithExtras(promptTokens, maxNew, false)
	plan = withResidentWeightsDemand(plan, p.residentWeightBytes())
	p.storePlanLedgerDiff(compute.DiffMemoryPlanAgainstLedger(MemoryPlanDiffPhaseFirstPrefill, plan, snap))
}

func (p *InKernelPlanner) storePlanLedgerDiff(d compute.MemoryPlanLedgerDiff) {
	log.Print(d.LogLine())
	p.planDiffMu.Lock()
	p.planDiff = &d
	p.planDiffMu.Unlock()
}

func (p *InKernelPlanner) residentWeightBytes() int64 {
	if p == nil || p.m == nil {
		return 0
	}
	if r := p.m.ResidentReport(); r != nil {
		return r.TotalResidentBytes
	}
	return 0
}

// withResidentWeightsDemand prepends a device weights demand when plan carries
// none and the model reports resident bytes.
func withResidentWeightsDemand(plan compute.MemoryPlan, residentBytes int64) compute.MemoryPlan {
	if residentBytes <= 0 {
		return plan
	}
	for _, d := range plan {
		if d.Class == compute.MemoryWeights && d.ScopeOrDefault() == compute.MemoryScopeDevice {
			return plan
		}
	}
	return append(compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: residentBytes, Detail: "resident-weights", DType: "mixed"}}, plan...)
}
