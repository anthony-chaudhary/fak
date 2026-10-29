package gateway

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type planDiffPlanner struct {
	stubPlanner
	d  compute.MemoryPlanLedgerDiff
	ok bool
}

func (p planDiffPlanner) MemoryPlanDiff() (compute.MemoryPlanLedgerDiff, bool) { return p.d, p.ok }

func fakeFirstPrefillDiff() compute.MemoryPlanLedgerDiff {
	return compute.DiffMemoryPlanAgainstLedger("first_prefill",
		compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 100}},
		compute.DeviceAllocSnapshot{Cells: []compute.DeviceAllocCell{
			{Class: compute.AllocWeights, Kind: compute.AllocDeviceLocal, LiveBytes: 100},
			{Class: compute.AllocIdlePool, Kind: compute.AllocDeviceLocal, LiveBytes: 30},
		}})
}

func TestDebugMemoryPlanDiffFromReporter(t *testing.T) {
	if debugMemoryPlanDiff(stubPlanner{}) != nil {
		t.Fatal("planner without reporter produced a diff")
	}
	if debugMemoryPlanDiff(planDiffPlanner{ok: false}) != nil {
		t.Fatal("unobserved diff rendered")
	}
	got := debugMemoryPlanDiff(planDiffPlanner{d: fakeFirstPrefillDiff(), ok: true})
	if got == nil || got.Phase != "first_prefill" || got.DeltaBytes != 30 {
		t.Fatalf("diff = %+v", got)
	}
	raw, err := json.Marshal(debugVarsResponse{MemoryPlanDiff: got})
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		MemoryPlanDiff struct {
			Phase       string `json:"phase"`
			DeltaBytes  int64  `json:"delta_bytes"`
			TopPositive []struct {
				Class      string `json:"class"`
				DeltaBytes int64  `json:"delta_bytes"`
			} `json:"top_positive"`
		} `json:"memory_plan_diff"`
	}
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if back.MemoryPlanDiff.DeltaBytes != 30 || len(back.MemoryPlanDiff.TopPositive) != 1 || back.MemoryPlanDiff.TopPositive[0].Class != "idle_pool" {
		t.Fatalf("json = %s", raw)
	}
}

func TestModelLoadProfileClonesPlanLedgerDiff(t *testing.T) {
	d := fakeFirstPrefillDiff()
	p := &ModelLoadProfile{PlanLedgerDiff: &d}
	c := p.clone()
	if c.PlanLedgerDiff == p.PlanLedgerDiff {
		t.Fatal("clone shares the diff pointer")
	}
	c.PlanLedgerDiff.Rows[0].ActualBytes = -1
	if p.PlanLedgerDiff.Rows[0].ActualBytes == -1 {
		t.Fatal("clone aliases diff rows")
	}
}
