package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestWithResidentWeightsDemand(t *testing.T) {
	kv := compute.MemoryPlan{{Class: compute.MemoryKVCache, Bytes: 10}}
	got := withResidentWeightsDemand(kv, 90)
	if len(got) != 2 || got[0].Class != compute.MemoryWeights || got[0].Bytes != 90 {
		t.Fatalf("weights not prepended: %+v", got)
	}
	withW := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 5}, {Class: compute.MemoryKVCache, Bytes: 10}}
	if got := withResidentWeightsDemand(withW, 90); len(got) != 2 || got[0].Bytes != 5 {
		t.Fatalf("device weights demand duplicated: %+v", got)
	}
	hostW := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 5, Scope: compute.MemoryScopeHost}}
	if got := withResidentWeightsDemand(hostW, 90); len(got) != 2 {
		t.Fatalf("host weights demand counted as device: %+v", got)
	}
	if got := withResidentWeightsDemand(kv, 0); len(got) != 1 {
		t.Fatalf("zero resident bytes prepended: %+v", got)
	}
}

func TestInKernelPlannerMemoryPlanDiffStoreAndOnce(t *testing.T) {
	var p InKernelPlanner
	var _ MemoryPlanDiffReporter = &p
	if _, ok := p.MemoryPlanDiff(); ok {
		t.Fatal("diff reported before observation")
	}
	// No backend: the first-prefill hook is a no-op and never latches.
	p.observeFirstPrefillPlanLedgerDiff(8, 8)
	if p.planDiffTaken {
		t.Fatal("hook latched without a backend")
	}
	d := compute.DiffMemoryPlanAgainstLedger(MemoryPlanDiffPhaseFirstPrefill,
		compute.MemoryPlan{{Class: compute.MemoryKVCache, Bytes: 40}},
		compute.DeviceAllocSnapshot{Cells: []compute.DeviceAllocCell{{Class: compute.AllocKVCache, Kind: compute.AllocDeviceLocal, LiveBytes: 50}}})
	p.storePlanLedgerDiff(d)
	got, ok := p.MemoryPlanDiff()
	if !ok || got.Phase != MemoryPlanDiffPhaseFirstPrefill || got.DeltaBytes != 10 {
		t.Fatalf("diff = %v %+v", ok, got)
	}
	got.Rows[0].ActualBytes = -1
	again, _ := p.MemoryPlanDiff()
	if again.Rows[0].ActualBytes != 50 {
		t.Fatal("MemoryPlanDiff aliases stored rows")
	}
}
