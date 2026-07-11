package modelroute

import "testing"

func TestCapacityBlockedTaskReroutesWithoutFailure(t *testing.T) {
	got := RerouteCapacity("local-7b", CapacitySignal{Blocked: true, Reason: CapacityReasonModelCeiling, RequiredModelB: 14, LocalModelCeilingB: 7}, []CapacityTarget{{Name: "gpu-70b", Model: "70b", Pool: "fleet-gpu", ModelB: 70, Available: true}, {Name: "cloud-14b", Model: "14b", Pool: "cloud", ModelB: 14, Available: true}})
	if !got.Rerouted || got.To.Name != "cloud-14b" || got.Reason != CapacityReasonModelCeiling {
		t.Fatalf("got=%+v", got)
	}
}
func TestContextBlockedTaskChoosesLargerWindow(t *testing.T) {
	got := RerouteCapacity("local", CapacitySignal{Blocked: true, Reason: CapacityReasonContextWindow, RequiredContext: 200000, UsableContext: 128000}, []CapacityTarget{{Name: "gpu", Pool: "fleet-gpu", Context: 256000, Available: true}})
	if !got.Rerouted || got.To.Name != "gpu" {
		t.Fatalf("got=%+v", got)
	}
}
func TestCapacityRerouteRefusesWithoutEligibleTarget(t *testing.T) {
	got := RerouteCapacity("local", CapacitySignal{Blocked: true, Reason: CapacityReasonModelCeiling, RequiredModelB: 70}, []CapacityTarget{{Name: "small", ModelB: 7, Available: true}})
	if got.Rerouted || got.Reason != CapacityReasonModelCeiling {
		t.Fatalf("got=%+v", got)
	}
}
