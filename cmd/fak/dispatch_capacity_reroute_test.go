package main

import (
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"testing"
)

func TestPrepareDispatchWorkerCommandAppliesCapacityReroute(t *testing.T) {
	payload := map[string]any{}
	opts := dispatchTickOptions{Backend: "claude", WorkerModel: "local-7b", CapacityReason: modelroute.CapacityReasonModelCeiling, CapacityFrom: "local-7b", RequiredModelB: 14, CapacityTargets: []modelroute.CapacityTarget{{Name: "gpu", Model: "gpu-70b", Pool: "fleet-gpu", ModelB: 70, Available: true}}}
	launch, _, _, err := prepareDispatchWorkerCommand(t.TempDir(), opts, dispatchLanePick{Lane: "docs"}, dispatchtick.Account{}, 1, 0, nil, nil, payload)
	if err != nil {
		t.Fatal(err)
	}
	if launch.Model != "gpu-70b" {
		t.Fatalf("model=%q payload=%+v", launch.Model, payload)
	}
	r, ok := payload["capacity_reroute"].(modelroute.CapacityReroute)
	if !ok || !r.Rerouted || r.From != "local-7b" {
		t.Fatalf("reroute=%+v", payload["capacity_reroute"])
	}
}
func TestPrepareDispatchWorkerCommandKeepsModelWithoutEligibleAlternate(t *testing.T) {
	payload := map[string]any{}
	opts := dispatchTickOptions{Backend: "claude", WorkerModel: "local-7b", CapacityReason: modelroute.CapacityReasonModelCeiling, CapacityFrom: "local-7b", RequiredModelB: 70}
	launch, _, _, err := prepareDispatchWorkerCommand(t.TempDir(), opts, dispatchLanePick{Lane: "docs"}, dispatchtick.Account{}, 1, 0, nil, nil, payload)
	if err != nil {
		t.Fatal(err)
	}
	if launch.Model != "local-7b" {
		t.Fatalf("model=%q", launch.Model)
	}
}
