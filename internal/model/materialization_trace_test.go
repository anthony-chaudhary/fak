package model

import "testing"

func TestEliminateRedundantMaterialization(t *testing.T) {
	steps := []MaterializationStep{
		{Stage: "load", From: "q4k", To: "q4k", Bytes: 1024},
		{Stage: "prefill", From: "device-panel", To: "host-panel", Bytes: 2048, HostStaged: true},
		{Stage: "prefill", From: "host-panel", To: "device-panel", Bytes: 2048, HostStaged: true},
		{Stage: "decode", From: "q4k", To: "f32", Bytes: 4096, Required: true},
		{Stage: "verify", From: "f32", To: "f32", Bytes: 512, Required: true},
	}
	kept, r, err := EliminateRedundantMaterialization(steps, 8)
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || r.RemovedSteps != 3 || r.RemovedBytes != 5120 || r.HostStagingBytesRemoved != 4096 || r.BytesRemovedPerAccepted != 640 {
		t.Fatalf("receipt=%+v", r)
	}
	if len(kept) != 2 || kept[0].Stage != "decode" || kept[1].Stage != "verify" {
		t.Fatalf("kept=%+v", kept)
	}
}

func TestEliminateRedundantMaterializationPreservesRequiredRoundTrip(t *testing.T) {
	steps := []MaterializationStep{{Stage: "cache", From: "device", To: "host", Bytes: 10, Required: true}, {Stage: "recovery", From: "host", To: "device", Bytes: 10}}
	kept, r, err := EliminateRedundantMaterialization(steps, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(kept) != 2 || r.RemovedBytes != 0 {
		t.Fatalf("required round trip removed: kept=%+v receipt=%+v", kept, r)
	}
	if _, _, err := EliminateRedundantMaterialization([]MaterializationStep{{Bytes: -1}}, 1); err == nil {
		t.Fatal("invalid step accepted")
	}
}
