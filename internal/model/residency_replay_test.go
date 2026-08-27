package model

import "testing"

func TestReplayResidencyReusesBaseAndBoundsChurn(t *testing.T) {
	arrivals := []ResidencyArrival{{"a", "qwen38", "adapter-a", "q4", "hot", 1000, 100, 10, 5}, {"b", "qwen38", "adapter-b", "q4", "hot", 1000, 100, 10, 7}, {"a", "qwen38", "adapter-a", "q4", "hot", 1000, 100, 10, 3}, {"c", "qwen38", "adapter-c", "q4", "hot", 1000, 100, 10, 9}}
	r, err := ReplayResidency(arrivals, 1200)
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || r.BaseLoads != 1 || r.DeltaLoads != 3 || r.EvictionBytes != 100 || r.PeakResidentBytes != 1200 || r.AcceptedTokens != 40 || r.ReloadBytesPerAccepted != 32.5 || r.MaxQueueNanoseconds != 9 {
		t.Fatalf("receipt=%+v", r)
	}
	if r.TenantRequests["a"] != 2 {
		t.Fatalf("fairness counts=%v", r.TenantRequests)
	}
}
func TestReplayResidencyRefusesOversizeEnvelope(t *testing.T) {
	_, err := ReplayResidency([]ResidencyArrival{{Tenant: "a", Base: "x", BaseBytes: 100, DeltaBytes: 1}}, 100)
	if err == nil {
		t.Fatal("oversize arrival admitted")
	}
}
