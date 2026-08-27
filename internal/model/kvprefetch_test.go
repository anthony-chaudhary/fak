package model

import "testing"

func TestKVPrefetchPredictiveBoundedWaste(t *testing.T) {
	trace := []KVPrefetchCandidate{
		{BlockID: 1, Bytes: 100, PositionDistance: 1, Recency: .9, AttentionScore: .9, Needed: true, ReadyByDemand: true},
		{BlockID: 2, Bytes: 100, PositionDistance: 2, RetrievalScore: .8, AttentionScore: .7, Needed: true, ReadyByDemand: true},
		{BlockID: 3, Bytes: 100, PositionDistance: 1, Recency: .8, AttentionScore: .6, Needed: false, ReadyByDemand: true},
		{BlockID: 4, Bytes: 100, PositionDistance: 20, Needed: true, ReadyByDemand: false},
	}
	predictive, err := EvaluateKVPrefetch(KVPrefetchPredictive, trace, 300, .34, 2)
	if err != nil {
		t.Fatal(err)
	}
	demand, _ := EvaluateKVPrefetch(KVPrefetchDemandOnly, trace, 0, 0, 2)
	oracle, _ := EvaluateKVPrefetch(KVPrefetchOracle, trace, 300, 0, 2)
	if predictive.Engine != "fak-native" || predictive.WasteRatio > .34 || predictive.UsefulBytes != 200 || predictive.PollutionBytes != 100 || predictive.FaultBytes != 100 {
		t.Fatalf("predictive=%+v", predictive)
	}
	if demand.FaultBytes != 300 || demand.FetchedBytes != 0 {
		t.Fatalf("demand=%+v", demand)
	}
	if oracle.PollutionBytes != 0 || oracle.UsefulBytes != 300 || oracle.FaultBytes != 100 {
		t.Fatalf("oracle=%+v", oracle)
	}
}

func TestKVPrefetchAdversarialTraceFallsBackWithinBudget(t *testing.T) {
	trace := []KVPrefetchCandidate{{BlockID: 1, Bytes: 128, AttentionScore: 1, Needed: false, ReadyByDemand: true}, {BlockID: 2, Bytes: 128, AttentionScore: .9, Needed: false, ReadyByDemand: true}, {BlockID: 3, Bytes: 128, Needed: true, ReadyByDemand: true}}
	r, err := EvaluateKVPrefetch(KVPrefetchPredictive, trace, 256, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.FetchedBytes != 128 || r.PollutionBytes != 0 || r.FaultBytes != 0 || len(r.SelectedBlocks) != 1 || r.SelectedBlocks[0] != 3 {
		t.Fatalf("bounded adversarial receipt=%+v", r)
	}
	if _, err := EvaluateKVPrefetch("guess", trace, 1, 0, 1); err == nil {
		t.Fatal("unknown policy accepted")
	}
}
