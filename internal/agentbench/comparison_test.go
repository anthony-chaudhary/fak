package agentbench

import "testing"

func TestAgentBenchFivePairedComparison(t *testing.T) {
	qualified := normalQualification{Qualified: true}
	arm := func(id string, delta int64) ComparisonInput {
		input := ComparisonInput{ArmID: id, ManifestDigest: "manifest", ModelID: "fixture-model", TokenizerID: "tokenizer", RendererID: "renderer", LoadDigest: "load", CachePreconditionDigest: "precondition", QualifiedConcurrency: 4}
		for pair := 0; pair < 5; pair++ {
			input.Replicates = append(input.Replicates, ComparisonReplicate{Pair: pair + 1, Seed: 0xA63E1001 + uint64(pair), Order: pair % 2, SharedC2: qualified, OwnCapacity: qualified, SharedC2TasksAccepted: true, OwnCapacityTasksAccepted: true, BootstrapMillis: 1000 + delta, MetricMillis: map[string]int64{"accepted_task_ms": 100 + delta}})
		}
		return input
	}
	a, b := arm("arm-a", 0), arm("arm-b", 20)
	result, err := ComparePaired(a, b)
	if err != nil || result.Status == "INCONCLUSIVE" || result.Pairs != 5 || len(result.ArmOrder) != 5 || result.MetricCohort != "shared-c2" || result.OwnQualifiedConcurrency["arm-a"] != 4 || result.OwnQualifiedConcurrency["arm-b"] != 4 {
		t.Fatalf("valid five-pair comparison = %+v err=%v", result, err)
	}
	metric, ok := result.Metrics["accepted_task_ms"]
	if !ok || metric.Pairs != 5 || metric.Confidence != .95 || metric.BootstrapLowMillis*metric.BootstrapHighMillis <= 0 {
		t.Fatalf("paired deterministic 95%% interval = %+v present=%v", metric, ok)
	}

	for name, mutate := range map[string]func(*ComparisonInput){
		"one pair":             func(v *ComparisonInput) { v.Replicates = v.Replicates[:1] },
		"manifest mismatch":    func(v *ComparisonInput) { v.ManifestDigest = "other" },
		"load mismatch":        func(v *ComparisonInput) { v.LoadDigest = "other" },
		"cache mismatch":       func(v *ComparisonInput) { v.CachePreconditionDigest = "other" },
		"nonalternating order": func(v *ComparisonInput) { v.Replicates[1].Order = v.Replicates[0].Order },
		"unqualified shared C2": func(v *ComparisonInput) {
			v.Replicates[2].SharedC2.Qualified = false
		},
		"unqualified own capacity": func(v *ComparisonInput) {
			v.Replicates[2].OwnCapacity.Qualified = false
		},
		"shared C2 tasks not accepted": func(v *ComparisonInput) { v.Replicates[2].SharedC2TasksAccepted = false },
		"own C tasks not accepted":     func(v *ComparisonInput) { v.Replicates[2].OwnCapacityTasksAccepted = false },
	} {
		t.Run(name, func(t *testing.T) {
			changed := b
			changed.Replicates = append([]ComparisonReplicate(nil), b.Replicates...)
			mutate(&changed)
			got, err := ComparePaired(a, changed)
			if err != nil || got.Status != "INCONCLUSIVE" || got.Reason == "" {
				t.Fatalf("non-comparable input = %+v err=%v", got, err)
			}
		})
	}
	differentCapacity := b
	differentCapacity.QualifiedConcurrency = 8
	if got, err := ComparePaired(a, differentCapacity); err != nil || got.Status == "INCONCLUSIVE" || got.OwnQualifiedConcurrency["arm-a"] != 4 || got.OwnQualifiedConcurrency["arm-b"] != 8 || got.MetricCohort != "shared-c2" {
		t.Fatalf("different own capacities invalidated shared-C2 comparison or were compared as peers: %+v err=%v", got, err)
	}
	missingIdentity := b
	missingIdentity.TokenizerID = ""
	if got, err := ComparePaired(a, missingIdentity); err != nil || got.Status != "INCONCLUSIVE" {
		t.Fatalf("unknown required identity compared: %+v err=%v", got, err)
	}

	duplicate := a
	duplicate.ArmID = b.ArmID
	if _, err := ComparePaired(duplicate, b); err == nil {
		t.Fatal("duplicate arm identity was not a structural error")
	}
	negative := b
	negative.Replicates = append([]ComparisonReplicate(nil), b.Replicates...)
	negative.Replicates[0].BootstrapMillis = -1
	if _, err := ComparePaired(a, negative); err == nil {
		t.Fatal("negative timing was not a structural error")
	}
}
