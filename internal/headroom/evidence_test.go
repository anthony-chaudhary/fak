package headroom

import (
	"strings"
	"testing"
)

func validLiveEvidence() LiveComparisonEvidence {
	metrics := LiveArmMetrics{
		TaskSuccess: 1, MetricFactRecall: 1, ProviderInputTokens: 100,
		TTFTMilliseconds: 10, RegrowthTaxTokens: 0, TotalCostUSD: 0.01,
	}
	return LiveComparisonEvidence{
		Schema: "fak-headroom-live-evidence/1", Witness: "ledger://independent/run-1",
		WorkloadDigest: "sha256:abc", Model: "model-v1", Provider: "provider-v1",
		CacheState: "warm-prefix", Grader: "grader-v1",
		Arms: map[string]LiveArmMetrics{"none": metrics, NativeName: metrics},
	}
}

func TestApplyLiveEvidenceCompletesOnlyFullIndependentReadback(t *testing.T) {
	report := CompareBench([]string{"none", NativeName}, BenchCorpus())
	got, err := ApplyLiveEvidence(report, validLiveEvidence())
	if err != nil {
		t.Fatal(err)
	}
	if !got.Complete || len(got.Pending) != 0 || len(got.Measured) != len(requiredComparisonMetrics) {
		t.Fatalf("joined report=%+v", got)
	}
}

func TestApplyLiveEvidenceRejectsPartialOrInvalidReadback(t *testing.T) {
	report := CompareBench([]string{"none", NativeName}, BenchCorpus())
	for name, mutate := range map[string]func(*LiveComparisonEvidence){
		"no witness":  func(e *LiveComparisonEvidence) { e.Witness = "" },
		"missing arm": func(e *LiveComparisonEvidence) { delete(e.Arms, NativeName) },
		"invalid recall": func(e *LiveComparisonEvidence) {
			m := e.Arms[NativeName]
			m.MetricFactRecall = 1.1
			e.Arms[NativeName] = m
		},
	} {
		t.Run(name, func(t *testing.T) {
			evidence := validLiveEvidence()
			mutate(&evidence)
			got, err := ApplyLiveEvidence(report, evidence)
			if err == nil || got.Complete {
				t.Fatalf("err=%v report=%+v", err, got)
			}
			if strings.TrimSpace(err.Error()) == "" {
				t.Fatal("empty diagnostic")
			}
		})
	}
}
