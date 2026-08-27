package nativeperf

import (
	"testing"
)

func TestSummarizeRepetitionsAllAndCleanOnly(t *testing.T) {
	reps := []Repetition{{TokensPerSecond: 10}, {TokensPerSecond: 100}, {TokensPerSecond: 20}, {TokensPerSecond: 999}}
	evidence := []AmbientEvidence{
		sealedAmbient(t, AmbientClean),
		sealedAmbient(t, AmbientInvestigate),
		sealedAmbient(t, AmbientClean),
		sealedAmbient(t, AmbientInvalid),
	}
	got := SummarizeRepetitions(reps, evidence)
	if got.AllSample.Included != 4 || got.AllSample.Total != 4 || got.AllSample.Median != 60 || got.AllSample.Mean != 282.25 {
		t.Fatalf("all=%+v", got.AllSample)
	}
	if got.CleanOnly.Included != 2 || got.CleanOnly.Total != 4 || got.CleanOnly.Mean != 15 || got.CleanOnly.Median != 15 || got.CleanOnly.StdDev != 5 {
		t.Fatalf("clean=%+v", got.CleanOnly)
	}
	if len(got.Exclusions) != 2 || got.Exclusions[0].Reason != "investigate" || got.Exclusions[1].Reason != "invalid" || got.Exclusions[0].AttestationDigest == "" {
		t.Fatalf("exclusions=%+v", got.Exclusions)
	}
}

func TestSummarizeRepetitionsMissingEvidenceIsExcluded(t *testing.T) {
	got := SummarizeRepetitions([]Repetition{{TokensPerSecond: 1}}, nil)
	if got.CleanOnly.Included != 0 || len(got.Exclusions) != 1 || got.Exclusions[0].Reason != "missing" || got.Exclusions[0].AttestationDigest != "" {
		t.Fatalf("summary=%+v", got)
	}
	if err := requireCleanSamples("candidate", got, 1); err == nil {
		t.Fatal("insufficient clean samples passed")
	}
}

func sealedAmbient(t *testing.T, verdict AmbientVerdict) AmbientEvidence {
	t.Helper()
	e := AmbientEvidence{Schema: AmbientEvidenceSchema, StartedAt: "2026-08-26T00:00:00Z", EndedAt: "2026-08-26T00:00:01Z", ElapsedMilliseconds: 1000, SampleIntervalMilliseconds: 100, Source: "test", Platform: "linux", HostCPUPercent: AmbientMetric{Availability: MetricMeasured, Value: 1}, AvailableMemoryBytes: AmbientMetric{Availability: MetricMeasured, Value: 1}, ProcessChurn: AmbientMetric{Availability: MetricMeasured, Value: 0}, NonSUTCPUPercent: AmbientMetric{Availability: MetricMeasured, Value: 0}, Verdict: verdict}
	if err := SealAmbientEvidence(&e); err != nil {
		t.Fatal(err)
	}
	return e
}
