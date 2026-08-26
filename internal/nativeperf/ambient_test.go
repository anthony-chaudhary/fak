package nativeperf

import (
	"strings"
	"testing"
)

func testAmbient(t *testing.T, verdict AmbientVerdict) AmbientEvidence {
	t.Helper()
	e := AmbientEvidence{
		Schema: AmbientEvidenceSchema, StartedAt: "2026-08-26T00:00:00Z", EndedAt: "2026-08-26T00:00:01Z",
		ElapsedMilliseconds: 1000, SampleIntervalMilliseconds: 100, Source: "procfs", Platform: "linux",
		SamplerOverheadMilliseconds: 1, HostCPUPercent: AmbientMetric{MetricMeasured, 20},
		AvailableMemoryBytes: AmbientMetric{MetricMeasured, 1 << 30}, ProcessChurn: AmbientMetric{MetricMeasured, 0},
		NonSUTCPUPercent: AmbientMetric{MetricMeasured, 2}, CommandExitCode: 0, Verdict: verdict,
	}
	if err := SealAmbientEvidence(&e); err != nil {
		t.Fatal(err)
	}
	return e
}

func addAmbient(t *testing.T, r *ExperimentReceipt, verdict AmbientVerdict) {
	t.Helper()
	r.AmbientEvidence = make([]AmbientEvidence, len(r.Repetitions))
	for i := range r.AmbientEvidence {
		r.AmbientEvidence[i] = testAmbient(t, verdict)
	}
}

func TestAmbientEvidenceDigestAndAlignment(t *testing.T) {
	r := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	addAmbient(t, &r, AmbientClean)
	if err := ValidateReceipt(ActiveGraph(), r); err != nil {
		t.Fatal(err)
	}
	r.AmbientEvidence[0].NonSUTCPUPercent.Value++
	if err := ValidateReceipt(ActiveGraph(), r); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("tamper err=%v", err)
	}
	addAmbient(t, &r, AmbientClean)
	r.AmbientEvidence = r.AmbientEvidence[:len(r.AmbientEvidence)-1]
	if err := ValidateReceipt(ActiveGraph(), r); err == nil || !strings.Contains(err.Error(), "align 1:1") {
		t.Fatalf("alignment err=%v", err)
	}
}

func TestAmbientEvidenceMissingAxisIsExplicit(t *testing.T) {
	e := testAmbient(t, AmbientInvalid)
	e.HostCPUPercent = AmbientMetric{Availability: MetricUnavailable}
	if err := SealAmbientEvidence(&e); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAmbientEvidence(e); err != nil {
		t.Fatal(err)
	}
	e.HostCPUPercent = AmbientMetric{Availability: MetricUnavailable, Value: 1}
	if err := SealAmbientEvidence(&e); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAmbientEvidence(e); err == nil || !strings.Contains(err.Error(), "cannot carry a value") {
		t.Fatalf("err=%v", err)
	}
}
