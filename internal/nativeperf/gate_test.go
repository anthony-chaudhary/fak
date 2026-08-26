package nativeperf

import (
	"strings"
	"testing"
)

func gateRequest(t *testing.T) GateRequest {
	t.Helper()
	a := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	c := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	a.Revision = "goodsha"
	c.Revision = "badsha"
	a.Quality = QualityMetric{"exact_match", 1, true}
	c.Quality = a.Quality
	a.ModuleVersions = []ModuleRevision{{"internal/model", "r10+ggood"}, {"internal/metalgemm", "r20+ggood"}}
	c.ModuleVersions = []ModuleRevision{{"internal/model", "r11+gbad"}, {"internal/metalgemm", "r20+ggood"}}
	for i := range a.Repetitions {
		a.Repetitions[i].TokensPerSecond = 100 + float64(i%2)
	}
	for i := range c.Repetitions {
		c.Repetitions[i].TokensPerSecond = 100 + float64(i%2)
	}
	return GateRequest{GateRequestSchema, GatePolicy{"fak-native-performance-gate-policy/v1", a.EnvelopeID, a.ChangedLeverID, a.Revision, 3, 2, 2, 5, 90, 1200, "exact_match", 1, true, "fak-native", a.Execution.ForwardPath, false}, a, c}
}
func TestGateNoiseBands(t *testing.T) {
	r := gateRequest(t)
	v, e := Gate(r)
	if e != nil || v.Classification != GatePass {
		t.Fatalf("pass: %+v %v", v, e)
	}
	for i := range r.Candidate.Repetitions {
		r.Candidate.Repetitions[i].TokensPerSecond = 97
	}
	v, e = Gate(r)
	if e != nil || v.Classification != GateInvestigate {
		t.Fatalf("investigate: %+v %v", v, e)
	}
	for i := range r.Candidate.Repetitions {
		r.Candidate.Repetitions[i].TokensPerSecond = 90
	}
	v, e = Gate(r)
	if e != nil || v.Classification != GateRegression || v.Bisect == nil || len(v.SuspectModules) != 1 || v.SuspectModules[0].Module != "internal/model" {
		t.Fatalf("regression: %+v %v", v, e)
	}
}
func TestGateHardFailures(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ExperimentReceipt)
	}{{"quality", func(r *ExperimentReceipt) { r.Quality.Score = .5 }}, {"fallback", func(r *ExperimentReceipt) { r.Execution.FallbackCount = 1 }}, {"memory", func(r *ExperimentReceipt) { r.Memory.PeakBytes = 2000 }}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := gateRequest(t)
			tt.edit(&r.Candidate)
			v, e := Gate(r)
			if e != nil || v.Classification != GateRegression {
				t.Fatalf("%+v %v", v, e)
			}
		})
	}
}
func TestGateRejectsIncomparableEnvelope(t *testing.T) {
	r := gateRequest(t)
	r.Policy.EnvelopeID = "other"
	_, e := Gate(r)
	if e == nil || !strings.Contains(e.Error(), "exact envelope") {
		t.Fatalf("err=%v", e)
	}
}
func TestGateSuspectRangeIsSortedAndBounded(t *testing.T) {
	r := gateRequest(t)
	r.LastAccepted.ModuleVersions = append(r.LastAccepted.ModuleVersions, ModuleRevision{"cmd/fak", "r3+ggood"})
	r.Candidate.ModuleVersions = append(r.Candidate.ModuleVersions, ModuleRevision{"cmd/fak", "r4+gbad"})
	for i := range r.Candidate.Repetitions {
		r.Candidate.Repetitions[i].TokensPerSecond = 80
	}
	v, e := Gate(r)
	if e != nil {
		t.Fatal(e)
	}
	if len(v.SuspectModules) != 2 || v.SuspectModules[0].Module != "cmd/fak" || v.Bisect.GoodRevision != "goodsha" || v.Bisect.BadRevision != "badsha" {
		t.Fatalf("%+v", v)
	}
}

func TestGateAmbientEvidencePolicy(t *testing.T) {
	clean := gateRequest(t)
	clean.Policy.RequireAmbientEvidence = true
	addAmbient(t, &clean.LastAccepted, AmbientClean)
	addAmbient(t, &clean.Candidate, AmbientClean)
	verdict, err := Gate(clean)
	if err != nil || verdict.Classification != GatePass {
		t.Fatalf("clean: %+v %v", verdict, err)
	}

	contaminated := clean
	addAmbient(t, &contaminated.Candidate, AmbientInvestigate)
	verdict, err = Gate(contaminated)
	if err != nil || verdict.Classification != GateInvestigate {
		t.Fatalf("contaminated: %+v %v", verdict, err)
	}

	unmeasured := clean
	unmeasured.Candidate.AmbientEvidence = nil
	if _, err = Gate(unmeasured); err == nil || !strings.Contains(err.Error(), "align 1:1") {
		t.Fatalf("unmeasured err=%v", err)
	}

	legacy := gateRequest(t)
	if verdict, err = Gate(legacy); err != nil || verdict.Classification != GatePass {
		t.Fatalf("legacy v1 compatibility: %+v %v", verdict, err)
	}
}
