package nativeperf

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/systembaseline"
)

func gateRequest(t *testing.T) GateRequest {
	t.Helper()
	a := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	c := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	a.Revision = "goodsha"
	c.Revision = "badsha"
	a.Quality = QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
	c.Quality = a.Quality
	a.ModuleVersions = []ModuleRevision{{"internal/model", "r10+ggood"}, {"internal/metalgemm", "r20+ggood"}}
	c.ModuleVersions = []ModuleRevision{{"internal/model", "r11+gbad"}, {"internal/metalgemm", "r20+ggood"}}
	for i := range a.Repetitions {
		a.Repetitions[i].TokensPerSecond = 100 + float64(i%2)
	}
	for i := range c.Repetitions {
		c.Repetitions[i].TokensPerSecond = 100 + float64(i%2)
	}
	return GateRequest{Schema: GateRequestSchema, Policy: GatePolicy{Schema: GatePolicySchema, EnvelopeID: a.EnvelopeID, ChangedLeverID: a.ChangedLeverID, AcceptedRevision: a.Revision, MinimumRepetitions: 3, MaximumNoisePercent: 2, InvestigateDropPercent: 2, RegressionDropPercent: 5, MinimumThroughput: 90, MaximumPeakBytes: 1200, QualityMetric: "exact_match", MinimumQualityScore: 1, QualityHigherIsBetter: true, RequiredEngine: "fak-native", RequiredForwardPath: a.Execution.ForwardPath}, LastAccepted: a, Candidate: c}
}

func baselineAttestation(verdict string, known bool) systembaseline.Report {
	m := systembaseline.Metric{Available: known, Value: 1, Unit: "percent", Source: "test"}
	r := systembaseline.Report{Schema: systembaseline.Schema, Verdict: verdict, Baseline: systembaseline.Window{StartedAtUTC: "2026-08-25T23:59:59Z", EndedAtUTC: "2026-08-26T00:00:00Z", DurationNS: 1e9, IntervalNS: 1e9, Samples: 2}, BaselineHost: systembaseline.HostTotals{CPUPercent: m}, BaselineSampler: systembaseline.SamplerOverhead{CountedSamples: 1, WallNS: 1e7, DutyPercent: m}, Window: systembaseline.Window{StartedAtUTC: "2026-08-26T00:00:00Z", EndedAtUTC: "2026-08-26T00:00:01Z", DurationNS: 1e9, IntervalNS: 1e9, Samples: 2}, CommandSampler: systembaseline.SamplerOverhead{CountedSamples: 1, WallNS: 1e7, DutyPercent: m}, Coverage: systembaseline.Coverage{SUTRootPID: 7, DescendantAttribution: "sampled_pid_ppid_tree"}, Host: systembaseline.HostTotals{CPUPercent: m}, Attribution: systembaseline.Attribution{SUTCPUPercentOfHost: m, NonSUTCPUPercentOfHost: m}, Policy: systembaseline.DefaultPolicy(), TopNonSUT: []systembaseline.Consumer{}, Findings: []systembaseline.Finding{}}
	if verdict == systembaseline.VerdictInvestigate && known {
		r.Attribution.NonSUTCPUPercentOfHost.Value = r.Policy.MaximumNonSUTCPUPercent + 1
	}
	r.Seal()
	return r
}

func attachSystemBaselines(r *ExperimentReceipt, verdict string, known bool) {
	salt := int64(0)
	if r.Role == RoleCandidate {
		salt = 100
	}
	for i := range r.Repetitions {
		attestation := baselineAttestation(verdict, known)
		attestation.Window.IntervalNS = salt + int64(i+1)
		attestation.Seal()
		r.SystemBaselines = append(r.SystemBaselines, attestation)
		r.Repetitions[i].SystemBaselineDigest = attestation.Digest
	}
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

func TestGateSystemBaselineLegacyAndStrictPolicy(t *testing.T) {
	legacy := gateRequest(t)
	if v, err := Gate(legacy); err != nil || v.Classification != GatePass {
		t.Fatalf("legacy=%+v err=%v", v, err)
	}
	recordedV1 := legacy
	recordedV1.Schema = GateRequestSchemaV1
	recordedV1.Policy.Schema = GatePolicySchemaV1
	recordedV1.LastAccepted.Schema = ReceiptSchemaV1
	recordedV1.Candidate.Schema = ReceiptSchemaV1
	if v, err := Gate(recordedV1); err != nil || v.Classification != GatePass {
		t.Fatalf("recorded v1=%+v err=%v", v, err)
	}

	strict := gateRequest(t)
	strict.Policy.RequireSystemBaseline = true
	attachSystemBaselines(&strict.LastAccepted, systembaseline.VerdictClean, true)
	attachSystemBaselines(&strict.Candidate, systembaseline.VerdictClean, true)
	if v, err := Gate(strict); err != nil || v.Classification != GateInvestigate || v.Bisect != nil {
		t.Fatalf("strict sampled without opt-in=%+v err=%v", v, err)
	}
	strict.Policy.AllowSampledSystemBaseline = true
	if v, err := Gate(strict); err != nil || v.Classification != GatePass {
		t.Fatalf("strict clean=%+v err=%v", v, err)
	}

	missing := strict
	missing.Candidate.SystemBaselines = nil
	if v, err := Gate(missing); err != nil || v.Classification != GateInvestigate || v.Bisect != nil || len(v.SuspectModules) != 0 {
		t.Fatalf("strict missing=%+v err=%v", v, err)
	}

	contaminated := strict
	contaminated.Candidate.SystemBaselines = append([]systembaseline.Report(nil), strict.Candidate.SystemBaselines...)
	contaminated.Candidate.SystemBaselines[0].Attribution.NonSUTCPUPercentOfHost.Value = contaminated.Candidate.SystemBaselines[0].Policy.MaximumNonSUTCPUPercent + 1
	contaminated.Candidate.SystemBaselines[0].Verdict = systembaseline.VerdictInvestigate
	contaminated.Candidate.SystemBaselines[0].Seal()
	if v, err := Gate(contaminated); err != nil || v.Classification != GateInvestigate {
		t.Fatalf("strict contaminated=%+v err=%v", v, err)
	}

	unknown := strict
	unknown.Candidate.SystemBaselines = append([]systembaseline.Report(nil), strict.Candidate.SystemBaselines...)
	unknown.Candidate.SystemBaselines[0].Attribution.NonSUTCPUPercentOfHost = systembaseline.Metric{Unit: "percent", Reason: "unknown"}
	unknown.Candidate.SystemBaselines[0].Verdict = systembaseline.VerdictInvestigate
	unknown.Candidate.SystemBaselines[0].Seal()
	if v, err := Gate(unknown); err != nil || v.Classification != GateInvestigate {
		t.Fatalf("strict unknown=%+v err=%v", v, err)
	}

	invalid := strict
	invalid.Candidate.SystemBaselines = append([]systembaseline.Report(nil), strict.Candidate.SystemBaselines...)
	invalid.Candidate.SystemBaselines[0].CommandExitCode = 9
	if v, err := Gate(invalid); err != nil || v.Classification != GateInvestigate {
		t.Fatalf("strict invalid=%+v err=%v", v, err)
	}

	missingBaselinePhase := strict
	missingBaselinePhase.Candidate.SystemBaselines = append([]systembaseline.Report(nil), strict.Candidate.SystemBaselines...)
	missingBaselinePhase.Candidate.SystemBaselines[0].Baseline = systembaseline.Window{}
	missingBaselinePhase.Candidate.SystemBaselines[0].Seal()
	if v, err := Gate(missingBaselinePhase); err != nil || v.Classification != GateInvestigate {
		t.Fatalf("strict no baseline phase=%+v err=%v", v, err)
	}

	hardWithMissingAmbient := missing
	hardWithMissingAmbient.Candidate.Memory.PeakBytes = 2000
	if v, err := Gate(hardWithMissingAmbient); err != nil || v.Classification != GateRegression {
		t.Fatalf("hard failure with missing ambient=%+v err=%v", v, err)
	}

	throughputWithMissingAmbient := missing
	for i := range throughputWithMissingAmbient.Candidate.Repetitions {
		throughputWithMissingAmbient.Candidate.Repetitions[i].TokensPerSecond = 50
	}
	if v, err := Gate(throughputWithMissingAmbient); err != nil || v.Classification != GateInvestigate || v.Bisect != nil || len(v.SuspectModules) != 0 {
		t.Fatalf("throughput with missing ambient=%+v err=%v", v, err)
	}

	highCardinality := strict
	highCardinality.Candidate.SystemBaselines = append([]systembaseline.Report(nil), strict.Candidate.SystemBaselines...)
	highCardinality.Candidate.SystemBaselines[0].Policy.IncludeTopConsumers = true
	highCardinality.Candidate.SystemBaselines[0].TopNonSUT = []systembaseline.Consumer{{Image: "other.exe", PID: 42, CPUAvailable: true}}
	highCardinality.Candidate.SystemBaselines[0].Seal()
	if v, err := Gate(highCardinality); err != nil || v.Classification != GateInvestigate {
		t.Fatalf("high-cardinality ambient=%+v err=%v", v, err)
	}

	crossArmReuse := strict
	crossArmReuse.Candidate.SystemBaselines = append([]systembaseline.Report(nil), strict.LastAccepted.SystemBaselines...)
	crossArmReuse.Candidate.Repetitions = append([]Repetition(nil), strict.Candidate.Repetitions...)
	for i := range crossArmReuse.Candidate.Repetitions {
		crossArmReuse.Candidate.Repetitions[i].SystemBaselineDigest = crossArmReuse.Candidate.SystemBaselines[i].Digest
	}
	if v, err := Gate(crossArmReuse); err != nil || v.Classification != GateInvestigate || v.Bisect != nil {
		t.Fatalf("cross-arm reuse=%+v err=%v", v, err)
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

func TestGateRejectsCompetingAmbientAuthorities(t *testing.T) {
	r := gateRequest(t)
	r.Policy.RequireAmbientEvidence = true
	r.Policy.RequireSystemBaseline = true
	if _, err := Gate(r); err == nil || !strings.Contains(err.Error(), "cannot require both") {
		t.Fatalf("policy err=%v", err)
	}

	r = gateRequest(t)
	r.Policy.RequireSystemBaseline = true
	r.Policy.AllowSampledSystemBaseline = true
	attachSystemBaselines(&r.LastAccepted, systembaseline.VerdictClean, true)
	attachSystemBaselines(&r.Candidate, systembaseline.VerdictClean, true)
	addAmbient(t, &r.Candidate, AmbientClean)
	verdict, err := Gate(r)
	if err != nil || verdict.Classification != GateInvestigate || verdict.Bisect != nil {
		t.Fatalf("mixed receipt: %+v %v", verdict, err)
	}
}

func TestGateUsesCleanOnlyStatisticsAndRefusesTooFewClean(t *testing.T) {
	r := gateRequest(t)
	r.Policy.RequireAmbientEvidence = true
	r.Policy.MinimumCleanRepetitions = 2
	addAmbient(t, &r.LastAccepted, AmbientClean)
	addAmbient(t, &r.Candidate, AmbientClean)
	r.LastAccepted.AmbientEvidence[1] = testAmbient(t, AmbientInvestigate)
	r.Candidate.AmbientEvidence[1] = testAmbient(t, AmbientInvestigate)
	r.LastAccepted.Repetitions[0].TokensPerSecond = 100
	r.LastAccepted.Repetitions[1].TokensPerSecond = 1
	r.LastAccepted.Repetitions[2].TokensPerSecond = 100
	r.Candidate.Repetitions[0].TokensPerSecond = 101
	r.Candidate.Repetitions[1].TokensPerSecond = 1000
	r.Candidate.Repetitions[2].TokensPerSecond = 101

	v, err := Gate(r)
	if err != nil {
		t.Fatal(err)
	}
	if v.Classification != GatePass || v.AcceptedMeanTokensPerS != 100 || v.CandidateMeanTokensPerS != 101 {
		t.Fatalf("clean-only gate=%+v", v)
	}
	if v.AcceptedSamples.AllSample.Included != 3 || v.AcceptedSamples.CleanOnly.Included != 2 || len(v.CandidateSamples.Exclusions) != 1 {
		t.Fatalf("sample provenance missing: accepted=%+v candidate=%+v", v.AcceptedSamples, v.CandidateSamples)
	}

	r.Policy.MinimumCleanRepetitions = 3
	if _, err := Gate(r); err == nil || !strings.Contains(err.Error(), "rerun with clean ambient attestations") {
		t.Fatalf("insufficient clean samples err=%v", err)
	}
}
