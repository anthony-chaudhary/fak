package benchcli

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

func TestSimulationEvidenceTypeClaimCeilingMatrix(t *testing.T) {
	types := []EvidenceType{
		EvidenceStructuralCount,
		EvidenceAnalyticalBound,
		EvidenceTraceSimulation,
		EvidenceCycleSimulation,
		EvidenceLearnedEstimate,
		EvidenceCalibratedSimulation,
		EvidenceHardwareMeasurement,
	}
	claims := []ClaimCeiling{
		ClaimCorrectnessOnly,
		ClaimBottleneckOnly,
		ClaimRelativeRank,
		ClaimAbsoluteEstimate,
		ClaimMeasuredAbsolute,
	}

	for _, evidenceType := range types {
		for _, claim := range claims {
			name := string(evidenceType) + "/" + string(claim)
			t.Run(name, func(t *testing.T) {
				ev := validSimulationEvidence(evidenceType, claim)
				err := ValidateSimulationEvidence(ev)
				wantValid := claimStrength[claim] <= claimStrength[evidenceMaxClaim[evidenceType]]
				if wantValid && err != nil {
					t.Fatalf("valid matrix cell rejected: %v", err)
				}
				if !wantValid && err == nil {
					t.Fatal("invalid promotion accepted")
				}
			})
		}
	}
}

func TestSimulationEvidenceVocabularyFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*SimulationEvidence)
		want   string
	}{
		{"unknown evidence", func(ev *SimulationEvidence) { ev.EvidenceType = "oracle" }, "unknown evidence_type"},
		{"unknown ceiling", func(ev *SimulationEvidence) { ev.ClaimCeiling = "headline" }, "unknown claim_ceiling"},
		{"wrong schema", func(ev *SimulationEvidence) { ev.Schema = "fak-simulation-evidence/99" }, "schema"},
		{"unknown engine revision", func(ev *SimulationEvidence) { ev.Engine.Revision = "unknown" }, "not blank or unknown"},
		{"case-folded unknown engine revision", func(ev *SimulationEvidence) { ev.Engine.Revision = " UNKNOWN " }, "not blank or unknown"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ev := validSimulationEvidence(EvidenceAnalyticalBound, ClaimBottleneckOnly)
			test.mutate(&ev)
			err := ValidateSimulationEvidence(ev)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ValidateSimulationEvidence error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestSimulationEvidenceJSONRoundTripPreservesEveryFieldAndDigest(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCalibratedSimulation, ClaimAbsoluteEstimate)
	ev.Trace = validTrace()
	ev.LearnedModel = &LearnedModelInfo{
		TrainingSetDigest:      SHA256Digest([]byte("training-corpus")),
		FeatureSchema:          "features/v4",
		Compiler:               "nvcc 13.0",
		Target:                 "sm_90",
		AbstainOutsideEnvelope: true,
	}
	ev.Replay.Stochastic = true
	ev.Replay.Repetitions = 8
	ev.Replay.IndependentStreams = 4
	ev.Replay.Uncertainty = "95% bootstrap CI"
	ev.ExcludedEffects = []string{"network jitter", "thermal throttling"}
	ev.Cost = SimulationCost{HostWallTimeMS: 1234.5, HostCPUTimeMS: 987.25, Bytes: 456789}

	t.Setenv("FAK_BENCH_UTC", "2026-08-27T12:34:56Z")
	t.Setenv("FAK_BENCH_COMMIT", "0123456789abcdef")
	t.Setenv("FAK_BENCH_NODE", "sim-host")
	raw, err := MarshalReportWithEvidence(map[string]any{
		"headline": "PROJECTED bottleneck estimate",
		"results":  map[string]any{"latency_ms": 10.0},
	}, ev)
	if err != nil {
		t.Fatalf("MarshalReportWithEvidence: %v", err)
	}
	art, ok := DecodeArtifact(raw)
	if !ok || art.SimulationEvidence == nil {
		t.Fatalf("DecodeArtifact did not preserve simulation block:\n%s", raw)
	}
	if !reflect.DeepEqual(*art.SimulationEvidence, ev) {
		got, _ := json.MarshalIndent(art.SimulationEvidence, "", "  ")
		want, _ := json.MarshalIndent(ev, "", "  ")
		t.Fatalf("round trip changed evidence\ngot: %s\nwant: %s", got, want)
	}
	for _, digest := range []string{
		ev.Engine.ConfigDigest,
		ev.Workload.Digest,
		ev.Trace.Digest,
		ev.Trace.InputDigest,
		ev.LearnedModel.TrainingSetDigest,
		ev.Calibration.Digest,
		ev.Calibration.HardwareProfileDigest,
	} {
		if !strings.Contains(string(raw), digest) {
			t.Fatalf("rendered artifact lost digest %q", digest)
		}
	}
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("round-tripped artifact does not validate: %v", err)
	}
}

func TestGradeSimulationCalibrationUsesPredictionCalibrationSemantics(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCalibratedSimulation, ClaimAbsoluteEstimate)
	cal := *ev.Calibration
	cal.Predicted = 10
	cal.Measured = 10.5
	cal.LowerIsBetter = true
	cal.ErrorBand = 0.10

	got := GradeSimulationCalibration(cal)
	want := dojo.Score("independent-holdout", dojo.Prediction{
		Lever: cal.Profile, Metric: cal.Metric, Claimed: cal.Predicted,
		Unit: cal.Unit, Basis: cal.Revision, LowerIsBetter: true,
	}, dojo.Outcome{
		Realized: cal.Measured, Provenance: dojo.Observed,
		Source: cal.MeasurementSource, Measured: true, Sample: cal.Sample,
	}, dojo.DefaultCalibBand())
	if got.Verdict != want.Verdict || got.Grade != want.Grade ||
		got.Residual != want.Residual || got.NormalizedError != want.CalibErr {
		t.Fatalf("grade = %+v, want dojo episode %+v", got, want)
	}
	if got.Verdict != CalibrationCalibrated || got.Grade != "A" {
		t.Fatalf("held-out 5%% residual = %s/%s, want CALIBRATED/A", got.Verdict, got.Grade)
	}

	cal.IndependentHoldout = false
	insufficient := GradeSimulationCalibration(cal)
	if insufficient.Verdict != CalibrationInsufficient || insufficient.Grade != "n/a" {
		t.Fatalf("non-independent measurement = %s/%s, want INSUFFICIENT/n/a", insufficient.Verdict, insufficient.Grade)
	}
	ev.Calibration = &insufficient
	if err := ValidateSimulationEvidence(ev); err == nil || !strings.Contains(err.Error(), "independent held-out") {
		t.Fatalf("calibrated_sim accepted non-independent calibration: %v", err)
	}
}

func TestTraceProvenanceRequiresCompleteIdentityAndInvalidationTriggers(t *testing.T) {
	ev := validSimulationEvidence(EvidenceTraceSimulation, ClaimRelativeRank)
	if err := ValidateSimulationEvidence(ev); err != nil {
		t.Fatalf("complete trace rejected: %v", err)
	}

	ev.Trace.CaptureGPU = ""
	ev.Trace.InvalidatedBy = []string{TraceInvalidationControlFlow}
	err := ValidateSimulationEvidence(ev)
	for _, want := range []string{"capture GPU", "trace invalidation triggers missing"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("incomplete trace error = %v, want %q", err, want)
		}
	}
}

func TestLearnedEstimateRequiresIdentityAndOutOfEnvelopeAbstention(t *testing.T) {
	ev := validSimulationEvidence(EvidenceLearnedEstimate, ClaimRelativeRank)
	ev.LearnedModel.FeatureSchema = ""
	ev.LearnedModel.AbstainOutsideEnvelope = false
	err := ValidateSimulationEvidence(ev)
	for _, want := range []string{"feature schema", "must abstain"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("learned identity error = %v, want %q", err, want)
		}
	}
}

func TestCalibrationRejectsDriftAndSelfAuthoredGrade(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCalibratedSimulation, ClaimAbsoluteEstimate)
	ev.Calibration.EngineRevision = "different-engine"
	ev.Calibration.Verdict = CalibrationUnderClaim
	err := ValidateSimulationEvidence(ev)
	for _, want := range []string{"drift identity", "grade mismatch"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("calibration error = %v, want %q", err, want)
		}
	}
}

func TestBenchmarkArtifactTailClaimsRequireIndependentReplication(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCycleSimulation, ClaimRelativeRank)
	art := BenchmarkArtifact{
		SimulationEvidence: &ev,
		Results: ResultSnapshot{
			Metrics:       map[string]any{"candidate_rank_p99": 1.0},
			ClaimLanguage: []string{"PROJECTED same-envelope relative rank"},
		},
	}
	err := ValidateBenchmarkArtifact(art)
	if err == nil || !strings.Contains(err.Error(), "stochastic/tail") {
		t.Fatalf("single-stream p99 accepted: %v", err)
	}

	art.SimulationEvidence.Replay.Repetitions = 10
	art.SimulationEvidence.Replay.IndependentStreams = 5
	art.SimulationEvidence.Replay.Uncertainty = "99% bootstrap CI"
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("replicated p99 rejected: %v", err)
	}
}

func TestBenchmarkArtifactRejectsSimulatedCompetitiveClaim(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCalibratedSimulation, ClaimAbsoluteEstimate)
	art := BenchmarkArtifact{
		SimulationEvidence: &ev,
		Results: ResultSnapshot{
			Metrics:  map[string]any{"throughput_tokens_per_second": 42.0},
			Baseline: map[string]any{"engine": "llama.cpp", "speedup": 1.2},
		},
	}
	if err := ValidateBenchmarkArtifact(art); err == nil || !strings.Contains(err.Error(), "competitive") {
		t.Fatalf("simulated competitive claim accepted: %v", err)
	}

	hw := validSimulationEvidence(EvidenceHardwareMeasurement, ClaimMeasuredAbsolute)
	art.SimulationEvidence = &hw
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("hardware competitive measurement rejected: %v", err)
	}
}

func TestBenchmarkArtifactStructuralCountRejectsPerformanceMetrics(t *testing.T) {
	blocked := []string{
		"elapsed_time_ms", "decode_latency_ms", "achieved_throughput_tokens_s", "energy_joules", "cost_usd",
	}
	for _, metric := range blocked {
		t.Run(metric, func(t *testing.T) {
			ev := validSimulationEvidence(EvidenceStructuralCount, ClaimCorrectnessOnly)
			art := BenchmarkArtifact{
				SimulationEvidence: &ev,
				Results: ResultSnapshot{
					Metrics:       map[string]any{metric: 1.0},
					ClaimLanguage: []string{"PROJECTED"},
				},
			}
			err := ValidateBenchmarkArtifact(art)
			if err == nil || !strings.Contains(err.Error(), "structural_count") {
				t.Fatalf("structural performance metric accepted: %v", err)
			}
		})
	}
}

func TestBenchmarkArtifactAnalyticalMetricsMustStayExplicitlyProjected(t *testing.T) {
	ev := validSimulationEvidence(EvidenceAnalyticalBound, ClaimBottleneckOnly)
	art := BenchmarkArtifact{
		SimulationEvidence: &ev,
		Results: ResultSnapshot{Metrics: map[string]any{
			"latency_ms": 10.0,
		}},
	}
	if err := ValidateBenchmarkArtifact(art); err == nil || !strings.Contains(err.Error(), "explicit projected") {
		t.Fatalf("unmarked analytical latency accepted: %v", err)
	}

	// This is the coalescebench shape: every result is globally PROJECTED, so
	// even the explicitly optimistic uncoalesced baseline remains a bound rather
	// than being mistaken for achieved throughput.
	art.Results = ResultSnapshot{
		Metrics: map[string]any{
			"results.projected_agent_seconds":         0.25,
			"results.projected_net_tokens_s":          16.0,
			"results.uncoalesced_shared_ssd_tokens_s": 8.0,
		},
		ClaimLanguage: []string{"PROJECTED"},
	}
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("explicitly projected analytical metrics rejected: %v", err)
	}
}

func TestBenchmarkArtifactNonHardwareRejectsMeasuredOrAchievedLanguage(t *testing.T) {
	for _, test := range []struct {
		name    string
		metrics map[string]any
		claims  []string
	}{
		{"measured metric", map[string]any{"measured_latency_ms": 7.0}, []string{"PROJECTED"}},
		{"achieved headline", map[string]any{"projected_latency_ms": 7.0}, []string{"ACHIEVED"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ev := validSimulationEvidence(EvidenceCalibratedSimulation, ClaimAbsoluteEstimate)
			art := BenchmarkArtifact{SimulationEvidence: &ev, Results: ResultSnapshot{Metrics: test.metrics, ClaimLanguage: test.claims}}
			if err := ValidateBenchmarkArtifact(art); err == nil || !strings.Contains(err.Error(), "measured or achieved") {
				t.Fatalf("positive hardware language accepted: %v", err)
			}
		})
	}

	// An honest negative fence is not a positive measurement claim.
	ev := validSimulationEvidence(EvidenceAnalyticalBound, ClaimBottleneckOnly)
	art := BenchmarkArtifact{
		SimulationEvidence: &ev,
		Results: ResultSnapshot{
			Metrics:       map[string]any{"projected_latency_ms": 7.0},
			ClaimLanguage: []string{"PROJECTED — not measured or achieved"},
		},
	}
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("negative measurement fence rejected: %v", err)
	}
}

func TestBenchmarkArtifactCalibratedAbsoluteEstimateAdmitsAbsoluteMetric(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCalibratedSimulation, ClaimAbsoluteEstimate)
	art := BenchmarkArtifact{
		SimulationEvidence: &ev,
		Results:            ResultSnapshot{Metrics: map[string]any{"latency_ms": 10.5}},
	}
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("calibrated absolute estimate rejected: %v", err)
	}
}

func TestRelativeRankEvidenceCannotEmitProjectedAbsolutePerformance(t *testing.T) {
	for _, evidenceType := range []EvidenceType{
		EvidenceTraceSimulation,
		EvidenceCycleSimulation,
		EvidenceLearnedEstimate,
		EvidenceCalibratedSimulation,
	} {
		t.Run(string(evidenceType), func(t *testing.T) {
			ev := validSimulationEvidence(evidenceType, ClaimRelativeRank)
			art := BenchmarkArtifact{
				SimulationEvidence: &ev,
				Results: ResultSnapshot{
					Metrics:       map[string]any{"projected_latency_ms": 10.5},
					ClaimLanguage: []string{"PROJECTED same-envelope relative rank"},
				},
			}
			err := ValidateBenchmarkArtifact(art)
			if err == nil || (!strings.Contains(err.Error(), "cannot emit absolute") && !strings.Contains(err.Error(), "require absolute_estimate")) {
				t.Fatalf("%s relative_rank emitted projected absolute latency: %v", evidenceType, err)
			}
		})
	}
}

func TestMarshalReportWithEvidenceRunsArtifactLevelGate(t *testing.T) {
	ev := validSimulationEvidence(EvidenceCycleSimulation, ClaimRelativeRank)
	report := map[string]any{
		"label": "PROJECTED",
		"results": map[string]any{
			"latency_p99_ms": 12.5,
		},
	}
	if _, err := MarshalReportWithEvidence(report, ev); err == nil || !strings.Contains(err.Error(), "stochastic/tail") {
		t.Fatalf("serialization seam emitted single-stream p99: %v", err)
	}

	report = map[string]any{
		"label":    "PROJECTED",
		"results":  map[string]any{"projected_throughput_tokens_s": 50.0},
		"baseline": map[string]any{"engine": "llama.cpp"},
	}
	if _, err := MarshalReportWithEvidence(report, ev); err == nil || !strings.Contains(err.Error(), "competitive") {
		t.Fatalf("serialization seam emitted simulated competitive result: %v", err)
	}
}

func TestMarshalReportWithEvidenceScansHeadlineAndMetricStringValues(t *testing.T) {
	ev := validSimulationEvidence(EvidenceAnalyticalBound, ClaimBottleneckOnly)
	for _, headline := range []string{
		"PROJECTED native faster",
		"PROJECTED comparison with llama.cpp",
	} {
		competitive := map[string]any{
			"headline": headline,
			"results":  map[string]any{"projected_throughput_tokens_s": 50.0},
		}
		if _, err := MarshalReportWithEvidence(competitive, ev); err == nil || !strings.Contains(err.Error(), "competitive") {
			t.Fatalf("headline %q escaped competitive serialization gate: %v", headline, err)
		}
	}

	measuredValue := map[string]any{
		"label": "PROJECTED",
		"results": map[string]any{
			"projected_throughput_tokens_s": 50.0,
			"status":                        "MEASURED",
		},
	}
	if _, err := MarshalReportWithEvidence(measuredValue, ev); err == nil || !strings.Contains(err.Error(), "measured or achieved") {
		t.Fatalf("metric string value MEASURED escaped serialization gate: %v", err)
	}

	measuredValue["results"].(map[string]any)["status"] = "NOT MEASURED"
	if _, err := MarshalReportWithEvidence(measuredValue, ev); err != nil {
		t.Fatalf("negative not-measured fence was treated as measured: %v", err)
	}
}

func TestHardwareMeasurementMayEmitMeasuredAbsolutePerformance(t *testing.T) {
	ev := validSimulationEvidence(EvidenceHardwareMeasurement, ClaimMeasuredAbsolute)
	art := BenchmarkArtifact{
		SimulationEvidence: &ev,
		Results: ResultSnapshot{
			Metrics:       map[string]any{"measured_latency_ms": 8.25},
			ClaimLanguage: []string{"ACHIEVED ON HARDWARE"},
		},
	}
	if err := ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("hardware measured absolute result rejected: %v", err)
	}
}

func TestDecodeArtifactMalformedExplicitEnvelopeDoesNotFallBackToLineage(t *testing.T) {
	lineage := map[string]any{
		"lineage_schema": LineageSchema,
		"app_version":    "1.0.0",
		"utc":            "2026-08-27T00:00:00Z",
		"git_commit":     "deadbeef",
		"go_version":     "go1.26.0",
		"node":           "test-node",
	}
	for _, explicit := range []any{
		"not-an-envelope",
		map[string]any{
			"schema": BenchmarkArtifactSchema,
			"run_id": "malformed-evidence",
			"simulation_evidence": map[string]any{
				"schema":        SimulationEvidenceSchema,
				"evidence_type": "oracle",
				"claim_ceiling": ClaimRelativeRank,
			},
		},
	} {
		raw, err := json.Marshal(map[string]any{
			"benchmark_artifact": explicit,
			"lineage":            lineage,
		})
		if err != nil {
			t.Fatalf("marshal fixture: %v", err)
		}
		if art, ok := DecodeArtifact(raw); ok {
			t.Fatalf("malformed explicit envelope fell back to lineage: %+v", art)
		}
	}
}

func TestMarshalReportWithEvidenceFailsClosedAndDefaultsSchema(t *testing.T) {
	ev := validSimulationEvidence(EvidenceStructuralCount, ClaimCorrectnessOnly)
	ev.Schema = ""
	raw, err := MarshalReportWithEvidence(map[string]int{"count": 7}, ev)
	if err != nil {
		t.Fatalf("schema default rejected: %v", err)
	}
	art, ok := DecodeArtifact(raw)
	if !ok || art.SimulationEvidence == nil || art.SimulationEvidence.Schema != SimulationEvidenceSchema {
		t.Fatalf("schema was not defaulted in artifact: %+v", art.SimulationEvidence)
	}

	ev.ClaimCeiling = ClaimMeasuredAbsolute
	if _, err := MarshalReportWithEvidence(map[string]int{"count": 7}, ev); err == nil {
		t.Fatal("MarshalReportWithEvidence emitted an invalid promotion")
	}
}

func TestMarshalReportWithEvidenceRefusesShapesThatCannotCarryEvidence(t *testing.T) {
	ev := validSimulationEvidence(EvidenceStructuralCount, ClaimCorrectnessOnly)
	for _, test := range []struct {
		name   string
		report any
	}{
		{"compact object bytes", []byte(`{"count":7}`)},
		{"array", []byte("[\n  1\n]")},
		{"scalar", json.RawMessage("42")},
	} {
		t.Run(test.name, func(t *testing.T) {
			if raw, err := MarshalReportWithEvidence(test.report, ev); err == nil || !strings.Contains(err.Error(), "cannot embed the exact simulation evidence") {
				t.Fatalf("unembeddable report succeeded: err=%v raw=%s", err, raw)
			}
		})
	}

	// The additive helper becomes strict, but the legacy helper keeps its
	// documented fail-soft/no-op behavior for compact and non-object reports.
	compact := []byte(`{"count":7}`)
	raw, err := MarshalReport(compact)
	if err != nil || string(raw) != string(compact) {
		t.Fatalf("legacy MarshalReport semantics changed: err=%v raw=%s", err, raw)
	}
}

func TestSHA256DigestCanonicalAndValidatorStrict(t *testing.T) {
	digest := SHA256Digest([]byte("config"))
	if len(digest) != len("sha256:")+64 || !strings.HasPrefix(digest, "sha256:") || !validSHA256Digest(digest) {
		t.Fatalf("SHA256Digest = %q, want canonical sha256:<64 hex>", digest)
	}
	for _, bad := range []string{"", strings.TrimPrefix(digest, "sha256:"), "sha256:xyz", digest + "00"} {
		if validSHA256Digest(bad) {
			t.Fatalf("invalid digest accepted: %q", bad)
		}
	}
}

func validSimulationEvidence(evidenceType EvidenceType, claim ClaimCeiling) SimulationEvidence {
	seed := int64(9424)
	ev := SimulationEvidence{
		Schema:       SimulationEvidenceSchema,
		EvidenceType: evidenceType,
		ClaimCeiling: claim,
		Engine: SimulationEngine{
			Name:         "fak-deterministic-model",
			Revision:     "r17",
			ConfigDigest: SHA256Digest([]byte("engine-config")),
			Toolchain:    "go1.26.0",
		},
		Workload: WorkloadProvenance{
			Name:   "qwen3.8-decode-shape",
			Source: "testdata/workload.json",
			Digest: SHA256Digest([]byte("workload")),
		},
		ValidityEnvelope: ValidityEnvelope{
			Description: "same model shape, compiler, target, and batch",
			Dimensions: map[string]string{
				"model": "qwen3.8", "target": "sm_90", "batch": "1",
			},
		},
		ExcludedEffects: []string{"host scheduling", "thermal throttling"},
		Replay: ReplaySpec{
			Seed: &seed, Stream: "candidate-order-v1", Repetitions: 1, IndependentStreams: 1,
		},
		Cost: SimulationCost{HostWallTimeMS: 3.5, HostCPUTimeMS: 2.25, Bytes: 4096},
	}
	switch evidenceType {
	case EvidenceStructuralCount, EvidenceAnalyticalBound, EvidenceCycleSimulation, EvidenceHardwareMeasurement:
		// These producers require no type-specific provenance block.
	case EvidenceTraceSimulation:
		ev.Trace = validTrace()
	case EvidenceLearnedEstimate:
		ev.LearnedModel = &LearnedModelInfo{
			TrainingSetDigest:      SHA256Digest([]byte("training")),
			FeatureSchema:          "features/v3",
			Compiler:               "nvcc 13.0",
			Target:                 "sm_90",
			AbstainOutsideEnvelope: true,
		}
	case EvidenceCalibratedSimulation:
		cal := SimulationCalibration{
			Profile:               "h100-qwen38-decode",
			Revision:              "r3",
			Digest:                SHA256Digest([]byte("calibration-profile")),
			Hardware:              "NVIDIA H100 SXM",
			HardwareProfile:       "h100-sxm-cuda13",
			HardwareProfileDigest: SHA256Digest([]byte("hardware-profile")),
			Date:                  "2026-08-27",
			IndependentHoldout:    true,
			MeasurementSource:     "lab/heldout/run-17.json",
			Metric:                "latency_ms",
			Unit:                  "ms",
			Sample:                40,
			Predicted:             10,
			Measured:              10.5,
			LowerIsBetter:         true,
			ErrorBand:             0.10,
			EngineRevision:        ev.Engine.Revision,
			EngineConfigDigest:    ev.Engine.ConfigDigest,
			Toolchain:             ev.Engine.Toolchain,
			WorkloadDigest:        ev.Workload.Digest,
		}
		cal = GradeSimulationCalibration(cal)
		ev.Calibration = &cal
	}
	return ev
}

func validTrace() *TraceProvenance {
	return &TraceProvenance{
		Artifact:        "traces/qwen38-decode.trace",
		Digest:          SHA256Digest([]byte("trace")),
		CaptureGPU:      "NVIDIA H100 SXM",
		Compiler:        "nvcc 13.0",
		Libraries:       []string{"CUDA 13.0", "cuBLAS 13.0"},
		KernelSelection: "fak-native qwen3.8 decode kernel r4",
		Input:           "fixtures/qwen38-decode.json",
		InputDigest:     SHA256Digest([]byte("input")),
		InvalidatedBy: []string{
			TraceInvalidationControlFlow,
			TraceInvalidationISA,
			TraceInvalidationSynchronization,
			TraceInvalidationAllocation,
			TraceInvalidationKernelChoice,
		},
	}
}
