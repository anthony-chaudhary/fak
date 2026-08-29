package webbench

import (
	"reflect"
	"strings"
	"testing"
)

func TestServingWorkloadDigestPinsOrderedRequestShape(t *testing.T) {
	workload := []ServingRequest{
		{ID: "a", Messages: []ChatMessage{{Role: "user", Content: "one"}}, MaxOutputTokens: 8, PromptTokensEstimate: 3},
		{ID: "b", Messages: []ChatMessage{{Role: "user", Content: "two"}}, MaxOutputTokens: 16, PromptTokensEstimate: 4},
	}
	first, err := ServingWorkloadDigest(workload)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ServingWorkloadDigest(append([]ServingRequest(nil), workload...))
	if err != nil {
		t.Fatal(err)
	}
	if first != second || !strings.HasPrefix(first, "sha256:") {
		t.Fatalf("stable digest = %q / %q", first, second)
	}
	workload[1].MaxOutputTokens++
	changed, err := ServingWorkloadDigest(workload)
	if err != nil {
		t.Fatal(err)
	}
	if changed == first {
		t.Fatal("changed output shape retained the same workload digest")
	}
}

func TestNormalizeServingConcurrenciesSortsAndRequiresSweep(t *testing.T) {
	got, err := normalizeServingConcurrencies([]int{8, 1, 4, 4, 2})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{1, 2, 4, 8}; !reflect.DeepEqual(got, want) {
		t.Fatalf("concurrencies = %v, want %v", got, want)
	}
	for _, input := range [][]int{{1}, {2, 2}, {1, 0}} {
		if _, err := normalizeServingConcurrencies(input); err == nil {
			t.Fatalf("normalizeServingConcurrencies(%v) succeeded, want refusal", input)
		}
	}
}

func TestEvaluateServingSweepSelectsCapacityValidPeakAndSLAKnee(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
		{concurrency: 4, throughput: 24, ttftP99: 50, itlP99: 7},
		{concurrency: 8, throughput: 23, ttftP99: 120, itlP99: 10},
	})
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	if len(report.Tracks) != 1 {
		t.Fatalf("summaries = %#v", report.Tracks)
	}
	summary := report.Tracks[0]
	if summary.Status != "measured" || summary.ValidPoints != 4 {
		t.Fatalf("summary = %#v, want 4 valid measured points", summary)
	}
	if summary.Peak == nil || summary.Peak.Concurrency != 4 || summary.Peak.ThroughputTokens != 24 {
		t.Fatalf("peak = %#v, want concurrency 4 / 24 tok/s", summary.Peak)
	}
	if summary.PeakStatus != "measured" || summary.EnvelopeDigest == "" {
		t.Fatalf("peak evidence status/digest = %q / %q", summary.PeakStatus, summary.EnvelopeDigest)
	}
	if summary.SLAStatus != "measured" || summary.SLAKnee == nil || summary.SLAKnee.Concurrency != 4 {
		t.Fatalf("SLA knee = status %q value %#v, want concurrency 4", summary.SLAStatus, summary.SLAKnee)
	}
}

func TestEvaluateServingSweepCensorsMonotonicTerminalPeakBeforeCapacity(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
		{concurrency: 4, throughput: 24, ttftP99: 50, itlP99: 7},
	})
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	summary := report.Tracks[0]
	if summary.PeakStatus != "right_censored" || summary.Peak == nil || summary.Peak.Concurrency != 4 {
		t.Fatalf("terminal peak = status %q value %+v", summary.PeakStatus, summary.Peak)
	}
}

func TestEvaluateServingSweepMissingPointMakesFindingsNotIdentifiable(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
		{concurrency: 4, throughput: 24, ttftP99: 50, itlP99: 7},
	})
	report.Points[1].Tracks[0].MeasurementStatus = "not_measured"
	report.Points[1].Tracks[0].Stats.OK = 0
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	summary := report.Tracks[0]
	if summary.PeakStatus != "not_identifiable" || summary.Peak != nil || summary.SLAStatus != "not_identifiable" {
		t.Fatalf("missing-point findings = %+v", summary)
	}
}

func TestEvaluateServingSweepAboveCapacityInvalidatesTrackClaim(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
		{concurrency: 16, throughput: 30, ttftP99: 200, itlP99: 20},
	})
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	overCap := report.Points[len(report.Points)-1].Tracks[0]
	if overCap.Status != "invalid" || overCap.ReasonCode != "capacity_exceeded" {
		t.Fatalf("above-cap point = %#v, want typed invalid refusal", overCap)
	}
	if summary := report.Tracks[0]; summary.Status != "invalid" || summary.Peak != nil || summary.SLAKnee != nil {
		t.Fatalf("above-cap sweep retained a claim: %#v", summary)
	}
}

func TestEvaluateServingSweepRefusesIdentityDriftWithoutZeroingIt(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
		{concurrency: 4, throughput: 24, ttftP99: 50, itlP99: 7},
	})
	report.Points[1].WorkloadDigest = "sha256:changed"
	report.Points[2].Tracks[0].EngineReceiptDigest = "sha256:other-engine"
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	if got := report.Points[1].Tracks[0]; got.Status != "invalid" || got.ReasonCode != "workload_identity_mismatch" {
		t.Fatalf("workload drift = %#v", got)
	}
	if got := report.Points[2].Tracks[0]; got.Status != "invalid" || got.ReasonCode != "engine_identity_mismatch" {
		t.Fatalf("engine drift = %#v", got)
	}
	summary := report.Tracks[0]
	if summary.Status != "invalid" || summary.Peak != nil || summary.ValidPoints != 1 {
		t.Fatalf("identity-drift summary = %#v, want invalid/no peak", summary)
	}
	if summary.PeakStatus != "invalid" || summary.SLAStatus != "invalid" {
		t.Fatalf("identity drift finding status = %+v", summary)
	}
	if report.Points[1].Tracks[0].Stats.ThroughputTokensS.Value == nil || *report.Points[1].Tracks[0].Stats.ThroughputTokensS.Value != 18 {
		t.Fatal("invalid point was rewritten as zero instead of preserving the observation")
	}
}

func TestEvaluateServingSweepUnknownCapacityCannotClaimPeak(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
	})
	report.Contracts[0].BatchCapacity = 0
	report.Contracts[0].CapacitySource = ""
	for i := range report.Points {
		report.Points[i].Tracks[0].BatchCapacity = 0
		report.Points[i].Tracks[0].CapacitySource = ""
	}
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	if report.Tracks[0].Peak != nil || report.Tracks[0].Status != "invalid" {
		t.Fatalf("unknown capacity produced a peak: %#v", report.Tracks[0])
	}
	for _, point := range report.Points {
		if point.Tracks[0].ReasonCode != "capacity_unknown" {
			t.Fatalf("point refusal = %#v, want capacity_unknown", point.Tracks[0])
		}
	}
}

func TestEvaluateServingSweepWithoutSLADoesNotInventKnee(t *testing.T) {
	report := syntheticSweepReport([]syntheticSweepPoint{
		{concurrency: 1, throughput: 10, ttftP99: 20, itlP99: 5},
		{concurrency: 2, throughput: 18, ttftP99: 30, itlP99: 6},
	})
	report.SLA = ServingSweepSLA{}
	if err := EvaluateServingSweep(report); err != nil {
		t.Fatal(err)
	}
	summary := report.Tracks[0]
	if summary.Peak == nil || summary.SLAStatus != "not_configured" || summary.SLAKnee != nil {
		t.Fatalf("no-SLA summary = %#v", summary)
	}
}

type syntheticSweepPoint struct {
	concurrency int
	throughput  float64
	ttftP99     float64
	itlP99      float64
}

func syntheticSweepReport(points []syntheticSweepPoint) *ServingSweepReport {
	const (
		digest        = "sha256:workload"
		engineReceipt = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	)
	contract := ServingSweepTrackContract{
		Track:               TrackOurs,
		Model:               "qwen3.8",
		Engine:              "fak-native",
		EngineReceiptDigest: engineReceipt,
		BatchCapacity:       8,
		CapacitySource:      "fixture:declared",
	}
	report := &ServingSweepReport{
		Schema:    ServingSweepSchema,
		Model:     contract.Model,
		Workload:  ServingSweepWorkload{Digest: digest, Requests: 8},
		Contracts: []ServingSweepTrackContract{contract},
		SLA:       ServingSweepSLA{TTFTP99Millis: 80, ITLP99Millis: 8},
	}
	for _, point := range points {
		throughput := point.throughput
		ttft := point.ttftP99
		itl := point.itlP99
		goodput := point.throughput / 10
		report.Workload.Concurrencies = append(report.Workload.Concurrencies, point.concurrency)
		report.Points = append(report.Points, ServingSweepPoint{
			Concurrency:    point.concurrency,
			WorkloadDigest: digest,
			Tracks: []ServingSweepTrackPoint{{
				Track:               TrackOurs,
				Status:              "measured",
				MeasurementStatus:   "measured",
				Model:               contract.Model,
				Engine:              contract.Engine,
				EngineReceiptDigest: contract.EngineReceiptDigest,
				BatchCapacity:       contract.BatchCapacity,
				CapacitySource:      contract.CapacitySource,
				Stats: ServingStats{
					OK:                1,
					ThroughputTokensS: ScalarMetric{Status: "measured", Value: &throughput},
					GoodputRPS:        ScalarMetric{Status: "measured", Value: &goodput},
					TTFTMillis:        QuantileMetric{Status: "measured", P99: &ttft},
					ITLMillis:         QuantileMetric{Status: "measured", P99: &itl},
				},
			}},
		})
	}
	return report
}
