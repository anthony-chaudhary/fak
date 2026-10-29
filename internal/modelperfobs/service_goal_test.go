package modelperfobs

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func serviceGoalFixture() ServiceGoal {
	return ServiceGoal{Schema: ServiceGoalSchema, ID: "interactive", WorkloadRef: "trace:fixture",
		TTFTLimitMS: 100, TPOTLimitMS: 20, WarmupPolicy: "exclude", SingleTokenTPOTPolicy: "ttft-only",
		Run: ServiceRunEvidence{SourceRef: "source:fixture", ModelRef: "model:fixture", ConfigRef: "config:fixture",
			WallSeconds: 10, OfferedRequests: 6, Quality: QualityUnknown}}
}

func serviceObservation(id string, tokens int64, ttft, tpot float64) Observation {
	return Observation{Schema: Schema, RequestID: id, Status: 200, Streaming: true,
		CompletionTokens: tokens, DurationMS: 500, TTFTMS: ttft, TPOTMS: tpot}
}

// fak-test:runtime fast est=10ms
func TestServiceGoalJointAccounting(t *testing.T) {
	g := serviceGoalFixture()
	rows := []Observation{serviceObservation("joint-pass", 3, 50, 10), serviceObservation("ttft-fail", 5, 150, 10),
		serviceObservation("tpot-fail", 5, 50, 30), serviceObservation("single", 1, 70, 0), serviceObservation("error", 0, 0, 0)}
	rows[4].Error, rows[4].Status = "rejected", 503
	warm := serviceObservation("warm", 100, 1, 1)
	warm.Warmup = true
	rows = append(rows, warm)
	r, err := EvaluateServiceGoal(g, rows)
	if err != nil {
		t.Fatal(err)
	}
	if r.ObservedRequests != 5 || r.MissingRequests != 1 || r.WarmupRequests != 1 || r.Errors != 1 ||
		r.SLOPassedRequests != 2 || r.SLOFailedRequests != 2 || r.SLOUnknownRequests != 1 || r.SLOAttainment != 2.0/6 ||
		r.MeasuredSLOGoodput.RequestsPerSecond != .2 || r.MeasuredSLOGoodput.CompletionTokensPerSecond != .4 {
		t.Fatalf("joint/offered accounting: %+v", r)
	}
	if r.TTFTP95.MeasuredRequests != 4 || r.TPOTP95.MeasuredRequests != 3 || r.TTFTP95.OfferedRequests != 6 ||
		r.TTFTP95.ValueMS == nil || *r.TTFTP95.ValueMS != 150 || r.QualityQualifiedGoodput != nil || r.CostPerQualifiedRequest != nil {
		t.Fatalf("coverage/unknowns: %+v", r)
	}
	cost := 12.0
	g.Run.Cost, g.Run.CostUnit, g.Run.Quality, g.Run.QualityRef = &cost, "fixture-credit", QualityQualified, "quality:external-fixture"
	r, err = EvaluateServiceGoal(g, rows)
	if err != nil || r.QualityQualifiedGoodput == nil || r.QualityQualifiedGoodput.RequestsPerSecond != .2 ||
		r.CostPerQualifiedRequest == nil || *r.CostPerQualifiedRequest != 6 {
		t.Fatalf("qualified result: %+v, %v", r, err)
	}
	g.Run.Quality = QualityFailed
	r, err = EvaluateServiceGoal(g, rows)
	if err != nil || r.QualityQualifiedGoodput == nil || r.QualityQualifiedGoodput.RequestsPerSecond != 0 || r.CostPerQualifiedRequest != nil {
		t.Fatalf("failed quality must be known zero: %+v, %v", r, err)
	}
	g.SingleTokenTPOTPolicy, g.Run.Quality = "unknown", QualityUnknown
	rows[0].Streaming = false
	r, err = EvaluateServiceGoal(g, rows)
	if err != nil || r.SLOPassedRequests != 0 || r.SLOUnknownRequests != 3 || r.TPOTNotApplicableRequests != 1 {
		t.Fatalf("undefined and non-stream timings: %+v, %v", r, err)
	}
	data, err := json.Marshal(r)
	if err != nil || !strings.Contains(string(data), `"quality_qualified_goodput":null`) {
		t.Fatalf("unknown serialized as a value: %s, %v", data, err)
	}
}

// fak-test:runtime fast est=10ms
func TestServiceGoalRejectsInvalidEvidence(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*ServiceGoal, *[]Observation)
	}{
		{"duplicate", func(_ *ServiceGoal, r *[]Observation) { *r = append(*r, (*r)[0]) }},
		{"missing-id", func(_ *ServiceGoal, r *[]Observation) { (*r)[0].RequestID = "" }},
		{"nonfinite", func(_ *ServiceGoal, r *[]Observation) { (*r)[0].TPOTMS = math.NaN() }},
		{"negative-tokens", func(_ *ServiceGoal, r *[]Observation) { (*r)[0].CompletionTokens = -1 }},
		{"token-overflow", func(_ *ServiceGoal, r *[]Observation) { (*r)[0].PromptTokens = math.MaxInt64 }},
		{"aggregate-overflow", func(_ *ServiceGoal, r *[]Observation) {
			(*r)[0].CompletionTokens = math.MaxInt64
			*r = append(*r, serviceObservation("second", 1, 5, 5))
		}},
		{"offered-smaller", func(g *ServiceGoal, r *[]Observation) {
			g.Run.OfferedRequests = 1
			*r = append(*r, serviceObservation("second", 1, 5, 0))
		}},
		{"zero-wall", func(g *ServiceGoal, _ *[]Observation) { g.Run.WallSeconds = 0 }},
		{"rate-overflow", func(g *ServiceGoal, _ *[]Observation) { g.Run.WallSeconds = math.SmallestNonzeroFloat64 }},
		{"missing-quality-ref", func(g *ServiceGoal, _ *[]Observation) { g.Run.Quality = QualityQualified }},
		{"missing-warmup-policy", func(g *ServiceGoal, _ *[]Observation) { g.WarmupPolicy = "" }},
		{"missing-single-policy", func(g *ServiceGoal, _ *[]Observation) { g.SingleTokenTPOTPolicy = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g, rows := serviceGoalFixture(), []Observation{serviceObservation("one", 3, 5, 5)}
			tc.edit(&g, &rows)
			if _, err := EvaluateServiceGoal(g, rows); err == nil {
				t.Fatal("accepted invalid evidence")
			}
		})
	}
}
