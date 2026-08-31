package webbench

import (
	"strings"
	"testing"
)

func TestValidateServingSweepClaim(t *testing.T) {
	tests := []struct {
		name    string
		claim   string
		mutate  func(*ServingSweepReport)
		wantErr string
	}{
		{name: "honest generic peak", claim: "ours serving peak is 120 tok/s"},
		{name: "honest capacity-valid peak", claim: "ours capacity-valid peak is 120 tok/s"},
		{name: "honest SLA knee", claim: "ours p99 SLA knee is concurrency 2"},
		{name: "above capacity", claim: "ours serving peak is 120 tok/s", mutate: func(r *ServingSweepReport) { r.Points[1].Concurrency, r.Workload.Concurrencies[1] = 3, 3 }, wantErr: "exceeds declared batch capacity"},
		{name: "identity drift", claim: "ours capacity-valid peak is 120 tok/s", mutate: func(r *ServingSweepReport) { r.Points[1].Tracks[0].Engine = "other-engine" }, wantErr: "engine identity differs"},
		{name: "missing SLA", claim: "ours SLA knee is concurrency 2", mutate: func(r *ServingSweepReport) { r.SLA = ServingSweepSLA{} }, wantErr: "requires a configured p99"},
		{name: "sparse points", claim: "ours serving peak is 120 tok/s", mutate: func(r *ServingSweepReport) { r.Points = r.Points[:1] }, wantErr: "requires at least two declared coordinates"},
		{name: "missing track", claim: "vllm serving peak is 120 tok/s", wantErr: "requires measured vllm track"},
		{name: "unrelated prose", claim: "this report records a planned serving comparison", mutate: func(r *ServingSweepReport) { r.Schema = "wrong" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := servingSweepClaimFixture()
			if tt.mutate != nil {
				tt.mutate(report)
			}
			err := ValidateServingSweepClaim(tt.claim, report)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateServingSweepClaim() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ValidateServingSweepClaim() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestValidateServingSweepClaimRequiresArtifactAndSchema(t *testing.T) {
	if err := ValidateServingSweepClaim("ours peak throughput is 120 tok/s", nil); err == nil || !strings.Contains(err.Error(), ServingSweepSchema) {
		t.Fatalf("missing artifact error = %v", err)
	}
	report := servingSweepClaimFixture()
	report.Schema = "wrong"
	if err := ValidateServingSweepClaim("ours peak throughput is 120 tok/s", report); err == nil || !strings.Contains(err.Error(), ServingSweepSchema) {
		t.Fatalf("wrong schema error = %v", err)
	}
}

func TestParseServingSweepClaim(t *testing.T) {
	tests := []struct {
		claim string
		kind  ServingSweepClaimKind
		track ServingTrack
	}{
		{"ours serving peak", ServingSweepClaimPeak, TrackOurs},
		{"vllm capacity-valid peak", ServingSweepClaimCapacityPeak, TrackVLLM},
		{"sglang p99 knee", ServingSweepClaimSLAKnee, TrackSGLang},
		{"no serving assertion", ServingSweepClaimNone, ""},
	}
	for _, tt := range tests {
		if got := ParseServingSweepClaim(tt.claim); got.Kind != tt.kind || got.Track != tt.track {
			t.Errorf("ParseServingSweepClaim(%q) = %#v, want kind=%q track=%q", tt.claim, got, tt.kind, tt.track)
		}
	}
}

func servingSweepClaimFixture() *ServingSweepReport {
	throughputs := []float64{80, 120}
	ttft := []float64{10, 20}
	digest := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	report := &ServingSweepReport{
		Schema: ServingSweepSchema,
		Workload: ServingSweepWorkload{
			Digest:        "workload-1",
			Concurrencies: []int{1, 2},
		},
		SLA: ServingSweepSLA{TTFTP99Millis: 50},
		Contracts: []ServingSweepTrackContract{{
			Track:               TrackOurs,
			Model:               "qwen3.8",
			Engine:              "fak-native",
			EngineReceiptDigest: digest,
			BatchCapacity:       2,
			CapacitySource:      "fixture",
		}},
	}
	for i, concurrency := range []int{1, 2} {
		report.Points = append(report.Points, ServingSweepPoint{
			Concurrency:    concurrency,
			WorkloadDigest: report.Workload.Digest,
			Tracks: []ServingSweepTrackPoint{{
				Track:               TrackOurs,
				Model:               "qwen3.8",
				Engine:              "fak-native",
				EngineReceiptDigest: digest,
				BatchCapacity:       2,
				CapacitySource:      "fixture",
				MeasurementStatus:   "measured",
				Stats: ServingStats{
					OK:                1,
					ThroughputTokensS: ScalarMetric{Status: "measured", Value: &throughputs[i]},
					TTFTMillis:        QuantileMetric{Status: "measured", P99: &ttft[i]},
				},
			}},
		})
	}
	return report
}
