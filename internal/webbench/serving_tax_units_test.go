package webbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms
func TestRunServingParityTaxRequiresComparableThroughput(t *testing.T) {
	for _, tc := range []struct {
		name      string
		oursUsage bool
		rawUsage  bool
	}{
		{name: "mixed", oursUsage: true},
		{name: "usage", oursUsage: true, rawUsage: true},
		{name: "events"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var usage bool
				switch r.URL.Path {
				case "/ours/chat/completions":
					usage = tc.oursUsage
				case "/raw/chat/completions":
					usage = tc.rawUsage
				default:
					http.NotFound(w, r)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n")
				if usage {
					fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"completion_tokens\":4}}\n\n")
				}
				fmt.Fprint(w, "data: [DONE]\n\n")
			}))
			defer server.Close()
			report, err := RunServingParity(context.Background(), ServingParityConfig{
				Model: "fixture", Client: server.Client(), Concurrency: 1, Timeout: time.Second,
				Tracks:   []ServingTrackConfig{{Track: TrackOurs, BaseURL: server.URL + "/ours"}, {Track: TrackSGLang, BaseURL: server.URL + "/raw"}},
				Workload: []ServingRequest{{ID: "one", Messages: []ChatMessage{{Role: "user", Content: "hello"}}, MaxOutputTokens: 8}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Tracks) != 2 {
				t.Fatalf("tracks=%d, want two actual HTTP tracks", len(report.Tracks))
			}
			for i, usage := range []bool{tc.oursUsage, tc.rawUsage} {
				track := report.Tracks[i]
				basis := "stream_content_events"
				if usage {
					basis = "usage.completion_tokens"
				}
				if track.Stats.OK != 1 || len(track.Samples) != 1 || track.Samples[0].OutputEvents != 2 || track.Stats.TokenCountBasis != basis || track.Stats.ThroughputTokensS.Unit != basis+"/s" {
					t.Fatalf("track evidence changed: %+v", track)
				}
			}
			tax := report.OursVsSGLangTax
			metric := tax.Metrics["throughput_tok_s"]
			if tc.oursUsage != tc.rawUsage {
				assertServingTaxUnavailable(t, metric)
				if !strings.Contains(metric.Reason, "differs between tracks") {
					t.Fatalf("missing mixed-unit reason: %+v", metric)
				}
			} else {
				ours, raw := report.Tracks[0].Stats.ThroughputTokensS, report.Tracks[1].Stats.ThroughputTokensS
				if metric.Status != "measured" || metric.Unit != ours.Unit || metric.Ours == nil || metric.Raw == nil || metric.OursMinusRaw == nil || metric.OursDivRaw == nil || *metric.Ours != *ours.Value || *metric.Raw != *raw.Value || *metric.OursMinusRaw != round(*ours.Value-*raw.Value) || *metric.OursDivRaw != round(*ours.Value / *raw.Value) {
					t.Fatalf("comparable tax lost original rates or arithmetic: %+v", metric)
				}
			}
			if tax.Status != "measured" || tax.Metrics["ttft_ms"].Status != "measured" || tax.Metrics["goodput_rps"].Status != "measured" {
				t.Fatalf("comparable latency/request evidence was discarded: %+v", tax)
			}
			encoded, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			var roundtrip ServingParityReport
			if err := json.Unmarshal(encoded, &roundtrip); err != nil || !reflect.DeepEqual(roundtrip.OursVsSGLangTax, tax) {
				t.Fatalf("tax serialization lost availability or identity: err=%v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestServingTaxRejectsUnknownThroughputIdentity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*ServingStats)
	}{
		{name: "missing_unit", mutate: func(s *ServingStats) { s.ThroughputTokensS.Unit = "" }},
		{name: "invalid_unit", mutate: func(s *ServingStats) { s.ThroughputTokensS.Unit = "tok/s" }},
		{name: "missing_basis", mutate: func(s *ServingStats) { s.TokenCountBasis = "" }},
		{name: "mismatched_basis", mutate: func(s *ServingStats) { s.TokenCountBasis = "stream_content_events" }},
		{name: "mixed_basis", mutate: func(s *ServingStats) {
			s.TokenCountBasis = "mixed"
			s.ThroughputTokensS.Unit = "output_token_estimate/s"
		}},
		{name: "unmeasured", mutate: func(s *ServingStats) { s.ThroughputTokensS.Status = "not_measured" }},
		{name: "missing_value", mutate: func(s *ServingStats) { s.ThroughputTokensS.Value = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ours := ServingStats{TokenCountBasis: "usage.completion_tokens", ThroughputTokensS: measuredScalar(80, "usage.completion_tokens/s", "fixture")}
			raw := ServingStats{TokenCountBasis: "usage.completion_tokens", ThroughputTokensS: measuredScalar(100, "usage.completion_tokens/s", "fixture")}
			tc.mutate(&raw)
			tracks := []ServingTrackResult{{Track: TrackOurs, Status: "measured", Stats: ours}, {Track: TrackSGLang, Status: "measured", Stats: raw}}
			before, _ := json.Marshal(tracks)
			tax := servingOursVsSGLangTax(tracks)
			assertServingTaxUnavailable(t, tax.Metrics["throughput_tok_s"])
			after, _ := json.Marshal(tracks)
			if string(before) != string(after) {
				t.Fatal("tax comparison mutated raw evidence")
			}
		})
	}
}

func assertServingTaxUnavailable(t *testing.T, metric ServingTaxMetric) {
	t.Helper()
	if metric.Status != "not_measured" || metric.Reason == "" || metric.Ours != nil || metric.Raw != nil || metric.OursMinusRaw != nil || metric.OursDivRaw != nil {
		t.Fatalf("incomparable tax reported numeric evidence: %+v", metric)
	}
}

// fak-test:runtime fast est=1ms
func TestServingTaxPreservesKnownThroughputBases(t *testing.T) {
	for _, basis := range []string{"usage.completion_tokens", "stream_content_events", "estimated_content_tokens"} {
		t.Run(basis, func(t *testing.T) {
			stats := func(value float64) ServingStats {
				return ServingStats{TokenCountBasis: basis, ThroughputTokensS: measuredScalar(value, basis+"/s", "fixture")}
			}
			tax := servingOursVsSGLangTax([]ServingTrackResult{{Track: TrackOurs, Status: "measured", Stats: stats(80)}, {Track: TrackSGLang, Status: "measured", Stats: stats(100)}})
			metric := tax.Metrics["throughput_tok_s"]
			if metric.Status != "measured" || metric.Unit != basis+"/s" || metric.OursMinusRaw == nil || *metric.OursMinusRaw != -20 || metric.OursDivRaw == nil || *metric.OursDivRaw != 0.8 {
				t.Fatalf("known comparable basis lost: %+v", metric)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestServingTaxGoodputRequiresRequestUnits(t *testing.T) {
	for _, tc := range []struct{ name, ours, raw string }{
		{name: "missing", ours: "requests/s"},
		{name: "mismatch", ours: "requests/s", raw: "usage.completion_tokens/s"},
		{name: "invalid_same", ours: "unknown/s", raw: "unknown/s"},
		{name: "requests", ours: "requests/s", raw: "requests/s"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tax := servingOursVsSGLangTax([]ServingTrackResult{
				{Track: TrackOurs, Status: "measured", Stats: ServingStats{GoodputRPS: measuredScalar(80, tc.ours, "fixture")}},
				{Track: TrackSGLang, Status: "measured", Stats: ServingStats{GoodputRPS: measuredScalar(100, tc.raw, "fixture")}},
			})
			metric := tax.Metrics["goodput_rps"]
			if tc.name != "requests" {
				assertServingTaxUnavailable(t, metric)
				return
			}
			if metric.Status != "measured" || metric.Unit != "requests/s" || metric.OursMinusRaw == nil || *metric.OursMinusRaw != -20 || metric.OursDivRaw == nil || *metric.OursDivRaw != 0.8 {
				t.Fatalf("request goodput changed: %+v", metric)
			}
		})
	}
}
