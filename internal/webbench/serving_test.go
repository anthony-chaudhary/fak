package webbench

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestServingTrackSelectorIncludesIssue44Tracks(t *testing.T) {
	for _, name := range []string{"ours", "sglang", "vllm", "fak-fronts-fleet", "fak_fronts_fleet"} {
		tr, err := ParseServingTrack(name)
		if err != nil {
			t.Fatalf("ParseServingTrack(%q): %v", name, err)
		}
		if name == "fak_fronts_fleet" && tr != TrackFakFrontsFleet {
			t.Fatalf("underscore alias = %q, want %q", tr, TrackFakFrontsFleet)
		}
		script, err := ScriptFor(ServingPlan{Track: tr, Model: "m", BaseURL: "http://fleet/v1", Replicas: 3})
		if err != nil {
			t.Fatalf("ScriptFor(%q): %v", tr, err)
		}
		if script == "" {
			t.Fatalf("ScriptFor(%q) returned empty script", tr)
		}
	}
	if _, err := ParseServingTrack("dynamo"); err == nil {
		t.Fatalf("unknown track parsed without error")
	}
}

func TestFoldServingSamplesMeasuredMetrics(t *testing.T) {
	ttft1, tpot1 := 100.0, 45.0
	ttft2, tpot2 := 200.0, 55.0
	stats := FoldServingSamples([]ServingSample{
		{
			ID: "a", Status: "ok", TTFTMillis: &ttft1, ITLMillis: []float64{40, 50},
			TPOTMillis: &tpot1, EndToEndMillis: 500, OutputTokenEstimate: 4,
		},
		{
			ID: "b", Status: "ok", TTFTMillis: &ttft2, ITLMillis: []float64{60, 70},
			TPOTMillis: &tpot2, EndToEndMillis: 900, OutputTokenEstimate: 6,
		},
		{ID: "c", Status: "fail", Error: "boom"},
	}, 2.0, time.Second)

	if stats.Requests != 3 || stats.OK != 2 || stats.Failed != 1 {
		t.Fatalf("counts = %+v, want requests=3 ok=2 failed=1", stats)
	}
	if stats.TTFTMillis.Status != "measured" || *stats.TTFTMillis.P50 != 100 || *stats.TTFTMillis.P90 != 200 || *stats.TTFTMillis.P95 != 200 {
		t.Fatalf("ttft quantiles wrong: %+v", stats.TTFTMillis)
	}
	if stats.ITLMillis.Status != "measured" || *stats.ITLMillis.P99 != 70 {
		t.Fatalf("itl quantiles wrong: %+v", stats.ITLMillis)
	}
	if stats.ThroughputTokensS.Status != "measured" || *stats.ThroughputTokensS.Value != 5 {
		t.Fatalf("throughput = %+v, want 5 events/s", stats.ThroughputTokensS)
	}
	if stats.GoodputRPS.Status != "measured" || *stats.GoodputRPS.Value != 1 {
		t.Fatalf("goodput = %+v, want 1 req/s", stats.GoodputRPS)
	}
}

func TestRunServingParityReportsMissingEndpointsAsNotMeasured(t *testing.T) {
	d := NewDataset([]Instance{{TaskID: "t1", Description: "Find a contact email."}})
	workload := BuildServingWorkload(d, DefaultGeometryModel(), 1, 1, 16, "shared prefix")
	rep, err := RunServingParity(context.Background(), ServingParityConfig{
		GeneratedAt: "2026-06-25T00:00:00Z",
		MachineID:   "dev",
		Model:       "m",
		Tracks: []ServingTrackConfig{
			{Track: TrackVLLM},
			{Track: TrackFakFrontsFleet},
		},
		Workload: workload,
	})
	if err != nil {
		t.Fatalf("RunServingParity: %v", err)
	}
	if rep.Schema != ServingParitySchema || len(rep.Tracks) != 2 {
		t.Fatalf("report shape wrong: %+v", rep)
	}
	for _, tr := range rep.Tracks {
		if tr.Status != "not_measured" {
			t.Fatalf("%s status = %q, want not_measured", tr.Track, tr.Status)
		}
		if tr.Stats.TTFTMillis.Status != "not_measured" {
			t.Fatalf("%s ttft = %+v, want not_measured", tr.Track, tr.Stats.TTFTMillis)
		}
	}
	if rep.OursVsSGLangTax.Status != "not_measured" {
		t.Fatalf("tax status = %+v, want not_measured", rep.OursVsSGLangTax)
	}
}

func TestParsePrefixCacheHitRateMetrics(t *testing.T) {
	v, src, ok := ParsePrefixCacheHitRateMetrics(`
# HELP vllm:gpu_prefix_cache_hit_rate hit rate
vllm:gpu_prefix_cache_hit_rate{gpu="0"} 0.875
`)
	if !ok || v != 0.875 || src != "vllm:gpu_prefix_cache_hit_rate" {
		t.Fatalf("parsed (%v, %q, %v), want 0.875 vllm metric true", v, src, ok)
	}
	v, src, ok = ParsePrefixCacheHitRateMetrics(`sglang_prefix_cache_hit_rate 87.5`)
	if !ok || v != 0.875 || src != "sglang_prefix_cache_hit_rate" {
		t.Fatalf("percent parsed (%v, %q, %v), want 0.875 sglang true", v, src, ok)
	}
}

func TestMeasureSSERequestReadsTokenStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n")
		flusher.Flush()
		time.Sleep(time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	s := MeasureSSERequest(
		context.Background(),
		srv.Client(),
		ServingTrackConfig{BaseURL: srv.URL + "/v1"},
		"m",
		ServingRequest{
			ID: "r1",
			Messages: []ChatMessage{
				{Role: "user", Content: "hello"},
			},
			MaxOutputTokens: 8,
		},
		time.Second,
	)
	if s.Status != "ok" || s.StreamMode != "sse" {
		t.Fatalf("sample status/mode = %+v", s)
	}
	if s.OutputEvents != 2 || s.OutputTokenEstimate != 2 {
		t.Fatalf("output events = %d estimate=%d, want 2/2", s.OutputEvents, s.OutputTokenEstimate)
	}
	if s.OutputTokensExact == nil || *s.OutputTokensExact != 2 || s.OutputTokenCountBasis != "usage.completion_tokens" {
		t.Fatalf("usage token count = exact:%v basis:%q, want 2 usage.completion_tokens", s.OutputTokensExact, s.OutputTokenCountBasis)
	}
	if s.TTFTMillis == nil || len(s.ITLMillis) != 1 || s.TPOTMillis == nil {
		t.Fatalf("stream timings missing: %+v", s)
	}
}

func TestMeasureSSERequestClassifiesHTTPFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	s := MeasureSSERequest(
		context.Background(),
		srv.Client(),
		ServingTrackConfig{BaseURL: srv.URL + "/v1"},
		"m",
		ServingRequest{ID: "r1", Messages: []ChatMessage{{Role: "user", Content: "hello"}}, MaxOutputTokens: 8},
		time.Second,
	)
	if s.Status != "fail" || s.HTTPStatus != http.StatusServiceUnavailable || s.ErrorClass != "http_status" {
		t.Fatalf("failure sample = %+v, want http_status classification", s)
	}
}

func TestServingOursVsSGLangTaxReportsDeltaAndRatio(t *testing.T) {
	oursTTFT, rawTTFT := 120.0, 100.0
	oursThroughput, rawThroughput := 80.0, 100.0
	tax := servingOursVsSGLangTax([]ServingTrackResult{
		{
			Track:  TrackOurs,
			Status: "measured",
			Stats: ServingStats{
				TTFTMillis:        QuantileMetric{Status: "measured", Unit: "ms", P50: &oursTTFT},
				ThroughputTokensS: ScalarMetric{Status: "measured", Unit: "usage.completion_tokens/s", Value: &oursThroughput},
			},
		},
		{
			Track:  TrackSGLang,
			Status: "measured",
			Stats: ServingStats{
				TTFTMillis:        QuantileMetric{Status: "measured", Unit: "ms", P50: &rawTTFT},
				ThroughputTokensS: ScalarMetric{Status: "measured", Unit: "usage.completion_tokens/s", Value: &rawThroughput},
			},
		},
	})
	if tax.Status != "measured" || tax.RawTrack != TrackSGLang || tax.OursTrack != TrackOurs {
		t.Fatalf("tax header = %+v, want measured ours-vs-sglang", tax)
	}
	ttft := tax.Metrics["ttft_ms"]
	if ttft.OursMinusRaw == nil || *ttft.OursMinusRaw != 20 || ttft.OursDivRaw == nil || *ttft.OursDivRaw != 1.2 {
		t.Fatalf("ttft tax = %+v, want delta 20 ratio 1.2", ttft)
	}
	throughput := tax.Metrics["throughput_tok_s"]
	if throughput.OursMinusRaw == nil || *throughput.OursMinusRaw != -20 || throughput.OursDivRaw == nil || *throughput.OursDivRaw != 0.8 {
		t.Fatalf("throughput tax = %+v, want delta -20 ratio 0.8", throughput)
	}
}

func TestDefaultServingArtifactPathIsDatedAndTrackLabeled(t *testing.T) {
	path := DefaultServingArtifactPath("experiments/benchmark/runs/by-machine", "GPU Node", "2026-06-25T12:34:56Z", []ServingTrack{TrackVLLM, TrackSGLang})
	for _, want := range []string{"gpu-node", "20260625T123456Z-serving-parity-vllm-sglang", "result.json"} {
		if !strings.Contains(path, want) {
			t.Fatalf("artifact path %q missing %q", path, want)
		}
	}
}

func TestValidateParityClaimRequiresMeasuredArtifact(t *testing.T) {
	claim := "fak is parity or better on the base serving items."
	if err := ValidateParityClaim(claim, nil); err == nil {
		t.Fatalf("nil artifact accepted for parity claim")
	}
	rep := &ServingParityReport{
		Schema: ServingParitySchema,
		Tracks: []ServingTrackResult{
			{Track: TrackVLLM, Status: "measured", Stats: ServingStats{OK: 1}},
			{Track: TrackSGLang, Status: "measured", Stats: ServingStats{OK: 1}},
			{Track: TrackFakFrontsFleet, Status: "not_measured"},
		},
	}
	if err := ValidateParityClaim(claim, rep); err == nil {
		t.Fatalf("unmeasured fak-fronts-fleet track accepted")
	}
	rep.Tracks[2].Status = "measured"
	rep.Tracks[2].Stats.OK = 1
	if err := ValidateParityClaim(claim, rep); err != nil {
		t.Fatalf("measured artifact rejected: %v", err)
	}
	if err := ValidateParityClaim("this only records a planned comparison", nil); err != nil {
		t.Fatalf("non-parity claim should not require artifact: %v", err)
	}
}
