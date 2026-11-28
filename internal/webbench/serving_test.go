package webbench

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
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

// fak-test:justify why=contract when=changed:internal/webbench/**
// fak-test:runtime fast est=10ms lane=default
func TestFoldServingSamplesJointGoodputRequiresAllSLOs(t *testing.T) {
	ttftOK, ttftBad, tpotOK, tpotBad := 100.0, 500.0, 40.0, 80.0
	samples := []ServingSample{
		{ID: "fast-e2e-over-tpot", Status: "ok", TTFTMillis: &ttftOK, TPOTMillis: &tpotBad, EndToEndMillis: 500, OutputTokenEstimate: 4},
		{ID: "slow-e2e-fast-tpot", Status: "ok", TTFTMillis: &ttftBad, TPOTMillis: &tpotOK, EndToEndMillis: 1500, OutputTokenEstimate: 4},
		{ID: "all-within", Status: "ok", TTFTMillis: &ttftOK, TPOTMillis: &tpotOK, EndToEndMillis: 900, OutputTokenEstimate: 4},
		{ID: "failed", Status: "fail", Error: "boom"},
	}
	slos := ServingSLOs{TTFT: 2 * time.Second, TPOT: 50 * time.Millisecond, EndToEnd: time.Second}

	t.Run("requires-all-configured-slos", func(t *testing.T) {
		stats := FoldServingSamplesWithSLOs(samples, 2.0, slos)
		if stats.OK != 3 || stats.Failed != 1 {
			t.Fatalf("counts = ok:%d failed:%d, want ok=3 failed=1", stats.OK, stats.Failed)
		}
		// Only "all-within" meets every SLO: fast-e2e-over-tpot is fast end to
		// end but over the TPOT SLO, and slow-e2e-fast-tpot breaches the
		// end-to-end SLO.
		if stats.GoodputRPS.Status != "measured" || stats.GoodputRPS.Value == nil || *stats.GoodputRPS.Value != 0.5 {
			t.Fatalf("joint goodput = %+v, want measured 0.5 req/s", stats.GoodputRPS)
		}
	})

	t.Run("unset-slos-ignored", func(t *testing.T) {
		lowTTFT := 40.0
		onlyTTFT := []ServingSample{
			{ID: "missing-ttft", Status: "ok", TPOTMillis: &tpotOK, EndToEndMillis: 300, OutputTokenEstimate: 2},
			{ID: "within-ttft", Status: "ok", TTFTMillis: &lowTTFT, TPOTMillis: &tpotBad, EndToEndMillis: 5000, OutputTokenEstimate: 2},
		}
		stats := FoldServingSamplesWithSLOs(onlyTTFT, 2.0, ServingSLOs{TTFT: 2 * time.Second})
		// Unset TPOT/EndToEnd thresholds are ignored; a configured threshold
		// with a missing measurement fails closed.
		if stats.GoodputRPS.Status != "measured" || stats.GoodputRPS.Value == nil || *stats.GoodputRPS.Value != 0.5 {
			t.Fatalf("ttft-only goodput = %+v, want measured 0.5 req/s (unset tpot/e2e ignored)", stats.GoodputRPS)
		}
	})

	t.Run("legacy-single-slo-path-identical", func(t *testing.T) {
		legacy := FoldServingSamples(samples, 2.0, time.Second)
		joint := FoldServingSamplesWithSLOs(samples, 2.0, ServingSLOs{EndToEnd: time.Second})
		if !reflect.DeepEqual(legacy, joint) {
			t.Fatalf("legacy single-SLO stats differ from joint e2e-only stats:\nlegacy=%+v\njoint=%+v", legacy, joint)
		}
		if legacy.GoodputRPS.Status != "measured" || *legacy.GoodputRPS.Value != 1 {
			t.Fatalf("legacy goodput = %+v, want measured 1 req/s", legacy.GoodputRPS)
		}
	})

	t.Run("no-slos-not-measured", func(t *testing.T) {
		stats := FoldServingSamplesWithSLOs(samples, 2.0, ServingSLOs{})
		if stats.GoodputRPS.Status != "not_measured" || stats.GoodputRPS.Value != nil {
			t.Fatalf("goodput without configured SLOs = %+v, want not_measured", stats.GoodputRPS)
		}
	})
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
				TokenCountBasis:   "usage.completion_tokens",
				TTFTMillis:        QuantileMetric{Status: "measured", Unit: "ms", P50: &oursTTFT},
				ThroughputTokensS: ScalarMetric{Status: "measured", Unit: "usage.completion_tokens/s", Value: &oursThroughput},
			},
		},
		{
			Track:  TrackSGLang,
			Status: "measured",
			Stats: ServingStats{
				TokenCountBasis:   "usage.completion_tokens",
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

// fak-test:justify why=contract when=changed:internal/webbench/**
// fak-test:runtime fast est=100ms lane=default
func TestMeasureSSERequestReasoningAndNonSSESemantics(t *testing.T) {
	t.Run("sse_reasoning_and_content", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":null}}]}\n\n")
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
			// Both reasoning and reasoning_content on the same frame must not double-count.
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\",\"reasoning_content\":\"think\"}}]}\n\n")
			flusher.Flush()
			time.Sleep(1 * time.Millisecond)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" more\"}}]}\n\n")
			flusher.Flush()
			time.Sleep(1 * time.Millisecond)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ans\"}}]}\n\n")
			flusher.Flush()
			time.Sleep(1 * time.Millisecond)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}\n\n")
			flusher.Flush()
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":16,\"completion_tokens\":4,\"total_tokens\":20,\"completion_tokens_details\":{\"reasoning_tokens\":2}}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer srv.Close()

		s := MeasureSSERequest(context.Background(), srv.Client(), ServingTrackConfig{BaseURL: srv.URL + "/v1"}, "qwen3.8", ServingRequest{
			ID: "r-reason", Messages: []ChatMessage{{Role: "user", Content: "hi"}}, MaxOutputTokens: 16,
		}, 2*time.Second)
		if s.Status != "ok" || s.StreamMode != "sse" {
			t.Fatalf("sample status/mode = %+v", s)
		}
		if s.OutputEvents != 4 || s.ReasoningEvents != 2 || s.OutputTokenEstimate != 4 {
			t.Fatalf("events=%d reasoning_events=%d estimate=%d, want 4/2/4", s.OutputEvents, s.ReasoningEvents, s.OutputTokenEstimate)
		}
		if s.OutputTokensExact == nil || *s.OutputTokensExact != 4 || s.OutputTokenCountBasis != "usage.completion_tokens" {
			t.Fatalf("exact tokens = %v basis=%q", s.OutputTokensExact, s.OutputTokenCountBasis)
		}
		if s.Usage == nil || s.Usage.ReasoningTokens == nil || *s.Usage.ReasoningTokens != 2 {
			t.Fatalf("usage.reasoning_tokens = %+v, want 2", s.Usage)
		}
		if s.TTFTMillis == nil || *s.TTFTMillis <= 0 || len(s.ITLMillis) != 3 || s.TPOTMillis == nil || *s.TPOTMillis <= 0 {
			t.Fatalf("sse timings = ttft:%v itl:%v tpot:%v", s.TTFTMillis, s.ITLMillis, s.TPOTMillis)
		}
	})

	t.Run("sse_reasoning_only_without_usage", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			time.Sleep(2 * time.Millisecond)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning\":\"step1\"}}]}\n\n")
			flusher.Flush()
			time.Sleep(1 * time.Millisecond)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step2\"}}]}\n\n")
			flusher.Flush()
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer srv.Close()

		s := MeasureSSERequest(context.Background(), srv.Client(), ServingTrackConfig{BaseURL: srv.URL + "/v1"}, "qwen3.8", ServingRequest{
			ID: "r-reason-only", Messages: []ChatMessage{{Role: "user", Content: "hi"}}, MaxOutputTokens: 8,
		}, 2*time.Second)
		if s.Status != "ok" || s.OutputEvents != 2 || s.ReasoningEvents != 2 || s.OutputTokenEstimate != 2 || s.OutputTokenCountBasis != "stream_content_events" {
			t.Fatalf("reasoning-only stream sample = %+v", s)
		}
		if s.TTFTMillis == nil || *s.TTFTMillis <= 0 || len(s.ITLMillis) != 1 || s.TPOTMillis == nil || *s.TPOTMillis <= 0 {
			t.Fatalf("reasoning-only timings = %+v", s)
		}
	})

	t.Run("sse_stream_error_fails_closed", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"error\":{\"message\":\"engine overloaded\"}}\n\n")
		}))
		defer srv.Close()

		s := MeasureSSERequest(context.Background(), srv.Client(), ServingTrackConfig{BaseURL: srv.URL + "/v1"}, "qwen3.8", ServingRequest{
			ID: "r-err", Messages: []ChatMessage{{Role: "user", Content: "hi"}}, MaxOutputTokens: 8,
		}, time.Second)
		if s.Status != "fail" || s.ErrorClass != "stream_error" || s.Error != "engine overloaded" {
			t.Fatalf("stream error sample = %+v, want fail/stream_error", s)
		}
	})

	t.Run("non_sse_fails_closed_on_stream_timings_and_accounts_reasoning", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(2 * time.Millisecond)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","reasoning_content":"thinking step ","content":"answer"}}]}`)
		}))
		defer srv.Close()

		s := MeasureSSERequest(context.Background(), srv.Client(), ServingTrackConfig{BaseURL: srv.URL + "/v1"}, "qwen3.8", ServingRequest{
			ID: "r-non-sse", Messages: []ChatMessage{{Role: "user", Content: "hi"}}, MaxOutputTokens: 8,
		}, 2*time.Second)
		if s.Status != "ok" || s.StreamMode != "non_sse" {
			t.Fatalf("non_sse status/mode = %+v", s)
		}
		if s.TTFTMillis != nil || s.ITLMillis != nil || s.TPOTMillis != nil || s.OutputEvents != 0 || s.ReasoningEvents != 0 {
			t.Fatalf("non_sse leaked stream timings/events: %+v", s)
		}
		wantEst := EstimateTokens("thinking step answer")
		if s.OutputTokenEstimate != wantEst || s.OutputTokenCountBasis != "estimated_content_tokens" || s.EndToEndMillis <= 0 {
			t.Fatalf("non_sse accounting = est:%d (want %d) basis:%q e2e:%v", s.OutputTokenEstimate, wantEst, s.OutputTokenCountBasis, s.EndToEndMillis)
		}
	})

	t.Run("non_sse_fold_strips_synthetic_stream_metrics_and_refuses_ttft_tpot_slo", func(t *testing.T) {
		bogusTTFT, bogusTPOT := 10.0, 5.0
		stats := FoldServingSamplesWithSLOs([]ServingSample{{
			ID: "non-sse", Status: "ok", StreamMode: "non_sse",
			TTFTMillis: &bogusTTFT, ITLMillis: []float64{4.0}, TPOTMillis: &bogusTPOT,
			EndToEndMillis: 50, OutputTokenEstimate: 4,
		}}, 1.0, ServingSLOs{TTFT: time.Second, TPOT: time.Second, EndToEnd: time.Second})
		if stats.TTFTMillis.Status != "not_measured" || stats.ITLMillis.Status != "not_measured" || stats.TPOTMillis.Status != "not_measured" {
			t.Fatalf("non_sse fold accepted synthetic stream timings: %+v", stats)
		}
		if stats.GoodputRPS.Status != "measured" || stats.GoodputRPS.Value == nil || *stats.GoodputRPS.Value != 0 {
			t.Fatalf("non_sse sample passed TTFT/TPOT SLO: %+v", stats.GoodputRPS)
		}
	})
}

// fak-test:justify why=contract when=changed:internal/webbench/**
// fak-test:runtime fast est=5ms lane=default
func TestObservedTrackWallSecondsFloorAndSkippedContract(t *testing.T) {
	if got := observedTrackWallSeconds(0, []ServingSample{{Status: "ok"}}); got <= 0 {
		t.Fatalf("executed wire sample with 0s duration got wall=%v, want > 0", got)
	}
	if got := observedTrackWallSeconds(0, []ServingSample{{Status: "fail", HTTPStatus: http.StatusServiceUnavailable}}); got <= 0 {
		t.Fatalf("HTTP error wire sample with 0s duration got wall=%v, want > 0", got)
	}
	if got := observedTrackWallSeconds(0, []ServingSample{{Status: "fail", ErrorClass: "cancelled"}}); got != 0 {
		t.Fatalf("cancelled non-wire sample got wall=%v, want 0", got)
	}
	if got := observedTrackWallSeconds(0, nil); got != 0 {
		t.Fatalf("empty skipped samples got wall=%v, want 0", got)
	}
	if got := observedTrackWallSeconds(25*time.Millisecond, nil); got != 0.025 {
		t.Fatalf("positive elapsed got wall=%v, want 0.025", got)
	}

	wall := observedTrackWallSeconds(0, []ServingSample{{Status: "ok"}})
	sample := ServingSample{Status: "ok", EndToEndMillis: 0, OutputTokensExact: intPtr(2), OutputTokenEstimate: 2}
	measured := FoldServingSamples([]ServingSample{sample}, wall, time.Second)
	if measured.WallSeconds == nil || *measured.WallSeconds <= 0 {
		t.Fatalf("WallSeconds = %v, want positive", measured.WallSeconds)
	}
	if measured.ThroughputTokensS.Status != "measured" || measured.GoodputRPS.Status != "measured" || measured.GoodputTokensS.Status != "measured" {
		t.Fatalf("wire-executed floor failed to keep metrics measured: %+v", measured)
	}
	unmeasured := FoldServingSamples([]ServingSample{sample}, 0, time.Second)
	if unmeasured.WallSeconds != nil || unmeasured.ThroughputTokensS.Status != "not_measured" || unmeasured.GoodputTokensS.Status != "not_measured" {
		t.Fatalf("wallSeconds=0 contract violated: %+v", unmeasured)
	}
}
