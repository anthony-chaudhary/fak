package macobs

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestParseMLXMetricsVLLM(t *testing.T) {
	mockVLLMMetrics := `
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 4
vllm:num_requests_waiting 2
vllm:kv_cache_usage_perc 65.5
vllm:time_to_first_token_seconds_sum 1.2
vllm:time_to_first_token_seconds_count 10
vllm:inter_token_latency_seconds_sum 0.25
vllm:inter_token_latency_seconds_count 50
vllm:avg_prompt_throughput_tok_per_s 120.5
vllm:avg_generation_throughput_tok_per_s 35.2
vllm:prefix_cache_hits 80
vllm:prefix_cache_queries 100
`
	srv, prefix := ParseMLXMetrics(mockVLLMMetrics)
	if !srv.Available {
		t.Fatalf("expected srv.Available to be true")
	}
	if srv.ServerType != "vllm-mlx" {
		t.Errorf("got ServerType %s, want vllm-mlx", srv.ServerType)
	}
	if srv.ActiveRequests != 4 {
		t.Errorf("got ActiveRequests %d, want 4", srv.ActiveRequests)
	}
	if srv.QueuedRequests != 2 {
		t.Errorf("got QueuedRequests %d, want 2", srv.QueuedRequests)
	}
	if srv.KVCacheUsagePct != 65.5 {
		t.Errorf("got KVCacheUsagePct %f, want 65.5", srv.KVCacheUsagePct)
	}
	// TTFT: (1.2 / 10) * 1000 = 120ms
	if srv.AvgTTFTMs != 120.0 {
		t.Errorf("got AvgTTFTMs %f, want 120.0", srv.AvgTTFTMs)
	}
	// ITL: (0.25 / 50) * 1000 = 5ms
	if srv.AvgITLMs != 5.0 {
		t.Errorf("got AvgITLMs %f, want 5.0", srv.AvgITLMs)
	}
	if srv.PromptTokensPerSec != 120.5 {
		t.Errorf("got PromptTokensPerSec %f, want 120.5", srv.PromptTokensPerSec)
	}
	if srv.DecodeTokensPerSec != 35.2 {
		t.Errorf("got DecodeTokensPerSec %f, want 35.2", srv.DecodeTokensPerSec)
	}

	// Prefix Cache
	if !prefix.Available {
		t.Fatalf("expected prefix.Available to be true")
	}
	if prefix.Hits != 80 {
		t.Errorf("got prefix.Hits %d, want 80", prefix.Hits)
	}
	if prefix.QueriedBlocks != 100 {
		t.Errorf("got prefix.QueriedBlocks %d, want 100", prefix.QueriedBlocks)
	}
	if prefix.HitRatio != 0.8 {
		t.Errorf("got prefix.HitRatio %f, want 0.8", prefix.HitRatio)
	}
	if srv.PrefixCacheHitRatio != 0.8 {
		t.Errorf("got srv.PrefixCacheHitRatio %f, want 0.8", srv.PrefixCacheHitRatio)
	}
}

func TestParseMLXMetricsMLXLM(t *testing.T) {
	mockMLXLMMetrics := `
mlx_active_requests 2
mlx_queued_requests 0
mlx_kv_cache_usage_pct 42.0
mlx_prompt_tokens_per_sec 95.0
mlx_decode_tokens_per_sec 28.5
mlx_ttft_seconds_sum 0.5
mlx_ttft_seconds_count 5
mlx_itl_seconds_sum 0.1
mlx_itl_seconds_count 10
mlx:prefix_cache_hit_ratio 0.75
`
	srv, prefix := ParseMLXMetrics(mockMLXLMMetrics)
	if !srv.Available {
		t.Fatalf("expected srv.Available to be true")
	}
	if srv.ServerType != "mlx-lm" {
		t.Errorf("got ServerType %s, want mlx-lm", srv.ServerType)
	}
	if srv.ActiveRequests != 2 {
		t.Errorf("got ActiveRequests %d, want 2", srv.ActiveRequests)
	}
	if srv.KVCacheUsagePct != 42.0 {
		t.Errorf("got KVCacheUsagePct %f, want 42.0", srv.KVCacheUsagePct)
	}
	if srv.AvgTTFTMs != 100.0 {
		t.Errorf("got AvgTTFTMs %f, want 100.0", srv.AvgTTFTMs)
	}
	if srv.AvgITLMs != 10.0 {
		t.Errorf("got AvgITLMs %f, want 10.0", srv.AvgITLMs)
	}
	if srv.PromptTokensPerSec != 95.0 {
		t.Errorf("got PromptTokensPerSec %f, want 95.0", srv.PromptTokensPerSec)
	}
	if srv.DecodeTokensPerSec != 28.5 {
		t.Errorf("got DecodeTokensPerSec %f, want 28.5", srv.DecodeTokensPerSec)
	}
	if prefix.HitRatio != 0.75 {
		t.Errorf("got prefix.HitRatio %f, want 0.75", prefix.HitRatio)
	}
}

func TestParseMLXMetricsEmpty(t *testing.T) {
	srv, prefix := ParseMLXMetrics("")
	if srv.Available {
		t.Errorf("expected srv.Available to be false for empty body")
	}
	if prefix.Available {
		t.Errorf("expected prefix.Available to be false for empty body")
	}
}

func TestScrapeMLXMetrics(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("vllm:num_requests_running 1\nvllm:kv_cache_usage_perc 25.0\n"))
	}))
	defer ts.Close()

	ctx := context.Background()
	srv, _, err := ScrapeMLXMetrics(ctx, ts.URL, ts.Client())
	if err != nil {
		t.Fatalf("ScrapeMLXMetrics failed: %v", err)
	}
	if !srv.Available {
		t.Fatalf("expected srv.Available to be true")
	}
	if srv.ActiveRequests != 1 {
		t.Errorf("got ActiveRequests %d, want 1", srv.ActiveRequests)
	}
	if srv.KVCacheUsagePct != 25.0 {
		t.Errorf("got KVCacheUsagePct %f, want 25.0", srv.KVCacheUsagePct)
	}
}

func TestScrapeMLXMetricsErrors(t *testing.T) {
	// Empty URL -> fail soft
	srv, _, err := ScrapeMLXMetrics(context.Background(), "", nil)
	if err != nil {
		t.Fatalf("expected nil error on empty URL, got: %v", err)
	}
	if srv.Available {
		t.Errorf("expected srv.Available false on empty URL")
	}

	// 500 Server Error
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts.Close()

	_, _, err = ScrapeMLXMetrics(context.Background(), ts.URL, ts.Client())
	if err == nil {
		t.Fatalf("expected error on 500 status")
	}
}

func TestParseMLXMetrics_EdgeCases(t *testing.T) {
	// 1. Lines with NaN, Inf, and malformed syntax
	corruptMetrics := `
# Valid comment
vllm:num_requests_running NaN
vllm:num_requests_waiting +Inf
vllm:kv_cache_usage_perc -Inf
vllm:invalid_line_no_val
vllm:broken_label{key= 42.0
vllm:valid_metric{type="prompt"} 125.0
`
	srv, _ := ParseMLXMetrics(corruptMetrics)
	if !srv.Available {
		t.Fatalf("expected srv.Available true for partial valid metrics")
	}

	// 2. GPU cache usage factor (0.0 to 1.0 -> multiplied by 100)
	factorMetrics := `
vllm:gpu_cache_usage_factor 0.685
mlx:tokens_per_second{type="prompt"} 145.0
mlx:tokens_per_second{type="decode"} 42.5
`
	srvFactor, _ := ParseMLXMetrics(factorMetrics)
	if math.Abs(srvFactor.KVCacheUsagePct-68.5) > 1e-4 {
		t.Errorf("got KVCacheUsagePct %f, want 68.5", srvFactor.KVCacheUsagePct)
	}
	if srvFactor.PromptTokensPerSec != 145.0 {
		t.Errorf("got PromptTokensPerSec %f, want 145.0", srvFactor.PromptTokensPerSec)
	}
	if srvFactor.DecodeTokensPerSec != 42.5 {
		t.Errorf("got DecodeTokensPerSec %f, want 42.5", srvFactor.DecodeTokensPerSec)
	}

	// 3. Prefix cache with hits and misses (total inferred as hits+misses)
	prefixHitMissMetrics := `
mlx:prefix_cache_hits 150
mlx:prefix_cache_misses 50
`
	_, prefixHM := ParseMLXMetrics(prefixHitMissMetrics)
	if !prefixHM.Available {
		t.Fatalf("expected prefix.Available true")
	}
	if prefixHM.Hits != 150 {
		t.Errorf("got Hits %d, want 150", prefixHM.Hits)
	}
	if prefixHM.Misses != 50 {
		t.Errorf("got Misses %d, want 50", prefixHM.Misses)
	}
	if prefixHM.QueriedBlocks != 200 {
		t.Errorf("got QueriedBlocks %d, want 200", prefixHM.QueriedBlocks)
	}
	if prefixHM.HitRatio != 0.75 {
		t.Errorf("got HitRatio %f, want 0.75", prefixHM.HitRatio)
	}
}
