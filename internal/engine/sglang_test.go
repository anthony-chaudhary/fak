package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

// TestSGLangIsRouterDispatchableNotProxyOnly pins acceptance #1: SGLang is a
// first-class registered EngineDriver reachable through the kernel engine registry
// (what the fleet router dispatches over), not only as the gateway's single
// hardcoded BaseURL proxy.
func TestSGLangIsRouterDispatchableNotProxyOnly(t *testing.T) {
	eng := abi.Engine(SGLangEngineID)
	if eng == nil {
		t.Fatalf("engine %q is not registered — SGLang is still reachable only as a proxy BaseURL", SGLangEngineID)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("engine %q must implement the lifecycle seam", SGLangEngineID)
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("engine %q must advertise lifecycle support", SGLangEngineID)
	}
	found := false
	for _, id := range abi.EngineIDs() {
		if id == SGLangEngineID {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("engine id %q absent from the dispatchable registry %v", SGLangEngineID, abi.EngineIDs())
	}
}

func TestSGLangNativeGenerateStreams(t *testing.T) {
	ctx := context.Background()
	type seenRequest struct {
		path string
		body map[string]any
	}
	seen := make(chan seenRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}
		if r.URL.Path != "/generate" {
			t.Errorf("path = %s, want /generate", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body JSON: %v", err)
		}
		seen <- seenRequest{path: r.URL.Path, body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		// SGLang streams CUMULATIVE text per chunk, with meta_info carrying token
		// accounting, the RadixAttention cached_tokens hit count, and the finish
		// reason (null mid-stream, an object on the terminal chunk).
		io.WriteString(w, "data: {\"text\":\"hel\",\"meta_info\":{\"prompt_tokens\":3,\"cached_tokens\":2,\"finish_reason\":null,\"id\":\"req-1\"}}\n\n")
		io.WriteString(w, "data: {\"text\":\"hello\",\"meta_info\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"cached_tokens\":2,\"finish_reason\":{\"type\":\"stop\"},\"id\":\"req-1\"}}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	rec := NewCacheEventRecorder()
	e := NewSGLangEngine(SGLangConfig{
		BaseURL:       srv.URL,
		Model:         "served",
		APIKey:        "test-key",
		WorkerID:      "worker-a",
		CacheRecorder: rec,
	})

	res, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "generate",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"text":"hi","sampling_params":{"max_new_tokens":2}}`)},
	})
	if err != nil {
		t.Fatalf("generate Complete: %v", err)
	}
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK", res)
	}
	if res.Meta["engine"] != SGLangEngineID || res.Meta["endpoint"] != "generate" || res.Meta["finish_reason"] != "stop" {
		t.Fatalf("unexpected result meta: %+v", res.Meta)
	}
	if res.Meta["input_tokens"] != "3" || res.Meta["output_tokens"] != "2" || res.Meta["total_tokens"] != "5" || res.Meta["cached_tokens"] != "2" {
		t.Fatalf("unexpected token meta: %+v", res.Meta)
	}
	if !strings.Contains(string(res.Payload.Inline), `"text":"hello"`) {
		t.Fatalf("payload missing assembled cumulative text: %s", res.Payload.Inline)
	}

	got := <-seen
	if got.body["stream"] != true {
		t.Fatalf("generate stream flag = %#v, want true", got.body["stream"])
	}
	if got.body["text"] != "hi" {
		t.Fatalf("generate body text not passed through: %#v", got.body)
	}

	// The RadixAttention prefix-cache hit (cached_tokens) is folded into the shared
	// cache-event stream as a typed restore, not left an invisible internal counter.
	snap := rec.Metrics().Snapshot()
	if snap.Events != 1 || snap.Hits != 1 {
		t.Fatalf("cached_tokens prefix hit not recorded as a cache event: %+v", snap)
	}
}

// TestSGLangRadixPollFeedsResidencyIndex pins acceptance #2: the RadixAttention
// cache-aware signal is consumed from a (stub) radix endpoint and feeds the
// per-worker prefix-residency index, replacing the worker's prior residency.
func TestSGLangRadixPollFeedsResidencyIndex(t *testing.T) {
	idx := NewPrefixResidencyIndex()
	idx.Store("worker-a", PrefixResidency{WorkerID: "worker-a", Digest: "stale"})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/radix" {
			t.Errorf("path = %s, want /radix", r.URL.Path)
		}
		io.WriteString(w, `{"worker_id":"worker-a","model_id":"m","tokenizer_id":"tok","ts":1.5,"resident":[`+
			`{"digest":"h1","tokens":16,"medium":"GPU","block_size":16},`+
			`{"hash":"h2","tokens":16,"medium":"CPU"}]}`)
	}))
	defer srv.Close()

	e := NewSGLangEngine(SGLangConfig{
		BaseURL:   srv.URL,
		Model:     "m",
		WorkerID:  "worker-a",
		RadixURL:  srv.URL + "/radix",
		Residency: idx,
	})

	snap, err := e.PollRadixResidency(context.Background())
	if err != nil {
		t.Fatalf("PollRadixResidency: %v", err)
	}
	if len(snap.Resident) != 2 {
		t.Fatalf("snapshot resident count = %d, want 2", len(snap.Resident))
	}
	if idx.Has("worker-a", "stale") {
		t.Fatal("stale residency was not replaced by the snapshot")
	}
	if !idx.Has("worker-a", "h1") || !idx.Has("worker-a", "h2") {
		t.Fatalf("snapshot prefixes not stored in the residency index: %+v", idx.Snapshot("worker-a"))
	}
	rows := idx.Snapshot("worker-a")
	if len(rows) != 2 {
		t.Fatalf("residency row count = %d, want 2: %+v", len(rows), rows)
	}
	for _, row := range rows {
		if row.ModelID != "m" || row.TokenizerID != "tok" || row.Tokens != 16 {
			t.Fatalf("residency row not normalized: %+v", row)
		}
	}
}

func TestSGLangPrometheusNormalization(t *testing.T) {
	snap := ParseSGLangPrometheus("worker-a", `
sglang:time_to_first_token_seconds_sum 1.5
sglang:time_to_first_token_seconds_count 3
sglang:time_per_output_token_seconds_sum 2.5
sglang:time_per_output_token_seconds_count 5
sglang:inter_token_latency_seconds_sum 0.25
sglang:inter_token_latency_seconds_count 4
sglang:queue_time_seconds_sum 0.75
sglang:queue_time_seconds_count 6
sglang:token_usage 0.8
sglang:num_running_reqs 2
sglang:num_waiting_reqs 1
sglang:cache_hit_rate 0.42
`)
	if snap.Engine != SGLangEngineID || snap.WorkerID != "worker-a" {
		t.Fatalf("snapshot identity = %q/%q, want sglang/worker-a", snap.Engine, snap.WorkerID)
	}
	if snap.TTFT.Sum != 1.5 || snap.TTFT.Count != 3 || snap.TPOT.Sum != 2.5 || snap.TPOT.Count != 5 {
		t.Fatalf("serving latency metrics not normalized: %+v", snap)
	}
	if snap.ITL.Sum != 0.25 || snap.Queue.Count != 6 || snap.KVCacheUsage != 0.8 || snap.RequestsRunning != 2 || snap.RequestsWaiting != 1 {
		t.Fatalf("serving gauges not normalized: %+v", snap)
	}
	if snap.PrefixCacheHitRatio == nil || *snap.PrefixCacheHitRatio != 0.42 {
		t.Fatalf("prefix cache hit ratio not normalized: %+v", snap.PrefixCacheHitRatio)
	}
	// The output uses the SHARED fak_serving_* schema (the same the vLLM emitter
	// targets), tagged per-engine/per-worker — never SGLang-only metric names.
	prom := snap.Prometheus()
	for _, want := range []string{
		`fak_serving_ttft_seconds_sum{engine="sglang",worker="worker-a"} 1.5`,
		`fak_serving_tpot_seconds_count{engine="sglang",worker="worker-a"} 5`,
		`fak_serving_kv_cache_usage_ratio{engine="sglang",worker="worker-a"} 0.8`,
		`fak_serving_prefix_cache_hit_ratio{engine="sglang",worker="worker-a"} 0.42`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
	if strings.Contains(prom, "sglang:") {
		t.Fatalf("normalized output leaks SGLang-only metric names:\n%s", prom)
	}
}

// TestSGLangGovernanceResolvesToEngineSGLang pins acceptance #4: the serving adapter
// and the enginecache cache referee agree on engine identity, and the honest
// whole-prefix boundary (SupportsExactSpan==false) is preserved — no duplicate flush
// logic is added here.
func TestSGLangGovernanceResolvesToEngineSGLang(t *testing.T) {
	if SGLangEngineID != string(enginecache.EngineSGLang) {
		t.Fatalf("adapter engine id %q != enginecache identity %q", SGLangEngineID, enginecache.EngineSGLang)
	}
	if enginecache.SupportsExactSpan(enginecache.EngineSGLang) {
		t.Fatal("SGLang must not claim exact-span eviction — its public control plane is whole-prefix flush_cache")
	}
}

// vLLM keeping a nil PrefixCacheHitRatio must emit NO ratio line, so a 0.0 is never
// misread as a measured 0% hit rate (the conflation the pointer field guards).
func TestVLLMOmitsPrefixHitRatioLine(t *testing.T) {
	snap := ParseVLLMPrometheus("worker-a", "vllm:prefix_cache_hits 7\nvllm:prefix_cache_queries 11\n")
	if snap.PrefixCacheHitRatio != nil {
		t.Fatalf("vLLM should not populate a directly-reported hit ratio: %+v", snap.PrefixCacheHitRatio)
	}
	if strings.Contains(snap.Prometheus(), "fak_serving_prefix_cache_hit_ratio") {
		t.Fatalf("vLLM emitted a hit-ratio line it cannot measure:\n%s", snap.Prometheus())
	}
}
