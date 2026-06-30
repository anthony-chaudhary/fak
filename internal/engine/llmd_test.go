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
)

func TestLLMDEngineIsRegisteredLifecycleDriver(t *testing.T) {
	eng := abi.Engine(LLMDEngineID)
	if eng == nil {
		t.Fatalf("engine %q is not registered", LLMDEngineID)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("engine %q must implement the lifecycle seam", LLMDEngineID)
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("engine %q must advertise lifecycle support", LLMDEngineID)
	}
	found := false
	for _, id := range abi.EngineIDs() {
		if id == LLMDEngineID {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("engine id %q absent from the dispatchable registry %v", LLMDEngineID, abi.EngineIDs())
	}
}

func TestLLMDHTTPAdapterStreamsThroughOpenAIFrontend(t *testing.T) {
	ctx := context.Background()
	seen := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s, want /v1/chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer llmd-key" {
			t.Fatalf("Authorization = %q, want Bearer llmd-key", got)
		}
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request body JSON: %v", err)
		}
		seen <- body
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"delta\":{\"content\":\"llm\"}}]}\n\n")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"-d\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	e := NewLLMDEngine(LLMDConfig{
		BaseURL:  srv.URL + "/v1",
		Model:    "served",
		APIKey:   "llmd-key",
		WorkerID: "llmd-front",
	})
	res, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK", res)
	}
	if res.Meta["engine"] != LLMDEngineID || res.Meta["worker"] != "llmd-front" || res.Meta["finish_reason"] != "stop" {
		t.Fatalf("unexpected result meta: %+v", res.Meta)
	}
	if res.Meta["input_tokens"] != "5" || res.Meta["output_tokens"] != "2" || res.Meta["total_tokens"] != "7" {
		t.Fatalf("unexpected token meta: %+v", res.Meta)
	}
	if !strings.Contains(string(res.Payload.Inline), `"text":"llm-d"`) {
		t.Fatalf("payload missing assembled llm-d text: %s", res.Payload.Inline)
	}
	body := <-seen
	if body["stream"] != true || body["stream_options"] == nil {
		t.Fatalf("llm-d request was not forced into streaming mode: %#v", body)
	}
}

func TestLLMDPrometheusNormalizationUsesLLMDEngineIdentity(t *testing.T) {
	snap := ParseLLMDPrometheus("", `
vllm:time_to_first_token_seconds_sum{model_name="qwen"} 0.9
vllm:time_to_first_token_seconds_count{model_name="qwen"} 3
vllm:request_time_per_output_token_seconds_sum 1.2
vllm:request_time_per_output_token_seconds_count 4
vllm:kv_cache_usage_perc 0.7
vllm:num_requests_running 2
vllm:num_requests_waiting 5
vllm:request_success_total 11
vllm:prefix_cache_queries 10
vllm:prefix_cache_hits 6
`)
	if snap.Engine != LLMDEngineID || snap.WorkerID != LLMDEngineID {
		t.Fatalf("snapshot identity = %q/%q, want llm-d/llm-d", snap.Engine, snap.WorkerID)
	}
	if snap.TTFT.Sum != 0.9 || snap.TTFT.Count != 3 || snap.TPOT.Count != 4 {
		t.Fatalf("latency metrics not normalized: %+v", snap)
	}
	if snap.KVCacheUsage != 0.7 || snap.RequestsRunning != 2 || snap.RequestsWaiting != 5 || snap.PrefixHits != 6 || snap.PrefixQueries != 10 {
		t.Fatalf("serving gauges/counters not normalized: %+v", snap)
	}
	prom := snap.Prometheus()
	for _, want := range []string{
		`fak_serving_ttft_seconds_sum{engine="llm-d",worker="llm-d"} 0.9`,
		`fak_serving_tpot_seconds_count{engine="llm-d",worker="llm-d"} 4`,
		`fak_serving_kv_cache_usage_ratio{engine="llm-d",worker="llm-d"} 0.7`,
		`fak_serving_prefix_cache_hits_total{engine="llm-d",worker="llm-d"} 6`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
	if strings.Contains(prom, "vllm:") {
		t.Fatalf("normalized output leaks vLLM metric names:\n%s", prom)
	}
}

func TestLLMDEnvConfigAcceptsHyphenFriendlyAliases(t *testing.T) {
	t.Setenv("FAK_LLMD_BASE_URL", "")
	t.Setenv("FAK_LLM_D_BASE_URL", "http://llmd.example/v1")
	t.Setenv("FAK_LLMD_MODEL", "")
	t.Setenv("FAK_LLM_D_MODEL", "model-a")
	t.Setenv("FAK_LLMD_API_KEY", "")
	t.Setenv("FAK_LLM_D_API_KEY", "secret")
	t.Setenv("FAK_LLMD_WORKER_ID", "")
	t.Setenv("FAK_LLM_D_WORKER_ID", "front-a")
	t.Setenv("FAK_LLMD_METRICS_URL", "")
	t.Setenv("FAK_LLM_D_METRICS_URL", "http://llmd.example/metrics")

	cfg := EnvLLMDConfig()
	if cfg.BaseURL != "http://llmd.example/v1" ||
		cfg.Model != "model-a" ||
		cfg.APIKey != "secret" ||
		cfg.WorkerID != "front-a" ||
		cfg.MetricsURL != "http://llmd.example/metrics" {
		t.Fatalf("EnvLLMDConfig did not read FAK_LLM_D_* aliases: %+v", cfg)
	}
}
