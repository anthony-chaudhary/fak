package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

func TestMLXEngineIsRegisteredLifecycleDriver(t *testing.T) {
	eng := abi.Engine(MLXEngineID)
	if eng == nil {
		t.Fatalf("engine %q is not registered", MLXEngineID)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("engine %q must implement the lifecycle seam", MLXEngineID)
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("engine %q must advertise lifecycle support", MLXEngineID)
	}
}

// TestMLXHonestyFenceStaysWholePrefix pins the same SupportsExactSpan=false honesty
// fence the three sibling ride adapters keep: MLX's public control plane exposes at
// most a whole-prefix reset, never fak's exact middle-span evict, so the adapter
// advertises engine.cache.whole-prefix and never an exact-span capability.
func TestMLXHonestyFenceStaysWholePrefix(t *testing.T) {
	if enginecache.SupportsExactSpan(enginecache.Engine(MLXEngineID)) {
		t.Fatalf("MLX must not claim exact-span eviction; public control plane is whole-prefix only")
	}
	caps := abi.Engine(MLXEngineID).Caps()
	var whole, exact bool
	for _, c := range caps {
		switch c {
		case "engine.cache.whole-prefix":
			whole = true
		case "engine.cache.exact-span":
			exact = true
		}
	}
	if !whole || exact {
		t.Fatalf("MLX caps must advertise whole-prefix and NOT exact-span: %v", caps)
	}
}

func TestMLXHTTPAdapterStreamsThroughOpenAIFrontend(t *testing.T) {
	ctx := context.Background()
	seen := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s, want /v1/chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want Bearer test-key", got)
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
		io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"delta\":{\"content\":\"ml\"}}]}\n\n")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	e := NewMLXEngine(MLXConfig{
		BaseURL:  srv.URL + "/v1",
		Model:    "served",
		APIKey:   "test-key",
		WorkerID: "mlx-metal",
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
	if res.Meta["engine"] != MLXEngineID || res.Meta["worker"] != "mlx-metal" || res.Meta["finish_reason"] != "stop" {
		t.Fatalf("unexpected result meta: %+v", res.Meta)
	}
	if res.Meta["input_tokens"] != "4" || res.Meta["output_tokens"] != "2" || res.Meta["total_tokens"] != "6" {
		t.Fatalf("unexpected token meta: %+v", res.Meta)
	}
	if !strings.Contains(string(res.Payload.Inline), `"text":"mlx"`) {
		t.Fatalf("payload missing assembled MLX text: %s", res.Payload.Inline)
	}
	body := <-seen
	if body["stream"] != true || body["stream_options"] == nil {
		t.Fatalf("MLX request was not forced into streaming mode: %#v", body)
	}
}

func TestMLXCompletionsEndpointRoutes(t *testing.T) {
	ctx := context.Background()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("path = %s, want /v1/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"text\":\"ok\",\"finish_reason\":\"stop\"}]}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	e := NewMLXEngine(MLXConfig{BaseURL: srv.URL + "/v1", Model: "served"})
	res, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "completions",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"prompt":"hi"}`)},
		Meta: map[string]string{"openai_endpoint": "completions"},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if res.Meta["engine"] != MLXEngineID || res.Meta["endpoint"] != "completions" {
		t.Fatalf("completions not adapter-tagged: %+v", res.Meta)
	}
}

// TestMLXPrometheusNormalizationRetagsVLLMSurface proves vllm-mlx's vLLM-format
// metrics normalize into fak_serving_* under engine="mlx" and that the raw vllm:*
// names never leak into the normalized output.
func TestMLXPrometheusNormalizationRetagsVLLMSurface(t *testing.T) {
	snap := ParseMLXPrometheus("mlx-metal", `
vllm:time_to_first_token_seconds_sum 0.42
vllm:time_to_first_token_seconds_count 3
vllm:num_requests_running 2
vllm:kv_cache_usage_perc 0.5
`)
	if snap.Engine != MLXEngineID || snap.WorkerID != "mlx-metal" {
		t.Fatalf("snapshot not tagged engine=mlx worker=mlx-metal: %+v", snap)
	}
	if snap.TTFT.Count != 3 || snap.RequestsRunning != 2 {
		t.Fatalf("vLLM-format metrics not normalized: %+v", snap)
	}
	prom := snap.Prometheus()
	if !strings.Contains(prom, `fak_serving_ttft_seconds_count{engine="mlx",worker="mlx-metal"} 3`) {
		t.Fatalf("normalized output missing engine=mlx TTFT row:\n%s", prom)
	}
	if strings.Contains(prom, "vllm:") {
		t.Fatalf("normalized output leaks raw vllm:* metric names:\n%s", prom)
	}
}

// TestMLXPrometheusEmptySurfaceFabricatesNothing pins the "document gaps rather than
// fabricate" fence: a plain mlx-lm server that exposes no vllm:* metrics normalizes
// to an empty (engine="mlx") snapshot, not invented counters.
func TestMLXPrometheusEmptySurfaceFabricatesNothing(t *testing.T) {
	snap := ParseMLXPrometheus("mlx", "# mlx-lm exposes no prometheus surface\n")
	if snap.Engine != MLXEngineID {
		t.Fatalf("engine tag lost on empty surface: %+v", snap)
	}
	if snap.TTFT.Count != 0 || snap.RequestsRunning != 0 || snap.PrefixCacheHitRatio != nil {
		t.Fatalf("empty MLX surface fabricated metrics: %+v", snap)
	}
}

// TestMLXLiveWorkerSmoke is the env-gated LIVE witness (mirroring
// vllm_live_smoke_test.go's GPU-gated pattern). The GPU-free tests above front a
// MOCK upstream and prove the protocol lowering adapter-free; what they cannot cover
// is driving a REAL mlx-lm/vllm-mlx server on Apple Silicon. Point this at one and
// it drives the adapter end-to-end over the server's public OpenAI surface:
//
//	FAK_MLX_BASE_URL=http://<host>:8080/v1 \
//	FAK_MLX_MODEL=<served-model-id> \
//	[FAK_MLX_METRICS_URL=http://<host>:8000/metrics] \
//	[FAK_MLX_API_KEY=<key>] \
//	go test ./internal/engine -run TestMLXLiveWorkerSmoke -count=1 -v
//
// Without FAK_MLX_BASE_URL / FAK_MLX_MODEL it SKIPS: the adapter-free CI proves it
// compiles and the gate is wired; it does NOT prove live serving (that needs an
// Apple Silicon box this dev host is not). This is the gen/next "prototype behind an
// explicit gate" — never a fabricated pass.
//
// Promotion evidence (toward gen/now): this passing against a real mlx-lm/vllm-mlx
// server is the live half of the issue's done condition. Demotion/retirement
// evidence: if the epic's native Metal Track B ships the base serving path first,
// the ride's value ("serve the current SOTA Mac runtime before native is ready")
// collapses and this gate retires with the adapter. Invalidating assumption: it
// assumes mlx-lm/vllm-mlx speaks the public OpenAI /chat/completions + /completions
// SSE contract the adapter lowers to — an MLX release changing that wire is the
// intended failure site, to be re-witnessed, not assumed.
func TestMLXLiveWorkerSmoke(t *testing.T) {
	cfg := EnvMLXConfig()
	if strings.TrimSpace(cfg.BaseURL) == "" || strings.TrimSpace(cfg.Model) == "" {
		t.Skip("live MLX smoke: set FAK_MLX_BASE_URL and FAK_MLX_MODEL to a real mlx-lm/vllm-mlx server; skipped off Apple Silicon")
	}
	e := NewMLXEngine(cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	chat, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(
			`{"messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":16,"temperature":0}`)},
	})
	if err != nil {
		t.Fatalf("live chat Complete: %v", err)
	}
	if chat == nil || chat.Status != abi.StatusOK {
		t.Fatalf("live chat result = %+v, want StatusOK", chat)
	}
	if chat.Meta["engine"] != MLXEngineID || chat.Meta["endpoint"] != "chat" {
		t.Fatalf("live chat result not adapter-tagged: %+v", chat.Meta)
	}
}
