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

func TestVLLMEngineIsRegisteredLifecycleDriver(t *testing.T) {
	eng := abi.Engine(VLLMEngineID)
	if eng == nil {
		t.Fatalf("engine %q is not registered", VLLMEngineID)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("engine %q must implement the lifecycle seam", VLLMEngineID)
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("engine %q must advertise lifecycle support", VLLMEngineID)
	}
}

func TestVLLMHTTPAdapterStreamsChatAndCompletions(t *testing.T) {
	ctx := context.Background()
	type seenRequest struct {
		path string
		body map[string]any
	}
	seen := make(chan seenRequest, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
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
		seen <- seenRequest{path: r.URL.Path, body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		switch r.URL.Path {
		case "/v1/chat/completions":
			io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
		case "/v1/completions":
			io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"text\":\"o\"}]}\n\n")
			io.WriteString(w, "data: {\"choices\":[{\"text\":\"k\",\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	e := NewVLLMEngine(VLLMConfig{
		BaseURL:  srv.URL + "/v1",
		Model:    "served",
		APIKey:   "test-key",
		WorkerID: "worker-a",
	})

	chat, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
	})
	if err != nil {
		t.Fatalf("chat Complete: %v", err)
	}
	assertVLLMResult(t, ctx, chat, "chat", "hello", "stop", "3", "2", "5")

	comp, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "completions",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"prompt":"hi"}`)},
		Meta: map[string]string{"openai_endpoint": "completions"},
	})
	if err != nil {
		t.Fatalf("completions Complete: %v", err)
	}
	assertVLLMResult(t, ctx, comp, "completions", "ok", "length", "7", "2", "9")

	first := <-seen
	if first.path != "/v1/chat/completions" {
		t.Fatalf("first path = %s, want chat completions", first.path)
	}
	if first.body["stream"] != true {
		t.Fatalf("chat stream flag = %#v, want true", first.body["stream"])
	}
	if first.body["stream_options"] == nil {
		t.Fatalf("chat request missing stream_options: %#v", first.body)
	}
	second := <-seen
	if second.path != "/v1/completions" {
		t.Fatalf("second path = %s, want completions", second.path)
	}
	if second.body["stream"] != true || second.body["prompt"] != "hi" {
		t.Fatalf("completion body not normalized: %#v", second.body)
	}
}

func assertVLLMResult(t *testing.T, ctx context.Context, res *abi.Result, endpoint, text, finish, in, out, total string) {
	t.Helper()
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK", res)
	}
	if res.Meta["engine"] != VLLMEngineID || res.Meta["endpoint"] != endpoint || res.Meta["finish_reason"] != finish {
		t.Fatalf("unexpected result meta: %+v", res.Meta)
	}
	if res.Meta["input_tokens"] != in || res.Meta["output_tokens"] != out || res.Meta["total_tokens"] != total {
		t.Fatalf("unexpected token meta: %+v", res.Meta)
	}
	body := res.Payload.Inline
	if res.Payload.Kind != abi.RefInline {
		resolver := abi.ActiveResolver()
		if resolver == nil {
			t.Fatalf("payload was %v but ActiveResolver is nil", res.Payload.Kind)
		}
		b, err := resolver.Resolve(ctx, res.Payload)
		if err != nil {
			t.Fatalf("resolve payload: %v", err)
		}
		body = b
	}
	if !strings.Contains(string(body), `"text":"`+text+`"`) {
		t.Fatalf("payload missing assembled text %q: %s", text, body)
	}
}

func TestVLLMKVEventSubscriptionFeedsResidencyAndCacheMetrics(t *testing.T) {
	idx := NewPrefixResidencyIndex()
	rec := NewCacheEventRecorder()
	src := NewVLLMJSONKVEventSource(io.NopCloser(strings.NewReader(strings.Join([]string{
		`{"ts":1.25,"worker_id":"worker-a","model_id":"m","tokenizer_id":"tok","events":[{"type":"BlockStored","block_hashes":["h1","h2"],"token_ids":[1,2,3,4],"block_size":2,"medium":"GPU","group_idx":0},{"type":"BlockRemoved","block_hashes":["h1"],"block_size":2,"medium":"CPU","group_idx":0}]}`,
		``,
	}, "\n"))))
	e := NewVLLMEngine(VLLMConfig{
		WorkerID:      "worker-a",
		Model:         "m",
		Residency:     idx,
		CacheRecorder: rec,
		KVEvents:      src,
	})

	if err := e.RunKVEventSubscription(context.Background()); err != nil {
		t.Fatalf("RunKVEventSubscription: %v", err)
	}
	if idx.Has("worker-a", "h1") {
		t.Fatal("removed block h1 is still resident")
	}
	if !idx.Has("worker-a", "h2") {
		t.Fatal("stored block h2 was not marked resident")
	}
	rows := idx.Snapshot("worker-a")
	if len(rows) != 1 || rows[0].ModelID != "m" || rows[0].TokenizerID != "tok" || rows[0].Tokens != 2 {
		t.Fatalf("residency row not normalized: %+v", rows)
	}
	snap := rec.Metrics().Snapshot()
	if snap.Events != 3 || snap.Hits != 3 {
		t.Fatalf("cache event metrics not fed by KV events: %+v", snap)
	}
}

func TestVLLMPrometheusNormalization(t *testing.T) {
	snap := ParseVLLMPrometheus("worker-a", `
vllm:time_to_first_token_seconds_sum{model_name="m"} 1.5
vllm:time_to_first_token_seconds_count{model_name="m"} 3
vllm:request_time_per_output_token_seconds_sum 2.5
vllm:request_time_per_output_token_seconds_count 5
vllm:inter_token_latency_seconds_sum 0.25
vllm:inter_token_latency_seconds_count 4
vllm:request_queue_time_seconds_sum 0.75
vllm:request_queue_time_seconds_count 6
vllm:kv_cache_usage_perc 0.8
vllm:num_requests_running 2
vllm:num_requests_waiting 1
vllm:num_requests_swapped 0
vllm:request_success_total 9
vllm:prefix_cache_queries 11
vllm:prefix_cache_hits 7
`)
	if snap.TTFT.Sum != 1.5 || snap.TTFT.Count != 3 || snap.TPOT.Sum != 2.5 || snap.TPOT.Count != 5 {
		t.Fatalf("serving latency metrics not normalized: %+v", snap)
	}
	if snap.KVCacheUsage != 0.8 || snap.RequestsRunning != 2 || snap.PrefixQueries != 11 || snap.PrefixHits != 7 {
		t.Fatalf("serving gauges/counters not normalized: %+v", snap)
	}
	prom := snap.Prometheus()
	for _, want := range []string{
		`fak_serving_ttft_seconds_sum{engine="vllm",worker="worker-a"} 1.5`,
		`fak_serving_tpot_seconds_count{engine="vllm",worker="worker-a"} 5`,
		`fak_serving_kv_cache_usage_ratio{engine="vllm",worker="worker-a"} 0.8`,
		`fak_serving_prefix_cache_hits_total{engine="vllm",worker="worker-a"} 7`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
}
