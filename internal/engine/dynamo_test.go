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

func TestDynamoEngineIsRegisteredLifecycleDriver(t *testing.T) {
	eng := abi.Engine(DynamoEngineID)
	if eng == nil {
		t.Fatalf("engine %q is not registered", DynamoEngineID)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("engine %q must implement the lifecycle seam", DynamoEngineID)
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("engine %q must advertise lifecycle support", DynamoEngineID)
	}
}

func TestDynamoHTTPAdapterStreamsThroughOpenAIFrontend(t *testing.T) {
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
		io.WriteString(w, "data: {\"model\":\"served\",\"choices\":[{\"delta\":{\"content\":\"dy\"}}]}\n\n")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"namo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	e := NewDynamoEngine(DynamoConfig{
		BaseURL:  srv.URL + "/v1",
		Model:    "served",
		APIKey:   "test-key",
		WorkerID: "dynamo-front",
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
	if res.Meta["engine"] != DynamoEngineID || res.Meta["worker"] != "dynamo-front" || res.Meta["finish_reason"] != "stop" {
		t.Fatalf("unexpected result meta: %+v", res.Meta)
	}
	if res.Meta["input_tokens"] != "4" || res.Meta["output_tokens"] != "2" || res.Meta["total_tokens"] != "6" {
		t.Fatalf("unexpected token meta: %+v", res.Meta)
	}
	if !strings.Contains(string(res.Payload.Inline), `"text":"dynamo"`) {
		t.Fatalf("payload missing assembled Dynamo text: %s", res.Payload.Inline)
	}
	body := <-seen
	if body["stream"] != true || body["stream_options"] == nil {
		t.Fatalf("Dynamo request was not forced into streaming mode: %#v", body)
	}
}

func TestDynamoPrometheusNormalizationSurfacesPDWorkers(t *testing.T) {
	snap := ParseDynamoPrometheus("frontend", `
dynamo_frontend_worker_last_time_to_first_token_seconds{worker_id="prefill-0",worker_type="prefill"} 0.21
dynamo_frontend_worker_active_prefill_tokens{worker_id="prefill-0",worker_type="prefill"} 128
dynamo_component_requests_total{worker_id="prefill-0",worker_type="prefill"} 7
dynamo_frontend_worker_last_inter_token_latency_seconds{worker_id="decode-0",worker_type="backend"} 0.013
dynamo_frontend_worker_active_decode_blocks{worker_id="decode-0",worker_type="backend"} 64
dynamo_component_inflight_requests{worker_id="decode-0",worker_type="backend"} 3
dynamo_component_router_kv_hit_rate_sum{worker_id="decode-0",worker_type="backend"} 8
dynamo_component_router_kv_hit_rate_count{worker_id="decode-0",worker_type="backend"} 10
`)
	if len(snap.Rows) != 2 {
		t.Fatalf("row count = %d, want 2: %+v", len(snap.Rows), snap.Rows)
	}
	var prefill, decode bool
	for _, row := range snap.Rows {
		switch row.WorkerID {
		case "prefill-0":
			prefill = true
			if row.WorkerRole != "prefill" || row.TTFT.Count != 1 || row.RequestsWaiting != 128 || row.ActivePrefillTokens == nil || *row.ActivePrefillTokens != 128 {
				t.Fatalf("prefill row not normalized: %+v", row)
			}
		case "decode-0":
			decode = true
			if row.WorkerRole != "decode" || row.ITL.Count != 1 || row.TPOT.Count != 1 || row.RequestsRunning != 67 || row.ActiveDecodeBlocks == nil || *row.ActiveDecodeBlocks != 64 {
				t.Fatalf("decode row not normalized: %+v", row)
			}
			if row.PrefixCacheHitRatio == nil || *row.PrefixCacheHitRatio != 0.8 {
				t.Fatalf("decode row prefix ratio = %+v, want 0.8", row.PrefixCacheHitRatio)
			}
		}
	}
	if !prefill || !decode {
		t.Fatalf("missing expected P/D rows: %+v", snap.Rows)
	}
	prom := snap.Prometheus()
	for _, want := range []string{
		`fak_serving_ttft_seconds_count{engine="dynamo",worker="prefill-0"} 1`,
		`fak_serving_requests_waiting{engine="dynamo",worker="prefill-0"} 128`,
		`fak_serving_worker_active_prefill_tokens{engine="dynamo",worker="prefill-0",role="prefill"} 128`,
		`fak_serving_tpot_seconds_count{engine="dynamo",worker="decode-0"} 1`,
		`fak_serving_requests_running{engine="dynamo",worker="decode-0"} 67`,
		`fak_serving_worker_active_decode_blocks{engine="dynamo",worker="decode-0",role="decode"} 64`,
		`fak_serving_prefix_cache_hit_ratio{engine="dynamo",worker="decode-0"} 0.8`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
	if strings.Contains(prom, "dynamo_frontend_") || strings.Contains(prom, "dynamo_component_") {
		t.Fatalf("normalized output leaks Dynamo-only metric names:\n%s", prom)
	}
}
