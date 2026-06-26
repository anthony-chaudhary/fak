package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestChatProxyForwardsStructuredOutputFieldsToRideEngine is the GPU-free witness
// for #907's pass-through deliverable: when a client asks for OpenAI-compatible
// structured outputs (`response_format`) and a `logit_bias` mask, those constraint
// carriers survive the fak gateway and reach the ride engine (vLLM/SGLang) VERBATIM,
// AND the tool candidate the constrained generation produced still enters fak's
// adjudication plane before any survivor is forwarded.
//
// This proves the answer to the issue's first question — "Which constraints are
// pass-through-only in ride mode?" — is wired, not prose: vLLM/SGLang enforce the
// JSON-schema/grammar during generation; fak forwards the constraint and adjudicates
// the result. The gateway is the proxy planner (a BaseURL is set), so the assertion
// runs against the actual bytes that crossed the upstream wire, not an internal seam.
//
// The constraint is generic structured-output JSON (a `response_format` object) — the
// shape vLLM `guided_json`/`response_format` and SGLang `json_schema` both accept — so
// the one test stands in for the whole ride-mode pass-through lane.
func TestChatProxyForwardsStructuredOutputFieldsToRideEngine(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	// The exact structured-output constraint the client sends. The gateway must forward
	// it byte-equivalent: a json_schema response_format pinning the tool-call shape.
	clientResponseFormat := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"tool_call","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"arguments":{"type":"object"}},"required":["name","arguments"]}}}`)
	clientLogitBias := map[int]float64{50256: -100, 1024: 12.5}

	// The upstream (a stand-in vLLM/SGLang OpenAI surface) captures what the gateway
	// actually forwarded, so the test asserts the constraint crossed the wire.
	var gotResponseFormat json.RawMessage
	var gotLogitBias map[int]float64
	upstreamHits := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		var req struct {
			ResponseFormat json.RawMessage `json:"response_format"`
			LogitBias      map[int]float64 `json:"logit_bias"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("decode upstream request: %v\n%s", err, raw)
		}
		gotResponseFormat = req.ResponseFormat
		gotLogitBias = req.LogitBias
		w.Header().Set("Content-Type", "application/json")
		// A constrained generation: one allow*, one deny* call — proof the proposed set
		// reaches the gate AFTER generation, where deny is dropped and allow is kept.
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-907",
			"object":"chat.completion",
			"created":1718900007,
			"model":"Qwen/Qwen3.6-27B",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"content":null,
					"tool_calls":[
						{"id":"call_0","type":"function","function":{"name":"allow_read","arguments":"{\"path\":\"/etc/hosts\"}"}},
						{"id":"call_1","type":"function","function":{"name":"deny_write","arguments":"{\"path\":\"/etc/passwd\"}"}}
					]
				},
				"logprobs":null,
				"finish_reason":"tool_calls"
			}],
			"usage":{"prompt_tokens":40,"completion_tokens":20,"total_tokens":60}
		}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test",
		Model:    "qwen3.6-27b",
		BaseURL:  upstream.URL + "/v1",
		Provider: "openai-compatible",
		VDSO:     true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, err := json.Marshal(map[string]any{
		"model": "qwen3.6-27b",
		"messages": []map[string]any{
			{"role": "user", "content": "read /etc/hosts"},
		},
		"tools": []map[string]any{
			{"type": "function", "function": map[string]any{"name": "allow_read", "description": "read", "parameters": map[string]any{"type": "object"}}},
			{"type": "function", "function": map[string]any{"name": "deny_write", "description": "write", "parameters": map[string]any{"type": "object"}}},
		},
		"response_format": clientResponseFormat,
		"logit_bias":      clientLogitBias,
	})
	if err != nil {
		t.Fatal(err)
	}
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}
	if upstreamHits != 1 {
		t.Fatalf("upstream hits = %d, want 1", upstreamHits)
	}

	// (1) The structured-output constraint reached the ride engine VERBATIM.
	if !jsonEqual(t, gotResponseFormat, clientResponseFormat) {
		t.Errorf("response_format forwarded to ride engine = %s\nwant (verbatim) = %s", gotResponseFormat, clientResponseFormat)
	}
	if len(gotLogitBias) != len(clientLogitBias) {
		t.Fatalf("logit_bias forwarded = %v, want %v", gotLogitBias, clientLogitBias)
	}
	for tok, bias := range clientLogitBias {
		if gotLogitBias[tok] != bias {
			t.Errorf("logit_bias[%d] forwarded = %v, want %v", tok, gotLogitBias[tok], bias)
		}
	}

	// (2) The candidate the constrained generation produced still entered the gate.
	var resp ChatResponse
	if err := json.Unmarshal(respRaw, &resp); err != nil {
		t.Fatalf("decode response: %v (%s)", err, respRaw)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	kept := resp.Choices[0].Message.ToolCalls
	if len(kept) != 1 {
		t.Fatalf("surviving tool calls = %d, want 1 (deny dropped): %+v", len(kept), kept)
	}
	if kept[0].Function.Name != "allow_read" {
		t.Errorf("surviving call = %q, want allow_read", kept[0].Function.Name)
	}
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 2 {
		t.Fatalf("fak adjudications = %+v, want 2 (every proposed call adjudicated AFTER generation)", resp.Fak)
	}
}

// TestChatProxyOmitsStructuredOutputFieldsWhenAbsent pins the bit-exact drop-in half:
// a client that sends NO response_format / logit_bias must produce an upstream body
// with neither key present (omitempty), so the unconstrained path is byte-identical to
// the pre-#907 wire and a non-structured client is never silently constrained.
func TestChatProxyOmitsStructuredOutputFieldsWhenAbsent(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var rawUpstream []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawUpstream, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	io.Copy(io.Discard, httpResp.Body)

	var up map[string]json.RawMessage
	if err := json.Unmarshal(rawUpstream, &up); err != nil {
		t.Fatalf("decode upstream body: %v\n%s", err, rawUpstream)
	}
	if _, ok := up["response_format"]; ok {
		t.Errorf("response_format present on upstream wire for an unconstrained request: %s", rawUpstream)
	}
	if _, ok := up["logit_bias"]; ok {
		t.Errorf("logit_bias present on upstream wire for an unconstrained request: %s", rawUpstream)
	}
}

// jsonEqual reports whether two JSON documents are semantically equal (key order /
// whitespace insensitive), so a verbatim-forwarding assertion does not pin a
// re-marshal's incidental key ordering.
func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		t.Fatalf("unmarshal a: %v (%s)", err, a)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		t.Fatalf("unmarshal b: %v (%s)", err, b)
	}
	na, _ := json.Marshal(av)
	nb, _ := json.Marshal(bv)
	return bytes.Equal(na, nb)
}
