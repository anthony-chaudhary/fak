package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestChatProxyFrontsVLLMAndSGLangServedToolCalls is the GPU-free witness for the
// "fak serve in front of vLLM / SGLang" drop-in lane (#451). The live end-to-end
// adjudication TAX (median latency / decode tok/s) is network-bound and stays
// host-gated behind a serving node (tools/vllm_tax_witness.py; docs/benchmarks/
// VLLM-HEADTOHEAD-RESULTS.md §3). But the PROTOCOL-level integration — that
// `fak serve` correctly fronts a vLLM-served (`--enable-auto-tool-choice`) and an
// SGLang-served model, decoding the exact tool-call wire each engine emits and
// running every proposed call through fak's adjudication plane before forwarding
// — needs no GPU and is proven here, in CI, so the drop-in lane is a TESTED
// integration rather than prose.
//
// Each case stands an OpenAI-compatible upstream up in the precise response shape
// the named engine produces (engine-specific tool-call id formats, served-model
// id divergence from the requested alias, vLLM's null/extra fields, SGLang's
// content-alongside-tool_calls + matched_stop), fronts it with the gateway in
// proxy mode, and asserts the full adjudication plane runs: allow* kept verbatim,
// deny* dropped, transform* redacted — the same toolAdj policy the rest of the
// suite uses (allow=ALLOW, deny=DENY, transform=TRANSFORM -> {"redacted":true}).
func TestChatProxyFrontsVLLMAndSGLangServedToolCalls(t *testing.T) {
	cases := []struct {
		name string
		// basePrefix is the upstream path prefix the engine's OpenAI surface is
		// mounted under (vLLM/SGLang both serve /v1; the gateway honors a BaseURL
		// path prefix and appends /chat/completions).
		basePrefix string
		// reqModel is the alias the client sends (forwarded to the upstream verbatim).
		reqModel string
		// servedModel is what the engine reports it actually served — vLLM echoes
		// the HF repo id, SGLang echoes its --served-model-name; the gateway echoes
		// THIS back to the client (#82), not the requested alias.
		servedModel  string
		clientTools  []string
		upstreamBody string
		wantKept     map[string]string
		wantContent  string
		wantFak      int
	}{
		{
			// vLLM with --enable-auto-tool-choice: standard OpenAI tool_calls,
			// chatcmpl-tool-* ids, content:null, and vLLM's extra null fields
			// (stop_reason / prompt_logprobs / prompt_tokens_details) the gateway
			// must ignore for a clean drop-in.
			name:        "vllm-auto-tool-choice",
			basePrefix:  "/v1",
			reqModel:    "qwen3.6-27b",
			servedModel: "Qwen/Qwen3.6-27B",
			clientTools: []string{"allow_read", "deny_write", "transform_exec"},
			upstreamBody: `{
				"id":"chatcmpl-9f2a",
				"object":"chat.completion",
				"created":1718900000,
				"model":"Qwen/Qwen3.6-27B",
				"choices":[{
					"index":0,
					"message":{
						"role":"assistant",
						"content":null,
						"tool_calls":[
							{"id":"chatcmpl-tool-abc123","type":"function","function":{"name":"allow_read","arguments":"{\"path\":\"/etc/hosts\"}"}},
							{"id":"chatcmpl-tool-def456","type":"function","function":{"name":"deny_write","arguments":"{\"path\":\"/etc/passwd\"}"}},
							{"id":"chatcmpl-tool-ghi789","type":"function","function":{"name":"transform_exec","arguments":"{\"cmd\":\"rm -rf /\"}"}}
						]
					},
					"logprobs":null,
					"finish_reason":"tool_calls",
					"stop_reason":null
				}],
				"usage":{"prompt_tokens":42,"completion_tokens":18,"total_tokens":60,"prompt_tokens_details":null},
				"prompt_logprobs":null
			}`,
			wantKept: map[string]string{
				"allow_read":     `{"path":"/etc/hosts"}`,
				"transform_exec": `{"redacted":true}`,
			},
			wantFak: 3,
		},
		{
			// SGLang's OpenAI-compatible tool parser: standard tool_calls with
			// call_* / index fields, content emitted ALONGSIDE the tool_calls, and
			// a matched_stop field the gateway must ignore. served-model id differs
			// from the requested alias (--served-model-name).
			name:        "sglang-tool-call",
			basePrefix:  "/v1",
			reqModel:    "sglang-served",
			servedModel: "qwen3.6-27b",
			clientTools: []string{"allow_search", "deny_delete"},
			upstreamBody: `{
				"id":"a1b2",
				"object":"chat.completion",
				"created":1718900001,
				"model":"qwen3.6-27b",
				"choices":[{
					"index":0,
					"message":{
						"role":"assistant",
						"content":"I'll search the index.",
						"tool_calls":[
							{"id":"call_0","index":0,"type":"function","function":{"name":"allow_search","arguments":"{\"q\":\"fak\"}"}},
							{"id":"call_1","index":1,"type":"function","function":{"name":"deny_delete","arguments":"{\"id\":7}"}}
						]
					},
					"logprobs":null,
					"finish_reason":"tool_calls",
					"matched_stop":null
				}],
				"usage":{"prompt_tokens":31,"completion_tokens":12,"total_tokens":43}
			}`,
			wantKept:    map[string]string{"allow_search": `{"q":"fak"}`},
			wantContent: "I'll search the index.",
			wantFak:     2,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterAdjudicator(0, toolAdj{})

			upstreamHits := 0
			wantPath := c.basePrefix + "/chat/completions"
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamHits++
				if r.URL.Path != wantPath {
					t.Errorf("upstream path = %q, want %q", r.URL.Path, wantPath)
				}
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatalf("read upstream body: %v", err)
				}
				var req struct {
					Model string          `json:"model"`
					Tools []agent.ToolDef `json:"tools"`
				}
				if err := json.Unmarshal(raw, &req); err != nil {
					t.Fatalf("decode upstream request: %v\n%s", err, raw)
				}
				// The client's alias is forwarded to the engine verbatim (#82).
				if req.Model != c.reqModel {
					t.Errorf("upstream model = %q, want %q", req.Model, c.reqModel)
				}
				if len(req.Tools) != len(c.clientTools) {
					t.Errorf("tools sent upstream = %d, want %d", len(req.Tools), len(c.clientTools))
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(c.upstreamBody))
			}))
			defer upstream.Close()

			srv, err := New(Config{
				EngineID: "test",
				Model:    c.reqModel,
				BaseURL:  upstream.URL + c.basePrefix,
				Provider: "openai-compatible",
				VDSO:     true,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			tools := make([]map[string]any, 0, len(c.clientTools))
			for _, name := range c.clientTools {
				tools = append(tools, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        name,
						"description": "drop-in tool",
						"parameters":  map[string]any{"type": "object"},
					},
				})
			}
			body, err := json.Marshal(map[string]any{
				"model": c.reqModel,
				"messages": []map[string]any{
					{"role": "system", "content": "system rules"},
					{"role": "user", "content": "call tools if needed"},
				},
				"tools": tools,
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

			var resp ChatResponse
			if err := json.Unmarshal(respRaw, &resp); err != nil {
				t.Fatalf("decode response: %v (%s)", err, respRaw)
			}
			// The gateway echoes the model the ENGINE reported it served (#82).
			if resp.Model != c.servedModel {
				t.Errorf("response model = %q, want engine-served %q", resp.Model, c.servedModel)
			}
			if len(resp.Choices) != 1 {
				t.Fatalf("choices = %d, want gateway-normalized single choice", len(resp.Choices))
			}
			msg := resp.Choices[0].Message
			if msg.Content != c.wantContent {
				t.Errorf("content = %q, want %q", msg.Content, c.wantContent)
			}
			if got := len(msg.ToolCalls); got != len(c.wantKept) {
				t.Fatalf("surviving tool calls = %d, want %d: %+v", got, len(c.wantKept), msg.ToolCalls)
			}
			for _, tc := range msg.ToolCalls {
				wantArgs, ok := c.wantKept[tc.Function.Name]
				if !ok {
					t.Fatalf("unexpected tool call survived adjudication: %+v", tc)
				}
				if tc.Function.Arguments != wantArgs {
					t.Errorf("%s args = %q, want %q", tc.Function.Name, tc.Function.Arguments, wantArgs)
				}
			}
			// Every proposed call was adjudicated before any survivor was forwarded.
			if resp.Fak == nil || len(resp.Fak.Adjudications) != c.wantFak {
				t.Fatalf("fak adjudications = %+v, want %d", resp.Fak, c.wantFak)
			}
		})
	}
}
