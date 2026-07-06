package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestMessagesWireFrontsVLLMServedToolCalls is the GPU-free witness for the
// Anthropic-wire half of issue #40 acceptance item 2 — "fak gateway serves
// /v1/chat/completions AND /v1/messages end-to-end through a live vLLM-V1 worker
// via the adapter." The /v1/chat/completions half is already proven GPU-free by
// TestChatProxyFrontsVLLMAndSGLangServedToolCalls; this is its /v1/messages twin.
//
// The LIVE overhead (median latency / decode tok/s of fak-fronted-vLLM vs
// raw-vLLM) stays host-gated behind a real serving node (issue #40 item 6,
// "would need measurement on real hardware"; docs/serving/dual-track-serving-plan.md).
// But the PROTOCOL-level integration on the wire Claude Code uses natively —
// that `fak serve` fronting a vLLM-served OpenAI upstream over the Anthropic
// /v1/messages wire translates Anthropic->OpenAI upstream, decodes vLLM's exact
// tool-call shape (chatcmpl-tool-* ids, content:null, vLLM's extra null fields:
// stop_reason / prompt_logprobs / prompt_tokens_details), runs every proposed
// call through fak's adjudication plane, and renders the survivors back as
// Anthropic content blocks with the `fak` extension — needs no GPU and is proven
// here, in CI, so both wires of the drop-in are TESTED integrations, not prose.
//
// The adjudication policy is the shared toolAdj: allow* -> ALLOW (kept verbatim),
// deny* -> DENY (dropped), transform* -> TRANSFORM (input redacted). The vLLM
// upstream never sees the deny; the client never sees a denied tool_use block.
func TestMessagesWireFrontsVLLMServedToolCalls(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	const reqModel = "qwen3.6-27b"
	// vLLM with --enable-auto-tool-choice emits standard OpenAI tool_calls with
	// chatcmpl-tool-* ids, content:null, and extra null fields the gateway must
	// ignore for a clean drop-in — the exact shape the chat-wire drop-in pins.
	const vllmBody = `{
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
	}`

	upstreamHits := 0
	var upstreamPath string
	var upstreamTools int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		upstreamPath = r.URL.Path
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		// The Anthropic /v1/messages request must have been translated into an
		// OpenAI /v1/chat/completions request before it reached the vLLM worker,
		// carrying the client's tools forward.
		var req struct {
			Tools []agent.ToolDef `json:"tools"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("decode upstream request: %v\n%s", err, raw)
		}
		upstreamTools = len(req.Tools)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(vllmBody))
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test",
		Model:    reqModel,
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

	// A Claude-Code-shaped Anthropic request: max_tokens is required on the wire,
	// tools carry input_schema (Anthropic), not parameters (OpenAI).
	body, err := json.Marshal(map[string]any{
		"model":      reqModel,
		"max_tokens": 256,
		"messages": []map[string]any{
			{"role": "user", "content": "read a file, write a file, run a command"},
		},
		"tools": []map[string]any{
			{"name": "allow_read", "input_schema": map[string]any{"type": "object"}},
			{"name": "deny_write", "input_schema": map[string]any{"type": "object"}},
			{"name": "transform_exec", "input_schema": map[string]any{"type": "object"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}

	// The Anthropic wire was translated to the OpenAI chat-completions surface the
	// vLLM worker serves, exactly once, tools forwarded.
	if upstreamHits != 1 {
		t.Fatalf("upstream hits = %d, want 1", upstreamHits)
	}
	if upstreamPath != "/v1/chat/completions" {
		t.Fatalf("upstream path = %q, want /v1/chat/completions", upstreamPath)
	}
	if upstreamTools != 3 {
		t.Fatalf("tools forwarded upstream = %d, want 3", upstreamTools)
	}

	var resp anthropicMessageResponse
	if err := json.Unmarshal(respRaw, &resp); err != nil {
		t.Fatalf("decode Anthropic response: %v (%s)", err, respRaw)
	}
	if resp.Type != "message" || resp.Role != "assistant" {
		t.Fatalf("envelope wrong: %+v", resp)
	}
	// The Anthropic wire echoes the REQUESTED model (unlike the OpenAI wire, which
	// echoes the engine-served id) — the vLLM served id must not leak here.
	if resp.Model != reqModel {
		t.Errorf("response model = %q, want requested %q", resp.Model, reqModel)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", resp.StopReason)
	}

	// allow_read kept verbatim; transform_exec redacted; deny_write must be gone.
	var allowInput, transformInput string
	toolUses := 0
	for _, b := range resp.Content {
		if b.Type != "tool_use" {
			continue
		}
		toolUses++
		switch b.Name {
		case "allow_read":
			allowInput = string(b.Input)
		case "transform_exec":
			transformInput = string(b.Input)
		case "deny_write":
			t.Error("denied vLLM-proposed tool call must NOT reach the caller")
		}
	}
	if toolUses != 2 {
		t.Fatalf("surviving tool_use blocks = %d, want 2: %+v", toolUses, resp.Content)
	}
	if allowInput != `{"path":"/etc/hosts"}` {
		t.Errorf("allow_read input not forwarded verbatim: %q", allowInput)
	}
	if transformInput != `{"redacted":true}` {
		t.Errorf("transform_exec input not redacted: %q", transformInput)
	}

	// The kernel's drop + repair is legible in-band (the wire Claude Code reads).
	if len(resp.Content) == 0 || resp.Content[0].Type != "text" {
		t.Fatalf("first block must be the [fak] decision note, got %+v", resp.Content)
	}
	note := resp.Content[0].Text
	if !strings.Contains(note, "[fak]") || !strings.Contains(note, "deny_write") || !strings.Contains(note, "transform_exec") {
		t.Errorf("in-band note must name the dropped + repaired calls, got %q", note)
	}

	// Every proposed call rode back as a structured adjudication on the `fak` twin.
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 3 {
		t.Fatalf("fak extension must carry all 3 adjudications, got %+v", resp.Fak)
	}
	var sawDeny, sawTransform bool
	for _, a := range resp.Fak.Adjudications {
		if a.Tool == "deny_write" && !a.Admitted {
			sawDeny = true
		}
		if a.Tool == "transform_exec" && a.Admitted && a.Verdict.Kind == "TRANSFORM" {
			sawTransform = true
		}
	}
	if !sawDeny || !sawTransform {
		t.Errorf("fak extension must record the deny + transform verdicts: %+v", resp.Fak.Adjudications)
	}
}

func TestMessagesWirePreservesToolResultNameForOpenAIUpstream(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var captured []agent.Message
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		var req struct {
			Messages []agent.Message `json:"messages"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("decode upstream request: %v\n%s", err, raw)
		}
		captured = req.Messages
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"saw the result"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}`))
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
		"model":      "qwen3.6-27b",
		"max_tokens": 128,
		"messages": []map[string]any{
			{"role": "user", "content": "read main.go"},
			{"role": "assistant", "content": []map[string]any{
				{"type": "tool_use", "id": "call_1", "name": "allow_read", "input": map[string]any{"path": "main.go"}},
			}},
			{"role": "user", "content": []map[string]any{
				{"type": "tool_result", "tool_use_id": "call_1", "content": "package main"},
			}},
		},
		"tools": []map[string]any{
			{"name": "allow_read", "input_schema": map[string]any{"type": "object"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}
	if len(captured) == 0 {
		t.Fatal("upstream did not capture a request")
	}
	var toolResult *agent.Message
	for i := range captured {
		if captured[i].Role == agent.RoleTool {
			toolResult = &captured[i]
			break
		}
	}
	if toolResult == nil {
		t.Fatalf("upstream request missing role=tool message: %+v", captured)
	}
	if toolResult.ToolCallID != "call_1" || toolResult.Name != "allow_read" {
		t.Fatalf("upstream tool result = %+v, want tool_call_id call_1 with name allow_read", *toolResult)
	}
}

func TestMessagesWireTextToolModeSkipsCtxViewForToolContinuation(t *testing.T) {
	t.Setenv("FAK_OPENAI_TOOL_MESSAGES_AS_TEXT", "1")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var captured []agent.Message
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		var req struct {
			Messages []agent.Message `json:"messages"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("decode upstream request: %v\n%s", err, raw)
		}
		captured = req.Messages
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"saw the result"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID:      "test",
		Model:         "qwen3.6-27b",
		BaseURL:       upstream.URL + "/v1",
		Provider:      "openai-compatible",
		VDSO:          true,
		CtxViewBudget: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, err := json.Marshal(map[string]any{
		"model":      "qwen3.6-27b",
		"max_tokens": 128,
		"messages": []map[string]any{
			{"role": "user", "content": "read main.go"},
			{"role": "assistant", "content": []map[string]any{
				{"type": "tool_use", "id": "call_1", "name": "allow_read", "input": map[string]any{"path": "main.go"}},
			}},
			{"role": "user", "content": []map[string]any{
				{"type": "tool_result", "tool_use_id": "call_1", "content": "package main"},
			}},
		},
		"tools": []map[string]any{
			{"name": "allow_read", "input_schema": map[string]any{"type": "object"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}
	var sawToolCallText, sawToolResponseText bool
	for _, msg := range captured {
		if msg.Role == agent.RoleTool {
			t.Fatalf("text tool mode sent native role=tool upstream: %+v", captured)
		}
		if msg.Role == agent.RoleAssistant && strings.TrimSpace(msg.Content) == "" {
			t.Fatalf("ctxview dropped the assistant tool_use into an empty assistant turn: %+v", captured)
		}
		if msg.Role == agent.RoleAssistant && strings.Contains(msg.Content, "<tool_call>") {
			sawToolCallText = true
		}
		if msg.Role == agent.RoleUser && strings.Contains(msg.Content, "<tool_response>") {
			sawToolResponseText = true
		}
	}
	if !sawToolCallText || !sawToolResponseText {
		t.Fatalf("tool continuation was not lowered to text blocks: sawToolCall=%v sawToolResponse=%v messages=%+v",
			sawToolCallText, sawToolResponseText, captured)
	}
}
