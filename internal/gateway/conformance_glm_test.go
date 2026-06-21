package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// conformance_glm_test.go — the tool-call conformance fail-closed.
//
// "The agent must work" with GLM-5.2 means the kernel's permission floor can
// NEVER be silently bypassed by a tool-call format fak failed to parse. A
// GLM-5.2 variant that announces finish_reason=tool_calls but emits the call in a
// shape the OpenAI adapter does not recognize (e.g. buried in reasoning_content,
// or a non-standard wrapper) would otherwise reach the gateway as a benign empty
// turn and skip adjudication entirely. The conformance gate makes that a
// fail-closed 502 instead.

// TestChatProxyFailsClosedOnUnparsedToolCallClaim proves that an upstream which
// claims tool_calls but delivers zero parseable calls is REFUSED, not passed
// through as an empty turn — so adjudication is never skipped on an unparsed call.
func TestChatProxyFailsClosedOnUnparsedToolCallClaim(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// finish_reason announces tool calls, but the message carries NONE in the
		// tool_calls field and NO recognizable <tool_call> text — the unparsed-call
		// silent-no-op a non-conformant GLM variant would produce.
		resp := map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{
					"role":              "assistant",
					"content":           "I'll use a tool now.",
					"reasoning_content": "calling Bash with command rm -rf /tmp/x",
				},
				"finish_reason": "tool_calls",
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode upstream response: %v", err)
		}
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "glm-5.2", BaseURL: upstream.URL, Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "client-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "do a thing"}},
		Tools: []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "Bash", Parameters: json.RawMessage(`{"type":"object"}`)}},
		},
	}, &resp)

	// Must fail closed (502), NOT return 200 with an empty/benign turn.
	if code == 200 {
		t.Fatalf("conformance hole: upstream claimed tool_calls but parsed none, and the gateway returned 200 (adjudication skipped). want a fail-closed 5xx. resp=%+v", resp)
	}
	if code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (fail-closed on unparsed tool-call claim)", code)
	}
}

// TestChatProxyParsesGLMThinkingModeToolCall is the "it works" complement: a
// GLM-5.2 thinking-mode response (reasoning_content present alongside a proper
// structured tool_calls field) must parse, adjudicate, and return the call —
// reasoning_content is an unknown field the OpenAI adapter ignores, so it must
// not break tool-call parsing.
func TestChatProxyParsesGLMThinkingModeToolCall(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{
					"role":              "assistant",
					"content":           "",
					"reasoning_content": "The user wants a listing. I should call the allow_glm tool.",
					"tool_calls": []any{map[string]any{
						"id":   "call_1",
						"type": "function",
						"function": map[string]any{
							"name":      "allow_glm",
							"arguments": `{"path":"."}`,
						},
					}},
				},
				"finish_reason": "tool_calls",
			}},
			"usage": map[string]any{"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode upstream response: %v", err)
		}
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "glm-5.2", BaseURL: upstream.URL, Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "client-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "list the dir"}},
		Tools: []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "allow_glm", Parameters: json.RawMessage(`{"type":"object"}`)}},
		},
	}, &resp)
	if code != 200 {
		t.Fatalf("GLM thinking-mode tool call: status = %d, want 200", code)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls (the GLM tool call must survive)", resp.Choices[0].FinishReason)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 || resp.Choices[0].Message.ToolCalls[0].Function.Name != "allow_glm" {
		t.Fatalf("expected the adjudicated allow_glm call to be returned; got %+v", resp.Choices[0].Message.ToolCalls)
	}
}

// TestChatProxyAllowsNormalStopTurn is the no-op guard: a plain content turn with
// finish_reason=stop (no tool-call claim) must still pass through 200, so the
// conformance gate fires ONLY on the dropped-call case and never on benign text.
func TestChatProxyAllowsNormalStopTurn(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"role": "assistant", "content": "Here is the answer: 42."},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode upstream response: %v", err)
		}
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "glm-5.2", BaseURL: upstream.URL, Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "client-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "what is 6*7?"}},
	}, &resp)
	if code != 200 {
		t.Fatalf("normal stop turn status = %d, want 200 (conformance gate must not fire on benign text)", code)
	}
}
