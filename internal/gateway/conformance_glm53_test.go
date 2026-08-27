package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestChatProxyMediatesGLM53FlashHostedToolCall keeps hosted-wire conformance
// independent from zaitask's request builder. The generic gateway must preserve
// reasoning and parse JSON-string tool arguments before the adjudicator sees the
// call; none of this response is evidence of fak-native model execution.
func TestChatProxyMediatesGLM53FlashHostedToolCall(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req["model"] != "glm-5.3-flash" {
			t.Fatalf("model = %v, want hosted GLM-5.3-Flash", req["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"glm-5.3-flash","request_id":"req-gateway","choices":[{"message":{"role":"assistant","content":"","reasoning_content":"Call the allowed listing tool.","tool_calls":[{"id":"call-53","type":"function","function":{"name":"allow_glm","arguments":"{\"path\":\".\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":3}}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "glm-5.3-flash", BaseURL: upstream.URL, Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "list files"}},
		Tools: []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{
			Name: "allow_glm", Parameters: json.RawMessage(`{"type":"object"}`),
		}}},
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "tool_calls" || choice.Message.ReasoningContent != "Call the allowed listing tool." {
		t.Fatalf("response = %+v", choice)
	}
	if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].Function.Name != "allow_glm" || choice.Message.ToolCalls[0].Function.Arguments != `{"path":"."}` {
		t.Fatalf("tool call was not parsed and mediated: %+v", choice.Message.ToolCalls)
	}
}
