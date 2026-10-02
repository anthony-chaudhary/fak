package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestToollessBufferedReplyWire exercises the real HTTP planner between two
// loopback servers. Provider text must not become an adjudicated call merely
// because a tool-less answer contains a name-bearing JSON object.
// fak-test:runtime fast est=1s lane=default
func TestToollessBufferedReplyWire(t *testing.T) {
	const content = " \n{\"name\":\"Alice\",\"age\":30}\n "
	quoted, _ := json.Marshal(content)
	providers := []struct {
		name string
		body string
	}{
		{"openai", fmt.Sprintf(`{"model":"served-model","choices":[{"message":{"role":"assistant","content":%s},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`, quoted)},
		{"xai", fmt.Sprintf(`{"model":"served-model","choices":[{"message":{"role":"assistant","content":%s},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`, quoted)},
		{"openai-responses", fmt.Sprintf(`{"model":"served-model","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":%s}]}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`, quoted)},
		{"gemini", fmt.Sprintf(`{"modelVersion":"served-model","candidates":[{"content":{"role":"model","parts":[{"text":%s}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":10}}`, quoted)},
	}
	for _, provider := range providers {
		t.Run(provider.name, func(t *testing.T) {
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Tools []json.RawMessage `json:"tools"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if len(req.Tools) != 0 {
					t.Errorf("tool-less request offered %d upstream tools", len(req.Tools))
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, provider.body)
			}))
			defer up.Close()
			planner, err := agent.NewProviderHTTPPlanner(provider.name, up.URL, "requested-model", "")
			if err != nil {
				t.Fatal(err)
			}
			srv := newTestServer(t)
			srv.planner = planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			request := json.RawMessage(`{"model":"requested-model","messages":[{"role":"user","content":"Give me a JSON record"}],"max_tokens":64}`)

			t.Run("chat", func(t *testing.T) {
				var resp ChatResponse
				if code := postJSON(t, ts.URL+"/v1/chat/completions", request, &resp); code != http.StatusOK {
					t.Fatalf("status = %d, want 200", code)
				}
				if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != content || len(resp.Choices[0].Message.ToolCalls) != 0 || resp.Choices[0].FinishReason != "stop" {
					t.Errorf("tool-less reply changed: %+v", resp.Choices)
				}
				if resp.Fak != nil && len(resp.Fak.Adjudications) != 0 {
					t.Errorf("tool-less text was adjudicated: %+v", resp.Fak.Adjudications)
				}
				if resp.Model != "served-model" || resp.Usage.TotalTokens != 10 {
					t.Errorf("provider metadata changed: model=%q usage=%+v", resp.Model, resp.Usage)
				}
			})
			t.Run("completions", func(t *testing.T) {
				var resp CompletionResponse
				if code := postJSON(t, ts.URL+"/v1/completions", json.RawMessage(`{"model":"requested-model","prompt":"Give me a JSON record","max_tokens":64}`), &resp); code != http.StatusOK {
					t.Fatalf("status = %d, want 200", code)
				}
				if len(resp.Choices) != 1 || resp.Choices[0].Text != content || resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
					t.Errorf("tool-less completion changed: %+v", resp.Choices)
				}
				if resp.Model != "served-model" || resp.Usage.TotalTokens != 10 {
					t.Errorf("provider metadata changed: model=%q usage=%+v", resp.Model, resp.Usage)
				}
			})
			t.Run("messages", func(t *testing.T) {
				var resp anthropicMessageResponse
				if code := postJSON(t, ts.URL+"/v1/messages", request, &resp); code != http.StatusOK {
					t.Fatalf("status = %d, want 200", code)
				}
				if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != content || resp.StopReason != "end_turn" {
					t.Errorf("tool-less message changed: %+v, stop=%q", resp.Content, resp.StopReason)
				}
				if resp.Fak != nil && len(resp.Fak.Adjudications) != 0 {
					t.Errorf("tool-less text was adjudicated: %+v", resp.Fak.Adjudications)
				}
				if resp.Model != "requested-model" || resp.Usage.InputTokens != 7 || resp.Usage.OutputTokens != 3 {
					t.Errorf("provider metadata changed: model=%q usage=%+v", resp.Model, resp.Usage)
				}
			})
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestToollessBufferedReplyControls(t *testing.T) {
	const textCall = " \n<tool_call>{\"name\":\"allow_a\",\"arguments\":{\"x\":1}}</tool_call>\n "
	for _, tc := range []struct {
		name       string
		message    agent.Message
		finish     string
		offerTools bool
		wantCalls  int
		wantText   string
		wantDrop   bool
	}{
		{"tool_shaped_text", agent.Message{Content: textCall}, "stop", false, 0, textCall, false},
		{"offered_text_call", agent.Message{Content: textCall}, "stop", true, 1, "", false},
		{"native_call", agent.Message{Content: "native", ToolCalls: []agent.ToolCall{{ThoughtSignature: "provider-signature", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}}}}, "stop", false, 1, "native", false},
		{"legacy_native_call", agent.Message{FunctionCall: &agent.Func{Name: "allow_a", Arguments: `{"x":1}`}}, "function_call", false, 1, "", false},
		{"announced_missing_call", agent.Message{Content: "no call"}, "tool_calls", false, 0, "no call", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.message.Role = agent.RoleAssistant
			tc.message.ReasoningContent = "provider reasoning"
			body, err := json.Marshal(map[string]any{
				"model":   "served-model",
				"choices": []any{map[string]any{"message": tc.message, "finish_reason": tc.finish}},
				"usage":   agent.Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
			})
			if err != nil {
				t.Fatal(err)
			}
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write(body)
			}))
			defer up.Close()
			planner := agent.NewHTTPPlanner(up.URL, "requested-model", "")
			var tools []agent.ToolDef
			if tc.offerTools {
				tools = []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "allow_a", Parameters: json.RawMessage(`{"type":"object"}`)}}}
			}
			comp, err := planner.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "reply"}}, tools)
			if err != nil {
				t.Fatal(err)
			}
			if len(comp.Message.ToolCalls) != tc.wantCalls || comp.Message.Content != tc.wantText || comp.ToolCallsDropped != tc.wantDrop {
				t.Fatalf("completion = %+v, want calls=%d content=%q dropped=%v", comp, tc.wantCalls, tc.wantText, tc.wantDrop)
			}
			wantFinish := tc.finish
			if tc.wantCalls > 0 {
				wantFinish = "tool_calls"
				call := comp.Message.ToolCalls[0]
				if call.ID == "" || call.Type != "function" || call.Function.Name != "allow_a" || call.Function.Arguments != `{"x":1}` || comp.Message.FunctionCall != nil {
					t.Errorf("call fields not preserved/normalized: %+v", call)
				}
				if tc.name == "native_call" && call.ThoughtSignature != "provider-signature" {
					t.Errorf("native thought signature lost: %+v", call)
				}
			}
			if comp.FinishReason != wantFinish || comp.Message.ReasoningContent != "provider reasoning" || comp.Model != "served-model" || comp.Usage.TotalTokens != 10 || string(comp.Raw) != string(body) {
				t.Errorf("provider metadata changed: %+v", comp)
			}
		})
	}
}
