package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestHTTPPlannerGeminiThoughtResponse(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	for _, tc := range []struct {
		name, parts, wantContent, wantReasoning, wantError string
		wantTool                                           bool
	}{
		{
			name:        "marked_summary",
			parts:       `[{"thought":true,"text":"provider summary"},{"text":"answer"}]`,
			wantContent: "answer", wantReasoning: "provider summary",
		},
		{
			name:          "marked_tool_syntax",
			parts:         `[{"thought":true,"text":"<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"SFO\"}}</tool_call>"}]`,
			wantReasoning: `<tool_call>{"name":"lookup","arguments":{"city":"SFO"}}</tool_call>`,
		},
		{
			name:        "unmarked_text",
			parts:       `[{"text":"first"},{"text":"second"}]`,
			wantContent: "first\nsecond",
		},
		{
			name:        "explicit_false",
			parts:       `[{"thought":false,"text":"visible answer"}]`,
			wantContent: "visible answer",
		},
		{
			name:        "legacy_delimiters",
			parts:       `[{"text":"<think>legacy summary</think>answer"}]`,
			wantContent: "answer", wantReasoning: "legacy summary",
		},
		{
			name:          "signed_function_call",
			parts:         `[{"thought":true,"text":"provider summary"},{"thoughtSignature":"opaque-signed-call","functionCall":{"name":"lookup","id":"call_gemini","args":{"city":"SFO"}}}]`,
			wantReasoning: "provider summary", wantTool: true,
		},
		{
			name:          "marked_text_and_function_call",
			parts:         `[{"thought":true,"text":"provider summary","thoughtSignature":"opaque-signed-call","functionCall":{"name":"lookup","id":"call_gemini","args":{"city":"SFO"}}}]`,
			wantReasoning: "provider summary", wantTool: true,
		},
		{
			name:      "malformed_thought",
			parts:     `[{"thought":"true","text":"invalid field"}]`,
			wantError: "decode:",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := fmt.Sprintf(`{"candidates":[{"content":{"role":"model","parts":%s},"finishReason":"STOP"}],"modelVersion":"gemini-test","usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`, tc.parts)
			var requests atomic.Int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/v1beta/models/gemini-test:generateContent" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				var request struct {
					Contents []json.RawMessage `json:"contents"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil || len(request.Contents) == 0 {
					t.Errorf("missing Gemini request contents: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, body)
			}))
			defer ts.Close()
			planner, err := NewProviderHTTPPlanner(string(ProviderGemini), ts.URL+"/v1beta", "gemini-test", "loopback-only")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = ts.Client()
			var clears int
			planner.PendingTurnCheckpoint = func(attempt, status int, retryAt int64) {
				clears++
				if attempt != 0 || status != 0 || retryAt != 0 {
					t.Errorf("unexpected retry checkpoint: %d %d %d", attempt, status, retryAt)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			comp, err := planner.Complete(ctx, []Message{{Role: RoleUser, Content: "respond to the fixture"}}, adapterTestTools())
			if requests.Load() != 1 {
				t.Errorf("requests=%d, want one", requests.Load())
			}
			if tc.wantError != "" {
				if err == nil || comp != nil || !strings.Contains(err.Error(), tc.wantError) || clears != 0 {
					t.Fatalf("invalid thought field completion=%+v error=%v clears=%d", comp, err, clears)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("completion=%+v error=%v", comp, err)
			}
			if comp.Message.Content != tc.wantContent || comp.Message.ReasoningContent != tc.wantReasoning {
				t.Fatalf("provider thought routing: content=%q reasoning=%q, want content=%q reasoning=%q", comp.Message.Content, comp.Message.ReasoningContent, tc.wantContent, tc.wantReasoning)
			}
			wantFinish := "stop"
			if tc.wantTool {
				wantFinish = "tool_calls"
				if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_gemini" || comp.Message.ToolCalls[0].ThoughtSignature != "opaque-signed-call" || comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` {
					t.Fatalf("signed function call changed: %+v", comp.Message.ToolCalls)
				}
			} else if len(comp.Message.ToolCalls) != 0 {
				t.Fatalf("thought summary released executable tool calls: %+v", comp.Message.ToolCalls)
			}
			if clears != 1 || comp.FinishReason != wantFinish || comp.Model != "gemini-test" || comp.Usage.PromptTokens != 3 || comp.Usage.CompletionTokens != 2 || comp.Usage.TotalTokens != 5 {
				t.Fatalf("completion identity/usage changed: %+v clears=%d", comp, clears)
			}
		})
	}
}
