package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestHTTPPlannerOpenAIBufferedEnvelope(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	for _, tc := range []struct {
		name, choice, body, wantError, wantContent, wantFinish, wantToolID string
	}{
		{name: "missing_message", choice: `{"finish_reason":"stop"}`, wantError: "missing or null message and delta"},
		{name: "null_message", choice: `{"message":null,"finish_reason":"stop"}`, wantError: "missing or null message and delta"},
		{name: "both_null", choice: `{"message":null,"delta":null,"finish_reason":"stop"}`, wantError: "missing or null message and delta"},
		{name: "null_choice", choice: `null`, wantError: "missing or null message and delta"},
		{name: "malformed_json", body: `{"choices":`, wantError: "decode:"},
		{name: "scalar_message", choice: `{"message":"invalid","finish_reason":"stop"}`, wantError: "decode:"},
		{name: "api_error_precedence", body: `{"error":{"message":"existing provider failure"},"choices":[{"message":null}]}`, wantError: "api error: existing provider failure"},
		{name: "empty_object", choice: `{"message":{},"finish_reason":"stop"}`, wantFinish: "stop"},
		{name: "null_content", choice: `{"message":{"role":"assistant","content":null},"finish_reason":"stop"}`, wantFinish: "stop"},
		{name: "text", choice: `{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}`, wantContent: "answer", wantFinish: "stop"},
		{name: "tool_only", choice: `{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_envelope","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"SFO\"}"}}]},"finish_reason":"tool_calls"}`, wantFinish: "tool_calls", wantToolID: "call_envelope"},
		{name: "legacy_function", choice: `{"message":{"role":"assistant","function_call":{"name":"lookup","arguments":"{\"city\":\"SFO\"}"}},"finish_reason":"function_call"}`, wantFinish: "tool_calls", wantToolID: "legacy_function_call"},
		{name: "delta_only", choice: `{"delta":{"role":"assistant","content":"delta"},"finish_reason":"stop"}`, wantContent: "delta", wantFinish: "stop"},
		{name: "null_message_with_delta", choice: `{"message":null,"delta":{"role":"assistant","content":"delta"},"finish_reason":"stop"}`, wantContent: "delta", wantFinish: "stop"},
		{name: "empty_message_with_delta", choice: `{"message":{},"delta":{"role":"assistant","content":"delta"},"finish_reason":"stop"}`, wantContent: "delta", wantFinish: "stop"},
		{name: "empty_delta", choice: `{"delta":{},"finish_reason":"stop"}`, wantFinish: "stop"},
		{name: "populated_message_wins", choice: `{"message":{"role":"assistant","content":"answer"},"delta":{"role":"assistant","content":"delta"},"finish_reason":"stop"}`, wantContent: "answer", wantFinish: "stop"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := tc.body
			if body == "" {
				body = `{"model":"openai-test","usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5},"choices":[` + tc.choice + `]}`
			}
			var requests atomic.Int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				var request struct {
					Stream bool `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode request: %v", err)
				}
				if request.Stream {
					t.Error("buffered fixture unexpectedly requested streaming")
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, body)
			}))
			defer ts.Close()
			planner, err := NewProviderHTTPPlanner(string(ProviderOpenAI), ts.URL, "openai-test", "loopback-only")
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
			comp, err := planner.Complete(ctx, []Message{{Role: RoleUser, Content: "look up SFO"}}, adapterTestTools())
			if requests.Load() != 1 {
				t.Errorf("requests=%d, want one", requests.Load())
			}
			if tc.wantError != "" {
				if err == nil || comp != nil || !strings.Contains(err.Error(), tc.wantError) || clears != 0 {
					t.Fatalf("invalid envelope released completion=%+v error=%v clears=%d; want %q", comp, err, clears, tc.wantError)
				}
				if tc.name == "api_error_precedence" && err.Error() != "planner: openai: api error: existing provider failure" {
					t.Fatalf("API error changed: %v", err)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("completion=%+v error=%v", comp, err)
			}
			if clears != 1 || comp.FinishReason != tc.wantFinish || comp.Message.Content != tc.wantContent || comp.Model != "openai-test" || comp.Usage.PromptTokens != 3 || comp.Usage.CompletionTokens != 2 || comp.Usage.TotalTokens != comp.Usage.PromptTokens+comp.Usage.CompletionTokens {
				t.Fatalf("healthy completion changed: %+v clears=%d", comp, clears)
			}
			if tc.wantToolID != "" {
				if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != tc.wantToolID || comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` || comp.Message.FunctionCall != nil {
					t.Fatalf("tool-only turn changed: %+v", comp.Message)
				}
			} else if len(comp.Message.ToolCalls) != 0 {
				t.Fatalf("unexpected tool calls: %+v", comp.Message.ToolCalls)
			}
		})
	}
}
