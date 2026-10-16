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
func TestHTTPPlannerResponsesSSEProviderError(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	const first = "first provider failure"
	const second = "later provider failure"
	const tool = `{"type":"function_call","call_id":"call_error","name":"lookup","arguments":"{\"city\":\"SFO\"}"}`
	frame := func(payload string) string { return "data: " + payload + "\n\n" }
	item := frame(`{"type":"response.output_item.done","output_index":0,"item":` + tool + `}`)
	completed := frame(`{"type":"response.completed","response":{"status":"completed","error":null,"model":"gpt-test","output":[],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`)
	text := frame(`{"type":"response.completed","response":{"status":"completed","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`)
	flat := frame(`{"type":"error","code":"server_error","message":"` + first + `","param":null,"sequence_number":2}`)
	nested := frame(`{"type":"error","error":{"code":"server_error","message":"` + first + `"}}`)
	for _, tc := range []struct {
		name, body, wantError string
		truncated, text       bool
	}{
		{name: "flat_before_completed", body: item + flat + completed, wantError: "api error: " + first},
		{name: "flat_after_completed", body: item + completed + flat, wantError: "api error: " + first},
		{name: "flat_only", body: item + flat, wantError: "api error: " + first},
		{name: "nested_before_completed", body: item + nested + completed, wantError: "api error: " + first},
		{name: "nested_after_completed", body: item + completed + nested, wantError: "api error: " + first},
		{name: "event_only", body: item + "event: error\n" + frame(`{"code":"server_error","message":"`+first+`","param":null}`) + completed, wantError: "api error: " + first},
		{name: "escaped_type", body: item + frame(`{"type":"err\u006fr","message":"`+first+`"}`) + completed, wantError: "api error: " + first},
		{name: "untyped_nested", body: item + frame(`{"error":{"message":"`+first+`"}}`) + completed, wantError: "api error: " + first},
		{name: "empty_nested_error", body: item + frame(`{"error":{}}`) + completed, wantError: "api error: "},
		{name: "malformed_nested_error", body: item + frame(`{"error":true}`) + completed, wantError: "decode:"},
		{name: "repeated_errors", body: item + nested + frame(`{"error":{"message":"`+second+`"}}`) + completed, wantError: "api error: " + first},
		{name: "completed_with_error", body: item + frame(`{"type":"response.completed","response":{"status":"completed","error":{"message":"`+first+`"},"output":[]}}`) + completed, wantError: "api error: " + first},
		{name: "malformed_json_after_error", body: item + flat + frame(`{`), wantError: "decode SSE data:"},
		{name: "read_error_before_decode", body: item + flat + completed, truncated: true, wantError: "read body: unexpected EOF"},
		{name: "healthy_tools", body: item + completed},
		{name: "healthy_text", body: text, text: true},
		{name: "null_error", body: item + frame(`{"error":null}`) + completed},
		{name: "raw_response_null_error", body: frame(`{"error":null,"status":"completed","model":"gpt-test","output":[` + tool + `],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/responses" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				var req struct {
					Stream bool `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil || !req.Stream {
					t.Errorf("Responses SSE request stream=%v decode=%v", req.Stream, err)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if tc.truncated {
					w.Header().Set("Content-Length", fmt.Sprint(len(tc.body)+1))
				}
				_, _ = io.WriteString(w, tc.body)
			}))
			defer ts.Close()
			planner, err := NewProviderHTTPPlanner(string(ProviderOpenAIResponses), ts.URL, "gpt-test", "loopback-only")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = ts.Client()
			planner.ForceResponsesStream = true
			var checkpointClears int
			planner.PendingTurnCheckpoint = func(attempt, status int, retryAt int64) {
				if attempt == 0 && status == 0 && retryAt == 0 {
					checkpointClears++
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			comp, err := planner.Complete(ctx, []Message{{Role: RoleUser, Content: "look up SFO"}}, adapterTestTools())
			if requests.Load() != 1 {
				t.Errorf("requests=%d, want one bounded attempt", requests.Load())
			}
			if tc.wantError != "" {
				if comp != nil || err == nil || !strings.Contains(err.Error(), tc.wantError) || checkpointClears != 0 {
					t.Fatalf("failed SSE released completion=%+v error=%v clears=%d; want %q", comp, err, checkpointClears, tc.wantError)
				}
				if strings.HasPrefix(tc.wantError, "api error:") && err.Error() != "planner: openai-responses: "+tc.wantError {
					t.Fatalf("provider error diagnostic changed: %v", err)
				}
				if tc.name == "repeated_errors" && strings.Contains(err.Error(), second) {
					t.Fatalf("first error overwritten: %v", err)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("healthy completion=%+v error=%v", comp, err)
			}
			if checkpointClears != 1 || comp.Model != "gpt-test" || comp.Usage.TotalTokens != 5 {
				t.Fatalf("healthy completion=%+v clears=%d", comp, checkpointClears)
			}
			if tc.text {
				if comp.Message.Content != "done" || len(comp.Message.ToolCalls) != 0 || comp.FinishReason != "stop" {
					t.Fatalf("healthy text changed: %+v", comp)
				}
			} else if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_error" ||
				comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` || comp.FinishReason != "tool_calls" {
				t.Fatalf("healthy tool reconstruction changed: %+v", comp)
			}
		})
	}
}
