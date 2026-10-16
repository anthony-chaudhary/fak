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
func TestHTTPPlannerResponsesSSENullEnvelope(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	frame := func(payload string) string { return "data: " + payload + "\n\n" }
	const tool = `{"type":"function_call","call_id":"call_null","name":"lookup","arguments":"{\"city\":\"SFO\"}"}`
	item := frame(`{"type":"response.output_item.done","output_index":0,"item":` + tool + `}`)
	null := frame(`{"type":"response.completed","response":null}`)
	emptyObject := frame(`{"type":"response.completed","response":{}}`)
	emptyOutput := frame(`{"type":"response.completed","response":{"status":"completed","output":[]}}`)
	providerError := frame(`{"type":"error","message":"provider failure"}`)
	text := frame(`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}}`)
	for _, tc := range []struct {
		name, body, wantError, wantContent string
		wantTool, truncated                bool
	}{
		{name: "null_with_item", body: item + null, wantError: "response.completed payload is null"},
		{name: "null_without_item", body: null, wantError: "response.completed payload is null"},
		{name: "empty_object_fill", body: item + emptyObject, wantTool: true},
		{name: "empty_output_fill", body: item + emptyOutput, wantTool: true},
		{name: "empty_output_without_items", body: emptyOutput},
		{name: "populated_output_preserved", body: item + text, wantContent: "done"},
		{name: "provider_error_before_null", body: item + providerError + null, wantError: "api error: provider failure"},
		{name: "provider_error_after_null", body: item + null + providerError, wantError: "api error: provider failure"},
		{name: "parser_error_after_null", body: item + null + frame(`{`), wantError: "decode SSE data:"},
		{name: "body_error_before_decode", body: item + null, truncated: true, wantError: "read body: unexpected EOF"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/responses" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				var request struct {
					Stream bool `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil || !request.Stream {
					t.Errorf("SSE request stream=%v decode=%v", request.Stream, err)
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
			var clears int
			planner.PendingTurnCheckpoint = func(attempt, status int, retryAt int64) {
				if attempt == 0 && status == 0 && retryAt == 0 {
					clears++
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			var comp *Completion
			var panicValue any
			func() {
				defer func() { panicValue = recover() }()
				comp, err = planner.Complete(ctx, []Message{{Role: RoleUser, Content: "look up SFO"}}, adapterTestTools())
			}()
			if panicValue != nil {
				t.Fatalf("completed envelope panicked: %v", panicValue)
			}
			if requests.Load() != 1 {
				t.Errorf("requests=%d, want one", requests.Load())
			}
			if tc.wantError != "" {
				if comp != nil || err == nil || !strings.Contains(err.Error(), tc.wantError) || clears != 0 {
					t.Fatalf("invalid envelope released completion=%+v error=%v clears=%d; want %q", comp, err, clears, tc.wantError)
				}
				if strings.HasPrefix(tc.wantError, "api error:") && err.Error() != "planner: openai-responses: "+tc.wantError {
					t.Fatalf("provider error precedence changed: %v", err)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("healthy completion=%+v error=%v", comp, err)
			}
			if clears != 1 || comp.Message.Content != tc.wantContent {
				t.Fatalf("healthy completion changed: %+v clears=%d", comp, clears)
			}
			if tc.wantTool {
				if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_null" || comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` || comp.FinishReason != "tool_calls" {
					t.Fatalf("item reconstruction changed: %+v", comp)
				}
			} else if len(comp.Message.ToolCalls) != 0 || comp.FinishReason != "stop" {
				t.Fatalf("final output changed: %+v", comp)
			}
		})
	}
}
