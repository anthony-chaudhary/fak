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
func TestHTTPPlannerResponsesBufferedStatus(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	const tool = `{"type":"function_call","call_id":"call_status","name":"lookup","arguments":"{\"city\":\"SFO\"}"}`
	const partialTool = `{"type":"function_call","call_id":"call_status","name":"lookup","arguments":"{\"city\":"}`
	const message = `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}`
	for _, tc := range []struct {
		name, status, output, extra, wantError, wantFinish string
	}{
		{"incomplete", `"incomplete"`, partialTool, `,"incomplete_details":{"reason":"max_output_tokens"}`, "max_output_tokens", ""},
		{"failed", `"failed"`, tool, "", `status "failed"`, ""},
		{"cancelled", `"cancelled"`, tool, "", `status "cancelled"`, ""},
		{"queued", `"queued"`, tool, "", `status "queued"`, ""},
		{"in_progress", `"in_progress"`, tool, "", `status "in_progress"`, ""},
		{"unknown", `"future_status"`, tool, "", `status "future_status"`, ""},
		{"completed_tool", `"completed"`, tool, "", "", "tool_calls"},
		{"completed_text", `"completed"`, message, "", "", "stop"},
		{"omitted_legacy", "", tool, "", "", "tool_calls"},
		{"empty_legacy", `""`, tool, "", "", "tool_calls"},
		{"null_legacy", "null", tool, "", "", "tool_calls"},
		{"api_error_precedence", `"failed"`, tool, `,"error":{"message":"existing provider failure"}`, "api error: existing provider failure", ""},
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
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("request decode: %v", err)
				}
				if req.Stream {
					t.Error("buffered witness unexpectedly requested SSE")
				}
				status := ""
				if tc.status != "" {
					status = `"status":` + tc.status + ","
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, "{"+status+`"model":"gpt-test","output":[`+tc.output+`],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}`+tc.extra+"}")
			}))
			defer ts.Close()
			planner, err := NewProviderHTTPPlanner(string(ProviderOpenAIResponses), ts.URL, "gpt-test", "loopback-only")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = ts.Client()
			var checkpointClears int
			planner.PendingTurnCheckpoint = func(attempt, status int, retryAt int64) {
				checkpointClears++
				if attempt != 0 || status != 0 || retryAt != 0 {
					t.Errorf("unexpected retry checkpoint: %d %d %d", attempt, status, retryAt)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			comp, err := planner.Complete(ctx, []Message{{Role: RoleUser, Content: "look up SFO"}}, adapterTestTools())
			if requests.Load() != 1 {
				t.Errorf("requests=%d, want one without status-driven retry", requests.Load())
			}
			if tc.wantError != "" {
				if comp != nil || err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("unsuccessful response released completion=%+v error=%v; want %q", comp, err, tc.wantError)
				}
				if checkpointClears != 0 {
					t.Fatalf("unsuccessful response cleared pending turn %d times", checkpointClears)
				}
				if tc.name == "api_error_precedence" && err.Error() != "planner: openai-responses: api error: existing provider failure" {
					t.Fatalf("API error changed: %v", err)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("completion=%+v error=%v", comp, err)
			}
			if checkpointClears != 1 || comp.FinishReason != tc.wantFinish || comp.Model != "gpt-test" || comp.Usage.TotalTokens != 5 {
				t.Fatalf("completed response drifted: completion=%+v clears=%d", comp, checkpointClears)
			}
			if tc.wantFinish == "tool_calls" {
				if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_status" ||
					comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` {
					t.Fatalf("completed tool identity changed: %+v", comp.Message)
				}
			} else if comp.Message.Content != "done" || len(comp.Message.ToolCalls) != 0 {
				t.Fatalf("completed text changed: %+v", comp.Message)
			}
		})
	}
}
