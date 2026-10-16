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
func TestHTTPPlannerAnthropicBufferedEnvelope(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	for _, tc := range []struct {
		name, body, wantError, wantFinish, wantContent string
		wantTool                                       bool
	}{
		{"null_response", `null`, "missing or null content array", "", "", false},
		{"missing_content", `{"type":"message","stop_reason":"end_turn"}`, "missing or null content array", "", "", false},
		{"null_content", `{"content":null,"stop_reason":"end_turn"}`, "missing or null content array", "", "", false},
		{"malformed_json", `{"content":`, "decode:", "", "", false},
		{"wrong_content_type", `{"content":{}}`, "decode:", "", "", false},
		{"api_error_precedence", `{"error":{"message":"existing provider failure"},"content":null}`, "api error: existing provider failure", "", "", false},
		{"empty_turn", `{"content":[],"stop_reason":"end_turn","model":"claude-test","usage":{"input_tokens":3,"output_tokens":2}}`, "", "stop", "", false},
		{"text_turn", `{"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","model":"claude-test","usage":{"input_tokens":3,"output_tokens":2}}`, "", "stop", "done", false},
		{"tool_only_turn", `{"content":[{"type":"tool_use","id":"call_envelope","name":"lookup","input":{"city":"SFO"}}],"stop_reason":"tool_use","model":"claude-test","usage":{"input_tokens":3,"output_tokens":2}}`, "", "tool_calls", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
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
				_, _ = io.WriteString(w, tc.body)
			}))
			defer ts.Close()
			planner, err := NewProviderHTTPPlanner(string(ProviderAnthropic), ts.URL, "claude-test", "loopback-only")
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
				if tc.name == "api_error_precedence" && err.Error() != "planner: anthropic: api error: existing provider failure" {
					t.Fatalf("API error changed: %v", err)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("completion=%+v error=%v", comp, err)
			}
			if clears != 1 || comp.FinishReason != tc.wantFinish || comp.Message.Content != tc.wantContent || comp.Model != "claude-test" || comp.Usage.TotalTokens != 5 {
				t.Fatalf("healthy completion changed: %+v clears=%d", comp, clears)
			}
			if tc.wantTool {
				if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_envelope" || comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` {
					t.Fatalf("tool-only turn changed: %+v", comp.Message.ToolCalls)
				}
			} else if len(comp.Message.ToolCalls) != 0 {
				t.Fatalf("unexpected tool calls: %+v", comp.Message.ToolCalls)
			}
		})
	}
}
