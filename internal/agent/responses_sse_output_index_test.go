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
func TestHTTPPlannerResponsesSSEOutputIndex(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	frame := func(payload string) string { return "data: " + payload + "\n\n" }
	item := func(index, id string) string {
		field := ""
		if index != "" {
			field = `"output_index":` + index + `,`
		}
		return frame(`{"type":"response.output_item.done",` + field + `"item":{"type":"function_call","call_id":"` + id + `","name":"lookup","arguments":"{\"city\":\"SFO\"}"}}`)
	}
	empty := frame(`{"type":"response.completed","response":{"status":"completed","output":[]}}`)
	text := frame(`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}}`)
	provider := frame(`{"type":"error","message":"provider failure"}`)
	duplicate := item("0", "call_a") + item("0", "call_a")
	for _, tc := range []struct {
		name, body, wantError string
		wantIDs               []string
		wantText, truncated   bool
	}{
		{name: "duplicate_tool", body: duplicate + empty, wantError: "duplicate output_index 0"},
		{name: "duplicate_distinct_calls", body: item("0", "call_a") + item("0", "call_b") + empty, wantError: "duplicate output_index 0"},
		{name: "malformed_index", body: item(`"zero"`, "call_a") + empty, wantError: "invalid output_index"},
		{name: "negative_index", body: item("-1", "call_a") + empty, wantError: "invalid output_index"},
		{name: "null_index", body: item("null", "call_a") + empty, wantError: "invalid output_index"},
		{name: "fractional_index", body: item("0.5", "call_a") + empty, wantError: "invalid output_index"},
		{name: "fallback_collision", body: item("1", "call_a") + item("", "call_b") + empty, wantError: "duplicate output_index 1"},
		{name: "out_of_order", body: item("1", "call_b") + item("0", "call_a") + empty, wantIDs: []string{"call_a", "call_b"}},
		{name: "omitted_index_compatibility", body: item("", "call_a") + item("", "call_b") + empty, wantIDs: []string{"call_a", "call_b"}},
		{name: "authoritative_final", body: duplicate + item("null", "call_b") + text, wantText: true},
		{name: "provider_precedence", body: duplicate + empty + provider, wantError: "api error: provider failure"},
		{name: "parser_precedence", body: duplicate + empty + frame(`{`), wantError: "decode SSE data:"},
		{name: "body_precedence", body: duplicate + empty, truncated: true, wantError: "read body: unexpected EOF"},
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
			planner.Client, planner.ForceResponsesStream = ts.Client(), true
			clears := 0
			planner.PendingTurnCheckpoint = func(attempt, status int, retryAt int64) {
				if attempt == 0 && status == 0 && retryAt == 0 {
					clears++
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			comp, err := planner.Complete(ctx, []Message{{Role: RoleUser, Content: "look up SFO"}}, adapterTestTools())
			if requests.Load() != 1 {
				t.Errorf("requests=%d, want one", requests.Load())
			}
			if tc.wantError != "" {
				if comp != nil || err == nil || !strings.Contains(err.Error(), tc.wantError) || clears != 0 {
					t.Fatalf("ambiguous reconstruction released completion=%+v error=%v clears=%d; want %q", comp, err, clears, tc.wantError)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("healthy completion=%+v error=%v", comp, err)
			}
			if clears != 1 || len(comp.Message.ToolCalls) != len(tc.wantIDs) {
				t.Fatalf("healthy output changed: %+v clears=%d", comp, clears)
			}
			for i, id := range tc.wantIDs {
				call := comp.Message.ToolCalls[i]
				if call.ID != id || call.Function.Name != "lookup" || call.Function.Arguments != `{"city":"SFO"}` || comp.FinishReason != "tool_calls" {
					t.Fatalf("ordered call %d changed: %+v", i, comp)
				}
			}
			wantContent := ""
			if tc.wantText {
				wantContent = "done"
			}
			if comp.Message.Content != wantContent || (tc.wantText && comp.FinishReason != "stop") {
				t.Fatalf("final content changed: %+v", comp)
			}
		})
	}
}
