package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Embedding only Planner deliberately hides streaming while retaining the actual
// HTTP provider parser, so the synthesized Responses path is exercised separately.
type responsesFinishBufferedPlanner struct{ agent.Planner }

// fak-test:runtime fast est=1s
func TestResponsesProviderFinishWire(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	type finishCase struct {
		name, route, finish string
		providerError       bool
	}
	var cases []finishCase
	for _, route := range []string{"buffered", "synthesized", "live"} {
		for _, finish := range []string{"content_filter", "length", "stop", "other"} {
			cases = append(cases, finishCase{name: route + "_" + finish, route: route, finish: finish})
		}
	}
	cases = append(cases, finishCase{name: "live_error_precedence", route: "live", finish: "content_filter", providerError: true})
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const answer = "partial answer"
			var calls atomic.Int32
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
					t.Errorf("provider route=%s %s", r.Method, r.URL.Path)
				}
				var request struct {
					Stream    bool              `json:"stream"`
					MaxTokens int               `json:"max_tokens"`
					Messages  []json.RawMessage `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("request: %v", err)
					return
				}
				if request.Stream != (tc.route == "live") || request.MaxTokens != 7 || len(request.Messages) == 0 {
					t.Errorf("provider request=%+v", request)
				}
				if request.Stream {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprintf(w, "data: {\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q}}]}\n\n", answer)
					w.(http.Flusher).Flush()
					if tc.providerError {
						fmt.Fprint(w, "data: {\"error\":{\"message\":\"fixture provider failure\",\"type\":\"server_error\"}}\n\n")
					}
					fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n", tc.finish)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":%q}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`, answer, tc.finish)
			}))
			defer up.Close()
			planner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), up.URL, "test-model", "")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = up.Client()
			srv := newTestServer(t)
			srv.planner = planner
			if tc.route == "synthesized" {
				srv.planner = responsesFinishBufferedPlanner{Planner: planner}
			}
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			client := ts.Client()
			client.Timeout = 2 * time.Second
			stream := tc.route != "buffered"
			res, err := client.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(fmt.Sprintf(`{"model":"test-model","input":"answer briefly","max_output_tokens":7,"store":false,"stream":%t}`, stream)))
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			raw, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatal(err)
			}
			if res.StatusCode != http.StatusOK || calls.Load() != 1 {
				t.Fatalf("HTTP status=%d provider calls=%d body=%s", res.StatusCode, calls.Load(), raw)
			}
			wantStatus, wantEvent, wantReason := "completed", "response.completed", ""
			switch tc.finish {
			case "content_filter":
				wantStatus, wantEvent, wantReason = "incomplete", "response.incomplete", "content_filter"
			case "length":
				wantStatus, wantEvent, wantReason = "incomplete", "response.incomplete", "max_output_tokens"
			}
			if tc.providerError {
				wantStatus, wantEvent, wantReason = "failed", "response.failed", ""
			}
			got := decodeResponsesFinishWire(t, raw, stream, wantEvent)
			if got.Status != wantStatus {
				t.Fatalf("terminal mapping status=%q want=%q body=%s", got.Status, wantStatus, raw)
			}
			if wantReason == "" {
				if got.IncompleteDetails != nil {
					t.Fatalf("unexpected incomplete details: %+v", got.IncompleteDetails)
				}
			} else if got.IncompleteDetails == nil || got.IncompleteDetails.Reason != wantReason {
				t.Fatalf("terminal mapping details=%+v want=%q", got.IncompleteDetails, wantReason)
			}
			if tc.providerError {
				return
			}
			if got.OutputText != answer || messageText(got.Output) != answer || len(functionCallItems(got.Output)) != 0 || got.Model != "test-model" || got.Usage.InputTokens != 3 || got.Usage.OutputTokens != 2 || got.Usage.TotalTokens != 5 {
				t.Fatalf("provider output/usage changed: %+v", got)
			}
		})
	}
}

func decodeResponsesFinishWire(t *testing.T, raw []byte, stream bool, wantEvent string) responsesResponse {
	t.Helper()
	var got responsesResponse
	if !stream {
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	terminals := 0
	for _, block := range strings.Split(string(raw), "\n\n") {
		var event, data string
		for _, line := range strings.Split(block, "\n") {
			if v, ok := strings.CutPrefix(line, "event: "); ok {
				event = v
			}
			if v, ok := strings.CutPrefix(line, "data: "); ok {
				data = v
			}
		}
		if data == "" {
			continue
		}
		var frame struct {
			Type     string            `json:"type"`
			Response responsesResponse `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			t.Fatal(err)
		}
		if event != frame.Type {
			t.Fatalf("event/type mismatch: %q/%q", event, frame.Type)
		}
		switch frame.Type {
		case "response.completed", "response.incomplete", "response.failed":
			terminals++
			if frame.Type != wantEvent {
				t.Fatalf("terminal mapping event=%q want=%q body=%s", frame.Type, wantEvent, raw)
			}
			got = frame.Response
		}
	}
	if terminals != 1 {
		t.Fatalf("terminal count=%d body=%s", terminals, raw)
	}
	return got
}

// fak-test:runtime fast est=1s
func TestResponsesContentFilterGuardPrecedence(t *testing.T) {
	for _, finish := range []string{"content_filter", "length", "stop"} {
		t.Run(finish, func(t *testing.T) {
			first := toolCallTurn("call_one", "deny_shell", `{"command":"bash install-openssh-linux.sh"}`)
			second := toolCallTurn("call_two", "selfmod_edit", `{"path":"internal/abi/abi.go"}`)
			first.FinishReason, second.FinishReason = finish, finish
			planner := &sequencePlanner{comps: []*agent.Completion{first, second}}
			srv := newTestServer(t)
			srv.planner = planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			code, resp := postResponses(t, ts.URL, map[string]any{"model": "m", "input": "fix the install script", "tools": []map[string]any{{"type": "function", "name": "deny_shell"}, {"type": "function", "name": "selfmod_edit"}}})
			wantReason := deniedGuardIncompleteReason
			if finish == "length" {
				wantReason = "max_output_tokens"
			} // preserve existing length-first precedence
			if code != http.StatusOK || planner.calls() != 2 || resp.Status != "incomplete" || resp.IncompleteDetails == nil || resp.IncompleteDetails.Reason != wantReason {
				t.Fatalf("guard precedence status=%d calls=%d response=%+v", code, planner.calls(), resp)
			}
			if len(functionCallItems(resp.Output)) != 0 || !strings.Contains(messageText(resp.Output), "BLOCKED_BY_GUARD") {
				t.Fatalf("guard enforcement changed: %+v", resp)
			}
		})
	}
}
