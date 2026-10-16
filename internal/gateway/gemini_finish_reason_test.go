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

// fak-test:runtime fast est=1s
func TestGeminiLimitResponsesWire(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	for _, tc := range []struct {
		name, finish string
		stream       bool
	}{
		{"capped_buffered", "MAX_TOKENS", false},
		{"capped_stream", "MAX_TOKENS", true},
		{"stopped_buffered", "STOP", false},
		{"stopped_stream", "STOP", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/v1beta/models/gemini-test:generateContent" {
					t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.Path)
				}
				var request struct {
					Contents         []json.RawMessage `json:"contents"`
					GenerationConfig struct {
						MaxOutputTokens int `json:"maxOutputTokens"`
					} `json:"generationConfig"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil || len(request.Contents) == 0 || request.GenerationConfig.MaxOutputTokens != 7 {
					t.Errorf("upstream request=%+v error=%v", request, err)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"partial answer"}]},"finishReason":%q}],"modelVersion":"gemini-test","usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":7,"totalTokenCount":10}}`, tc.finish)
			}))
			defer up.Close()
			planner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderGemini), up.URL+"/v1beta", "gemini-test", "")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = up.Client()
			srv := newTestServer(t)
			srv.planner = planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			client := ts.Client()
			client.Timeout = 2 * time.Second
			res, err := client.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(fmt.Sprintf(`{"model":"gemini-test","input":"answer briefly","max_output_tokens":7,"store":false,"stream":%t}`, tc.stream)))
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			raw, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatal(err)
			}
			if res.StatusCode != http.StatusOK || calls.Load() != 1 {
				t.Fatalf("status=%d calls=%d body=%s", res.StatusCode, calls.Load(), raw)
			}
			var got responsesResponse
			wantStatus, wantEvent := "completed", "response.completed"
			if tc.finish == "MAX_TOKENS" {
				wantStatus, wantEvent = "incomplete", "response.incomplete"
			}
			if !tc.stream {
				if err := json.Unmarshal(raw, &got); err != nil {
					t.Fatal(err)
				}
			} else {
				terminals := 0
				for _, line := range strings.Split(string(raw), "\n") {
					if !strings.HasPrefix(line, "data: ") {
						continue
					}
					var frame struct {
						Type     string            `json:"type"`
						Response responsesResponse `json:"response"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
						t.Fatal(err)
					}
					if frame.Type == "response.completed" || frame.Type == "response.incomplete" {
						terminals++
						if frame.Type != wantEvent {
							t.Errorf("terminal=%q want %q", frame.Type, wantEvent)
						}
						got = frame.Response
					}
				}
				if terminals != 1 {
					t.Fatalf("terminal count=%d body=%s", terminals, raw)
				}
			}
			if got.Status != wantStatus {
				t.Fatalf("Gemini capped response status=%q want %q; body=%s", got.Status, wantStatus, raw)
			}
			if tc.finish == "MAX_TOKENS" {
				if got.IncompleteDetails == nil || got.IncompleteDetails.Reason != "max_output_tokens" {
					t.Fatalf("missing token-limit details: %+v", got)
				}
			} else if got.IncompleteDetails != nil {
				t.Fatalf("normal stop gained incomplete details: %+v", got)
			}
			if got.OutputText != "partial answer" || got.Model != "gemini-test" || got.Usage.InputTokens != 3 || got.Usage.OutputTokens != 7 || got.Usage.TotalTokens != 10 {
				t.Fatalf("provider content/usage changed: %+v", got)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestGeminiFinishNormalizationControls(t *testing.T) {
	adapter, err := agent.NewTranscriptAdapter(agent.ProviderGemini)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, finish, parts, want string }{
		{"unknown_preserved", "OTHER", `[{"text":"answer"}]`, "other"},
		{"omitted_preserved", "", `[{"text":"answer"}]`, ""},
		{"tool_precedence", "MAX_TOKENS", `[{"thoughtSignature":"opaque","functionCall":{"id":"call_1","name":"lookup","args":{"x":1}}}]`, "tool_calls"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			comp, err := adapter.ParseResponse([]byte(fmt.Sprintf(`{"candidates":[{"content":{"parts":%s},"finishReason":%q}]}`, tc.parts, tc.finish)))
			if err != nil || comp == nil {
				t.Fatalf("completion=%+v err=%v", comp, err)
			}
			if comp.FinishReason != tc.want {
				t.Fatalf("finish=%q want %q", comp.FinishReason, tc.want)
			}
			if tc.want == "tool_calls" && (len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ThoughtSignature != "opaque" || comp.Message.ToolCalls[0].ID != "call_1" || comp.Message.ToolCalls[0].Function.Arguments != `{"x":1}`) {
				t.Fatalf("tool metadata changed: %+v", comp.Message.ToolCalls)
			}
		})
	}
}
