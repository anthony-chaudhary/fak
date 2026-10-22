package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

const completionsStructuredResponseFormat = `{"type":"json_schema","json_schema":{"name":"o","schema":{"type":"object"}}}`

var completionsStructuredGuided = map[string]string{
	"guided_json":             `{"type":"object"}`,
	"guided_regex":            `"[ab]{2}"`,
	"guided_choice":           `["ab","ba"]`,
	"guided_grammar":          `"root ::= \"a\""`,
	"guided_decoding_backend": `"xgrammar"`,
	"json_schema":             `{"type":"object"}`,
	"regex":                   `"[ab]{2}"`,
	"ebnf":                    `"root ::= \"a\""`,
	"structured_outputs":      `{"regex":"[ab]{2}"}`,
}

func completionsStructuredExtra(carriers map[string]string) string {
	parts := make([]string, 0, len(carriers))
	for _, key := range slices.Sorted(maps.Keys(carriers)) {
		parts = append(parts, fmt.Sprintf("%q:%s", key, carriers[key]))
	}
	return strings.Join(parts, ",")
}

func completionsStructuredBody(model string, stream bool, extra string) string {
	body := fmt.Sprintf(`{"model":%q,"prompt":"answer","max_tokens":4,"temperature":0,"stream":%t`, model, stream)
	if extra != "" {
		body += "," + extra
	}
	return body + "}"
}

func completionsStructuredChatBody(model string, stream bool, extra string) string {
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"answer"}],"max_tokens":4,"temperature":0,"stream":%t`, model, stream)
	if extra != "" {
		body += "," + extra
	}
	return body + "}"
}

func completionsStructuredPost(t *testing.T, srv *Server, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

type completionsStructuredRecorder struct {
	mu             sync.Mutex
	calls          int
	responseFormat json.RawMessage
	guided         map[string]json.RawMessage
}

func (p *completionsStructuredRecorder) Model() string { return "recording" }

func (p *completionsStructuredRecorder) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.mu.Lock()
	p.calls++
	p.responseFormat = sp.ResponseFormat
	p.guided = sp.GuidedDecode
	p.mu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (p *completionsStructuredRecorder) snapshot() (int, json.RawMessage, map[string]json.RawMessage) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.responseFormat, p.guided
}

type completionsStructuredFrame struct {
	Choices []struct {
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Code string `json:"code"`
	} `json:"error"`
}

func completionsStructuredReadStream(t *testing.T, body string) (text, finish, errCode string, done bool) {
	t.Helper()
	var b strings.Builder
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if payload == "[DONE]" {
			done = true
			continue
		}
		var frame completionsStructuredFrame
		if err := json.Unmarshal([]byte(payload), &frame); err != nil {
			t.Fatalf("SSE frame %q: %v", payload, err)
		}
		if frame.Error != nil {
			errCode = frame.Error.Code
		}
		for _, c := range frame.Choices {
			b.WriteString(c.Text)
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
		}
	}
	return b.String(), finish, errCode, done
}

func completionsStructuredEqual(a, b map[string]json.RawMessage) bool {
	return maps.EqualFunc(a, b, func(x, y json.RawMessage) bool { return bytes.Equal(x, y) })
}

// fak-test:runtime fast est=80ms lane=default
func TestCompletionsStructuredCarriersReachPlanner(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	cases := []struct {
		name       string
		extra      string
		wantRF     string
		wantGuided map[string]string
	}{
		{
			name:       "all",
			extra:      completionsStructuredExtra(completionsStructuredGuided) + `,"response_format":` + completionsStructuredResponseFormat,
			wantRF:     completionsStructuredResponseFormat,
			wantGuided: completionsStructuredGuided,
		},
		{name: "null", extra: `"response_format":null,"structured_outputs":null,"guided_regex":null`},
		{name: "none"},
	}
	for _, stream := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tc.name), func(t *testing.T) {
				srv := newTestServer(t)
				legacy := &completionsStructuredRecorder{}
				srv.planner = legacy
				rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, tc.extra))
				calls, rf, guided := legacy.snapshot()
				if rr.Code != http.StatusOK || calls != 1 {
					t.Fatalf("status=%d calls=%d body=%s", rr.Code, calls, rr.Body.String())
				}

				chat := &completionsStructuredRecorder{}
				srv.planner = chat
				crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("m", stream, tc.extra))
				chatCalls, chatRF, chatGuided := chat.snapshot()
				if crr.Code != http.StatusOK || chatCalls != 1 {
					t.Fatalf("chat precondition: status=%d calls=%d body=%s", crr.Code, chatCalls, crr.Body.String())
				}
				if !bytes.Equal(rf, chatRF) || !completionsStructuredEqual(guided, chatGuided) {
					t.Fatalf("completions carriers rf=%s guided=%s, want chat parity rf=%s guided=%s", rf, guided, chatRF, chatGuided)
				}

				switch tc.name {
				case "all":
					if string(rf) != tc.wantRF {
						t.Fatalf("response_format = %s, want %s", rf, tc.wantRF)
					}
					if len(guided) != len(tc.wantGuided) {
						t.Fatalf("guided keys = %v, want %v", slices.Sorted(maps.Keys(guided)), slices.Sorted(maps.Keys(tc.wantGuided)))
					}
					for key, want := range tc.wantGuided {
						if string(guided[key]) != want {
							t.Fatalf("guided %s = %s, want %s", key, guided[key], want)
						}
					}
				case "none":
					if rf != nil || guided != nil {
						t.Fatalf("absent carriers rf=%s guided=%s, want nil", rf, guided)
					}
				}
			})
		}
	}
}

// fak-test:runtime fast est=120ms lane=default
func TestCompletionsStructuredIngressRefusedBeforePlanner(t *testing.T) {
	deliveries := make(chan string, 64)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		deliveries <- string(raw)
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(follower.Close)
	overlong := `"` + strings.Repeat("x", maxGuidedRegexBytes+1) + `"`
	cases := []struct {
		name, extra string
		status      int
		code        string
	}{
		{"structured_outputs_string", `"structured_outputs":"regex"`, http.StatusBadRequest, "invalid_structured_outputs"},
		{"structured_outputs_array", `"structured_outputs":[]`, http.StatusBadRequest, "invalid_structured_outputs"},
		{"guided_regex_overlong", `"guided_regex":` + overlong, http.StatusBadRequest, "parameter_out_of_range"},
		{"regex_overlong", `"regex":` + overlong, http.StatusBadRequest, "parameter_out_of_range"},
		{"response_format_string", `"response_format":"json_schema"`, http.StatusUnprocessableEntity, "invalid_response_format"},
		{"response_format_schema_missing", `"response_format":{"type":"json_schema"}`, http.StatusUnprocessableEntity, "json_schema_missing"},
	}
	for _, ep := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, tc := range cases {
				t.Run(fmt.Sprintf("ep=%t/stream=%t/%s", ep, stream, tc.name), func(t *testing.T) {
					t.Setenv("FAK_EP_FANOUT_ADDRS", "")
					if ep {
						t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
					}
					for len(deliveries) > 0 {
						<-deliveries
					}
					srv := newTestServer(t)
					p := &completionsStructuredRecorder{}
					srv.planner = p
					rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, tc.extra))
					calls, _, _ := p.snapshot()
					if rr.Code != tc.status || chatErrorCode(t, rr.Body.Bytes()) != tc.code {
						t.Fatalf("status=%d code=%q, want %d %s", rr.Code, chatErrorCode(t, rr.Body.Bytes()), tc.status, tc.code)
					}
					if ct := rr.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
						t.Fatalf("Content-Type=%q, want application/json", ct)
					}
					if calls != 0 || len(deliveries) != 0 {
						t.Fatalf("planner calls=%d follower deliveries=%d, want 0 and 0", calls, len(deliveries))
					}
					t.Setenv("FAK_EP_FANOUT_ADDRS", "")
					crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("m", stream, tc.extra))
					if crr.Code != rr.Code || chatErrorCode(t, crr.Body.Bytes()) != tc.code {
						t.Fatalf("chat parity: status=%d code=%q, want %d %s", crr.Code, chatErrorCode(t, crr.Body.Bytes()), rr.Code, tc.code)
					}
				})
			}
		}
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestCompletionsStructuredRideForwardsVerbatim(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	var mu sync.Mutex
	var forwarded map[string]json.RawMessage
	hits := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		mu.Lock()
		forwarded = body
		hits++
		mu.Unlock()
		if string(body["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"model":"m","choices":[{"delta":{"role":"assistant","content":"ab"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"ab"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	t.Cleanup(upstream.Close)
	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	all := maps.Clone(completionsStructuredGuided)
	all["response_format"] = completionsStructuredResponseFormat
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			name string
			want map[string]string
		}{
			{"all", all},
			{"absent", map[string]string{}},
		} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tc.name), func(t *testing.T) {
				mu.Lock()
				before := hits
				mu.Unlock()
				rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, completionsStructuredExtra(tc.want)))
				if rr.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
				}
				mu.Lock()
				got, calls := forwarded, hits-before
				mu.Unlock()
				if calls != 1 {
					t.Fatalf("upstream hits=%d, want 1", calls)
				}
				for _, key := range slices.Sorted(maps.Keys(all)) {
					want, wanted := tc.want[key]
					raw, present := got[key]
					if present != wanted {
						t.Fatalf("upstream %s present=%t, want %t (body keys %v)", key, present, wanted, slices.Sorted(maps.Keys(got)))
					}
					if wanted && string(raw) != want {
						t.Fatalf("upstream %s = %s, want verbatim %s", key, raw, want)
					}
				}
			})
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestCompletionsStructuredNativeMatchesChat(t *testing.T) {
	srv := nativeReceiptServer(t)
	enforced := []struct {
		name, extra string
		valid       func(string) bool
	}{
		{"guided_regex", `"guided_regex":"[ab]{2}"`, regexp.MustCompile(`^[ab]{2}$`).MatchString},
		{"structured_outputs_regex", `"structured_outputs":{"regex":"[ab]{2}"}`, regexp.MustCompile(`^[ab]{2}$`).MatchString},
		{"guided_choice", `"guided_choice":["ab","ba"]`, func(s string) bool { return s == "ab" || s == "ba" }},
		{"guided_json", `"guided_json":{"type":"object"}`, func(s string) bool {
			var obj map[string]any
			return json.Unmarshal([]byte(s), &obj) == nil && obj != nil
		}},
	}
	refused := []string{
		`"structured_outputs":{"grammar":"root ::= \"a\""}`,
		`"guided_grammar":"root ::= \"a\""`,
		`"ebnf":"root ::= \"a\""`,
		`"guided_decoding_backend":"xgrammar"`,
		`"structured_outputs":{"regex":"[ab]{2}"},"guided_regex":"[ab]{3}"`,
	}
	for _, stream := range []bool{false, true} {
		for _, tc := range enforced {
			t.Run(fmt.Sprintf("stream=%t/enforced/%s", stream, tc.name), func(t *testing.T) {
				crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("synthetic-live", false, tc.extra))
				var chat ChatResponse
				if crr.Code != http.StatusOK || json.Unmarshal(crr.Body.Bytes(), &chat) != nil || len(chat.Choices) != 1 {
					t.Fatalf("chat precondition: status=%d body=%s", crr.Code, crr.Body.String())
				}
				rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("synthetic-live", stream, tc.extra))
				if rr.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
				}
				var text, finish string
				if stream {
					var errCode string
					var done bool
					text, finish, errCode, done = completionsStructuredReadStream(t, rr.Body.String())
					if errCode != "" || !done {
						t.Fatalf("stream error=%q done=%t body=%s", errCode, done, rr.Body.String())
					}
				} else {
					var resp CompletionResponse
					if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil || len(resp.Choices) != 1 || resp.Choices[0].FinishReason == nil {
						t.Fatalf("decode completion: %v body=%s", err, rr.Body.String())
					}
					text, finish = resp.Choices[0].Text, *resp.Choices[0].FinishReason
				}
				if finish != "stop" || !tc.valid(text) || text != chat.Choices[0].Message.Content {
					t.Fatalf("text=%q finish=%q, want a constrained stop equal to chat %q", text, finish, chat.Choices[0].Message.Content)
				}
			})
		}
		for _, extra := range refused {
			t.Run(fmt.Sprintf("stream=%t/refused/%s", stream, extra), func(t *testing.T) {
				crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("synthetic-live", false, extra))
				chatCode := chatErrorCode(t, crr.Body.Bytes())
				if crr.Code == http.StatusOK || chatCode == "" {
					t.Fatalf("chat precondition: status=%d body=%s, want a typed refusal", crr.Code, crr.Body.String())
				}
				rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("synthetic-live", stream, extra))
				if !stream {
					if rr.Code != crr.Code || chatErrorCode(t, rr.Body.Bytes()) != chatCode {
						t.Fatalf("status=%d code=%q, want chat %d %s", rr.Code, chatErrorCode(t, rr.Body.Bytes()), crr.Code, chatCode)
					}
					return
				}
				if rr.Code != http.StatusOK || !strings.HasPrefix(rr.Header().Get("Content-Type"), "text/event-stream") {
					t.Fatalf("stream status=%d Content-Type=%q, want 200 text/event-stream", rr.Code, rr.Header().Get("Content-Type"))
				}
				text, _, errCode, done := completionsStructuredReadStream(t, rr.Body.String())
				if errCode != chatCode || !done || text != "" {
					t.Fatalf("in-band error=%q done=%t text=%q, want %s, [DONE] and no text", errCode, done, text, chatCode)
				}
			})
		}
	}
}
