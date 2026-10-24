package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var upStructuredGuided = map[string]string{
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

const upStructuredResponseFormat = `{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object"}}}`

func upStructuredExtra(carriers map[string]string) string {
	parts := make([]string, 0, len(carriers))
	for _, key := range slices.Sorted(maps.Keys(carriers)) {
		parts = append(parts, fmt.Sprintf("%q:%s", key, carriers[key]))
	}
	return strings.Join(parts, ",")
}

func upStructuredBody(route string, stream bool, extra string) string {
	body := fmt.Sprintf(`{"model":"local","max_tokens":4,"temperature":0,"stream":%t`, stream)
	if route == "completions" {
		body += `,"prompt":"answer"`
	} else {
		body += `,"messages":[{"role":"user","content":"answer"}]`
	}
	if extra != "" {
		body += "," + extra
	}
	return body + "}"
}

type upStructuredRecorder struct {
	mu             sync.Mutex
	calls          int
	responseFormat json.RawMessage
	guided         map[string]json.RawMessage
}

func (p *upStructuredRecorder) Model() string            { return "local" }
func (p *upStructuredRecorder) ContextWindow() int       { return 8192 }
func (p *upStructuredRecorder) StreamingSupported() bool { return true }

func (p *upStructuredRecorder) record(opts []agent.SampleOpt) *agent.Completion {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.mu.Lock()
	p.calls++
	p.responseFormat = sp.ResponseFormat
	p.guided = sp.GuidedDecode
	p.mu.Unlock()
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}
}

func (p *upStructuredRecorder) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.record(opts), nil
}

func (p *upStructuredRecorder) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	comp := p.record(opts)
	if sink != nil {
		if err := sink(comp.Message.Content); err != nil {
			return nil, err
		}
	}
	return comp, nil
}

func (p *upStructuredRecorder) snapshot() (int, json.RawMessage, map[string]json.RawMessage) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.responseFormat, p.guided
}

func upStructuredServe(s *turnkeyServer, route, body string) *httptest.ResponseRecorder {
	path, handler := "/v1/chat/completions", s.handleChatCompletions
	if route == "completions" {
		path, handler = "/v1/completions", s.handleCompletions
	}
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler(rr, req)
	return rr
}

func upStructuredGuidedEqual(a, b map[string]json.RawMessage) bool {
	return maps.EqualFunc(a, b, func(x, y json.RawMessage) bool { return bytes.Equal(x, y) })
}

func upStructuredErrorCode(t *testing.T, raw []byte) string {
	t.Helper()
	var envelope struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("error body is not a JSON envelope: %v; body=%s", err, raw)
	}
	return envelope.Error.Type + "/" + envelope.Error.Code
}

// fak-test:runtime fast est=50ms lane=default
func TestUpStructuredCarriersReachPlanner(t *testing.T) {
	cases := []struct {
		name, extra string
		wantRF      string
		wantGuided  map[string]string
	}{
		{
			name:       "all",
			extra:      upStructuredExtra(upStructuredGuided) + `,"response_format":` + upStructuredResponseFormat,
			wantRF:     upStructuredResponseFormat,
			wantGuided: upStructuredGuided,
		},
		{name: "null", extra: `"response_format":null,"structured_outputs":null,"guided_regex":null`},
		{name: "none"},
	}
	for _, stream := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tc.name), func(t *testing.T) {
				var got [2]struct {
					rf     json.RawMessage
					guided map[string]json.RawMessage
				}
				for i, route := range []string{"chat", "completions"} {
					rec := &upStructuredRecorder{}
					s := &turnkeyServer{planner: rec, plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
					body := upStructuredBody(route, stream, tc.extra)
					rr := upStructuredServe(s, route, body)
					calls, rf, guided := rec.snapshot()
					if rr.Code != http.StatusOK || calls != 1 {
						t.Fatalf("%s: status=%d calls=%d body=%s", route, rr.Code, calls, rr.Body.String())
					}
					if route == "chat" {
						var wire gateway.ChatRequest
						if err := json.Unmarshal([]byte(body), &wire); err != nil {
							t.Fatal(err)
						}
						if !upStructuredGuidedEqual(guided, wire.GuidedDecodeFields()) {
							t.Fatalf("chat guided = %s, want GuidedDecodeFields %s", guided, wire.GuidedDecodeFields())
						}
					}
					got[i].rf, got[i].guided = rf, guided
				}
				chat, completions := got[0], got[1]
				if !bytes.Equal(completions.rf, chat.rf) || !upStructuredGuidedEqual(completions.guided, chat.guided) {
					t.Fatalf("completions rf=%s guided=%s, want chat parity rf=%s guided=%s", completions.rf, completions.guided, chat.rf, chat.guided)
				}
				switch tc.name {
				case "all":
					if string(completions.rf) != tc.wantRF {
						t.Fatalf("response_format = %s, want %s", completions.rf, tc.wantRF)
					}
					if len(completions.guided) != len(tc.wantGuided) {
						t.Fatalf("guided keys = %v, want %v", slices.Sorted(maps.Keys(completions.guided)), slices.Sorted(maps.Keys(tc.wantGuided)))
					}
					for key, want := range tc.wantGuided {
						if string(completions.guided[key]) != want {
							t.Fatalf("guided %s = %s, want %s", key, completions.guided[key], want)
						}
					}
				case "none":
					if completions.rf != nil || completions.guided != nil {
						t.Fatalf("absent carriers rf=%s guided=%s, want nil", completions.rf, completions.guided)
					}
				}
			})
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestUpStructuredCarriersErrorEnvelope(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{"unsupported", fmt.Errorf("%w: guided carrier %q", agent.ErrNativeStructuredUnsupported, "ebnf"), http.StatusBadRequest, "invalid_request_error/native_structured_output_unsupported"},
		{"no_token", fmt.Errorf("decode: %w", agent.ErrNativeStructuredNoToken), http.StatusInternalServerError, "server_error/native_structured_decode_failed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			writeTurnkeyInferenceError(rr, tc.err)
			if rr.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", rr.Code, tc.wantStatus, rr.Body.String())
			}
			if got := upStructuredErrorCode(t, rr.Body.Bytes()); got != tc.wantCode {
				t.Fatalf("error = %s, want %s", got, tc.wantCode)
			}
		})
	}
	rr := httptest.NewRecorder()
	writeTurnkeyInferenceError(rr, errors.New("other"))
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("unrelated error status = %d, want 500", rr.Code)
	}
}

func upStructuredNativeServer(t *testing.T) *turnkeyServer {
	t.Helper()
	m := model.NewSynthetic(upContextSyntheticConfig())
	m.Quantize()
	m.Cfg.MaxPositionEmbeddings = 4096
	planner := newTurnkeyInKernelPlanner(m, testProbeTokenizer(t), "local", false, nil, false, 4096)
	plan := macfit.TurnkeyProfile{Tier: macfit.ModelTier{ModelID: "local"}, ContextBudgetTokens: 4096}
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Shutdown(context.Background()) })
	return server
}

func upStructuredPost(t *testing.T, server *turnkeyServer, route, body string) (int, []byte) {
	t.Helper()
	path := "/v1/chat/completions"
	if route == "completions" {
		path = "/v1/completions"
	}
	resp, err := http.Post("http://"+server.Addr()+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, raw
}

// fak-test:runtime medium est=500ms lane=default
func TestUpStructuredCarriersNativeRefusalCode(t *testing.T) {
	server := upStructuredNativeServer(t)
	refused := []struct{ name, extra string }{
		{"regex_conflict", `"structured_outputs":{"regex":"[ab]{2}"},"guided_regex":"[ab]{3}"`},
		{"regex_json_conflict", `"guided_regex":"[ab]{2}","guided_json":{"type":"object"}`},
		{"regex_json_object_format", `"guided_regex":"[ab]{2}","response_format":{"type":"json_object"}`},
	}
	for _, route := range []string{"chat", "completions"} {
		for _, stream := range []bool{false, true} {
			for _, tc := range refused {
				t.Run(fmt.Sprintf("%s/stream=%t/%s", route, stream, tc.name), func(t *testing.T) {
					status, raw := upStructuredPost(t, server, route, upStructuredBody(route, stream, tc.extra))
					if status != http.StatusBadRequest {
						t.Fatalf("status = %d, want 400; body=%s", status, raw)
					}
					if got := upStructuredErrorCode(t, raw); got != "invalid_request_error/native_structured_output_unsupported" {
						t.Fatalf("error = %s, want invalid_request_error/native_structured_output_unsupported", got)
					}
				})
			}
			t.Run(fmt.Sprintf("%s/stream=%t/null_carriers", route, stream), func(t *testing.T) {
				extra := `"response_format":null,"structured_outputs":null,"guided_regex":null,"ebnf":null`
				if status, raw := upStructuredPost(t, server, route, upStructuredBody(route, stream, extra)); status != http.StatusOK {
					t.Fatalf("status = %d, want 200; body=%s", status, raw)
				}
			})
		}
	}
}

// fak-test:runtime medium est=400ms lane=default
func TestUpStructuredCarriersNativeEnforced(t *testing.T) {
	server := upStructuredNativeServer(t)
	extra := `"guided_regex":"[ab]{2}"`
	pattern := regexp.MustCompile(`^[ab]{2}$`)
	texts := map[string]string{}
	for _, route := range []string{"chat", "completions"} {
		status, raw := upStructuredPost(t, server, route, upStructuredBody(route, false, extra))
		if status != http.StatusOK {
			t.Fatalf("%s: status = %d, want 200; body=%s", route, status, raw)
		}
		var resp struct {
			Choices []struct {
				Text         string `json:"text"`
				FinishReason string `json:"finish_reason"`
				Message      struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil || len(resp.Choices) != 1 {
			t.Fatalf("%s: decode %v choices=%d body=%s", route, err, len(resp.Choices), raw)
		}
		text := resp.Choices[0].Text + resp.Choices[0].Message.Content
		if !pattern.MatchString(text) || resp.Choices[0].FinishReason != "stop" {
			t.Fatalf("%s: text=%q finish=%q, want ^[ab]{2}$ and stop", route, text, resp.Choices[0].FinishReason)
		}
		texts[route] = text
	}
	if texts["chat"] != texts["completions"] {
		t.Fatalf("completions text %q != chat text %q", texts["completions"], texts["chat"])
	}
}
