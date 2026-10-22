package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
)

type upStructuredCarrierPlanner struct {
	turnkeyCompletionPlanner
	params      agent.SampleParams
	calls       int
	streamCalls int
}

func (p *upStructuredCarrierPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	p.params = agent.SampleParams{}
	for _, opt := range opts {
		opt(&p.params)
	}
	if string(p.params.ResponseFormat) == `{"type":"unsupported"}` {
		return nil, fmt.Errorf("wrapped: %w", agent.ErrNativeStructuredUnsupported)
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: `{"ok":true}`}}, nil
}

func (*upStructuredCarrierPlanner) StreamingSupported() bool { return true }

func (p *upStructuredCarrierPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.streamCalls++
	comp, err := p.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	return comp, sink(comp.Message.Content)
}

// TestUpStructuredCarriers witnesses the real HTTP handlers' carrier/options seam
// and typed refusal mapping, not native schema enforcement or model generation.
// fak-test:runtime fast est=10ms lane=default
func TestUpStructuredCarriers(t *testing.T) {
	const carriers = `"response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}}}}},"guided_json":{"type":"object"},"guided_regex":"[ab]","guided_choice":["a","b"],"guided_grammar":"root ::= \"a\"","guided_decoding_backend":"xgrammar","json_schema":{"type":"object"},"regex":"[ab]","ebnf":"root ::= \"a\"","structured_outputs":{"regex":"[ab]"},"unknown_constraint":{"not":"forwarded"}`
	var want gateway.ChatRequest
	if err := json.Unmarshal([]byte("{"+carriers+"}"), &want); err != nil {
		t.Fatal(err)
	}
	for _, route := range []string{"/v1/chat/completions", "/v1/completions"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", route, stream), func(t *testing.T) {
				p := &upStructuredCarrierPlanner{}
				s := &turnkeyServer{planner: p, plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
				handler := s.handleChatCompletions
				input := `"messages":[{"role":"user","content":"hello"}]`
				if route == "/v1/completions" {
					handler = s.handleCompletions
					input = `"prompt":"hello"`
				}
				post := func(extra string) *httptest.ResponseRecorder {
					rr := httptest.NewRecorder()
					body := fmt.Sprintf(`{%s,"stream":%t,%s}`, input, stream, extra)
					handler(rr, httptest.NewRequest(http.MethodPost, route, strings.NewReader(body)))
					return rr
				}
				rr := post(carriers)
				if rr.Code != http.StatusOK || string(p.params.ResponseFormat) != string(want.ResponseFormat) || !reflect.DeepEqual(p.params.GuidedDecode, want.GuidedDecodeFields()) {
					t.Fatalf("status=%d body=%s response_format=%s guided=%s", rr.Code, rr.Body.String(), p.params.ResponseFormat, p.params.GuidedDecode)
				}
				if p.calls != 1 || (stream && p.streamCalls != 1) || (!stream && p.streamCalls != 0) {
					t.Fatalf("planner calls=%d stream calls=%d", p.calls, p.streamCalls)
				}
				rr = post(`"response_format":{"type":"unsupported"}`)
				if rr.Code != http.StatusBadRequest || !strings.Contains(rr.Body.String(), `"code":"native_structured_output_unsupported"`) {
					t.Fatalf("typed refusal: status=%d body=%s", rr.Code, rr.Body.String())
				}
				calls := p.calls
				for _, bad := range []struct{ field, code string }{
					{`"response_format":{"type":"json_schema"}`, "json_schema_missing"},
					{`"structured_outputs":[]`, "invalid_structured_outputs"},
				} {
					rr = post(bad.field)
					if rr.Code < 400 || !strings.Contains(rr.Body.String(), bad.code) || p.calls != calls {
						t.Fatalf("malformed carrier reached planner: status=%d calls=%d body=%s", rr.Code, p.calls, rr.Body.String())
					}
				}
			})
		}
	}
}
