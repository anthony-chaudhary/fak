package gateway

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/wirescreen"
)

type nativeBeamWirePlanner struct {
	recordingPlanner
	calls     int
	supported bool
}

func (p *nativeBeamWirePlanner) NativeBeamSearchSupported() bool { return p.supported }
func (p *nativeBeamWirePlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	return p.recordingPlanner.Complete(ctx, m, t, opts...)
}

func nativeBeamWireCompletion() *agent.Completion {
	best := agent.Message{Role: agent.RoleAssistant, Content: "winner"}
	return &agent.Completion{Message: best, Model: "test-model", FinishReason: "length", Usage: agent.Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7},
		NativeBeamSearch: &agent.NativeBeamSearchReceipt{Schema: "fak.native-beam-search/1", Width: 2, LengthPenalty: 1, ForwardCalls: 3, ReplayedTokens: 17, ReplayTokenBound: 17, SessionOwnership: "serial-fresh-owned-cpu-replay",
			Sequences: []agent.NativeBeamSearchSequence{
				{Message: best, TokenIDs: []int{3, 4}, GeneratedTokens: 2, CumulativeLogprob: -1, Score: -.5, FinishReason: "length"},
				{Message: agent.Message{Role: agent.RoleAssistant, Content: "second"}, TokenIDs: []int{5, 6}, GeneratedTokens: 2, CumulativeLogprob: -2, Score: -1, FinishReason: "length"},
			}}}
}
func nativeBeamWireRequest(s *Server, path, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	r.RemoteAddr = "127.0.0.1:12345"
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	return w
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeBeamActualChatCarrierAndRankedResponse(t *testing.T) {
	s := newTestServer(t)
	p := &nativeBeamWirePlanner{supported: true, recordingPlanner: recordingPlanner{comp: nativeBeamWireCompletion()}}
	s.planner = p
	w := nativeBeamWireRequest(s, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"fak":{"beam_search":{"width":2,"length_penalty":0}}}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if p.calls != 1 || p.got.NativeBeamSearch == nil || p.got.NativeBeamSearch.Width != 2 ||
		p.got.NativeBeamSearch.LengthPenalty == nil || *p.got.NativeBeamSearch.LengthPenalty != 0 {
		t.Fatalf("carrier not forwarded %+v", p.got.NativeBeamSearch)
	}
	var response ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Message.Content != "winner" || response.Fak == nil || response.Fak.BeamSearch == nil ||
		len(response.Fak.BeamSearch.Sequences) != 2 || response.Fak.BeamSearch.ReplayedTokens != 17 ||
		response.Fak.BeamSearch.Schema != "fak.native-beam-search-scores/1" || response.Fak.BeamSearch.Sequences[1].Rank != 2 ||
		response.Usage.CompletionTokens != 2 {
		t.Fatalf("ranked wire lost or exploration mischarged: %+v", response)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeBeamUnsupportedIngressNeverCallsPlanner(t *testing.T) {
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":true,"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":129,"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"top_p":1,"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"response_format":{"type":"json_object"},"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"tool","tool_call_id":"x","content":"result"}],"max_tokens":2,"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":2,"fak":{"beam_search":{"width":2}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"fak_decode_trace":true,"fak":{"beam_search":{"width":2}}}`,
	} {
		s := newTestServer(t)
		p := &nativeBeamWirePlanner{supported: true, recordingPlanner: recordingPlanner{comp: nativeBeamWireCompletion()}}
		s.planner = p
		w := nativeBeamWireRequest(s, "/v1/chat/completions", body)
		if w.Code != http.StatusBadRequest || p.calls != 0 || !strings.Contains(w.Body.String(), "native_beam_search_unsupported") {
			t.Fatalf("unsupported ingress status=%d calls=%d body=%s", w.Code, p.calls, w.Body.String())
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeBeamRouteAndMissingResultRefuse(t *testing.T) {
	body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"fak":{"beam_search":{"width":2}}}`
	s := newTestServer(t)
	p := &nativeBeamWirePlanner{recordingPlanner: recordingPlanner{comp: nativeBeamWireCompletion()}}
	s.planner = p
	w := nativeBeamWireRequest(s, "/v1/chat/completions", body)
	if w.Code != http.StatusBadRequest || p.calls != 0 {
		t.Fatal("unsupported route dispatched")
	}
	p.supported = true
	p.comp.NativeBeamSearch = nil
	w = nativeBeamWireRequest(s, "/v1/chat/completions", body)
	if w.Code != http.StatusBadGateway || p.calls != 1 {
		t.Fatalf("ordinary completion silently substituted: %d %s", w.Code, w.Body.String())
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeBeamLegacyRefusalAndOmittedDefault(t *testing.T) {
	s := newTestServer(t)
	p := &nativeBeamWirePlanner{supported: true, recordingPlanner: recordingPlanner{comp: nativeBeamWireCompletion()}}
	s.planner = p
	w := nativeBeamWireRequest(s, "/v1/completions", `{"prompt":"hi","max_tokens":2,"fak":{"beam_search":{"width":2}}}`)
	if w.Code != http.StatusBadRequest || p.calls != 0 {
		t.Fatalf("legacy silently ignored selection: %d", w.Code)
	}
	p.comp.NativeBeamSearch = nil
	w = nativeBeamWireRequest(s, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}`)
	if w.Code != http.StatusOK || p.calls != 1 || p.got.NativeBeamSearch != nil || strings.Contains(w.Body.String(), "\"beam_search\"") {
		t.Fatalf("default changed: %d %s", w.Code, w.Body.String())
	}
}

func assertNativeBeamNoContent(t *testing.T, body []byte, forbidden ...string) {
	t.Helper()
	for _, value := range forbidden {
		if strings.Contains(string(body), value) {
			t.Fatalf("public response resurrected private beam value %q", value)
		}
	}
	var wire struct {
		Fak struct {
			BeamSearch json.RawMessage `json:"beam_search"`
		} `json:"fak"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Fak.BeamSearch) == 0 {
		t.Fatal("valid non-content beam receipt missing")
	}
	for _, key := range []string{"message", "token_ids", "content", "reasoning_content", "prompt", "digest"} {
		if strings.Contains(string(wire.Fak.BeamSearch), "\""+key+"\"") {
			t.Fatalf("public beam receipt contains forbidden field %q", key)
		}
	}
}

type nativeBeamRedactionBackend struct{ *blob.Store }

func (nativeBeamRedactionBackend) Caps() []abi.Capability { return nil }

// fak-test:runtime fast est=50ms lane=default
func TestNativeBeamWireCannotResurrectRedactedWinnerOrAlternatives(t *testing.T) {
	s := newTestServer(t)
	// Exercise the real redactor with its witnessed in-memory CAS, not a made-up
	// replacement string. This models an ALREADY-redacted winner; it does not
	// claim that pre-send input redaction is an assistant-output policy.
	const codec = "native-beam-redaction-fixture"
	abi.RegisterPageOutBackend(codec, nativeBeamRedactionBackend{blob.NewWithBudget(4096)})
	t.Setenv("FAK_PAGEOUT_BACKEND", codec)
	const secret = "alice@example.com"
	raw := "reach " + secret + " safely"
	redacted, ok := wirescreen.Apply(context.Background(), wirescreen.PIIRedactor(), []byte(raw), "assistant")
	if !ok || strings.Contains(string(redacted.Redacted), secret) || len(redacted.Spans) == 0 {
		t.Fatal("fixture did not actually redact its synthetic email")
	}
	comp := nativeBeamWireCompletion()
	comp.Message.Content = string(redacted.Redacted)
	comp.NativeBeamSearch.Sequences[0].Message.Content = raw
	comp.NativeBeamSearch.Sequences[0].TokenIDs = []int{810001, 810002}
	comp.NativeBeamSearch.Sequences[1].Message.Content = "unadmitted-alternative-only"
	comp.NativeBeamSearch.Sequences[1].Message.ReasoningContent = "private-alternative-reasoning"
	comp.NativeBeamSearch.Sequences[1].TokenIDs = []int{810003, 810004}
	p := &nativeBeamWirePlanner{supported: true, recordingPlanner: recordingPlanner{comp: comp}}
	s.planner = p
	w := nativeBeamWireRequest(s, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"fak":{"beam_search":{"width":2}}}`)
	if w.Code != http.StatusOK || p.calls != 1 {
		t.Fatalf("status=%d calls=%d body=%s", w.Code, p.calls, w.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Message.Content != string(redacted.Redacted) {
		t.Fatal("ordinary redacted winner changed")
	}
	assertNativeBeamNoContent(t, w.Body.Bytes(), secret, "unadmitted-alternative-only", "private-alternative-reasoning",
		"810001", "810002", "810003", "810004")
	if comp.NativeBeamSearch.Sequences[0].Message.Content != raw || comp.NativeBeamSearch.Sequences[0].TokenIDs[0] != 810001 {
		t.Fatal("wire projection mutated trusted internal search state")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeBeamWireCannotResurrectBodyRefusal(t *testing.T) {
	s := newTestServer(t)
	comp := nativeBeamWireCompletion()
	const raw = "I updated the file README.md. Private-body-marker."
	comp.Message.Content = raw
	// The test's allow_* adjudicator admits this read-only proposal, but no
	// effect-capable call witnesses the completed-edit claim. Exercise the real
	// handler's body-witness refusal rather than merely passing a boolean.
	comp.Message.ToolCalls = []agent.ToolCall{{ID: "beam-read", Type: "function",
		Function: agent.Func{Name: "allow_read", Arguments: "{}"}}}
	comp.NativeBeamSearch.Sequences[0].Message = comp.Message
	comp.NativeBeamSearch.Sequences[0].TokenIDs = []int{820001, 820002}
	comp.NativeBeamSearch.Sequences[1].Message.Content = "body-refusal-alternative"
	comp.NativeBeamSearch.Sequences[1].TokenIDs = []int{820003, 820004}
	p := &nativeBeamWirePlanner{supported: true, recordingPlanner: recordingPlanner{comp: comp}}
	s.planner = p
	w := nativeBeamWireRequest(s, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"fak":{"beam_search":{"width":2}}}`)
	if w.Code != http.StatusOK || p.calls != 1 || !strings.Contains(w.Body.String(), ReasonLoopBodyUnwitnessed) {
		t.Fatalf("real body refusal not reached: status=%d calls=%d body=%s", w.Code, p.calls, w.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].FinishReason != "stop" || len(response.Choices[0].Message.ToolCalls) != 0 {
		t.Fatal("ordinary refusal fold changed")
	}
	assertNativeBeamNoContent(t, w.Body.Bytes(), "Private-body-marker", "body-refusal-alternative",
		"820001", "820002", "820003", "820004")
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeBeamPublicProjectionOwnsOnlyBoundedClosedFields(t *testing.T) {
	in := nativeBeamWireCompletion().NativeBeamSearch
	out := nativeBeamWireReceipt(in)
	if out == nil {
		t.Fatal("valid receipt refused")
	}
	in.Sequences[0].Score = -99
	in.Sequences[0].FinishReason = "private-finish-text"
	if out.Sequences[0].Score != -.5 || out.Sequences[0].FinishReason != "length" {
		t.Fatal("wire projection aliases raw receipt")
	}
	for _, corrupt := range []func(*agent.NativeBeamSearchReceipt){
		func(r *agent.NativeBeamSearchReceipt) { r.Schema = "private-schema-text" },
		func(r *agent.NativeBeamSearchReceipt) { r.SessionOwnership = "private-ownership-text" },
		func(r *agent.NativeBeamSearchReceipt) { r.Width = 9 },
		func(r *agent.NativeBeamSearchReceipt) { r.ReplayTokenBound = 1 << 21 },
		func(r *agent.NativeBeamSearchReceipt) { r.Sequences[0].Score = math.NaN() },
		func(r *agent.NativeBeamSearchReceipt) { r.Sequences[0].FinishReason = "private-finish-text" },
	} {
		bad := nativeBeamWireCompletion().NativeBeamSearch
		corrupt(bad)
		if nativeBeamWireReceipt(bad) != nil {
			t.Fatal("unbounded or open-text metadata published")
		}
	}
	if nativeBeamWireReceipt(nil) != nil {
		t.Fatal("nil default changed")
	}
}
