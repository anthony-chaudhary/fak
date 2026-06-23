package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// sseServer is an httptest server that replays a fixed OpenAI-compatible SSE body and
// captures the request body the planner sent, so a test can assert both the streamed
// result AND that the gateway asked the upstream to stream (stream:true + usage).
func sseServer(t *testing.T, body string) (*httptest.Server, *[]byte) {
	t.Helper()
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, &gotBody
}

func TestCompleteStreamForwardsContentLiveAndReportsUsage(t *testing.T) {
	const body = "data: {\"model\":\"served-m\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\", world\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n" +
		"data: [DONE]\n\n"
	srv, gotBody := sseServer(t, body)

	p := NewHTTPPlanner(srv.URL, "m", "")
	var got []string
	sink := func(frag string) error { got = append(got, frag); return nil }
	comp, err := p.CompleteStream(context.Background(), sink, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	// The sink saw each content fragment, in order, exactly as the upstream emitted —
	// the live half (a real time-to-first-token, not a single buffered blob).
	if strings.Join(got, "|") != "Hello|, world" {
		t.Fatalf("sink fragments = %q, want [Hello , world]", got)
	}
	if comp.Message.Content != "Hello, world" {
		t.Fatalf("content = %q, want %q", comp.Message.Content, "Hello, world")
	}
	if comp.FinishReason != "stop" {
		t.Fatalf("finish = %q, want stop", comp.FinishReason)
	}
	if comp.Model != "served-m" {
		t.Fatalf("model = %q, want served-m", comp.Model)
	}
	if comp.Usage.PromptTokens != 5 || comp.Usage.CompletionTokens != 2 {
		t.Fatalf("usage = %+v, want prompt 5 completion 2", comp.Usage)
	}

	// The gateway asked the upstream to stream AND to include usage on the final chunk.
	var sent struct {
		Stream        bool `json:"stream"`
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(*gotBody, &sent); err != nil {
		t.Fatalf("decode upstream body: %v\n%s", err, *gotBody)
	}
	if !sent.Stream {
		t.Fatalf("upstream request stream=false, want true: %s", *gotBody)
	}
	if sent.StreamOptions == nil || !sent.StreamOptions.IncludeUsage {
		t.Fatalf("upstream request missing stream_options.include_usage: %s", *gotBody)
	}
}

func TestCompleteStreamBuffersToolCallsAndNeverStreamsThem(t *testing.T) {
	const body = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"sf\\\"}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"
	srv, _ := sseServer(t, body)

	p := NewHTTPPlanner(srv.URL, "m", "")
	sinkCalled := false
	sink := func(string) error { sinkCalled = true; return nil }
	comp, err := p.CompleteStream(context.Background(), sink, []Message{{Role: RoleUser, Content: "weather?"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	// A tool-call turn carries no content, so the live content sink must never fire —
	// the tool call is buffered for the caller to adjudicate, never streamed.
	if sinkCalled {
		t.Fatal("content sink fired on a pure tool-call turn (tool call must stay buffered)")
	}
	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1: %+v", len(comp.Message.ToolCalls), comp.Message.ToolCalls)
	}
	tc := comp.Message.ToolCalls[0]
	if tc.ID != "call_1" || tc.Function.Name != "get_weather" {
		t.Fatalf("tool call id/name = %q/%q, want call_1/get_weather", tc.ID, tc.Function.Name)
	}
	if tc.Function.Arguments != `{"city":"sf"}` {
		t.Fatalf("accumulated args = %q, want %q", tc.Function.Arguments, `{"city":"sf"}`)
	}
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("finish = %q, want tool_calls", comp.FinishReason)
	}
}

func TestCompleteStreamFallsBackWhenUpstreamIgnoresStream(t *testing.T) {
	// A server that advertises OpenAI-compat but ignores stream:true and returns a
	// single JSON body must still yield the correct turn, not an empty stream.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"buffered reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	}))
	t.Cleanup(srv.Close)
	p := NewHTTPPlanner(srv.URL, "m", "")
	var got []string
	comp, err := p.CompleteStream(context.Background(), func(f string) error { got = append(got, f); return nil }, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if comp.Message.Content != "buffered reply" {
		t.Fatalf("content = %q, want %q", comp.Message.Content, "buffered reply")
	}
	if len(got) != 1 || got[0] != "buffered reply" {
		t.Fatalf("sink fragments = %q, want one full-content fragment", got)
	}
	if comp.FinishReason != "stop" || comp.Usage.PromptTokens != 3 {
		t.Fatalf("finish/usage = %q/%+v, want stop / prompt 3", comp.FinishReason, comp.Usage)
	}
}

func TestStreamingSupportedByProvider(t *testing.T) {
	cases := []struct {
		provider Provider
		want     bool
	}{
		{ProviderOpenAI, true},
		{ProviderXAI, true},
		{"", true},
		{ProviderAnthropic, false},
		{ProviderGemini, false},
		{ProviderOpenAIResponses, false},
	}
	for _, c := range cases {
		p := &HTTPPlanner{Provider: c.provider}
		if got := p.StreamingSupported(); got != c.want {
			t.Errorf("StreamingSupported(%q) = %v, want %v", c.provider, got, c.want)
		}
	}
}

func TestCompleteStreamUnsupportedWireReturnsSentinel(t *testing.T) {
	// A non-OpenAI wire returns the sentinel WITHOUT a network call, so the gateway can
	// fall back to its buffered path having written nothing.
	p := &HTTPPlanner{Provider: ProviderAnthropic, BaseURL: "http://127.0.0.1:0", Client: http.DefaultClient}
	_, err := p.CompleteStream(context.Background(), nil, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if !errors.Is(err, ErrStreamingUnsupported) {
		t.Fatalf("err = %v, want ErrStreamingUnsupported", err)
	}
}

func TestCompleteStreamSurfacesUpstreamStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"bad model"}}`)
	}))
	t.Cleanup(srv.Close)
	p := NewHTTPPlanner(srv.URL, "m", "")
	_, err := p.CompleteStream(context.Background(), func(string) error { return nil }, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	var se *UpstreamStatusError
	if !errors.As(err, &se) || se.Status != http.StatusBadRequest {
		t.Fatalf("err = %v, want UpstreamStatusError status 400", err)
	}
}
