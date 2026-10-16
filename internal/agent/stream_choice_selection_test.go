package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=300ms lane=default
func TestCompleteStreamSelectsChoiceZero(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	frame := func(choices string) string { return "data: {\"choices\":[" + choices + "]}\n\n" }
	const zero = `{"index":0,"delta":{"content":"selected","reasoning_content":"selected reasoning"},"finish_reason":"stop"}`
	const one = `{"index":1,"delta":{"content":"alternative","reasoning_content":"other reasoning"},"finish_reason":"length"}`
	const implicit = `{"delta":{"content":"selected","reasoning_content":"selected reasoning"},"finish_reason":"stop"}`
	const done = "data: [DONE]\n\n"
	for _, tc := range []struct {
		name, body, extra, content, reasoning, finish string
		refuse, providerError, tool                   bool
	}{
		{name: "reversed_frame", body: frame(one+","+zero) + done, extra: `{"n":2}`, content: "selected", reasoning: "selected reasoning", finish: "stop"},
		{name: "separate_alternatives", body: frame(one) + frame(zero) + done, extra: `{"n":2}`, content: "selected", reasoning: "selected reasoning", finish: "stop"},
		{name: "all_finishes_without_done", body: frame(zero) + frame(one), extra: `{"n":2}`, content: "selected", reasoning: "selected reasoning", finish: "stop"},
		{name: "missing_other_finish", body: frame(zero), extra: `{"n":2}`, content: "selected", refuse: true},
		{name: "omitted_singleton", body: frame(implicit) + done, content: "selected", reasoning: "selected reasoning", finish: "stop"},
		{name: "omitted_legacy_multi", body: frame(implicit) + frame(one) + done, extra: `{"n":2}`, content: "selected", reasoning: "selected reasoning", finish: "stop"},
		{name: "missing_zero", body: frame(one) + done, extra: `{"n":2}`, refuse: true},
		{name: "fractional_index_after_selected", body: frame(zero) + frame(`{"index":0.5,"delta":{"content":"bad"}}`) + done, content: "selected", refuse: true},
		{name: "string_index_after_selected", body: frame(zero) + frame(`{"index":"0","delta":{"content":"bad"}}`) + done, content: "selected", refuse: true},
		{name: "negative_index", body: frame(`{"index":-1,"delta":{"content":"bad"}}`) + done, refuse: true},
		{name: "out_of_range", body: frame(one) + done, refuse: true},
		{name: "duplicate_frame", body: frame(zero+","+zero) + done, refuse: true},
		{name: "ambiguous_omitted", body: frame(implicit+","+one) + done, extra: `{"n":2}`, refuse: true},
		{name: "provider_after_invalid", body: frame(one) + "data: {\"error\":{\"message\":\"provider failed\"}}\n\n" + done, refuse: true, providerError: true},
		{name: "provider_after_selected", body: frame(zero) + frame(one) + "data: {\"error\":{\"message\":\"provider failed\"}}\n\n" + done, extra: `{"n":2}`, content: "selected", refuse: true, providerError: true},
		{name: "tool_indices_are_per_choice", body: frame(`{"index":0,"delta":{"tool_calls":[{"index":0,"id":"selected_call","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"SFO\"}"}}]},"finish_reason":"tool_calls"},{"index":1,"delta":{"tool_calls":[{"index":0,"id":"other_call","type":"function","function":{"name":"lookup","arguments":" "}}]},"finish_reason":"stop"}`) + done, extra: `{"n":2}`, finish: "tool_calls", tool: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			up, captured := sseServer(t, tc.body)
			p := NewHTTPPlanner(up.URL, "model", "")
			if tc.extra != "" {
				if err := p.SetExtraBodyJSON(tc.extra); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			var sink strings.Builder
			comp, err := p.CompleteStream(ctx, func(s string) error { sink.WriteString(s); return nil }, []Message{{Role: RoleUser, Content: "hi"}}, adapterTestTools())
			var sent map[string]json.RawMessage
			if decodeErr := json.Unmarshal(*captured, &sent); decodeErr != nil || string(sent["stream"]) != "true" {
				t.Fatalf("request=%s decode=%v", *captured, decodeErr)
			}
			if tc.extra != "" && string(sent["n"]) != "2" {
				t.Fatalf("requested alternatives changed: %s", *captured)
			}
			if sink.String() != tc.content {
				t.Fatalf("sink merged alternatives: got %q want %q", sink.String(), tc.content)
			}
			if tc.refuse {
				if comp != nil || err == nil {
					t.Fatalf("invalid choice stream released completion=%+v error=%v", comp, err)
				}
				if tc.providerError && !errors.Is(err, ErrUpstreamStreamError) {
					t.Fatalf("provider error precedence lost: %v", err)
				}
				return
			}
			if err != nil || comp == nil {
				t.Fatalf("selected completion=%+v error=%v", comp, err)
			}
			if comp.Message.Content != tc.content || comp.Message.ReasoningContent != tc.reasoning || comp.FinishReason != tc.finish {
				t.Fatalf("selected identity changed: %+v", comp)
			}
			if tc.tool {
				if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "selected_call" || comp.Message.ToolCalls[0].Function.Name != "lookup" || comp.Message.ToolCalls[0].Function.Arguments != `{"city":"SFO"}` {
					t.Fatalf("mixed tool alternatives: %+v", comp.Message.ToolCalls)
				}
			} else if len(comp.Message.ToolCalls) != 0 {
				t.Fatalf("unexpected tools: %+v", comp.Message.ToolCalls)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCompleteStreamChoiceRefusalPreservesReadError(t *testing.T) {
	cause := errors.New("choice stream reset")
	body := "data: {\"choices\":[{\"index\":-1,\"delta\":{\"content\":\"bad\"}}]}\n\n"
	reader := &completionErrorReader{body: []byte(body), err: cause}
	p := NewHTTPPlanner("http://upstream.test", "model", "")
	p.Client = &http.Client{Transport: issue10321RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return issue10321Response("text/event-stream", reader), nil
	})}
	comp, err := p.CompleteStream(context.Background(), nil, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if comp != nil || !errors.Is(err, cause) || errors.Is(err, io.EOF) {
		t.Fatalf("read error replaced: completion=%+v error=%v", comp, err)
	}
}
