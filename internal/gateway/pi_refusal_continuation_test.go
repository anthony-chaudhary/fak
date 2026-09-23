package gateway

import (
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// A Pi child can only continue after an all-denied OpenAI turn if the refusal
// survives in the model-visible content even when the model also wrote prose.
func TestPiRefusalContinuationBufferedOpenAIProse(t *testing.T) {
	srv := newTestServer(t)
	asst := agent.Message{Role: agent.RoleAssistant, Content: "I will update the file now."}
	denied := []ToolAdjudication{{
		Tool: "Write", Verdict: WireVerdict{Kind: "DENY", Reason: "DEFAULT_DENY", Disposition: "TERMINAL"},
	}}
	finish := srv.applyAdjudicatedTurn(&asst, denied, nil, 1, 0, "", false, "tool_calls")
	if finish != "stop" || len(asst.ToolCalls) != 0 {
		t.Fatalf("all-denied turn = finish %q, tool calls %+v; want stop with no calls", finish, asst.ToolCalls)
	}
	for _, want := range []string{
		"[fak] Allowed next step for 1 refused tool call(s):",
		"Constraint: Write",
		"I will update the file now.",
	} {
		if !strings.Contains(asst.Content, want) {
			t.Errorf("buffered response omitted %q: %q", want, asst.Content)
		}
	}
}

// The streamed OpenAI route must preserve model prose already sent on the SSE
// wire, then add the refusal note after the held tool call is denied.
func TestPiRefusalContinuationStreamedOpenAIProse(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"I will update the file now."}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"denied-1","type":"function","function":{"name":"deny_write","arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`,
		"data: [DONE]", "",
	}, "\n\n")
	hits := 0
	up := sseUpstream(t, &hits, sse)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	code, raw := postStream(t, ts.URL, map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "update the file"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "deny_write"}}},
		"stream":   true,
	})
	if code != http.StatusOK || hits != 1 {
		t.Fatalf("status=%d upstream hits=%d; want 200 and one hit: %s", code, hits, raw)
	}
	chunks, done := parseSSEChunks(t, raw)
	if !done {
		t.Fatalf("stream omitted [DONE]: %s", raw)
	}
	var content strings.Builder
	var tools []ChatDeltaToolCall
	var finish string
	for _, chunk := range chunks {
		if len(chunk.Choices) == 0 {
			continue
		}
		content.WriteString(chunk.Choices[0].Delta.Content)
		tools = append(tools, chunk.Choices[0].Delta.ToolCalls...)
		if chunk.Choices[0].FinishReason != nil {
			finish = *chunk.Choices[0].FinishReason
		}
	}
	if len(tools) != 0 || finish != "stop" {
		t.Fatalf("streamed all-denied turn kept %d tool calls, finish=%q; want zero and stop", len(tools), finish)
	}
	for _, want := range []string{"I will update the file now.", "[fak] Allowed next step for 1 refused tool call(s):", "Constraint: deny_write"} {
		if !strings.Contains(content.String(), want) {
			t.Errorf("stream omitted %q: %q", want, content.String())
		}
	}
}
