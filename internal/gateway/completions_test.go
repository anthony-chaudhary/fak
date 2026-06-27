package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestCompletionsBasic checks the legacy /v1/completions route returns the
// text_completion object shape (choices carry a bare `text`, not a chat message)
// over the same served path the chat route uses.
func TestCompletionsBasic(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "Paris."},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 5, CompletionTokens: 1, TotalTokens: 6},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp CompletionResponse
	code := postJSON(t, ts.URL+"/v1/completions", CompletionRequest{
		Model:  "test-model",
		Prompt: json.RawMessage(`"The capital of France is"`),
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d", code)
	}
	if resp.Object != "text_completion" {
		t.Errorf("object = %q, want text_completion", resp.Object)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Text != "Paris." {
		t.Fatalf("choices = %+v, want one choice with text=Paris.", resp.Choices)
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.TotalTokens != 6 {
		t.Errorf("usage.total_tokens = %d, want 6", resp.Usage.TotalTokens)
	}
}

// TestCompletionsEmptyPromptIsBadRequest checks the well-formedness floor: a missing
// or empty prompt is a client 400, not a forwarded degenerate request.
func TestCompletionsEmptyPromptIsBadRequest(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, body := range []string{`{"model":"m"}`, `{"model":"m","prompt":""}`, `{"model":"m","prompt":"   "}`} {
		r, err := http.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, r.StatusCode)
		}
	}
}

// capturePlanner records the last user message it was asked to complete, so a test
// can assert how the handler shaped the prompt before the planner saw it.
type capturePlanner struct {
	comp *agent.Completion
	last *string
}

func (c capturePlanner) Complete(_ context.Context, m []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if c.last != nil && len(m) > 0 {
		*c.last = m[len(m)-1].Content
	}
	return c.comp, nil
}

func (capturePlanner) Model() string { return "capture" }

// TestCompletionsArrayPromptJoined checks the OpenAI array-prompt form is folded into
// one prompt (the planner sees a single user message with the lines newline-joined).
func TestCompletionsArrayPromptJoined(t *testing.T) {
	srv := newTestServer(t)
	var seen string
	srv.planner = capturePlanner{
		comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"},
		last: &seen,
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp CompletionResponse
	code := postJSON(t, ts.URL+"/v1/completions", CompletionRequest{
		Model:  "test-model",
		Prompt: json.RawMessage(`["line one","line two"]`),
	}, &resp)
	if code != 200 {
		t.Fatalf("status = %d", code)
	}
	if seen != "line one\nline two" {
		t.Errorf("planner saw prompt %q, want the two lines joined with newline", seen)
	}
}

// TestCompletionsStreamEmitsTextSSE checks stream=true yields an SSE stream whose
// chunks carry a `text` fragment (the legacy shape) and a [DONE] terminator, and
// that concatenating the text reproduces the completion.
func TestCompletionsStreamEmitsTextSSE(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "one two three"},
		FinishReason: "stop",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(CompletionRequest{
		Model:  "test-model",
		Prompt: json.RawMessage(`"go"`),
		Stream: true,
	})
	r, err := http.Post(ts.URL+"/v1/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	var text strings.Builder
	sawDone := false
	sc := bufio.NewScanner(r.Body)
	for sc.Scan() {
		line := sc.Text()
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			sawDone = true
			continue
		}
		var chunk CompletionStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", data, err)
		}
		if chunk.Object != "text_completion" {
			t.Errorf("chunk object = %q, want text_completion", chunk.Object)
		}
		if len(chunk.Choices) > 0 {
			text.WriteString(chunk.Choices[0].Text)
		}
	}
	if !sawDone {
		t.Error("stream did not end with data: [DONE]")
	}
	if got := text.String(); got != "one two three" {
		t.Errorf("reassembled text = %q, want %q", got, "one two three")
	}
}
