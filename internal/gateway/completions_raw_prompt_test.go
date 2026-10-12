package gateway

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type rawPromptPlanner struct{ raw *[]bool }

func (r rawPromptPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	var sp agent.SampleParams
	for _, o := range opts {
		o(&sp)
	}
	*r.raw = append(*r.raw, sp.RawPrompt)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (rawPromptPlanner) Model() string { return "raw" }

// The legacy completions wire marks its prompt raw so a native V4.1 planner can
// skip the chat template; the chat wire never does.
// fak-test:runtime fast est=50ms
func TestCompletionsMarkPromptRawChatDoesNot(t *testing.T) {
	srv := newTestServer(t)
	var seen []bool
	srv.planner = rawPromptPlanner{raw: &seen}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	var comp CompletionResponse
	if code := postJSON(t, ts.URL+"/v1/completions", CompletionRequest{Model: "m", Prompt: []byte(`"The chemical symbol for gold is"`)}, &comp); code != 200 {
		t.Fatalf("completions status = %d", code)
	}
	var chat map[string]any
	if code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{Model: "m", Messages: []agent.Message{{Role: agent.RoleUser, Content: "hi"}}}, &chat); code != 200 {
		t.Fatalf("chat status = %d", code)
	}
	if len(seen) != 2 || !seen[0] || seen[1] {
		t.Fatalf("RawPrompt per request = %v, want [true false] (completions, chat)", seen)
	}
}
