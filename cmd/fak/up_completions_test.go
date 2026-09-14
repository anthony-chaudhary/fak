package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
)

// turnkeyCompletionPlanner is a buffered-only planner used to prove the legacy
// /v1/completions wire wraps the prompt as one user message and returns a
// text_completion object.
type turnkeyCompletionPlanner struct {
	messages [][]agent.Message
}

func (p *turnkeyCompletionPlanner) Model() string      { return "local" }
func (p *turnkeyCompletionPlanner) ContextWindow() int { return 8192 }

func (p *turnkeyCompletionPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.messages = append(p.messages, append([]agent.Message(nil), messages...))
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "legacy wire says hello"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 4, CompletionTokens: 4, TotalTokens: 8},
	}, nil
}

func newTurnkeyCompletionsTestServer(p agent.Planner) *httptest.Server {
	s := &turnkeyServer{planner: p, plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
	return httptest.NewServer(http.HandlerFunc(s.handleCompletions))
}

func postTurnkeyCompletion(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestTurnkeyCompletionsBufferedTextCompletionShape(t *testing.T) {
	p := &turnkeyCompletionPlanner{}
	ts := newTurnkeyCompletionsTestServer(p)
	defer ts.Close()

	resp := postTurnkeyCompletion(t, ts.URL, `{"model":"local","prompt":"hello","max_tokens":8}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var got gateway.CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Object != "text_completion" {
		t.Errorf("object = %q, want text_completion", got.Object)
	}
	if len(got.Choices) != 1 || got.Choices[0].Text != "legacy wire says hello" {
		t.Fatalf("choices = %+v, want one text choice", got.Choices)
	}
	if len(p.messages) != 1 || len(p.messages[0]) != 1 {
		t.Fatalf("planner saw %d message batches, want one single-message batch", len(p.messages))
	}
	if m := p.messages[0][0]; m.Role != agent.RoleUser || m.Content != "hello" {
		t.Errorf("wrapped message = %+v, want user/hello", m)
	}
}

func TestTurnkeyCompletionsArrayPromptJoined(t *testing.T) {
	p := &turnkeyCompletionPlanner{}
	ts := newTurnkeyCompletionsTestServer(p)
	defer ts.Close()

	resp := postTurnkeyCompletion(t, ts.URL, `{"model":"local","prompt":["a","b"],"max_tokens":8}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if m := p.messages[0][0]; m.Content != "a\nb" {
		t.Errorf("array prompt joined = %q, want %q", m.Content, "a\nb")
	}
}

func TestTurnkeyCompletionsRejectsEmptyPrompt(t *testing.T) {
	p := &turnkeyCompletionPlanner{}
	ts := newTurnkeyCompletionsTestServer(p)
	defer ts.Close()

	resp := postTurnkeyCompletion(t, ts.URL, `{"model":"local","prompt":"   ","max_tokens":8}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	if len(p.messages) != 0 {
		t.Fatalf("planner invoked for empty prompt")
	}
}

func TestTurnkeyCompletionsStreamingTextChunks(t *testing.T) {
	p := &turnkeyBarrierStreamPlanner{
		release:   make(chan struct{}),
		firstSent: make(chan struct{}),
		canceled:  make(chan struct{}),
		completion: &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, Content: "hello world"},
			Usage:   agent.Usage{PromptTokens: 2, CompletionTokens: 2, TotalTokens: 4},
		},
	}
	close(p.release)
	ts := newTurnkeyCompletionsTestServer(p)
	defer ts.Close()

	resp := postTurnkeyCompletion(t, ts.URL, `{"model":"local","prompt":"go","stream":true,"max_tokens":8}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var sawText, sawDone bool
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if line == "data: [DONE]" {
			sawDone = true
			break
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("bad stream chunk %q: %v", line, err)
		}
		if chunk.Object != "text_completion" {
			t.Errorf("chunk object = %q, want text_completion", chunk.Object)
		}
		if len(chunk.Choices) > 0 && strings.Contains(chunk.Choices[0].Text, "hello") {
			sawText = true
		}
	}
	if !sawText {
		t.Error("no streamed text fragment observed")
	}
	if !sawDone {
		t.Error("stream did not terminate with data: [DONE]")
	}
}

func TestTurnkeyNormalizePrompt(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"bare", `"x"`, "x"},
		{"array", `["a","b"]`, "a\nb"},
		{"empty", ``, ""},
		{"null", `null`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := turnkeyNormalizePrompt(json.RawMessage(tc.raw))
			if got != tc.want {
				t.Errorf("turnkeyNormalizePrompt(%s) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}
