package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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

type turnkeyPromptArrayPlanner struct {
	turnkeyCompletionPlanner
	streamCalls int
}

func (p *turnkeyPromptArrayPlanner) StreamingSupported() bool { return true }
func (p *turnkeyPromptArrayPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.streamCalls++
	completion, err := p.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	return completion, sink(completion.Message.Content)
}

// Replaces TestTurnkeyCompletionsArrayPromptJoined: independent prompts must not
// be silently combined into one completion, even when streaming is requested.
// fak-test:runtime fast est=10ms lane=default
func TestTurnkeyCompletionsPromptArrayBatchRefused(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, prompt := range []string{`["a","b"]`, `["a","b","c"]`, `["",""]`, `["a",""]`} {
			t.Run(fmt.Sprintf("stream=%t/prompt=%s", stream, prompt), func(t *testing.T) {
				p := &turnkeyPromptArrayPlanner{}
				s := &turnkeyServer{planner: p, plan: macfit.TurnkeyProfile{
					ContextBudgetTokens: 8192, KVBytesPerToken: 1, KVPoolBytes: 8192,
					Tier: macfit.ModelTier{ModelID: "local"},
				}}
				request := func(prompt string) *httptest.ResponseRecorder {
					r := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(fmt.Sprintf(`{"prompt":%s,"stream":%t}`, prompt, stream)))
					w := httptest.NewRecorder()
					s.handleCompletions(w, r)
					return w
				}
				w := request(prompt)
				if w.Code != http.StatusBadRequest || w.Header().Get("Content-Type") != "application/json" {
					t.Fatalf("response = %d %q, want 400 application/json: %s", w.Code, w.Header().Get("Content-Type"), w.Body.String())
				}
				var got struct {
					Error struct{ Code, Message, Type string } `json:"error"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
					t.Fatal(err)
				}
				if got.Error.Code != "prompt_batch_unsupported" || got.Error.Message == "" || got.Error.Type != "invalid_request_error" {
					t.Fatalf("error = %+v", got.Error)
				}
				if strings.Contains(w.Body.String(), "data:") || len(p.messages) != 0 || p.streamCalls != 0 {
					t.Fatal("batch opened a stream or invoked a planner")
				}
				if stats := s.capacityStats(); stats.MaxSessions != 1 || stats.LiveSessions != 0 || stats.AdmittedTotal != 1 {
					t.Fatalf("batch admission was not released: %+v", stats)
				}
				// Both compatible prompt forms still serve after the refused request,
				// preserving bytes and proving the bounded admission slot is reusable.
				for _, valid := range []string{`" a\nb "`, `[" a\nb "]`} {
					w = request(valid)
					if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"text_completion"`) {
						t.Fatalf("valid prompt response = %d: %s", w.Code, w.Body.String())
					}
					if stream && (!strings.HasPrefix(w.Header().Get("Content-Type"), "text/event-stream") || !strings.Contains(w.Body.String(), "data: [DONE]")) {
						t.Fatalf("valid prompt did not complete the stream: %s", w.Body.String())
					}
					messages := p.messages[len(p.messages)-1]
					if len(messages) != 1 || messages[0].Role != agent.RoleUser || messages[0].Content != " a\nb " {
						t.Fatalf("accepted prompt changed: %+v", messages)
					}
				}
				if len(p.messages) != 2 || (stream && p.streamCalls != 2) || (!stream && p.streamCalls != 0) {
					t.Fatalf("planner calls = %d buffered, %d streamed", len(p.messages), p.streamCalls)
				}
			})
		}
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
		name    string
		raw     string
		want    string
		batched bool
	}{
		{"bare", `"x"`, "x", false},
		{"singleton", `[" a\nb "]`, " a\nb ", false},
		{"array", `["a","b"]`, "", true},
		{"empty-array", `[]`, "", false},
		{"empty-singleton", `[""]`, "", false},
		{"empty", ``, "", false},
		{"null", `null`, "", false},
		{"malformed", `[`, "", false},
		{"token-ids", `[1,2]`, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, batched := turnkeyNormalizePrompt(json.RawMessage(tc.raw))
			if got != tc.want || batched != tc.batched {
				t.Errorf("turnkeyNormalizePrompt(%s) = (%q, %t), want (%q, %t)", tc.raw, got, batched, tc.want, tc.batched)
			}
		})
	}
}
