package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type chatReasoningPlanner struct {
	completion agent.Completion
	err        error
}

func (p *chatReasoningPlanner) Model() string { return "test-model" }

func (p *chatReasoningPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	comp := p.completion
	return &comp, p.err
}

type chatReasoningStreamingPlanner struct{ *chatReasoningPlanner }

func (*chatReasoningStreamingPlanner) StreamingSupported() bool { return true }

func (p *chatReasoningStreamingPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	for _, text := range []string{"an", "swer"} {
		if p.completion.Message.Content == "" {
			break
		}
		if err := sink(text); err != nil {
			return nil, err
		}
	}
	comp := p.completion
	return &comp, p.err
}

var _ agent.StreamingPlanner = (*chatReasoningStreamingPlanner)(nil)

type chatReasoningWire struct {
	Choices []struct {
		Message      agent.Message `json:"message"`
		FinishReason *string       `json:"finish_reason"`
		Delta        struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning_content"`
			ToolCalls []struct {
				Function agent.Func `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

func chatReasoningRequest(t *testing.T, srv *Server, stream, tools bool) *httptest.ResponseRecorder {
	t.Helper()
	req := ChatRequest{Model: "test-model", Messages: []agent.Message{{Role: agent.RoleUser, Content: "fixture"}}, Stream: stream}
	if tools {
		req.Tools = []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "allow_reasoning", Parameters: json.RawMessage(`{"type":"object"}`)}}}
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, r)
	return w
}

func chatReasoningFrames(t *testing.T, body string) (string, string, int, int, int) {
	t.Helper()
	var reasoning, content strings.Builder
	calls, finishes, done := 0, 0, 0
	for _, line := range strings.Split(body, "\n") {
		if line == "data: [DONE]" {
			done++
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var frame chatReasoningWire
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
			t.Fatal(err)
		}
		for _, choice := range frame.Choices {
			reasoning.WriteString(choice.Delta.Reasoning)
			content.WriteString(choice.Delta.Content)
			if choice.FinishReason != nil {
				finishes++
			}
			for _, call := range choice.Delta.ToolCalls {
				if call.Function.Name == "allow_reasoning" {
					calls++
				}
			}
		}
	}
	return reasoning.String(), content.String(), calls, finishes, done
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestChatReasoningStreamMatchesBuffered(t *testing.T) {
	for _, live := range []bool{false, true} {
		for _, name := range []string{"reasoning-only", "unicode-whitespace", "long-unicode", "allowed-tool"} {
			path := "buffered/"
			if live {
				path = "live/"
			}
			t.Run(path+name, func(t *testing.T) {
				srv := newTestServer(t)
				comp := agent.Completion{Model: "test-model", Message: agent.Message{Role: agent.RoleAssistant, ReasoningContent: " \t réfléchir 世界\n\u2003 ", Content: "answer"}, FinishReason: "stop"}
				if name == "long-unicode" {
					comp.Message.ReasoningContent = " \t" + strings.Repeat("réfléchir 世界 🧠\u2003\n", 128) + "\n "
				}
				if name == "reasoning-only" {
					comp.Message.Content = ""
				}
				if name == "allowed-tool" {
					comp.Message.ToolCalls = []agent.ToolCall{{ID: "call_reasoning", Type: "function", Function: agent.Func{Name: "allow_reasoning", Arguments: `{}`}}}
					comp.FinishReason = "tool_calls"
				}
				planner := &chatReasoningPlanner{completion: comp}
				srv.planner = planner
				if live {
					srv.planner = &chatReasoningStreamingPlanner{planner}
				}
				whole := chatReasoningRequest(t, srv, false, name == "allowed-tool")
				if whole.Code != http.StatusOK {
					t.Fatalf("nonstream status=%d body=%s", whole.Code, whole.Body.String())
				}
				var expected chatReasoningWire
				if err := json.Unmarshal(whole.Body.Bytes(), &expected); err != nil || len(expected.Choices) != 1 {
					t.Fatalf("nonstream response: err=%v body=%s", err, whole.Body.String())
				}
				if expected.Choices[0].Message.ReasoningContent != comp.Message.ReasoningContent {
					t.Fatal("nonstream fixture lost its reasoning before stream comparison")
				}
				streamed := chatReasoningRequest(t, srv, true, name == "allowed-tool")
				if streamed.Code != http.StatusOK || !strings.Contains(streamed.Body.String(), "data: [DONE]") {
					t.Fatalf("stream did not complete: status=%d body=%s", streamed.Code, streamed.Body.String())
				}
				got, content, calls, finishes, done := chatReasoningFrames(t, streamed.Body.String())
				if got != expected.Choices[0].Message.ReasoningContent {
					t.Errorf("stream reasoning_content=%q, want nonstream reasoning_content=%q", got, expected.Choices[0].Message.ReasoningContent)
				}
				if name == "allowed-tool" && (len(expected.Choices[0].Message.ToolCalls) != 1 || calls != 1) {
					t.Errorf("allowed tool adjudication changed: nonstream=%d stream=%d", len(expected.Choices[0].Message.ToolCalls), calls)
				}
				if content != expected.Choices[0].Message.Content {
					t.Errorf("stream content=%q, want nonstream content=%q", content, expected.Choices[0].Message.Content)
				}
				if finishes != 1 || done != 1 {
					t.Errorf("stream terminal sequence finishes=%d done=%d, want exactly one each", finishes, done)
				}
			})
		}
	}
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=100ms lane=default
func TestChatReasoningStreamFailureDoesNotPublishFinalReasoning(t *testing.T) {
	for _, cause := range []error{errors.New("fixture generation failure"), context.Canceled} {
		t.Run(cause.Error(), func(t *testing.T) {
			srv := newTestServer(t)
			srv.planner = &chatReasoningStreamingPlanner{&chatReasoningPlanner{completion: agent.Completion{Model: "test-model", Message: agent.Message{Role: agent.RoleAssistant, Content: "answer", ReasoningContent: "uncommitted reasoning"}, FinishReason: "stop"}, err: cause}}
			w := chatReasoningRequest(t, srv, true, false)
			reasoning, _, _, _, _ := chatReasoningFrames(t, w.Body.String())
			if reasoning != "" {
				t.Fatalf("failed turn published finalized reasoning=%q", reasoning)
			}
			if !strings.Contains(w.Body.String(), `"error"`) {
				t.Fatalf("fixture failed without an error envelope: status=%d body=%s", w.Code, w.Body.String())
			}
		})
	}
}
