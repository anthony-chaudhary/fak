package main

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/macfit"
)

type turnkeyBarrierStreamPlanner struct {
	release       chan struct{}
	firstSent     chan struct{}
	canceled      chan struct{}
	once          sync.Once
	completion    *agent.Completion
	err           error
	waitForCancel bool
}

type turnkeyUnsupportedStreamPlanner struct {
	completeCalled bool
}

func (p *turnkeyUnsupportedStreamPlanner) Model() string            { return "local" }
func (p *turnkeyUnsupportedStreamPlanner) StreamingSupported() bool { return false }
func (p *turnkeyUnsupportedStreamPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.completeCalled = true
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "buffered"}}, nil
}
func (p *turnkeyUnsupportedStreamPlanner) CompleteStream(context.Context, agent.StreamSink, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	panic("CompleteStream called for unsupported planner")
}

func (p *turnkeyBarrierStreamPlanner) Model() string            { return "local" }
func (p *turnkeyBarrierStreamPlanner) StreamingSupported() bool { return true }
func (p *turnkeyBarrierStreamPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	panic("buffered Complete called for eligible stream")
}
func (p *turnkeyBarrierStreamPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if err := sink("hello"); err != nil {
		return nil, err
	}
	p.once.Do(func() { close(p.firstSent) })
	if p.waitForCancel {
		<-ctx.Done()
		close(p.canceled)
		return nil, ctx.Err()
	}
	<-p.release
	return p.completion, p.err
}

func newTurnkeyStreamTestServer(p agent.Planner) *httptest.Server {
	s := &turnkeyServer{planner: p, plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
	return httptest.NewServer(http.HandlerFunc(s.handleChatCompletions))
}

func postTurnkeyStream(t *testing.T, ctx context.Context, url string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(`{"model":"local","stream":true,"messages":[{"role":"user","content":"go"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestTurnkeyIncrementalStreamFlushesBeforeGenerationCompletes(t *testing.T) {
	p := &turnkeyBarrierStreamPlanner{
		release:   make(chan struct{}),
		firstSent: make(chan struct{}),
		canceled:  make(chan struct{}),
		completion: &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, Content: "hello world"},
			Usage:   agent.Usage{PromptTokens: 2, CompletionTokens: 2, TotalTokens: 4},
		},
	}
	ts := newTurnkeyStreamTestServer(p)
	defer ts.Close()
	released := false
	defer func() {
		if !released {
			close(p.release)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp := postTurnkeyStream(t, ctx, ts.URL)
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	var before bytes.Buffer
	for !strings.Contains(before.String(), `"content":"hello"`) {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("read first content: %v (%q)", err, before.String())
		}
		before.WriteString(line)
	}
	select {
	case <-p.firstSent:
	default:
		t.Fatal("client observed content before planner recorded sink delivery")
	}
	close(p.release)
	released = true
	rest, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	body := before.String() + string(rest)
	if strings.Count(body, `"content":"hello"`) != 1 || !strings.Contains(body, `"content":" world"`) {
		t.Fatalf("content was duplicated or not reconciled: %s", body)
	}
	for _, want := range []string{`"completion_tokens":2`, `"finish_reason":"stop"`, "data: [DONE]"} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream missing %s: %s", want, body)
		}
	}
}

func TestTurnkeyIncrementalStreamFallsBackForUnsupportedPlanner(t *testing.T) {
	p := &turnkeyUnsupportedStreamPlanner{}
	ts := newTurnkeyStreamTestServer(p)
	defer ts.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp := postTurnkeyStream(t, ctx, ts.URL)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !p.completeCalled || !bytes.Contains(body, []byte(`"content":"buffered"`)) || !bytes.Contains(body, []byte("[DONE]")) {
		t.Fatalf("unsupported planner did not retain buffered streaming response: %s", body)
	}
}

func TestTurnkeyIncrementalStreamBuffersToolCalls(t *testing.T) {
	p := &turnkeyBarrierStreamPlanner{
		release:    make(chan struct{}),
		firstSent:  make(chan struct{}),
		canceled:   make(chan struct{}),
		completion: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "hello", ToolCalls: []agent.ToolCall{{ID: "call_1", Type: "function", Function: agent.Func{Name: "read_file", Arguments: `{}`}}}}},
	}
	ts := newTurnkeyStreamTestServer(p)
	defer ts.Close()
	released := false
	defer func() {
		if !released {
			close(p.release)
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp := postTurnkeyStream(t, ctx, ts.URL)
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	line, _ := reader.ReadString('\n')
	for !strings.Contains(line, `"content":"hello"`) {
		next, err := reader.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		line += next
	}
	if strings.Contains(line, "call_1") {
		t.Fatalf("tool call leaked before completion: %s", line)
	}
	close(p.release)
	released = true
	rest, _ := io.ReadAll(reader)
	if got := string(rest); !strings.Contains(got, "call_1") || !strings.Contains(got, `"finish_reason":"tool_calls"`) {
		t.Fatalf("buffered tool call missing: %s", got)
	}
}

func TestTurnkeyIncrementalStreamFailsInBandWithoutDone(t *testing.T) {
	p := &turnkeyBarrierStreamPlanner{
		release: make(chan struct{}), firstSent: make(chan struct{}), canceled: make(chan struct{}),
		completion: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "hello"}, ToolCallsDropped: true},
	}
	close(p.release)
	ts := newTurnkeyStreamTestServer(p)
	defer ts.Close()
	resp := postTurnkeyStream(t, context.Background(), ts.URL)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !bytes.Contains(body, []byte(`"code":"tool_call_conformance"`)) || bytes.Contains(body, []byte("[DONE]")) {
		t.Fatalf("post-start conformance failure shape: %s", body)
	}
}

func TestTurnkeyIncrementalStreamRejectsContradictoryCompletion(t *testing.T) {
	p := &turnkeyBarrierStreamPlanner{
		release: make(chan struct{}), firstSent: make(chan struct{}), canceled: make(chan struct{}),
		completion: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "different"}},
	}
	close(p.release)
	ts := newTurnkeyStreamTestServer(p)
	defer ts.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp := postTurnkeyStream(t, ctx, ts.URL)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !bytes.Contains(body, []byte(`"code":"stream_content_mismatch"`)) || bytes.Contains(body, []byte("[DONE]")) {
		t.Fatalf("contradictory completion failure shape: %s", body)
	}
}

func TestTurnkeyIncrementalStreamCancellationStopsPlanner(t *testing.T) {
	p := &turnkeyBarrierStreamPlanner{firstSent: make(chan struct{}), canceled: make(chan struct{}), waitForCancel: true}
	ts := newTurnkeyStreamTestServer(p)
	defer ts.Close()
	ctx, cancel := context.WithCancel(context.Background())
	resp := postTurnkeyStream(t, ctx, ts.URL)
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(line, `"content":"hello"`) {
			break
		}
	}
	cancel()
	_ = resp.Body.Close()
	select {
	case <-p.canceled:
	case <-time.After(2 * time.Second):
		t.Fatal("request cancellation did not reach planner")
	}
}

func TestTurnkeyStreamRemainder(t *testing.T) {
	for _, tc := range []struct {
		streamed, completed, want string
		wantErr                   bool
	}{
		{"", "whole", "whole", false}, {"hello", "hello world", " world", false}, {"different", "whole", "", true},
	} {
		got, err := turnkeyStreamRemainder(tc.streamed, tc.completed)
		if (err != nil) != tc.wantErr {
			t.Fatalf("remainder(%q, %q) error=%v wantErr %v", tc.streamed, tc.completed, err, tc.wantErr)
		}
		if got != tc.want {
			t.Fatalf("remainder(%q, %q)=%q want %q", tc.streamed, tc.completed, got, tc.want)
		}
	}
}
