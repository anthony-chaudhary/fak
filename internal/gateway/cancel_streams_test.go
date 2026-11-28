package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type failingSSEWriter struct {
	hdr        http.Header
	failAfter  int
	writes     int
	statusCode int
}

func (w *failingSSEWriter) Header() http.Header {
	if w.hdr == nil {
		w.hdr = make(http.Header)
	}
	return w.hdr
}

func (w *failingSSEWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes > w.failAfter {
		return 0, errors.New("broken pipe")
	}
	return len(p), nil
}

func (w *failingSSEWriter) WriteHeader(statusCode int) {
	w.statusCode = statusCode
}

func (w *failingSSEWriter) Flush() {}

type cancelCheckStreamingPlanner struct {
	canceled atomic.Bool
	started  chan struct{}
}

func (p *cancelCheckStreamingPlanner) Model() string { return "gpt-test" }

func (p *cancelCheckStreamingPlanner) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p *cancelCheckStreamingPlanner) StreamingSupported() bool {
	return true
}

func (p *cancelCheckStreamingPlanner) CompleteStream(ctx context.Context, emit agent.StreamSink, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	_ = emit("hello ")
	_ = emit("world")
	select {
	case <-p.started:
	default:
		close(p.started)
	}
	<-ctx.Done()
	p.canceled.Store(true)
	return nil, ctx.Err()
}

func TestAnthropicPassthroughPreStartCancelDoesNotFallbackOrCountError(t *testing.T) {
	srv := newTestServer(t)
	releaseUpstream := make(chan struct{})
	var upstreamCalls atomic.Int32

	ctx, cancel := context.WithCancel(context.Background())
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		cancel()
		select {
		case <-r.Context().Done():
		case <-releaseUpstream:
		}
	}))
	defer func() {
		close(releaseUpstream)
		upstream.CloseClientConnections()
		upstream.Close()
	}()

	hp := agent.NewHTTPPlanner(upstream.URL, "k", "claude-sonnet-4-5")
	hp.Provider = "anthropic"
	srv.planner = hp

	reqBody := []byte(`{"model":"claude-sonnet-4-5","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(string(reqBody))).WithContext(ctx)
	w := httptest.NewRecorder()
	req := &agent.AnthropicMessagesRequest{
		Model:  "claude-sonnet-4-5",
		Stream: true,
		Raw:    reqBody,
	}

	handled := srv.streamAnthropicPassthroughLive(w, r, req, "trace-prestart-cancel", servedSessionTurn{}, "", "", false, false, harnessCoherenceInputs{})
	if !handled {
		t.Fatal("streamAnthropicPassthroughLive returned false on client cancel, want true (must not fall back to buffered)")
	}
	if w.Body.Len() != 0 {
		t.Fatalf("expected empty body on pre-start cancel, got %q", w.Body.String())
	}
	if got := upstreamCalls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}
}

func TestAnthropicPassthroughWriteErrorCancelsUpstream(t *testing.T) {
	srv := newTestServer(t)
	upstreamCanceled := make(chan struct{})
	releaseUpstream := make(chan struct{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"))
		if fl != nil {
			fl.Flush()
		}
		select {
		case <-r.Context().Done():
			close(upstreamCanceled)
		case <-releaseUpstream:
		}
	}))
	defer func() {
		close(releaseUpstream)
		upstream.CloseClientConnections()
		upstream.Close()
	}()

	hp := agent.NewHTTPPlanner(upstream.URL, "k", "claude-sonnet-4-5")
	hp.Provider = "anthropic"
	srv.planner = hp
	reqBody := []byte(`{"model":"claude-sonnet-4-5","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(string(reqBody)))
	w := &failingSSEWriter{failAfter: 1}
	req := &agent.AnthropicMessagesRequest{
		Model:  "claude-sonnet-4-5",
		Stream: true,
		Raw:    reqBody,
	}

	handled := srv.streamAnthropicPassthroughLive(w, r, req, "trace-write-err", servedSessionTurn{}, "", "", false, false, harnessCoherenceInputs{})
	if !handled {
		t.Fatal("streamAnthropicPassthroughLive returned false on mid-stream write error, want true")
	}
	select {
	case <-upstreamCanceled:
	case <-time.After(2 * time.Second):
		t.Fatal("downstream write error did not cancel upstream StreamAnthropicRaw")
	}
}

func TestResponsesLiveWriteErrorCancelsTurn(t *testing.T) {
	srv := newTestServer(t)
	sp := &cancelCheckStreamingPlanner{started: make(chan struct{})}
	srv.planner = sp

	w := &failingSSEWriter{failAfter: 1}
	handled := srv.streamResponsesLive(context.Background(), w, "gpt-test", "trace-resp-write-err", servedSessionTurn{}, nil, responsesLiveTurn{
		messages: []agent.Message{{Role: agent.RoleUser, Content: "hi"}},
	}, newResponsesRestoreContinuation(nil), func(responsesResponse, agent.Message) {})
	if !handled {
		t.Fatal("streamResponsesLive returned false, want true")
	}
	if !sp.canceled.Load() {
		t.Fatal("downstream write error did not cancel planner CompleteStream ctx")
	}
}
