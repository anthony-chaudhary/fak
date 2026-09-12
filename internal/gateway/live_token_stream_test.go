package gateway

// live_token_stream_test.go — #920: the LIVE token-stream dedupe witness. A planner
// that forwards one sink delta per token must reach the client as one content chunk per
// token, and the reassembled chunks must equal the planner's returned Completion content.
// The gateway reconciles the live bytes against the buffered post-lift content
// (stream_proxy.go liftRemainder), so a token-sink-served turn is NOT replayed whole.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// liveTokenPlanner is a stub StreamingPlanner whose CompleteStream invokes sink once per
// token of a known multi-token string, then returns a Completion whose Content is the
// concatenation of those tokens — the exact shape a per-token in-kernel turn produces
// with FAK_STREAM_INKERNEL_PER_TOKEN on.
type liveTokenPlanner struct {
	model  string
	tokens []string
}

func (p *liveTokenPlanner) Model() string { return p.model }

func (p *liveTokenPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, agent.ErrStreamingUnsupported
}

func (p *liveTokenPlanner) StreamingSupported() bool { return true }

func (p *liveTokenPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	var full strings.Builder
	for _, tok := range p.tokens {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		if err := sink(tok); err != nil {
			return nil, err
		}
		full.WriteString(tok)
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: full.String()},
		FinishReason: "stop",
		Model:        p.model,
	}, nil
}

var _ agent.StreamingPlanner = (*liveTokenPlanner)(nil)

// TestLiveTokenStreamOneChunkPerToken drives the live SSE path with a per-token planner
// and asserts (i) one content chunk per token and (ii) reassembled content equals the
// completion content — the no-whole-turn-replay dedupe contract.
func TestLiveTokenStreamOneChunkPerToken(t *testing.T) {
	tokens := []string{"alpha", " beta", " gamma", " delta"}
	planner := &liveTokenPlanner{model: "test-model", tokens: tokens}

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"probe"}],"stream":true}`)
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)

	resp := tap.waitHead(t, hbBudget, "no HTTP head on the live token stream")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	rest := tap.drain(t, 15*time.Second)
	var content strings.Builder
	contentChunks := 0
	sawDone := false
	for _, line := range rest {
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		if strings.HasPrefix(line, ": fak-heartbeat") {
			continue
		}
		chunk := decodeSSEChunk(t, line)
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content
		if delta != "" {
			contentChunks++
			content.WriteString(delta)
		}
	}

	want := strings.Join(tokens, "")
	if got := content.String(); got != want {
		t.Fatalf("reassembled content = %q, want %q", got, want)
	}
	if contentChunks != len(tokens) {
		t.Fatalf("content chunks = %d, want one per token (%d); the turn was replayed/collapsed", contentChunks, len(tokens))
	}
	if !sawDone {
		t.Fatal("stream never terminated with [DONE]")
	}
}
