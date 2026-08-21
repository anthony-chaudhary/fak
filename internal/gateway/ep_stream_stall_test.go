package gateway

// ep_stream_stall_test.go — the CPU-modeled, GPU-free repro for #5399 (the streaming
// half of #4855 that survived its closure).
//
// It models the persistent-rank EP topology in-process, with no weights and no device:
// a fake FOLLOWER rank receives the fanout mirror, and a planner implementing ONLY
// agent.Complete (exactly like agent.InKernelPlanner, which is why streamChatLive
// declines the in-kernel serve) blocks inside a modeled per-step collective, standing
// in for the sharded decode. The question the test asks is the one the original
// 8-rank EP report asked: does ANY byte — status line, headers, first SSE chunk — reach the client
// while that collective is still pending?
//
// Every wait in here is BOUNDED, including the modeled collective itself, so the test
// can fail loudly but can never hang a suite.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// epStallBudget is how long the test is willing to wait for the preamble. It is far
// longer than an in-process flush needs and far shorter than a real sharded decode, so
// it separates "the gateway opened the stream immediately" from "the gateway waited
// for the whole turn" without being flaky.
const epStallBudget = 3 * time.Second

// collectivePlanner is the CPU model of a rank-0 in-kernel decode: it implements the
// bare agent.Planner surface — Complete + Model, nothing else — so, exactly like
// agent.InKernelPlanner, it is NOT an agent.StreamingPlanner and streamChatLive must
// decline it. Complete parks on a modeled per-step collective until the test releases
// it, which is what a real front rank does while it waits on the process-group
// AllReduce with the follower ranks.
type collectivePlanner struct {
	model   string
	content string
	// failWith, when non-nil, is the error the modeled decode returns INSTEAD of a
	// turn — a post-preamble upstream failure.
	failWith error

	joinOnce sync.Once
	joined   chan struct{} // closed the moment Complete enters the modeled collective
	release  chan struct{} // closed by the test to let the modeled collective finish
}

func newCollectivePlanner(model, content string, failWith error) *collectivePlanner {
	return &collectivePlanner{
		model:    model,
		content:  content,
		failWith: failWith,
		joined:   make(chan struct{}),
		release:  make(chan struct{}),
	}
}

func (p *collectivePlanner) Model() string { return p.model }

func (p *collectivePlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.joinOnce.Do(func() { close(p.joined) })
	select {
	case <-p.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(30 * time.Second):
		// Hard bound on the model itself: a wedged test fails, it never hangs.
		return nil, errors.New("modeled EP collective was never released")
	}
	if p.failWith != nil {
		return nil, p.failWith
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: p.content},
		FinishReason: "stop",
		Model:        p.model,
		Usage:        agent.Usage{PromptTokens: 6, CompletionTokens: 4, TotalTokens: 10},
	}, nil
}

// collectivePlanner must model the real thing: a planner the gateway can only serve
// through its BUFFERED path.
var _ agent.Planner = (*collectivePlanner)(nil)

// sseTap runs one streaming chat request and hands the response head and each SSE line
// back over channels, so the test can assert on WHEN each arrived instead of only on
// the final body. Nothing here blocks the test goroutine.
type sseTap struct {
	head   chan sseHead
	frames chan string
}

type sseHead struct {
	resp *http.Response
	err  error
}

func tapChatStream(url string, body []byte) *sseTap {
	tap := &sseTap{head: make(chan sseHead, 1), frames: make(chan string, 64)}
	go func() {
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			tap.head <- sseHead{err: err}
			close(tap.frames)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		tap.head <- sseHead{resp: resp, err: err}
		if err != nil {
			close(tap.frames)
			return
		}
		defer resp.Body.Close()
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 8*1024), 1024*1024)
		for sc.Scan() {
			if line := strings.TrimSpace(sc.Text()); line != "" {
				tap.frames <- line
			}
		}
		close(tap.frames)
	}()
	return tap
}

// waitHead returns the response head within the budget, or fails the test.
func (tap *sseTap) waitHead(t *testing.T, budget time.Duration, stallMsg string) *http.Response {
	t.Helper()
	select {
	case h := <-tap.head:
		if h.err != nil {
			t.Fatalf("streaming request failed: %v", h.err)
		}
		return h.resp
	case <-time.After(budget):
		t.Fatal(stallMsg)
		return nil
	}
}

// waitFrame returns the next SSE line within the budget, or fails the test.
func (tap *sseTap) waitFrame(t *testing.T, budget time.Duration, stallMsg string) string {
	t.Helper()
	select {
	case line, ok := <-tap.frames:
		if !ok {
			t.Fatal("stream closed before the expected SSE frame arrived")
		}
		return line
	case <-time.After(budget):
		t.Fatal(stallMsg)
		return ""
	}
}

// drain collects every remaining SSE line until the stream closes, bounded.
func (tap *sseTap) drain(t *testing.T, budget time.Duration) []string {
	t.Helper()
	var out []string
	deadline := time.After(budget)
	for {
		select {
		case line, ok := <-tap.frames:
			if !ok {
				return out
			}
			out = append(out, line)
		case <-deadline:
			t.Fatalf("stream never terminated; collected so far: %v", out)
			return out
		}
	}
}

func decodeSSEChunk(t *testing.T, line string) ChatStreamResponse {
	t.Helper()
	data, ok := strings.CutPrefix(line, "data: ")
	if !ok {
		t.Fatalf("non-SSE line on the wire: %q", line)
	}
	var chunk ChatStreamResponse
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		t.Fatalf("decode SSE chunk %q: %v", data, err)
	}
	return chunk
}

// startFollowerRank stands up a fake EP follower rank and points FAK_EP_FANOUT_ADDRS at
// it, returning the channel on which the mirrored request body arrives.
func startFollowerRank(t *testing.T) chan string {
	t.Helper()
	mirrored := make(chan string, 1)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get(epFollowerHeader); got != "1" {
			t.Errorf("follower rank %s = %q, want 1", epFollowerHeader, got)
		}
		raw, _ := io.ReadAll(r.Body)
		select {
		case mirrored <- string(raw):
		default:
		}
		// A follower answers its own stream and terminates, so the front rank's
		// bounded fanout wait returns.
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"rank\"}}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(follower.Close)
	t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
	return mirrored
}

// TestEPStreamingPlannerContracts pins both paths exercised below: the real in-kernel
// planner supports live streaming, while collectivePlanner deliberately models the
// buffered fallback whose preamble must still open before its collective completes.
func TestEPStreamingPlannerContracts(t *testing.T) {
	var p agent.Planner = (*agent.InKernelPlanner)(nil)
	if _, ok := p.(agent.StreamingPlanner); !ok {
		t.Fatal("agent.InKernelPlanner must implement agent.StreamingPlanner")
	}
	var modeled agent.Planner = newCollectivePlanner("m", "", nil)
	if _, ok := modeled.(agent.StreamingPlanner); ok {
		t.Fatal("collectivePlanner must model a Complete-ONLY planner, not a streaming one")
	}
}

// TestEPStreamingEmitsFirstSSEChunkBeforeCollectiveJoin is the #5399 witness. The
// client asks for stream:true; the front rank mirrors the request to the follower rank
// and then parks in the modeled collective. Before that collective is released the
// client must ALREADY hold the status line, the SSE headers, and the opening role
// chunk — otherwise a multi-rank decode is indistinguishable from a dead socket for its
// entire duration (1120+ seconds on the original 8-rank GLM-5.2 EP report).
//
// Against the pre-fix gateway nothing at all reaches the client until completeServed
// returns, so the preamble waits fail here.
func TestEPStreamingEmitsFirstSSEChunkBeforeCollectiveJoin(t *testing.T) {
	mirrored := startFollowerRank(t)

	planner := newCollectivePlanner("test-model", "rank0 decoded through the collective", nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// Registered AFTER ts.Close so LIFO cleanup frees the parked decode FIRST: a failing
	// run then tears down immediately instead of waiting out the modeled collective's
	// own hard bound while httptest.Server.Close blocks on the live connection.
	t.Cleanup(release)

	body, err := json.Marshal(map[string]any{
		"model":      "test-model",
		"messages":   []map[string]string{{"role": "user", "content": "decode across ranks"}},
		"stream":     true,
		"max_tokens": 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)

	resp := tap.waitHead(t, epStallBudget,
		"no HTTP status line/headers before the EP collective join — the #4855 streaming stall")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	opening := decodeSSEChunk(t, tap.waitFrame(t, epStallBudget,
		"no first SSE byte before the EP collective join — the #4855 streaming stall"))
	if opening.Object != "chat.completion.chunk" {
		t.Fatalf("opening object = %q, want chat.completion.chunk", opening.Object)
	}
	if got := opening.Choices[0].Delta.Role; got != agent.RoleAssistant {
		t.Fatalf("opening delta role = %q, want assistant", got)
	}
	if opening.Model != "test-model" {
		t.Fatalf("opening model = %q, want the request model", opening.Model)
	}

	// The preamble is not a race against a no-op planner: the decode really is parked
	// inside the modeled collective at this point, and only this test can free it.
	select {
	case <-planner.joined:
	case <-time.After(epStallBudget):
		t.Fatal("the modeled decode never entered the collective — the repro is not exercising the buffered path")
	}

	// The early flush AUGMENTS the fanout; it must never replace it. The follower rank
	// still has to receive the identical stream:true body.
	select {
	case got := <-mirrored:
		if got != string(body) {
			t.Fatalf("follower rank body = %q, want the mirrored request %q", got, body)
		}
	case <-time.After(epStallBudget):
		t.Fatal("follower rank never received the mirrored stream:true body")
	}

	// Release the collective: the turn now exists and the rest of the stream must land.
	release()
	rest := tap.drain(t, 15*time.Second)
	var content strings.Builder
	var finish string
	var sawDone bool
	var sawUsage bool
	for _, line := range rest {
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		chunk := decodeSSEChunk(t, line)
		if chunk.Model != "test-model" {
			t.Fatalf("chunk model = %q, want a constant %q across the stream", chunk.Model, "test-model")
		}
		content.WriteString(chunk.Choices[0].Delta.Content)
		if chunk.Choices[0].FinishReason != nil {
			finish = *chunk.Choices[0].FinishReason
		}
		if chunk.Usage != nil && chunk.Usage.TotalTokens == 10 {
			sawUsage = true
		}
	}
	if got := content.String(); got != planner.content {
		t.Fatalf("reassembled streamed content = %q, want %q", got, planner.content)
	}
	if finish != "stop" {
		t.Fatalf("finish_reason = %q, want stop", finish)
	}
	if !sawUsage {
		t.Fatalf("terminal chunk carried no usage: %v", rest)
	}
	if !sawDone {
		t.Fatalf("stream never terminated with [DONE]: %v", rest)
	}
}

// TestEPStreamingPostPreambleUpstreamErrorIsSSEErrorEvent covers the cost of opening
// the stream early: once the 200 and the SSE headers are on the wire the gateway can no
// longer answer with a real HTTP status, so an upstream failure that lands AFTER the
// preamble has to arrive as an OpenAI-shaped SSE error event followed by [DONE] — never
// as a silently truncated stream, and never leaking the upstream's raw body.
func TestEPStreamingPostPreambleUpstreamErrorIsSSEErrorEvent(t *testing.T) {
	const upstreamSecret = "raw-upstream-body-must-not-cross-the-boundary"
	planner := newCollectivePlanner("test-model", "", &agent.UpstreamStatusError{
		Status:     http.StatusServiceUnavailable,
		Body:       upstreamSecret,
		RetryAfter: "7",
	})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// See the sibling test: LIFO order frees the parked decode before Close waits on it.
	t.Cleanup(release)

	body, err := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "decode across ranks"}},
		"stream":   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	tap := tapChatStream(ts.URL+"/v1/chat/completions", body)

	resp := tap.waitHead(t, epStallBudget, "no HTTP status line/headers before the modeled collective join")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want the 200 the preamble already committed to", resp.StatusCode)
	}
	opening := decodeSSEChunk(t, tap.waitFrame(t, epStallBudget, "no opening SSE chunk before the modeled collective join"))
	if got := opening.Choices[0].Delta.Role; got != agent.RoleAssistant {
		t.Fatalf("opening delta role = %q, want assistant", got)
	}

	release()
	rest := tap.drain(t, 15*time.Second)
	joined := strings.Join(rest, "\n")
	if strings.Contains(joined, upstreamSecret) {
		t.Fatalf("upstream raw body leaked into the SSE error event: %s", joined)
	}
	if !strings.Contains(joined, "data: [DONE]") {
		t.Fatalf("errored stream did not terminate with [DONE]: %s", joined)
	}
	var sawError bool
	for _, line := range rest {
		if line == "data: [DONE]" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			t.Fatalf("non-SSE line on the wire: %q", line)
		}
		var frame struct {
			Error *struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			t.Fatalf("decode SSE frame %q: %v", data, err)
		}
		if frame.Error != nil {
			sawError = true
			if strings.TrimSpace(frame.Error.Message) == "" {
				t.Fatalf("SSE error event carried no message: %s", data)
			}
			if strings.TrimSpace(frame.Error.Type) == "" {
				t.Fatalf("SSE error event carried no type: %s", data)
			}
		}
	}
	if !sawError {
		t.Fatalf("a post-preamble upstream failure truncated the stream instead of emitting an SSE error event: %s", joined)
	}
}
