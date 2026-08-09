package gateway

// ep_completions_stall_test.go — the legacy-completions sibling of
// ep_stream_stall_test.go (#5514).
//
// #5399 gave POST /v1/chat/completions an EARLY SSE preamble so a stream:true turn
// served by the BUFFERED path proves the socket alive before the decode starts. The
// legacy text-completion wire (POST /v1/completions) shares the same served path and
// the same Complete-only planners, but wrote its first byte — status line, headers,
// first chunk — only after completeServed returned, so it kept the identical stall.
//
// These tests model that topology in-process with no weights and no device, reusing
// collectivePlanner (a Complete-ONLY planner that parks in a modeled per-step
// collective) and the SSE tap from ep_stream_stall_test.go. Every wait is bounded, so
// they can fail loudly but never hang a suite.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// decodeCompletionSSEChunk parses one legacy `text_completion` SSE line. It is the
// text-wire counterpart of decodeSSEChunk: the legacy chunk carries a bare `text`
// per choice rather than a chat `delta`, which is exactly the shape difference this
// route must PRESERVE while it adopts the chat route's preamble timing.
func decodeCompletionSSEChunk(t *testing.T, line string) CompletionStreamResponse {
	t.Helper()
	data, ok := strings.CutPrefix(line, "data: ")
	if !ok {
		t.Fatalf("non-SSE line on the wire: %q", line)
	}
	var chunk CompletionStreamResponse
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		t.Fatalf("decode legacy SSE chunk %q: %v", data, err)
	}
	return chunk
}

// TestCompletionsStreamingEmitsPreambleBeforeCollectiveJoin is the #5514 witness. A
// legacy-completions client asks for stream:true against a Complete-only planner that
// parks in the modeled collective. Before that collective is released the client must
// ALREADY hold the status line, the SSE headers, and an opening `text_completion`
// chunk — otherwise a multi-rank decode is indistinguishable from a dead socket for
// its whole duration, the same 1120+ second window the chat fix removed.
//
// Against the pre-fix gateway nothing reaches the client until completeServed returns
// (writeCompletionStream is what wrote the headers), so the preamble waits fail here.
func TestCompletionsStreamingEmitsPreambleBeforeCollectiveJoin(t *testing.T) {
	planner := newCollectivePlanner("test-model", "rank0 decoded through the collective", nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// Registered AFTER ts.Close so LIFO cleanup frees the parked decode FIRST: a failing
	// run tears down immediately instead of waiting out the modeled collective's own
	// hard bound while httptest.Server.Close blocks on the live connection.
	t.Cleanup(release)

	body, err := json.Marshal(CompletionRequest{
		Model:     "test-model",
		Prompt:    json.RawMessage(`"decode across ranks"`),
		Stream:    true,
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	// tapChatStream is wire-agnostic: it POSTs a body and hands back the response head
	// and each SSE line as they arrive. Point it at the legacy route.
	tap := tapChatStream(ts.URL+"/v1/completions", body)

	resp := tap.waitHead(t, epStallBudget,
		"no HTTP status line/headers before the modeled collective join — the #5514 legacy-completions stall")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	opening := decodeCompletionSSEChunk(t, tap.waitFrame(t, epStallBudget,
		"no first SSE byte before the modeled collective join — the #5514 legacy-completions stall"))
	// Schema, not shape-drift: the preamble chunk must still be a legacy text_completion
	// object with a `text` field. Emitting a chat-shaped frame here would "fix" the
	// stall by breaking every real legacy client.
	if opening.Object != "text_completion" {
		t.Fatalf("opening object = %q, want text_completion", opening.Object)
	}
	if len(opening.Choices) != 1 {
		t.Fatalf("opening choices = %+v, want exactly one", opening.Choices)
	}
	if opening.Choices[0].Text != "" {
		t.Fatalf("opening chunk text = %q, want empty — no model byte exists yet", opening.Choices[0].Text)
	}
	if opening.Choices[0].FinishReason != nil {
		t.Fatalf("opening chunk finish_reason = %v, want null", *opening.Choices[0].FinishReason)
	}
	if opening.Model != "test-model" {
		t.Fatalf("opening model = %q, want the request model", opening.Model)
	}
	if !strings.HasPrefix(opening.ID, "cmpl-") {
		t.Fatalf("opening id = %q, want the legacy cmpl- prefix", opening.ID)
	}

	// The preamble is not a race against a no-op planner: the decode really is parked
	// inside the modeled collective at this point, and only this test can free it.
	select {
	case <-planner.joined:
	case <-time.After(epStallBudget):
		t.Fatal("the modeled decode never entered the collective — the repro is not exercising the buffered path")
	}

	// Release the collective: the turn now exists and the rest of the stream must land,
	// byte-identical to what the pre-preamble writer produced.
	release()
	rest := tap.drain(t, 15*time.Second)
	var text strings.Builder
	var finish string
	var sawDone, sawUsage bool
	for _, line := range rest {
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		chunk := decodeCompletionSSEChunk(t, line)
		if chunk.Object != "text_completion" {
			t.Fatalf("chunk object = %q, want text_completion", chunk.Object)
		}
		if chunk.ID != opening.ID {
			t.Fatalf("chunk id = %q, want the stream identity %q announced in the preamble", chunk.ID, opening.ID)
		}
		if chunk.Model != "test-model" {
			t.Fatalf("chunk model = %q, want a constant %q across the stream", chunk.Model, "test-model")
		}
		text.WriteString(chunk.Choices[0].Text)
		if chunk.Choices[0].FinishReason != nil {
			finish = *chunk.Choices[0].FinishReason
		}
		if chunk.Usage != nil && chunk.Usage.TotalTokens == 10 {
			sawUsage = true
		}
	}
	if got := text.String(); got != planner.content {
		t.Fatalf("reassembled streamed text = %q, want %q", got, planner.content)
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

// TestCompletionsStreamingPostPreambleUpstreamErrorIsSSEErrorEvent covers the cost of
// opening the legacy stream early: once the 200 and the SSE headers are on the wire
// the handler can no longer answer with a real HTTP status, so an upstream failure
// landing AFTER the preamble has to arrive as an OpenAI-shaped SSE error event
// followed by [DONE] — never a silently truncated stream, and never the upstream's
// raw body.
func TestCompletionsStreamingPostPreambleUpstreamErrorIsSSEErrorEvent(t *testing.T) {
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

	body, err := json.Marshal(CompletionRequest{
		Model:  "test-model",
		Prompt: json.RawMessage(`"decode across ranks"`),
		Stream: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	tap := tapChatStream(ts.URL+"/v1/completions", body)

	resp := tap.waitHead(t, epStallBudget, "no HTTP status line/headers before the modeled collective join")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want the 200 the preamble already committed to", resp.StatusCode)
	}
	opening := decodeCompletionSSEChunk(t, tap.waitFrame(t, epStallBudget, "no opening SSE chunk before the modeled collective join"))
	if opening.Object != "text_completion" {
		t.Fatalf("opening object = %q, want text_completion", opening.Object)
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
		t.Fatalf("a post-preamble upstream failure truncated the legacy stream instead of emitting an SSE error event: %s", joined)
	}
}
