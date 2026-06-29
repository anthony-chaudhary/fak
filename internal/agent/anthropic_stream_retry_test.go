package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// anthropicStreamRetrySSE is a complete, well-formed Anthropic Messages SSE turn: one
// text block carrying `text`, then the terminal frames — enough for parseAnthropicSSE to
// deliver a full event sequence the gateway relay would forward.
func anthropicStreamRetrySSE(text string) string {
	return strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + text + `"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
}

// StreamAnthropicRaw — the flagship `fak guard -- claude` passthrough — must now RETRY a
// pre-stream retryable status (here 529 "Overloaded") and then relay the recovered turn,
// instead of failing on the first 529 and collapsing the live stream to the buffered
// fallback. The RetryNotify hook must also fire (with the triggering status), so the
// gateway's `fak-turn … retry` observability line covers this path like the others.
func TestStreamAnthropicRaw_Retries529ThenStreams(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.WriteHeader(statusOverloaded) // 529 once, before any SSE byte
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, anthropicStreamRetrySSE("ok"))
	}))
	t.Cleanup(srv.Close)

	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	var notifyN, notifyStatus int32
	p.RetryNotify = func(attempt, status int, wait time.Duration) {
		atomic.AddInt32(&notifyN, 1)
		atomic.StoreInt32(&notifyStatus, int32(status))
	}

	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	var gotText strings.Builder
	var sawStop bool
	err = p.StreamAnthropicRaw(context.Background(), rawBody, "k", "", func(ev AnthropicSSEEvent) error {
		switch ev.Event {
		case "content_block_delta":
			if strings.Contains(string(ev.Data), `"ok"`) {
				gotText.WriteString("ok")
			}
		case "message_stop":
			sawStop = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("StreamAnthropicRaw should retry 529 and stream, got: %v", err)
	}
	if gotText.String() != "ok" {
		t.Fatalf("streamed text = %q, want ok", gotText.String())
	}
	if !sawStop {
		t.Fatal("did not see message_stop — the recovered stream did not complete")
	}
	if got := atomic.LoadInt32(&n); got != 2 {
		t.Fatalf("upstream hit %d times, want 2 (one 529 + one streamed success)", got)
	}
	if got := atomic.LoadInt32(&notifyN); got != 1 {
		t.Fatalf("RetryNotify fired %d times, want 1", got)
	}
	if got := atomic.LoadInt32(&notifyStatus); got != statusOverloaded {
		t.Fatalf("RetryNotify status = %d, want %d (529)", got, statusOverloaded)
	}
}

// A persistently-overloaded upstream must fail AFTER exactly maxAttempts streamed tries
// (pinned low via env so the test is fast), and the returned error must carry the upstream
// status so the gateway can surface a real 429/503 (and any Retry-After) to the client.
func TestStreamAnthropicRaw_FailsAfterMaxAttempts(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.Header().Set("Retry-After", "0") // honored as a zero wait => the test does not sleep
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	gotErr := p.StreamAnthropicRaw(context.Background(), rawBody, "k", "", func(AnthropicSSEEvent) error { return nil })
	if gotErr == nil {
		t.Fatal("StreamAnthropicRaw against a persistent 503 should fail, got nil")
	}
	var se *UpstreamStatusError
	if !errors.As(gotErr, &se) || se.Status != http.StatusServiceUnavailable {
		t.Fatalf("err = %v, want a wrapped *UpstreamStatusError{503}", gotErr)
	}
	if got := atomic.LoadInt32(&n); got != 2 {
		t.Fatalf("upstream hit %d times, want 2 (FAK_PLANNER_MAX_ATTEMPTS)", got)
	}
}

// A cancelled context must abort the backoff wait promptly rather than sleeping out the
// retry schedule — so a user who Ctrl-C's a stalled streamed turn is not held hostage by
// the (now present) retry budget. Mirrors TestComplete_ContextCancelAbortsBackoff.
func TestStreamAnthropicRaw_ContextCancelAbortsBackoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable) // always 503, no Retry-After => real backoff
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	start := time.Now()
	if e := p.StreamAnthropicRaw(ctx, rawBody, "k", "", func(AnthropicSSEEvent) error { return nil }); e == nil {
		t.Fatal("StreamAnthropicRaw should fail when the context is cancelled mid-backoff")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("StreamAnthropicRaw took %s after a 200ms ctx timeout — backoff did not honor cancellation", elapsed)
	}
}
