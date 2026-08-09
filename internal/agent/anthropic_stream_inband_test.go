package agent

// anthropic_stream_inband_test.go — witnesses for #5491: Anthropic refuses a streamed turn
// TWO ways, and the second one wears a 200. Instead of a 429/503/529 on the wire it answers
// HTTP 200 + text/event-stream and then sends an SSE `error` frame as its FIRST event, before
// any message_start. Nothing has been relayed at that point, so the refusal must take the same
// arms as its equivalent status: a transient one re-sends invisibly under the same budget, a
// request error surfaces at once with the status it maps to, and an `error` frame arriving
// AFTER message_start stays a mid-stream failure the caller forwards.

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

// anthropicInBandRefusal writes the shape the real API sends when it refuses a streamed turn
// after the headers are already out: a 200 with the event-stream content type, then a single
// `error` frame. Retry-After: 0 is a test-speed lever, not part of the shape — the loop honors
// it exactly as it does on a 429/503, so the retries here do not sleep.
func anthropicInBandRefusal(w http.ResponseWriter, errType string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Retry-After", "0")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "event: error\n"+
		`data: {"type":"error","error":{"type":"`+errType+`","message":"refused in-band"}}`+"\n\n")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func inBandTestPlanner(t *testing.T, url string) *HTTPPlanner {
	t.Helper()
	p, err := NewProviderHTTPPlanner("anthropic", url, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	return p
}

var inBandRawBody = []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

// A transient in-band refusal (overloaded_error on a 200) must be RETRIED with the same
// backoff budget the HTTP 529 path uses, and the recovered turn relayed — the whole point of
// #5491. Before the fix the 200 broke out of the retry loop and the frame surfaced from the
// SSE reader as a terminal failure after exactly ONE upstream try.
func TestStreamAnthropicRaw_RetriesInBandOverloadedThenStreams(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0") // attempt-count bound only; Retry-After: 0 => no sleep

	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			anthropicInBandRefusal(w, "overloaded_error") // 200, then refuse in-band
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, anthropicStreamRetrySSE("ok"))
	}))
	t.Cleanup(srv.Close)

	p := inBandTestPlanner(t, srv.URL)
	var notifyN, notifyStatus int32
	p.RetryNotify = func(attempt, status int, wait time.Duration) {
		atomic.AddInt32(&notifyN, 1)
		atomic.StoreInt32(&notifyStatus, int32(status))
	}

	var gotText strings.Builder
	var sawStop, sawError bool
	err := p.StreamAnthropicRaw(context.Background(), inBandRawBody, "k", "", func(ev AnthropicSSEEvent) error {
		switch ev.Event {
		case "error":
			sawError = true
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
		t.Fatalf("an in-band overloaded_error must be retried and the recovered turn relayed, got: %v", err)
	}
	if got := atomic.LoadInt32(&n); got != 2 {
		t.Fatalf("upstream hit %d times, want 2 (one in-band refusal + one streamed success)", got)
	}
	if gotText.String() != "ok" || !sawStop {
		t.Fatalf("recovered stream incomplete: text=%q sawStop=%v", gotText.String(), sawStop)
	}
	if sawError {
		t.Fatal("the pre-start refusal frame must be absorbed by the retry, never handed to the relay callback")
	}
	// The retry must be OBSERVABLE on the same hook the HTTP-status path fires, carrying the
	// status the in-band type maps to (529), so the gateway's `fak-turn … retry` line covers it.
	if got := atomic.LoadInt32(&notifyN); got != 1 {
		t.Fatalf("RetryNotify fired %d times, want 1", got)
	}
	if got := atomic.LoadInt32(&notifyStatus); got != statusOverloaded {
		t.Fatalf("RetryNotify status = %d, want %d (the in-band overloaded_error mapped onto its status)", got, statusOverloaded)
	}
}

// A PERSISTENT in-band refusal must spend the whole pinned attempt budget and then surface the
// upstream's OWN status — a real *UpstreamStatusError{529}, which is what lets the gateway map
// it to upstream_overloaded/overloaded_error instead of an opaque 502/server_error.
func TestStreamAnthropicRaw_InBandRefusalExhaustsBudgetAndSurfacesStatus(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")

	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		anthropicInBandRefusal(w, "overloaded_error")
	}))
	t.Cleanup(srv.Close)

	err := inBandTestPlanner(t, srv.URL).StreamAnthropicRaw(context.Background(), inBandRawBody, "k", "", func(AnthropicSSEEvent) error {
		t.Error("no event may reach the relay callback: the refusal arrived before message_start")
		return nil
	})
	if err == nil {
		t.Fatal("a persistent in-band refusal must fail, got nil")
	}
	var se *UpstreamStatusError
	if !errors.As(err, &se) || se.Status != statusOverloaded {
		t.Fatalf("err = %v, want a wrapped *UpstreamStatusError{529} carrying the upstream's own signal", err)
	}
	if got := atomic.LoadInt32(&n); got != 3 {
		t.Fatalf("upstream hit %d times, want 3 (FAK_PLANNER_MAX_ATTEMPTS)", got)
	}
}

// The negative fence: an in-band REQUEST error is not transient, so it must surface on the
// FIRST try with the status it maps to — no retry burst against a wall no retry can move, and
// no 502 costume over a 400.
func TestStreamAnthropicRaw_InBandRequestErrorSurfacesImmediately(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")

	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		anthropicInBandRefusal(w, "invalid_request_error")
	}))
	t.Cleanup(srv.Close)

	err := inBandTestPlanner(t, srv.URL).StreamAnthropicRaw(context.Background(), inBandRawBody, "k", "", func(AnthropicSSEEvent) error { return nil })
	var se *UpstreamStatusError
	if !errors.As(err, &se) || se.Status != http.StatusBadRequest {
		t.Fatalf("err = %v, want *UpstreamStatusError{400} (an in-band invalid_request_error is a request error)", err)
	}
	if got := atomic.LoadInt32(&n); got != 1 {
		t.Fatalf("upstream hit %d times, want 1 (a request error must not be retried)", got)
	}
}

// The non-regression fence for the OTHER half of the arm: once message_start has opened the
// caller's stream, an `error` frame is a MID-stream failure the caller owns — it must still be
// delivered to the relay callback verbatim and must NOT be re-sent (the client already holds a
// live 200, and a retry would double-generate).
func TestStreamAnthropicRaw_MidStreamErrorFrameIsRelayedNotRetried(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")

	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
			``,
			`event: error`,
			`data: {"type":"error","error":{"type":"overloaded_error","message":"died mid-turn"}}`,
			``,
		}, "\n"))
	}))
	t.Cleanup(srv.Close)

	var events []string
	err := inBandTestPlanner(t, srv.URL).StreamAnthropicRaw(context.Background(), inBandRawBody, "k", "", func(ev AnthropicSSEEvent) error {
		events = append(events, ev.Event)
		return nil
	})
	if err != nil {
		t.Fatalf("a mid-stream error frame is relayed, not turned into a transport failure: %v", err)
	}
	if got := strings.Join(events, ","); got != "message_start,error" {
		t.Fatalf("relayed events = %q, want %q (the post-start error frame must reach the caller)", got, "message_start,error")
	}
	if got := atomic.LoadInt32(&n); got != 1 {
		t.Fatalf("upstream hit %d times, want 1 (a post-start refusal must never be re-sent)", got)
	}
}

// anthropicInBandErrorStatus is the single classification both halves lean on, so pin the
// mapping directly: each Anthropic error type lands on the status the same refusal carries
// when the API reports it on the wire, and an unrecognized type is NOT guessed into one.
func TestAnthropicInBandErrorStatusMapping(t *testing.T) {
	for _, c := range []struct {
		typ  string
		want int
	}{
		{"invalid_request_error", http.StatusBadRequest},
		{"authentication_error", http.StatusUnauthorized},
		{"permission_error", http.StatusForbidden},
		{"not_found_error", http.StatusNotFound},
		{"request_too_large", http.StatusRequestEntityTooLarge},
		{"timeout_error", http.StatusRequestTimeout},
		{"rate_limit_error", http.StatusTooManyRequests},
		{"api_error", http.StatusInternalServerError},
		{"overloaded_error", statusOverloaded},
	} {
		got, ok := anthropicInBandErrorStatus([]byte(`{"type":"error","error":{"type":"` + c.typ + `","message":"m"}}`))
		if !ok || got != c.want {
			t.Errorf("%s -> (%d, %v), want (%d, true)", c.typ, got, ok, c.want)
		}
		// Whichever way it maps, the retry verdict must agree with the status ladder — that
		// agreement is what makes ONE classification serve both refusal wires.
		if retryableStatus(got) != retryableStatus(c.want) {
			t.Errorf("%s: retry verdict disagrees with its status", c.typ)
		}
	}
	for _, bad := range []string{
		`{"type":"error","error":{"type":"some_future_error","message":"m"}}`,
		`{"type":"error"}`,
		`not json`,
	} {
		if got, ok := anthropicInBandErrorStatus([]byte(bad)); ok {
			t.Errorf("%s -> (%d, true), want ok=false (an unknown refusal must not be guessed into a status)", bad, got)
		}
	}
}
