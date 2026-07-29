package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// newErrorFrameServer builds a gateway fronting `upstream` on the Anthropic wire, the
// same fixture shape the sibling rate-limit retry test uses.
func newErrorFrameServer(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstreamURL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func postStreamingMessages(t *testing.T, ts *httptest.Server) *http.Response {
	t.Helper()
	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// TestAnthropicMessagesPassthroughStreamHTTP502IsRetried is the control for the
// error-frame test below: a real HTTP 502 IS in retryableStatus, so the streaming
// passthrough must retry it to the pinned attempt budget before surfacing anything.
// If this control ever goes red, the 502 complaint is about the STATUS path, not the
// in-band error-frame path.
func TestAnthropicMessagesPassthroughStreamHTTP502IsRetried(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Retry-After", "0") // honored as a zero wait => the test does not sleep
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer upstream.Close()

	resp := postStreamingMessages(t, newErrorFrameServer(t, upstream.URL))
	io.Copy(io.Discard, resp.Body)

	if got := atomic.LoadInt32(&hits); got != 3 {
		t.Fatalf("upstream hit %d times, want 3 (a real HTTP 502 is retryableStatus)", got)
	}
}

// TestAnthropicMessagesPassthroughStreamErrorFrameBeforeStartIsRetried pins the OTHER
// way an overloaded Anthropic upstream refuses a streamed turn: HTTP 200 +
// text/event-stream, then an in-band SSE `error` frame as the FIRST event, before any
// message_start. That is exactly as transient as the 529 the status path already
// retries — but the retry loop in StreamAnthropicRaw decides purely on r.StatusCode, so
// a 200 escapes it and the error frame reaches onEvent, which (pre-fix) flattened it
// into a bare 502 and marked the response written. Net effect for a live
// `fak guard -- claude` session: ONE upstream hit, no retry, no buffered fallback, and
// an opaque 502 whose `server_error` type has lost the upstream's own overloaded_error
// signal — the turn just dies.
//
// The decisive assertion is the upstream hit COUNT: a retried transient must be tried
// more than once.
func TestAnthropicMessagesPassthroughStreamErrorFrameBeforeStartIsRetried(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Anthropic's in-band overload refusal, as the very first frame.
		io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	resp := postStreamingMessages(t, newErrorFrameServer(t, upstream.URL))
	body, _ := io.ReadAll(resp.Body)

	if got := atomic.LoadInt32(&hits); got != 3 {
		t.Errorf("upstream hit %d times, want 3 — an in-band overloaded_error frame before message_start is as transient as a 529 and must be retried, not surfaced on the first hit", got)
	}

	// And once the budget IS spent, the client must still be able to tell an overload
	// apart from a generic server crash: the upstream's own error type survives.
	var env struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("error envelope did not decode: %v (body %q)", err, body)
	}
	if env.Error.Type != "overloaded_error" {
		t.Errorf("error type = %q, want overloaded_error (a flattened 502/server_error loses the upstream's retryable signal; status was %d, body %q)", env.Error.Type, resp.StatusCode, body)
	}
}

// TestAnthropicMessagesPassthroughStreamInBandRequestErrorIsNotRetried is the other half of
// the in-band classification: only the TRANSIENT family earns a re-send. An in-band
// invalid_request_error is a request error no retry can fix, so it must be surfaced on the
// FIRST hit — with the upstream's real 400, not a 502 and not an overload's backoff.
// Without this the "retry an in-band error" fix would turn every malformed request into a
// full attempt-budget burst against the upstream.
func TestAnthropicMessagesPassthroughStreamInBandRequestErrorIsNotRetried(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad\"}}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	resp := postStreamingMessages(t, newErrorFrameServer(t, upstream.URL))
	body, _ := io.ReadAll(resp.Body)

	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Errorf("upstream hit %d times, want 1 — an in-band invalid_request_error is not retryable", got)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("client status = %d, want 400 (the upstream's own status, body %q)", resp.StatusCode, body)
	}
}

// TestAnthropicMessagesPassthroughStreamErrorFrameAfterStartStillRelays guards the
// boundary the fix must NOT cross. Once message_start has been relayed the client owns a
// live 200 SSE stream, so a later error frame cannot be retried (the status is already
// sent) and must keep being forwarded verbatim as a terminal SSE error — the pre-existing
// behavior. A fix that routed EVERY error frame back into the retry loop would silently
// break mid-stream termination, so this test is the one that would catch it.
func TestAnthropicMessagesPassthroughStreamErrorFrameAfterStartStillRelays(t *testing.T) {
	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
		io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	resp := postStreamingMessages(t, newErrorFrameServer(t, upstream.URL))
	body, _ := io.ReadAll(resp.Body)

	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Errorf("upstream hit %d times, want 1 — a POST-message_start error frame cannot be retried, the client already owns the 200", got)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("client status = %d, want 200 (the stream had already opened)", resp.StatusCode)
	}
	if !bytes.Contains(body, []byte("overloaded_error")) {
		t.Errorf("relayed stream lost the terminal error frame; body = %q", body)
	}
}
