package agent

// stream_stall_progress_test.go covers the CONTENT-progress deadline (#5486): the half of
// stallReader that separates "the turn is advancing" from "the socket is warm". The byte
// (idle) deadline's own coverage lives in stream_stall_test.go and is unchanged — these
// tests all drive upstreams that keep BYTES flowing, so the idle deadline can never fire
// and only the progress deadline can end the turn.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// keepaliveOnlySSEServer is the stalled-but-WARM upstream: it writes a healthy-looking
// prefix, flushes it, then emits `keepalive` every gap FOREVER without ever sending another
// content frame. Bytes never stop arriving, so an inter-byte deadline alone can never fire —
// this is exactly the wedged-generation-behind-a-live-socket shape #5486 is about. The loop
// exits when the client goes away (the stall reader closes the body) or when the test ends.
func keepaliveOnlySSEServer(t *testing.T, contentType, prefix, keepalive string, gap time.Duration) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, prefix)
		if f != nil {
			f.Flush()
		}
		for {
			select {
			case <-r.Context().Done(): // client hung up (the deadline tripped and closed the body)
				return
			case <-release: // test over
				return
			case <-time.After(gap):
			}
			if _, err := io.WriteString(w, keepalive); err != nil {
				return
			}
			if f != nil {
				f.Flush()
			}
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}

const anthropicWarmPrefix = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":3,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

`

// TestStreamAnthropicRawStallTripsOnPingOnlyUpstream is the witness for #5486. The upstream
// opens a healthy stream and then emits nothing but `ping` keepalives, once a second — well
// inside the 5s idle window, so the byte deadline is re-armed forever and by itself would
// let this turn ride the 600s whole-request ceiling. The progress deadline must still fire,
// and must report the no-progress cause rather than claiming the upstream went silent.
func TestStreamAnthropicRawStallTripsOnPingOnlyUpstream(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	const ping = "event: ping\ndata: {\"type\":\"ping\"}\n\n"
	srv := keepaliveOnlySSEServer(t, "text/event-stream", anthropicWarmPrefix, ping, time.Second)

	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	p.StreamProgressTimeout = 6 * time.Second // the config surface, not an env var
	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	var pings int
	done := make(chan error, 1)
	start := time.Now()
	go func() {
		done <- p.StreamAnthropicRaw(context.Background(), rawBody, "k", "", func(ev AnthropicSSEEvent) error {
			if ev.Event == "ping" {
				pings++
			}
			return nil
		})
	}()

	select {
	case err := <-done:
		elapsed := time.Since(start)
		var stalled *UpstreamStalledError
		if !errors.As(err, &stalled) {
			t.Fatalf("err = %v, want *UpstreamStalledError", err)
		}
		if !errors.Is(err, ErrUpstreamStalled) {
			t.Fatalf("errors.Is(err, ErrUpstreamStalled) = false, err = %v", err)
		}
		if stalled.Kind != stallKindNoProgress {
			t.Fatalf("Kind = %q, want %q — a pinging upstream is warm, not silent", stalled.Kind, stallKindNoProgress)
		}
		if stalled.Idle != 6*time.Second {
			t.Fatalf("Idle = %s, want the 6s PROGRESS window that actually elapsed", stalled.Idle)
		}
		if !strings.Contains(stalled.Error(), "no content progress") {
			t.Fatalf("Error() = %q, want it to name the no-progress cause", stalled.Error())
		}
		// The pings really did arrive and really did keep the socket warm: without them the
		// 5s idle deadline would have been the one to fire, and this test would prove nothing.
		if pings < 2 {
			t.Fatalf("saw %d ping frames, want >=2 — the upstream was not warm, so the idle deadline may have fired instead", pings)
		}
		if elapsed < 5*time.Second {
			t.Fatalf("returned after %s, sooner than the 6s progress window — something else ended the turn", elapsed)
		}
	case <-time.After(60 * time.Second):
		t.Fatal("StreamAnthropicRaw did not return within 60s — a ping-only upstream still rides the whole-request ceiling (#5486)")
	}
}

// TestCompleteStreamTripsOnKeepaliveOnlyUpstream is the OpenAI-wire twin: SSE comment
// heartbeats plus empty-delta chunks keep the bytes flowing without advancing the turn, so
// only the progress deadline can end it.
func TestCompleteStreamTripsOnKeepaliveOnlyUpstream(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	const prefix = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
	const keepalive = ": keepalive\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n"
	srv := keepaliveOnlySSEServer(t, "text/event-stream", prefix, keepalive, time.Second)

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.StreamProgressTimeout = 6 * time.Second // the config surface, not an env var
	done := make(chan error, 1)
	go func() {
		_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
			[]Message{{Role: RoleUser, Content: "hi"}}, nil)
		done <- err
	}()

	select {
	case err := <-done:
		var stalled *UpstreamStalledError
		if !errors.As(err, &stalled) {
			t.Fatalf("err = %v, want *UpstreamStalledError", err)
		}
		if stalled.Kind != stallKindNoProgress {
			t.Fatalf("Kind = %q, want %q", stalled.Kind, stallKindNoProgress)
		}
	case <-time.After(60 * time.Second):
		t.Fatal("CompleteStream did not return within 60s — a keepalive-only upstream still hangs (#5486)")
	}
}

// TestStreamProgressDeadlineSurvivesSlowProgress is the other half of the contract: a stream
// that IS advancing, only slowly, must never be cut. Deltas arrive every 2s for 10s total —
// longer than the 6s progress window — and every one of them re-arms it, so the turn
// completes with its full content. Without this a "fix" that simply caps every stream at the
// progress window would pass the trip test above.
func TestStreamProgressDeadlineSurvivesSlowProgress(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	frames := []string{
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"c\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"d\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
		"data: [DONE]\n\n",
	}
	srv := steadySSEServer(t, "text/event-stream", frames, 2*time.Second)

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.StreamProgressTimeout = 6 * time.Second // the config surface, not an env var
	comp, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("a slow but PROGRESSING stream was cut: %v", err)
	}
	if comp.Message.Content != "abcd" {
		t.Fatalf("content = %q, want abcd", comp.Message.Content)
	}
}

// TestAnthropicFrameAdvancesTurn pins the classifier that decides what re-arms the progress
// deadline: every real turn frame does, `ping` does not — including a bare `data:` ping with
// no `event:` line, which is the shape that would otherwise slip through by name.
func TestAnthropicFrameAdvancesTurn(t *testing.T) {
	cases := []struct {
		name string
		ev   AnthropicSSEEvent
		want bool
	}{
		{"message_start", AnthropicSSEEvent{Event: "message_start", Data: json.RawMessage(`{"type":"message_start"}`)}, true},
		{"content_block_start", AnthropicSSEEvent{Event: "content_block_start", Data: json.RawMessage(`{"type":"content_block_start"}`)}, true},
		{"content_block_delta", AnthropicSSEEvent{Event: "content_block_delta", Data: json.RawMessage(`{"type":"content_block_delta"}`)}, true},
		{"message_delta", AnthropicSSEEvent{Event: "message_delta", Data: json.RawMessage(`{"type":"message_delta"}`)}, true},
		{"message_stop", AnthropicSSEEvent{Event: "message_stop", Data: json.RawMessage(`{"type":"message_stop"}`)}, true},
		{"error", AnthropicSSEEvent{Event: "error", Data: json.RawMessage(`{"type":"error"}`)}, true},
		{"ping", AnthropicSSEEvent{Event: "ping", Data: json.RawMessage(`{"type":"ping"}`)}, false},
		{"ping with no event line", AnthropicSSEEvent{Data: json.RawMessage(`{"type":"ping"}`)}, false},
		{"unparseable payload", AnthropicSSEEvent{Data: json.RawMessage(`not json`)}, true},
	}
	for _, c := range cases {
		if got := anthropicFrameAdvancesTurn(c.ev); got != c.want {
			t.Errorf("anthropicFrameAdvancesTurn(%s) = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestOpenAIChunkAdvancesTurn pins the OpenAI-wire twin: content / reasoning / a tool-call
// fragment / a finish reason count, while a role-only opener or an empty-delta keepalive
// chunk does not.
func TestOpenAIChunkAdvancesTurn(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want bool
	}{
		{"content", `{"choices":[{"delta":{"content":"hi"}}]}`, true},
		{"reasoning", `{"choices":[{"delta":{"reasoning_content":"hm"}}]}`, true},
		{"tool call", `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}`, true},
		{"finish", `{"choices":[{"delta":{},"finish_reason":"stop"}]}`, true},
		{"role only", `{"choices":[{"delta":{"role":"assistant"}}]}`, false},
		{"empty delta", `{"choices":[{"delta":{}}]}`, false},
		{"usage only", `{"usage":{"prompt_tokens":1}}`, false},
		{"model only", `{"model":"m"}`, false},
	}
	for _, c := range cases {
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(c.raw), &chunk); err != nil {
			t.Fatalf("%s: unmarshal fixture: %v", c.name, err)
		}
		if got := openAIChunkAdvancesTurn(chunk); got != c.want {
			t.Errorf("openAIChunkAdvancesTurn(%s) = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestStreamProgressWindowResolvesTheConfigField pins the CONFIG-SURFACE resolver: the band
// a configured window must land in, the zero-means-default rule every unconfigured planner
// takes, and the explicit off switch — the one escape hatch for an operator whose provider
// legitimately prefills longer than the window. The knob is a planner field, never an env
// var: a behavioral deadline is configuration, not a credential (CONFIG_NOT_ENV).
func TestStreamProgressWindowResolvesTheConfigField(t *testing.T) {
	cases := []struct {
		name string
		set  time.Duration
		want time.Duration
	}{
		{"unset", 0, DefaultStreamProgressTimeout},
		{"below the floor", 4 * time.Second, DefaultStreamProgressTimeout},
		{"floor", streamProgressMinWindow, 5 * time.Second},
		{"ceiling", streamProgressMaxWindow, 600 * time.Second},
		{"above the ceiling", 601 * time.Second, DefaultStreamProgressTimeout},
		{"explicit off", -1, 0},
	}
	for _, c := range cases {
		p := &HTTPPlanner{StreamProgressTimeout: c.set}
		if got := p.streamProgressWindow(); got != c.want {
			t.Errorf("%s: streamProgressWindow() with StreamProgressTimeout=%s = %s, want %s", c.name, c.set, got, c.want)
		}
	}
	// The default is the documented 300s, and an unconfigured planner really does take it —
	// the behavior b66a2dba5 shipped, preserved across the move off the environment.
	if DefaultStreamProgressTimeout != 300*time.Second {
		t.Errorf("DefaultStreamProgressTimeout = %s, want 300s", DefaultStreamProgressTimeout)
	}
	if got := NewHTTPPlanner("http://x", "m", "").streamProgressWindow(); got != 300*time.Second {
		t.Errorf("an unconfigured planner's progress window = %s, want the 300s default", got)
	}
}

// TestStallReaderProgressWindowNeverUndercutsIdle pins the constructor invariant: the
// progress deadline is the OUTER one. A shorter progress window is raised to the idle
// window, so a plain dead socket is never mislabelled as a no-progress stall. A zero
// progress window still means "disabled", not "raised".
func TestStallReaderProgressWindowNeverUndercutsIdle(t *testing.T) {
	raised := newStallReader(io.NopCloser(strings.NewReader("")), 60*time.Second, 5*time.Second, 0)
	defer raised.Close()
	if raised.progressWindow != 60*time.Second {
		t.Errorf("progressWindow = %s, want it raised to the 60s idle window", raised.progressWindow)
	}
	off := newStallReader(io.NopCloser(strings.NewReader("")), 60*time.Second, 0, 0)
	defer off.Close()
	if off.progressWindow != 0 || off.progressTimer != nil {
		t.Errorf("progressWindow = %s / timer set = %v, want the deadline disabled", off.progressWindow, off.progressTimer != nil)
	}
	if kind, window := off.stallCause(); kind != stallKindIdle || window != 60*time.Second {
		t.Errorf("stallCause() = (%q, %s), want (%q, 60s) for an untripped reader", kind, window, stallKindIdle)
	}
}
