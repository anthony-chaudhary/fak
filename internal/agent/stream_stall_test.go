package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// blockingSSEServer is an httptest server that writes a fixed SSE prefix, FLUSHES it (so the
// client genuinely receives those bytes and opens the stream), then BLOCKS without closing
// the connection — the upstream-stall failure mode. The block is released only when the test
// ends (release is closed in t.Cleanup), so the server goroutine never leaks. contentType
// selects the wire ("text/event-stream" for both the OpenAI and Anthropic SSE shapes here).
func blockingSSEServer(t *testing.T, contentType, prefix string) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, prefix)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-release // go silent: bytes stop arriving but the stream stays open (a stall)
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}

// steadySSEServer writes each frame in frames, flushing and sleeping `gap` between them, then
// closes. A healthy-but-slow stream whose inter-byte gaps stay under the idle window must NOT
// trip the stall reader even though the WHOLE turn outlasts the window.
func steadySSEServer(t *testing.T, contentType string, frames []string, gap time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		for _, fr := range frames {
			_, _ = io.WriteString(w, fr)
			if f != nil {
				f.Flush()
			}
			time.Sleep(gap)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestStreamStallTripsOnSilentUpstream drives CompleteStream against an OpenAI-compatible
// upstream that emits a couple of frames then goes silent. With a short stall window the
// read must abort as *UpstreamStalledError well before the planner's whole-request timeout —
// the bug this change fixes (a stalled stream used to hang for the full Client.Timeout).
func TestStreamStallTripsOnSilentUpstream(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	const prefix = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
	srv := blockingSSEServer(t, "text/event-stream", prefix)

	p := NewHTTPPlanner(srv.URL, "m", "")
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
		if !errors.Is(err, ErrUpstreamStalled) {
			t.Fatalf("errors.Is(err, ErrUpstreamStalled) = false, err = %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("CompleteStream did not return within 30s — the stall deadline did not fire (the hang regressed)")
	}
}

// TestStreamStallDoesNotTripOnSteadyStream proves a healthy-but-slow stream is never tripped:
// the frames arrive with inter-byte gaps UNDER the window, even though the whole turn (sum of
// gaps) outlasts a single window. The turn must complete with full content and no error.
func TestStreamStallDoesNotTripOnSteadyStream(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	frames := []string{
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"c\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
		"data: [DONE]\n\n",
	}
	// 2s gap < 5s window, but 6 frames * 2s = 12s > one window: a whole-request-timeout
	// approach would either cut this off or fail to detect a real stall; the idle window
	// distinguishes them.
	srv := steadySSEServer(t, "text/event-stream", frames, 2*time.Second)

	p := NewHTTPPlanner(srv.URL, "m", "")
	comp, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream on a steady stream: unexpected error %v", err)
	}
	if comp.Message.Content != "abc" {
		t.Fatalf("content = %q, want %q", comp.Message.Content, "abc")
	}
	if comp.FinishReason != "stop" {
		t.Fatalf("finish = %q, want stop", comp.FinishReason)
	}
}

// TestStreamAnthropicRawStallTrips is the Anthropic twin of the OpenAI stall test: the
// flagship `fak guard -- claude` passthrough goes through StreamAnthropicRaw, so the stall
// deadline must protect it too. The upstream opens with message_start + a delta, then stalls.
func TestStreamAnthropicRawStallTrips(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	prefix := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":3,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"chec"}}`,
		``,
		``,
	}, "\n")
	srv := blockingSSEServer(t, "text/event-stream", prefix)

	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	done := make(chan error, 1)
	go func() {
		done <- p.StreamAnthropicRaw(context.Background(), rawBody, "k", "", func(AnthropicSSEEvent) error { return nil })
	}()

	select {
	case err := <-done:
		var stalled *UpstreamStalledError
		if !errors.As(err, &stalled) {
			t.Fatalf("err = %v, want *UpstreamStalledError", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("StreamAnthropicRaw did not return within 30s — the stall deadline did not fire")
	}
}

// TestStreamStallSentinelDistinctFromEOF proves the stall mapping is gated on the timer
// actually firing: a clean, fully-closed stream returns nil (not a spurious stall), and a
// stream the server closes early surfaces as a normal read outcome, NEVER as a stall.
func TestStreamStallSentinelDistinctFromEOF(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")

	// (a) Clean, complete stream — no stall.
	const body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	srv, _ := sseServer(t, body)
	p := NewHTTPPlanner(srv.URL, "m", "")
	comp, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("clean stream returned error %v, want nil", err)
	}
	if errors.Is(err, ErrUpstreamStalled) {
		t.Fatal("clean stream reported as stalled")
	}
	if comp.Message.Content != "hi" {
		t.Fatalf("content = %q, want hi", comp.Message.Content)
	}

	// (b) Server closes mid-stream (a truncated stream, NOT silence): the scanner sees EOF
	// promptly, so the turn ends without the timer firing — and is never tagged as a stall.
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"par\"},\"finish_reason\":null}]}\n\n")
		// handler returns -> connection closes -> client read hits EOF, not a stall.
	}))
	t.Cleanup(srv2.Close)
	p2 := NewHTTPPlanner(srv2.URL, "m", "")
	comp2, err2 := p2.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	if errors.Is(err2, ErrUpstreamStalled) {
		t.Fatalf("a server-closed (EOF) stream was misreported as a stall: %v", err2)
	}
	// A truncated stream with no [DONE] is a clean scanner EOF here, so err2 is nil and the
	// partial content is returned — the point is only that it is NOT a stall.
	if comp2 != nil && comp2.Message.Content != "par" {
		t.Fatalf("partial content = %q, want par", comp2.Message.Content)
	}
}

// TestStreamStallTimeoutClamp pins streamStallTimeout's env parse + clamp band, the same
// shape plannerTimeout uses: unset -> 60s default, in-band honored, out-of-band / unparseable
// fall back to the default.
func TestStreamStallTimeoutClamp(t *testing.T) {
	cases := []struct {
		env  string
		want time.Duration
	}{
		{"", 60 * time.Second},
		{"1", 60 * time.Second},    // below the 5s floor -> default
		{"5", 5 * time.Second},     // floor
		{"600", 600 * time.Second}, // ceiling
		{"601", 60 * time.Second},  // above the 600s ceiling -> default
		{"x", 60 * time.Second},    // unparseable -> default
	}
	for _, c := range cases {
		t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", c.env)
		if got := streamStallTimeout(); got != c.want {
			t.Errorf("streamStallTimeout() with env %q = %s, want %s", c.env, got, c.want)
		}
	}
}
