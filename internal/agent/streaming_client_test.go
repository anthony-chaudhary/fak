package agent

import (
	"context"
	"testing"
	"time"
)

// streaming_client_test.go pins the SSE no-total-deadline contract on the public
// planner: a streamed upstream request must NOT inherit the planner's whole-request
// Client.Timeout, because that deadline covers the streamed body and severs a
// slow-but-live turn (the private `cmd/fak-server` proxy hit the same class as
// "502 upstream model error (stream): context canceled"). The bound for a stream
// is the request context plus the idle/progress/max-duration stall watchdog; a
// non-streaming caller keeps the bounded client.

// TestStreamingClientDropsTotalTimeout pins the mechanism: the streaming client
// zeroes Timeout while preserving the transport, and the configured client is
// untouched so the non-streaming paths stay bounded.
func TestStreamingClientDropsTotalTimeout(t *testing.T) {
	p := NewHTTPPlanner("http://127.0.0.1:0", "m", "")
	if p.Client == nil {
		t.Fatal("NewHTTPPlanner left Client nil")
	}
	if p.Client.Timeout == 0 {
		t.Fatal("baseline planner client has no total timeout; the test cannot distinguish the fix")
	}

	sc := p.streamingClient()
	if sc.Timeout != 0 {
		t.Fatalf("streamingClient().Timeout = %v, want 0 (no total deadline)", sc.Timeout)
	}
	if sc == p.Client {
		t.Fatal("streamingClient returned the configured client itself; the shared client would lose its bound")
	}
	if p.Client.Timeout == 0 {
		t.Fatal("streamingClient mutated the configured client's Timeout")
	}
	if sc.Transport != p.Client.Transport || sc.Jar != p.Client.Jar {
		t.Fatal("streamingClient did not preserve Transport/Jar")
	}
}

// TestCompleteStreamSurvivesBeyondPlannerTimeout is the end-to-end witness: a
// healthy SSE upstream whose whole turn takes LONGER than the planner's configured
// total timeout still completes, because the streaming request dropped that
// deadline. FAK_PLANNER_TIMEOUT_S is clamped to a 5s floor, so the upstream sleeps
// past 5s between a couple of frames (well under the default 60s stall window).
func TestCompleteStreamSurvivesBeyondPlannerTimeout(t *testing.T) {
	t.Setenv("FAK_PLANNER_TIMEOUT_S", "5")
	const frame = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"
	frames := []string{
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
		frame,
		frame,
		frame,
		"data: [DONE]\n\n",
	}
	// 6s per gap: the total turn (~24s) exceeds the 5s planner total timeout, but
	// each inter-byte gap is far under the 60s idle window, so a live stream must
	// survive. Before the fix the 5s total deadline canceled it mid-stream.
	srv := steadySSEServer(t, "text/event-stream", frames, 6*time.Second)

	p := NewHTTPPlanner(srv.URL, "m", "")
	done := make(chan error, 1)
	go func() {
		_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
			[]Message{{Role: RoleUser, Content: "hi"}}, nil)
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("CompleteStream over a slow-but-live stream = %v, want nil (the total timeout must not sever it)", err)
		}
	case <-time.After(90 * time.Second):
		t.Fatal("CompleteStream did not return within 90s")
	}
}

// TestStreamingClientNilClientStillUnbounded guards the hand-built-planner path: a
// planner with no Client must still produce an unbounded streaming client rather
// than panicking or inheriting a bound.
func TestStreamingClientNilClientStillUnbounded(t *testing.T) {
	p := &HTTPPlanner{}
	if sc := p.streamingClient(); sc == nil || sc.Timeout != 0 {
		t.Fatalf("nil-client streamingClient() = %+v, want non-nil Timeout 0", sc)
	}
}
