package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// prefillSilentServer opens a 200 SSE stream, flushes the headers, stays silent for
// `prefill` (a llama.cpp-style server emits no body bytes while it prefills a long prompt),
// then streams a short completion. hits counts how many times the upstream was asked, so a
// cancel-and-retry of the prefill is observable.
func prefillSilentServer(t *testing.T, prefill time.Duration, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		if f != nil {
			f.Flush()
		}
		select {
		case <-time.After(prefill):
		case <-r.Context().Done():
			return
		}
		for _, fr := range []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
			"data: [DONE]\n\n",
		} {
			_, _ = io.WriteString(w, fr)
			if f != nil {
				f.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestStreamStallPrefillSilenceWithinProgressWindowCompletes: a silence before the FIRST body
// byte that outlasts the inter-byte idle window but not the content-progress window is a
// prefill, not a stall. The turn must complete on the first upstream attempt.
//
// fak-test:runtime slow est=8s lane=default
func TestStreamStallPrefillSilenceWithinProgressWindowCompletes(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	var hits atomic.Int32
	srv := prefillSilentServer(t, 7*time.Second, &hits)

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.StreamProgressTimeout = 20 * time.Second
	comp, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	var stalled *UpstreamStalledError
	if errors.As(err, &stalled) {
		t.Fatalf("prefill silence under the progress window tripped the stall reader: kind=%q idle=%s", stalled.Kind, stalled.Idle)
	}
	if err != nil {
		t.Fatalf("CompleteStream: unexpected error %v", err)
	}
	if comp.Message.Content != "ok" || comp.FinishReason != "stop" {
		t.Fatalf("content=%q finish=%q, want ok/stop", comp.Message.Content, comp.FinishReason)
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("upstream attempts = %d, want 1 (the prefill was cancelled and retried)", got)
	}
}

// TestStreamStallMidStreamSilenceTripsAtIdleWindow: once the first byte has arrived, the
// inter-byte idle window governs again, well before the content-progress window.
//
// fak-test:runtime slow est=6s lane=default
func TestStreamStallMidStreamSilenceTripsAtIdleWindow(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	const prefix = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
	srv := blockingSSEServer(t, "text/event-stream", prefix)

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.StreamProgressTimeout = 60 * time.Second
	began := time.Now()
	_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	elapsed := time.Since(began)
	var stalled *UpstreamStalledError
	if !errors.As(err, &stalled) {
		t.Fatalf("err = %v, want *UpstreamStalledError", err)
	}
	if stalled.Kind != stallKindIdle || stalled.Idle != 5*time.Second {
		t.Fatalf("stall kind=%q idle=%s, want %q/5s", stalled.Kind, stalled.Idle, stallKindIdle)
	}
	if elapsed >= 30*time.Second {
		t.Fatalf("mid-stream silence tripped after %s, want the 5s idle window (not the 60s progress window)", elapsed)
	}
}

// TestStreamStallSilentUpstreamTripsAtProgressWindow: an upstream that never sends a body
// byte still fails, at the content-progress window rather than hanging.
//
// fak-test:runtime slow est=8s lane=default
func TestStreamStallSilentUpstreamTripsAtProgressWindow(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	srv := blockingSSEServer(t, "text/event-stream", "")

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.StreamProgressTimeout = 8 * time.Second
	began := time.Now()
	_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	elapsed := time.Since(began)
	var stalled *UpstreamStalledError
	if !errors.As(err, &stalled) {
		t.Fatalf("err = %v, want *UpstreamStalledError", err)
	}
	if stalled.Idle != 8*time.Second || (stalled.Kind != stallKindIdle && stalled.Kind != stallKindNoProgress) {
		t.Fatalf("stall kind=%q idle=%s, want idle or no-progress at 8s", stalled.Kind, stalled.Idle)
	}
	if elapsed < 7*time.Second || elapsed >= 30*time.Second {
		t.Fatalf("silent upstream tripped after %s, want ~8s (the progress window)", elapsed)
	}
}
