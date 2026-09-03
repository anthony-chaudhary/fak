package agent

// stream_maxduration_test.go — #10672: the bounded stream POLICY. The stall reader's
// idle and progress deadlines answer "is the socket alive?" and "is the turn advancing?";
// neither bounds a HEALTHY stream that legitimately runs for hours. The max-duration
// deadline is the third, ABSOLUTE bound: armed once when the stream opens, never re-armed
// by bytes or by progress, it ends a stream that outlives its configured total budget with
// the same typed error the other two use, so the gateway maps, logs, and receipts it
// identically. Default OFF: no existing stream behavior changes until an operator sets
// FAK_STREAM_MAX_DURATION_S.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// slowDripSSEServer is the HEALTHY long-stream upstream: after a role-opener it emits one
// content delta every gap FOREVER (until the client hangs up or the test releases it).
// Bytes arrive steadily, so the idle deadline can never fire; every delta is content, so
// each one re-arms the progress deadline. A max-duration bound is the only deadline that
// can end this turn — which is exactly what makes it the right fixture for the policy.
func slowDripSSEServer(t *testing.T, gap time.Duration) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			frame := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"d%d\"},\"finish_reason\":null}]}\n\n", i)
			if _, err := io.WriteString(w, frame); err != nil {
				return
			}
			if f != nil {
				f.Flush()
			}
			select {
			case <-r.Context().Done(): // the deadline tripped and closed the body
				return
			case <-release: // test over
				return
			case <-time.After(gap):
			}
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}

// slowDripAnthropicSSEServer is the Anthropic-wire twin: message_start + content_block_start,
// then one text_delta every gap forever. Same "healthy but unbounded" shape.
func slowDripAnthropicSSEServer(t *testing.T, gap time.Duration) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, anthropicWarmPrefix)
		if f != nil {
			f.Flush()
		}
		for i := 0; ; i++ {
			frame := fmt.Sprintf("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"d%d\"}}\n\n", i)
			if _, err := io.WriteString(w, frame); err != nil {
				return
			}
			if f != nil {
				f.Flush()
			}
			select {
			case <-r.Context().Done():
				return
			case <-release:
				return
			case <-time.After(gap):
			}
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}

// TestCompleteStreamMaxDurationBoundsAHealthyLongStream is the OpenAI-wire witness for the
// bounded stream policy: a stream that is ALIVE and ADVANCING — the one shape the idle and
// progress deadlines exist to protect — must still end when it outlives the operator's
// total-duration budget, with the typed stall error carrying the max-duration cause, and
// only after the budget actually elapsed (a policy that trips early would cut healthy turns).
func TestCompleteStreamMaxDurationBoundsAHealthyLongStream(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5") // idle deadline armed but drip-fed, never fires
	t.Setenv("FAK_STREAM_MAX_DURATION_S", "5")
	srv := slowDripSSEServer(t, time.Second)

	p := NewHTTPPlanner(srv.URL, "m", "")
	done := make(chan error, 1)
	start := time.Now()
	go func() {
		_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
			[]Message{{Role: RoleUser, Content: "hi"}}, nil)
		done <- err
	}()

	select {
	case err := <-done:
		var stalled *UpstreamStalledError
		if !errors.As(err, &stalled) {
			t.Fatalf("err = %v (%T), want *UpstreamStalledError — the max-duration bound never ended the stream", err, err)
		}
		if stalled.Kind != "max-duration" {
			t.Fatalf("Kind = %q, want \"max-duration\" — the turn was ended by the wrong deadline", stalled.Kind)
		}
		if !errors.Is(err, ErrUpstreamStalled) {
			t.Fatalf("errors.Is(err, ErrUpstreamStalled) = false, err = %v", err)
		}
		if elapsed := time.Since(start); elapsed < 5*time.Second {
			t.Fatalf("returned after %s, sooner than the 5s max-duration budget — the bound tripped early", elapsed)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("CompleteStream did not return within 30s — a healthy stream with no total-duration bound still rides the whole-request ceiling (#10672)")
	}
}

// TestStreamAnthropicRawMaxDurationBoundsAHealthyLongStream is the Anthropic-wire twin on
// the flagship passthrough wire: same healthy drip, same budget, same typed cause.
func TestStreamAnthropicRawMaxDurationBoundsAHealthyLongStream(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	t.Setenv("FAK_STREAM_MAX_DURATION_S", "5")
	srv := slowDripAnthropicSSEServer(t, time.Second)

	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "k")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	done := make(chan error, 1)
	start := time.Now()
	go func() {
		done <- p.StreamAnthropicRaw(context.Background(), rawBody, "k", "", func(AnthropicSSEEvent) error { return nil })
	}()

	select {
	case err := <-done:
		var stalled *UpstreamStalledError
		if !errors.As(err, &stalled) {
			t.Fatalf("err = %v (%T), want *UpstreamStalledError — the max-duration bound never ended the stream", err, err)
		}
		if stalled.Kind != "max-duration" {
			t.Fatalf("Kind = %q, want \"max-duration\"", stalled.Kind)
		}
		if elapsed := time.Since(start); elapsed < 5*time.Second {
			t.Fatalf("returned after %s, sooner than the 5s max-duration budget", elapsed)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("StreamAnthropicRaw did not return within 30s — no max-duration bound on the Anthropic wire (#10672)")
	}
}

// TestStreamMaxDurationDoesNotCutAStreamInsideItsBudget is the safety half of the policy:
// a stream that finishes WITHIN the budget is never touched, even though its total runtime
// outlasts a single idle window. Without this arm, a "fix" that caps every stream at the
// budget's FIRST idle window would pass the trip tests above while breaking healthy turns.
func TestStreamMaxDurationDoesNotCutAStreamInsideItsBudget(t *testing.T) {
	t.Setenv("FAK_STREAM_MAX_DURATION_S", "5")
	frames := []string{
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"c\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
		"data: [DONE]\n\n",
	}
	// 5 frames x 600ms = ~3s total: inside the 5s budget, past a single 60s-idle-window
	// comparison point. The turn must complete with full content, never touched by the bound.
	srv := steadySSEServer(t, "text/event-stream", frames, 600*time.Millisecond)

	p := NewHTTPPlanner(srv.URL, "m", "")
	comp, err := p.CompleteStream(context.Background(), func(string) error { return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("a stream that finished inside its max-duration budget was cut: %v", err)
	}
	if comp.Message.Content != "abc" {
		t.Fatalf("content = %q, want abc", comp.Message.Content)
	}
}
