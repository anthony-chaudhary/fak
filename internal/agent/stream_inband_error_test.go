package agent

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const inBandTruncatedFrame = "data: {\"error\":{\"type\":\"upstream_truncated\",\"code\":\"upstream_truncated\",\"message\":\"x\"}}\n\n"

func countingSSEServer(t *testing.T, body string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

// TestCompleteStreamFailsOnInBandRouterErrorFrame drives a router that streams one
// content chunk, then an in-band error frame, then [DONE]: the turn must fail, keep the
// fragment the sink already saw, and not re-send the request.
//
// fak-test:runtime fast est=50ms
func TestCompleteStreamFailsOnInBandRouterErrorFrame(t *testing.T) {
	for name, frame := range map[string]string{
		"string_code":  inBandTruncatedFrame,
		"numeric_code": "data: {\"error\":{\"type\":\"server_error\",\"code\":502,\"message\":\"x\"}}\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			body := "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n" +
				frame + "data: [DONE]\n\n"
			srv, hits := countingSSEServer(t, body)
			p := NewHTTPPlanner(srv.URL, "m", "")
			var got []string
			sink := func(frag string) error { got = append(got, frag); return nil }
			comp, err := p.CompleteStream(context.Background(), sink, []Message{{Role: RoleUser, Content: "hi"}}, nil)
			if err == nil {
				t.Fatalf("CompleteStream returned nil error (finish=%q) for an in-band error frame", comp.FinishReason)
			}
			if comp != nil {
				t.Fatalf("CompleteStream returned a completion alongside the error")
			}
			if strings.Join(got, "|") != "partial" {
				t.Fatalf("sink fragments = %q, want [partial]", got)
			}
			if n := hits.Load(); n != 1 {
				t.Fatalf("upstream hits = %d, want 1 (no retry after bytes reached the sink)", n)
			}
		})
	}
}

// TestCompleteStreamWithoutErrorFrameStillFinishesStop keeps the clean path unchanged.
//
// fak-test:runtime fast est=50ms
func TestCompleteStreamWithoutErrorFrameStillFinishesStop(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"error\":null}\n\n" +
		"data: [DONE]\n\n"
	srv, _ := countingSSEServer(t, body)
	p := NewHTTPPlanner(srv.URL, "m", "")
	comp, err := p.CompleteStream(context.Background(), nil, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if comp.Message.Content != "ok" || comp.FinishReason != "stop" {
		t.Fatalf("completion = (%q, %q), want (ok, stop)", comp.Message.Content, comp.FinishReason)
	}
}
