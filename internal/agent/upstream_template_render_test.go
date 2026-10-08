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

// llamacppTemplate500 is the body llama.cpp's server returns when its Jinja chat template
// rejects the message shape (witnessed 2026-10-07 on strix3 behind `fak serve --provider openai`).
const llamacppTemplate500 = `{"error":{"code":500,"message":"Error: Jinja Exception: System message must be at the beginning.","type":"server_error"}}`

func TestClassifyUpstreamTemplateRenderFailureIsTerminal(t *testing.T) {
	for _, body := range []string{
		llamacppTemplate500,
		`{"error":{"code":500,"message":"raise_exception: Conversation roles must alternate user/assistant","type":"server_error"}}`,
	} {
		if got := classifyUpstream(http.StatusInternalServerError, []byte(body), http.Header{}); got != RemedyTerminal {
			t.Errorf("classifyUpstream(500, %s) = %v, want %v", body, got, RemedyTerminal)
		}
	}
	if got := classifyUpstream(http.StatusInternalServerError,
		[]byte(`{"error":{"code":500,"message":"Internal Server Error","type":"server_error"}}`), http.Header{}); got != RemedyBackoff {
		t.Fatalf("generic 500 = %v, want %v", got, RemedyBackoff)
	}
}

func TestPlannerTemplateRender500NotRetried(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func(ctx context.Context, p *HTTPPlanner) error
	}{
		{"buffered", func(ctx context.Context, p *HTTPPlanner) error {
			_, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "hi"}}, nil)
			return err
		}},
		{"stream", func(ctx context.Context, p *HTTPPlanner) error {
			_, err := p.CompleteStream(ctx, nil, []Message{{Role: RoleUser, Content: "hi"}}, nil)
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hits int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&hits, 1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, llamacppTemplate500)
			}))
			t.Cleanup(srv.Close)

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			err := tc.call(ctx, NewHTTPPlanner(srv.URL, "m", ""))

			var se *UpstreamStatusError
			if !errors.As(err, &se) {
				t.Fatalf("err = %v, want *UpstreamStatusError", err)
			}
			if se.Status != http.StatusInternalServerError {
				t.Fatalf("status = %d, want 500", se.Status)
			}
			if got := classifyUpstream(se.Status, []byte(se.Body), nil); got != RemedyTerminal {
				t.Fatalf("surfaced error remedy = %v, want %v", got, RemedyTerminal)
			}
			if got := atomic.LoadInt32(&hits); got != 1 {
				t.Fatalf("upstream attempts = %d, want exactly 1", got)
			}
		})
	}
}
