package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestComplete_Transient502RecoversOnAlternateTarget(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")

	var mu sync.Mutex
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		mu.Lock()
		auths = append(auths, auth)
		mu.Unlock()
		if auth != "Bearer alternate" {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(w, `{"error":"temporary"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_1","model":"m","choices":[{"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "primary")
	var callbacks int32
	p.TransientTargetFunc = func(status int) (string, bool) {
		atomic.AddInt32(&callbacks, 1)
		if status != http.StatusBadGateway {
			t.Fatalf("transient status = %d, want 502", status)
		}
		return "alternate", true
	}

	got, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Message.Content != "recovered" {
		t.Fatalf("content = %q, want recovered", got.Message.Content)
	}
	if n := atomic.LoadInt32(&callbacks); n != 1 {
		t.Fatalf("TransientTargetFunc calls = %d, want 1", n)
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"Bearer primary", "Bearer primary", "Bearer alternate"}
	if strings.Join(auths, ",") != strings.Join(want, ",") {
		t.Fatalf("authorization sequence = %q, want %q", auths, want)
	}
}

func TestCompleteStream_429DoesNotAdvanceTransientTarget(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")

	var hits int32
	var authsMu sync.Mutex
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authsMu.Lock()
		auths = append(auths, r.Header.Get("Authorization"))
		authsMu.Unlock()
		if atomic.AddInt32(&hits, 1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":"throttled"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "primary")
	var callbacks int32
	p.TransientTargetFunc = func(status int) (string, bool) {
		atomic.AddInt32(&callbacks, 1)
		return "alternate", true
	}
	var streamed strings.Builder
	_, err := p.CompleteStream(context.Background(), func(delta string) error {
		streamed.WriteString(delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if streamed.String() != "ok" {
		t.Fatalf("streamed = %q, want ok", streamed.String())
	}
	if n := atomic.LoadInt32(&callbacks); n != 0 {
		t.Fatalf("TransientTargetFunc calls = %d, want 0 for 429", n)
	}
	authsMu.Lock()
	defer authsMu.Unlock()
	if len(auths) != 2 || auths[0] != "Bearer primary" || auths[1] != "Bearer primary" {
		t.Fatalf("authorization sequence = %q, want primary retained", auths)
	}
}

func TestStreamAnthropicRaw_Transient502NoAlternatePreservesTypedStatus(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")

	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"temporary"}}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewProviderHTTPPlanner("anthropic", srv.URL, "claude-test", "primary")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	var callbacks int32
	p.TransientTargetFunc = func(status int) (string, bool) {
		atomic.AddInt32(&callbacks, 1)
		return "", false
	}
	rawBody := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	err = p.StreamAnthropicRaw(context.Background(), rawBody, "primary", "", func(AnthropicSSEEvent) error { return nil })
	if err == nil {
		t.Fatal("StreamAnthropicRaw unexpectedly succeeded")
	}
	var statusErr *UpstreamStatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("error type = %T (%v), want *UpstreamStatusError", err, err)
	}
	if statusErr.Status != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", statusErr.Status)
	}
	if n := atomic.LoadInt32(&hits); n != 2 {
		t.Fatalf("upstream hits = %d, want 2 (initial + one quick same-target retry)", n)
	}
	if n := atomic.LoadInt32(&callbacks); n != 1 {
		t.Fatalf("TransientTargetFunc calls = %d, want 1", n)
	}
}
