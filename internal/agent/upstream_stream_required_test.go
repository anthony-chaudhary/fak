package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// TestCompleteSelfHealsStreamOnlyUpstream is the witness for the long-term transport
// negotiation fix (fak-private#1019): a buffered `Complete` against a STREAM-ONLY
// OpenAI-compatible upstream must self-heal by reissuing the SAME turn with stream:true,
// instead of blind-retrying the refusal (the witnessed hive-ai hang).
//
// The fake upstream is decisive: it 500s with a stream-required body for ANY request whose
// wire body does not set "stream":true, and answers a normal SSE stream when it does. The
// assertions pin the two facts the fix exists for:
//
//  1. the buffered refusal is NOT retried — exactly ONE non-stream request reaches the
//     upstream (the pre-fix behavior retried it on a backoff cadence until the budget drained);
//  2. the streaming path IS selected — the self-heal reissues with stream:true and the
//     turn completes with the streamed content.
func TestCompleteSelfHealsStreamOnlyUpstream(t *testing.T) {
	const sse = "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"healed via stream\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3,\"total_tokens\":7}}\n\n" +
		"data: [DONE]\n\n"

	var nonStreamHits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(raw, &req)
		if !req.Stream {
			atomic.AddInt32(&nonStreamHits, 1)
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"status_code":500,"message":"stream required"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, sse)
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete must self-heal the stream-only upstream, got: %v", err)
	}
	if comp.Message.Content != "healed via stream" {
		t.Fatalf("content = %q, want %q", comp.Message.Content, "healed via stream")
	}
	if comp.FinishReason != "stop" {
		t.Fatalf("finish = %q, want stop", comp.FinishReason)
	}
	if comp.Usage.CompletionTokens != 3 {
		t.Fatalf("usage = %+v, want completion 3 (usage from the terminal stream chunk)", comp.Usage)
	}
	// The load-bearing anti-hang assertion: the 500 was issued exactly once and never retried.
	if got := atomic.LoadInt32(&nonStreamHits); got != 1 {
		t.Fatalf("non-stream (500) requests = %d, want exactly 1 (the refusal must NOT be retried)", got)
	}
}

// TestCompleteStreamRequiredNotRetriedOnIncapableWire pins the guard: a non-streaming wire
// (Anthropic) has no streaming arm to heal into, so a stream-required 500 must surface as a
// classified status error, never a silent loop and never an ErrStreamingUnsupported leak.
func TestCompleteStreamRequiredNotRetriedOnIncapableWire(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"message":"stream required"}}`)
	}))
	t.Cleanup(srv.Close)

	p := &HTTPPlanner{Provider: ProviderAnthropic, BaseURL: srv.URL, Client: srv.Client(), ModelID: "m"}
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("a stream-required refusal on a non-streaming wire must surface, not loop")
	}
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("requests = %d, want 1 (no retry on an incapable wire)", got)
	}
}
