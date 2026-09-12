package gateway

// responses_terminal_receipt_test.go — #11567. A terminal Responses failure that the
// transport observer CANNOT witness (upstream answered HTTP 200, then the completion
// failed to decode/semantically validate) must still leave EXACTLY ONE truthful terminal
// receipt at the gateway boundary, distinct from a provider HTTP status or a transport
// attempt. Before the fix, writeUpstreamErr incremented counters without ever emitting an
// UpstreamFailureObserver receipt, so a live guard showed 97 "other" upstream errors and a
// journal with no terminal receipt for them.
//
// Both tests below use ONLY APIs present before the fix, so they build on the parent and
// the first one genuinely fails there (red-then-green): on the parent it observes zero
// receipts for the post-200 decode failure.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// TestResponsesHTTP200InvalidCompletionEmitsTerminalReceipt drives a real loopback HTTP
// upstream that returns 200 with an unusable body (empty choices), through the real
// POST /v1/responses route (real HTTP planner, real gateway). It asserts the downstream
// turn fails as a 502 AND the gateway emits exactly one terminal receipt naming itself —
// not zero (the pre-fix gap) and not a transport/provider mislabel.
func TestResponsesHTTP200InvalidCompletionEmitsTerminalReceipt(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// HTTP 200, valid JSON, but ZERO choices: the planner decodes this to an
		// untyped "no choices" error AFTER the transport already saw a 200. The
		// transport observer emits nothing for this class; only a terminal receipt
		// at the gateway boundary can.
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":0,"total_tokens":11}}`))
	}))
	defer upstream.Close()

	var mu sync.Mutex
	var receipts []UpstreamFailureReceipt
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
		BaseURL:  upstream.URL,
		Provider: "openai-compatible",
		UpstreamFailureObserver: func(r UpstreamFailureReceipt) {
			mu.Lock()
			receipts = append(receipts, r)
			mu.Unlock()
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, err := json.Marshal(map[string]any{
		"model":  "test-model",
		"input":  "hello",
		"stream": false,
	})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/responses", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Session-Id", "sess-11567")
	req.Header.Set("X-Fak-Call-Id", "call-11567")
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)

	if httpResp.StatusCode != http.StatusBadGateway {
		t.Fatalf("downstream status = %d, want 502 (body: %s)", httpResp.StatusCode, respRaw)
	}

	mu.Lock()
	got := append([]UpstreamFailureReceipt(nil), receipts...)
	mu.Unlock()

	if len(got) != 1 {
		t.Fatalf("terminal receipts = %d, want exactly 1 (receipts: %+v)", len(got), got)
	}
	r := got[0]
	if r.EmittingLayer != "gateway" {
		t.Fatalf("emitting_layer = %q, want gateway (a 200-then-decode failure is a gateway-onset terminal failure, not provider/transport)", r.EmittingLayer)
	}
	if r.Outcome != "terminal" {
		t.Fatalf("outcome = %q, want terminal", r.Outcome)
	}
	if r.SessionID != "sess-11567" || r.CallID != "call-11567" {
		t.Fatalf("correlation lost: session=%q call=%q", r.SessionID, r.CallID)
	}
	if r.TraceID == "" {
		t.Fatal("terminal receipt must carry the request correlation trace")
	}
	if r.Method != http.MethodPost {
		t.Fatalf("method = %q, want POST", r.Method)
	}
	if r.PathClass != "/v1/responses" {
		t.Fatalf("path_class = %q, want /v1/responses", r.PathClass)
	}
}

// TestResponsesProviderStatusDoesNotEmitGatewayReceipt pins the discrimination: a provider
// HTTP status failure is witnessed by the TRANSPORT observer (provider layer), so the
// gateway terminal path must NOT relabel it as a "gateway" terminal failure. A 400 is
// non-retryable, so this stays fast.
func TestResponsesProviderStatusDoesNotEmitGatewayReceipt(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Openai-Request-Id", "prov-400")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad model"}}`))
	}))
	defer upstream.Close()

	var mu sync.Mutex
	var receipts []UpstreamFailureReceipt
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
		BaseURL:  upstream.URL,
		Provider: "openai-compatible",
		UpstreamFailureObserver: func(r UpstreamFailureReceipt) {
			mu.Lock()
			receipts = append(receipts, r)
			mu.Unlock()
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{"model": "test-model", "input": "hello", "stream": false})
	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()
	_, _ = io.ReadAll(httpResp.Body)

	mu.Lock()
	got := append([]UpstreamFailureReceipt(nil), receipts...)
	mu.Unlock()

	for _, r := range got {
		if r.EmittingLayer == "gateway" {
			t.Fatalf("provider HTTP 400 must not be relabeled a gateway terminal failure: %+v", r)
		}
	}
}
