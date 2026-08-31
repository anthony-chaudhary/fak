package engine

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestDefaultHTTPClientStreamingTransportDeadlines witnesses #3489: the fallback
// client keeps Client.Timeout at 0 for healthy streams while its private transport bounds
// connection setup and the response-header wait without mutating http.DefaultTransport.
func TestDefaultHTTPClientStreamingTransportDeadlines(t *testing.T) {
	defaultTransport, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		t.Fatalf("http.DefaultTransport must be an *http.Transport, got %T", http.DefaultTransport)
	}
	defaultTLSHandshakeTimeout := defaultTransport.TLSHandshakeTimeout
	defaultResponseHeaderTimeout := defaultTransport.ResponseHeaderTimeout

	c := defaultHTTPClient(nil)
	if c.Timeout != 0 {
		t.Fatalf("Client.Timeout must stay 0 for unbounded healthy streams, got %v", c.Timeout)
	}
	tr, ok := c.Transport.(*http.Transport)
	if !ok || tr == nil {
		t.Fatalf("fallback client must carry an *http.Transport, got %T", c.Transport)
	}
	if tr == defaultTransport {
		t.Fatal("fallback client must use a private clone, not http.DefaultTransport")
	}
	if tr.DialContext == nil || engineDialTimeout <= 0 || engineDialTimeout > time.Minute {
		t.Fatalf("Transport.DialContext must use a positive bounded dial timeout, got %v", engineDialTimeout)
	}
	if tr.TLSHandshakeTimeout != engineTLSHandshakeTimeout || tr.TLSHandshakeTimeout <= 0 || tr.TLSHandshakeTimeout > time.Minute {
		t.Fatalf("Transport.TLSHandshakeTimeout = %v, want positive bound %v", tr.TLSHandshakeTimeout, engineTLSHandshakeTimeout)
	}
	if tr.ResponseHeaderTimeout != engineResponseHeaderTimeout || tr.ResponseHeaderTimeout <= 0 || tr.ResponseHeaderTimeout > time.Minute {
		t.Fatalf("Transport.ResponseHeaderTimeout = %v, want positive bound %v", tr.ResponseHeaderTimeout, engineResponseHeaderTimeout)
	}
	if defaultTransport.TLSHandshakeTimeout != defaultTLSHandshakeTimeout ||
		defaultTransport.ResponseHeaderTimeout != defaultResponseHeaderTimeout {
		t.Fatal("defaultHTTPClient mutated http.DefaultTransport")
	}
}

// TestDefaultHTTPClientPassesInjectedClientThrough guards that an explicitly configured
// client and transport are returned unchanged; fallback deadlines never override callers.
func TestDefaultHTTPClientPassesInjectedClientThrough(t *testing.T) {
	transport := &http.Transport{ResponseHeaderTimeout: 3 * time.Second}
	injected := &http.Client{Timeout: 7 * time.Second, Transport: transport}

	got := defaultHTTPClient(injected)
	if got != injected {
		t.Fatal("an injected *http.Client must pass through unchanged")
	}
	if got.Timeout != 7*time.Second || got.Transport != transport || transport.ResponseHeaderTimeout != 3*time.Second {
		t.Fatalf("injected client was mutated: timeout=%v transport=%p header_timeout=%v", got.Timeout, got.Transport, transport.ResponseHeaderTimeout)
	}
}

// TestNewIdleTimeoutReaderDisabledPassThrough guards the opt-out branch: a non-positive
// idle window returns the underlying body unchanged (no timer armed), so a caller can
// disable the guard without a branch at the call site.
func TestNewIdleTimeoutReaderDisabledPassThrough(t *testing.T) {
	rc := io.NopCloser(nil)
	if got := newIdleTimeoutReader(rc, 0, func() {}); got != rc {
		t.Fatal("a non-positive idle window must return the body unchanged")
	}
}

// TestRunSSEPumpIdleReadDeadlineUnblocksStall witnesses the second DoD of #3476: an SSE
// stream that opens (200 + a keepalive line) then goes silent mid-stream is unblocked by
// the idle-read deadline instead of parking sc.Scan forever — so finish() returns with an
// error and the request (and kernel.Reap behind it) completes in bounded time.
func TestRunSSEPumpIdleReadDeadlineUnblocksStall(t *testing.T) {
	old := sseIdleTimeout
	sseIdleTimeout = 100 * time.Millisecond
	defer func() { sseIdleTimeout = old }()

	// Server: send one SSE keepalive comment, flush, then hold the response open until the
	// client aborts (which the idle deadline triggers). This is the mid-stream-stall vector.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		_, _ = io.WriteString(w, ": keepalive\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	cctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(cctx, http.MethodGet, srv.URL, nil)
	if err != nil {
		cancel()
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		cancel()
		t.Fatalf("do request: %v", err)
	}

	tokens := make(chan abi.EngineToken, 4)
	done := make(chan error, 1)
	go runSSEPump(cctx, resp.Body, cancel, tokens,
		func(_ *abi.Result, e error) { done <- e },
		func() *abi.Result { return &abi.Result{} },
		func(string) (string, error) { return "", nil })

	select {
	case e := <-done:
		if e == nil {
			t.Fatal("a stalled SSE stream must finish with an error, got nil")
		}
	case <-time.After(3 * time.Second):
		cancel()
		t.Fatal("runSSEPump did not unblock within 3s of an idle stall")
	}
}
