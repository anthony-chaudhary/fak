package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestWithMetricsRecoversHandlerPanic pins #2773: a panic in a downstream handler must be
// contained by withMetrics into a 500 + exactly one structured log line + a counted turn —
// never allowed to unwind into net/http (whose recovery dumps a goroutine stack to ErrorLog,
// i.e. the guarded child's TTY, #2772) and never dropping the request from /metrics (#2775).
func TestWithMetricsRecoversHandlerPanic(t *testing.T) {
	var logs []string
	s := &Server{
		metrics: newGatewayMetrics(time.Now()),
		logf:    func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) },
	}
	h := s.withMetrics(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom in handler")
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set(traceHeader, "gw-panic-1")
	rr := httptest.NewRecorder()

	// If the recover were missing this ServeHTTP would re-panic and fail the test outright.
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusInternalServerError)
	}
	panicLines := 0
	for _, l := range logs {
		if strings.Contains(l, "recovered handler panic") {
			panicLines++
			if !strings.Contains(l, "gw-panic-1") || !strings.Contains(l, "boom in handler") {
				t.Fatalf("panic line missing trace/value: %q", l)
			}
		}
		if strings.Contains(l, "goroutine ") {
			t.Fatalf("a goroutine stack reached logf (want a single line, no dump): %q", l)
		}
	}
	if panicLines != 1 {
		t.Fatalf("recovered-panic log lines = %d, want 1; logs=%v", panicLines, logs)
	}

	// The failed turn is counted as a 500 in the HTTP metric family.
	found := false
	s.metrics.mu.Lock()
	for k := range s.metrics.http {
		if k.status == "500" {
			found = true
		}
	}
	s.metrics.mu.Unlock()
	if !found {
		t.Fatal("no status=\"500\" counter recorded; the panicking turn was dropped from /metrics")
	}
}

// TestWithMetricsReRaisesErrAbortHandler pins the one panic withMetrics must NOT swallow:
// http.ErrAbortHandler is net/http's intentional silent-abort sentinel and has to propagate
// so the server aborts the response as designed rather than reporting a spurious 500.
func TestWithMetricsReRaisesErrAbortHandler(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now()), logf: func(string, ...any) {}}
	h := s.withMetrics(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic(http.ErrAbortHandler)
	}))

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set(traceHeader, "gw-panic-2")
	rr := httptest.NewRecorder()

	defer func() {
		if r := recover(); r != http.ErrAbortHandler {
			t.Fatalf("recover = %v, want http.ErrAbortHandler re-raised untouched", r)
		}
	}()
	h.ServeHTTP(rr, req)
	t.Fatal("ServeHTTP returned normally; want http.ErrAbortHandler to propagate")
}
