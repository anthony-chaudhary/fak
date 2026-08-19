package gateway

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestParseTraceparentStrictV00(t *testing.T) {
	valid := "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	got, err := parseTraceparent(valid)
	if err != nil || got.TraceID != "4bf92f3577b34da6a3ce929d0e0e4736" || got.Flags != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	bad := []string{"", "00-xyz-00f067aa0ba902b7-01", "00-00000000000000000000000000000000-00f067aa0ba902b7-01", "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra", "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
	for _, raw := range bad {
		if _, err := parseTraceparent(raw); err == nil {
			t.Fatalf("accepted %q", raw)
		}
	}
}
func TestMetricsWrapperPropagatesW3CAndLegacyTrace(t *testing.T) {
	srv := &Server{metrics: newGatewayMetrics(time.Now())}
	h := srv.withMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requestTraceID(r) == "" {
			t.Fatal("handler lacks trace")
		}
		w.WriteHeader(204)
	}))
	req := httptest.NewRequest("GET", "/healthz", nil)
	req.Header.Set(traceparentHeader, "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	tp, err := parseTraceparent(rec.Header().Get(traceparentHeader))
	if err != nil {
		t.Fatal(err)
	}
	if tp.TraceID != "4bf92f3577b34da6a3ce929d0e0e4736" || tp.ParentID == "00f067aa0ba902b7" || rec.Header().Get(traceHeader) != tp.TraceID {
		t.Fatalf("headers=%v", rec.Header())
	}
	req = httptest.NewRequest("GET", "/healthz", nil)
	req.Header.Set(traceparentHeader, "malformed-secret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Header().Get(traceparentHeader) == "malformed-secret" {
		t.Fatal("reflected malformed context")
	}
	if _, err := parseTraceparent(rec.Header().Get(traceparentHeader)); err != nil {
		t.Fatal(err)
	}
}
func TestTraceContextConcurrentUniqueness(t *testing.T) {
	const n = 100
	ids := map[string]bool{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := newTraceContext("", 0)
			mu.Lock()
			if ids[c.TraceID+":"+c.ParentID] {
				t.Errorf("duplicate context")
			}
			ids[c.TraceID+":"+c.ParentID] = true
			mu.Unlock()
		}()
	}
	wg.Wait()
	if len(ids) != n {
		t.Fatalf("unique=%d", len(ids))
	}
}
