package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestMetricsWrapperExportsJoinedOTLPSpan(t *testing.T) {
	payload := make(chan string, 1)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p any
		_ = json.NewDecoder(r.Body).Decode(&p)
		b, _ := json.Marshal(p)
		payload <- string(b)
		w.WriteHeader(200)
	}))
	defer collector.Close()
	e, err := newOTLPExporter(collector.URL, 4, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{metrics: newGatewayMetrics(time.Now()), otlp: e}
	h := s.withMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) }))
	req := httptest.NewRequest("GET", "/healthz", nil)
	req.Header.Set(traceparentHeader, "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := e.close(ctx); err != nil {
		t.Fatal(err)
	}
	var body string
	select {
	case body = <-payload:
	case <-time.After(time.Second):
		t.Fatalf("no payload stats=%+v header=%s", e.stats(), rec.Header().Get(traceparentHeader))
	}
	tp, _ := parseTraceparent(rec.Header().Get(traceparentHeader))
	if !strings.Contains(body, tp.TraceID) || !strings.Contains(body, tp.ParentID) || !strings.Contains(body, "/healthz") {
		t.Fatalf("tp=%+v body=%s", tp, body)
	}
	stats := e.stats()
	if stats.Accepted != 1 || stats.Exported != 1 || stats.QueueDepth != 0 {
		t.Fatalf("stats=%+v", stats)
	}
}
