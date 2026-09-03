package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestOTLPExporterCollectorPayloadAndDrain(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	var payloads []map[string]any
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p map[string]any
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			t.Error(err)
		}
		mu.Lock()
		paths = append(paths, r.URL.Path)
		payloads = append(payloads, p)
		mu.Unlock()
		w.WriteHeader(200)
	}))
	defer collector.Close()
	e, err := newOTLPExporter(collector.URL, 4, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	e.enqueue(otlpSpan{TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", SpanID: "00f067aa0ba902b7", Name: "POST /v1/chat/completions", Route: "/v1/chat/completions", Method: "POST", Status: 200, Start: time.Unix(1, 0), End: time.Unix(1, 100)})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := e.close(ctx); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(payloads) != 1 || paths[0] != "/v1/traces" {
		t.Fatalf("paths=%v payloads=%v", paths, payloads)
	}
	body, _ := json.Marshal(payloads[0])
	text := string(body)
	for _, want := range []string{"4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", "http.route", "/v1/chat/completions", "fak-gateway"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q: %s", want, text)
		}
	}
	st := e.stats()
	if st.Accepted != 1 || st.Exported != 1 || st.Failed != 0 {
		t.Fatalf("stats=%+v", st)
	}
}
func TestOTLPExporterUnavailableNeverBlocksAdmission(t *testing.T) {
	e, err := newOTLPExporter("http://127.0.0.1:1", 1, 20*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for i := 0; i < 100; i++ {
		e.enqueue(otlpSpan{})
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatal("enqueue blocked")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = e.close(ctx)
	st := e.stats()
	if st.Accepted < 1 || st.Dropped < 1 || st.Failed < 1 {
		t.Fatalf("stats=%+v", st)
	}
}
func TestOTLPExporterConfigFailClosed(t *testing.T) {
	for _, endpoint := range []string{"file:///tmp/x", "relative", "ftp://x"} {
		if _, err := newOTLPExporter(endpoint, 1, time.Second); err == nil {
			t.Fatalf("accepted %s", endpoint)
		}
	}
	if e, err := newOTLPExporter("", 256, time.Second); err != nil || e != nil {
		t.Fatalf("disabled=%v err=%v", e, err)
	}
}

func TestOTLPExporterConcurrentEnqueueAndCloseRaceSafe(t *testing.T) {
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))
	defer collector.Close()

	for iter := 0; iter < 50; iter++ {
		e, err := newOTLPExporter(collector.URL, 16, time.Second)
		if err != nil {
			t.Fatal(err)
		}

		var wg sync.WaitGroup
		// Launch concurrent enqueuers
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < 50; j++ {
					e.enqueue(otlpSpan{TraceID: "test-trace", SpanID: "test-span", Name: "GET /"})
				}
			}()
		}

		// Launch concurrent closers (asserting idempotency and race safety)
		for i := 0; i < 3; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				_ = e.close(ctx)
			}()
		}

		wg.Wait()
		// Late enqueue after close should drop cleanly, never panic on closed channel
		e.enqueue(otlpSpan{TraceID: "late-trace"})
	}
}
