package gateway

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

type unsafeWriter struct {
	header  http.Header
	buf     bytes.Buffer
	flushes int
	status  int
}

func (u *unsafeWriter) Header() http.Header {
	return u.header
}

func (u *unsafeWriter) Write(p []byte) (int, error) {
	return u.buf.Write(p)
}

func (u *unsafeWriter) WriteHeader(code int) {
	u.status = code
}

func (u *unsafeWriter) Flush() {
	u.flushes++
}

func TestSyncResponseWriter_BasicAndIdempotence(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := newSyncResponseWriter(rec)
	if sw == nil {
		t.Fatal("expected non-nil syncResponseWriter")
	}
	if got := sw.Unwrap(); got != rec {
		t.Fatalf("Unwrap = %v, want %v", got, rec)
	}
	// Idempotence check: wrapping an existing syncResponseWriter returns it as-is.
	sw2 := newSyncResponseWriter(sw)
	if sw2 != sw {
		t.Fatalf("newSyncResponseWriter(sw) returned %v, want %v", sw2, sw)
	}
	sw.Header().Set("X-Test", "val")
	sw.WriteHeader(http.StatusAccepted)
	n, err := sw.Write([]byte("hello"))
	if err != nil || n != 5 {
		t.Fatalf("Write: n=%d, err=%v", n, err)
	}
	sw.Flush()
	if rec.Code != http.StatusAccepted {
		t.Fatalf("Code = %d, want %d", rec.Code, http.StatusAccepted)
	}
	if rec.Body.String() != "hello" {
		t.Fatalf("Body = %q, want hello", rec.Body.String())
	}
	if rec.Header().Get("X-Test") != "val" {
		t.Fatalf("Header = %q, want val", rec.Header().Get("X-Test"))
	}
}

func TestSyncResponseWriter_ConcurrentWritesAndFlushes(t *testing.T) {
	raw := &unsafeWriter{
		header: make(http.Header),
	}
	sw := newSyncResponseWriter(raw)

	const numGoroutines = 64
	const iterationsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			msg := fmt.Sprintf("chunk-%02d;", id)
			for j := 0; j < iterationsPerGoroutine; j++ {
				_, _ = sw.Write([]byte(msg))
				if j%5 == 0 {
					sw.Flush()
				}
			}
		}(i)
	}

	wg.Wait()

	expectedBytes := numGoroutines * iterationsPerGoroutine * len(fmt.Sprintf("chunk-%02d;", 0))
	if raw.buf.Len() != expectedBytes {
		t.Fatalf("total bytes = %d, want %d", raw.buf.Len(), expectedBytes)
	}
	if raw.flushes == 0 {
		t.Fatal("expected flushes > 0")
	}
}
