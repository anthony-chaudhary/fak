package gateway

import (
	"errors"
	"net/http"
	"reflect"
	"testing"
)

type metricsCapabilityWriter struct {
	header http.Header
	codes  []int
	body   []byte
}

func (w *metricsCapabilityWriter) Header() http.Header  { return w.header }
func (w *metricsCapabilityWriter) WriteHeader(code int) { w.codes = append(w.codes, code) }
func (w *metricsCapabilityWriter) Write(p []byte) (int, error) {
	w.body = append(w.body, p...)
	return len(p), nil
}

type metricsFlushingWriter struct {
	*metricsCapabilityWriter
	flushes int
}

func (w *metricsFlushingWriter) Flush() { w.flushes++ }

type metricsErrorFlushingWriter struct {
	*metricsCapabilityWriter
	flushes int
	err     error
}

func (w *metricsErrorFlushingWriter) FlushError() error { w.flushes++; return w.err }

func TestMetricsWriterPreservesStreamingCapabilities(t *testing.T) {
	base := &metricsCapabilityWriter{header: make(http.Header)}
	rec, wrapped := newStatusRecorder(base)
	if _, ok := wrapped.(http.Flusher); ok {
		t.Fatal("metrics invented Flusher on an unsupported writer")
	}
	if err := http.NewResponseController(wrapped).Flush(); !errors.Is(err, http.ErrNotSupported) {
		t.Fatalf("unsupported Flush=%v", err)
	}
	wrapped.WriteHeader(http.StatusEarlyHints)
	if rec.status != 0 {
		t.Fatalf("informational response recorded as final status %d", rec.status)
	}
	wrapped.WriteHeader(http.StatusAccepted)
	wrapped.WriteHeader(http.StatusBadRequest)
	if _, err := wrapped.Write([]byte("ok")); err != nil {
		t.Fatal(err)
	}
	if rec.status != http.StatusAccepted || rec.bytes != 2 || !reflect.DeepEqual(base.codes, []int{103, 202}) {
		t.Fatalf("recorder status=%d bytes=%d forwarded codes=%v", rec.status, rec.bytes, base.codes)
	}

	flushing := &metricsFlushingWriter{metricsCapabilityWriter: &metricsCapabilityWriter{header: make(http.Header)}}
	rec, wrapped = newStatusRecorder(flushing)
	f, ok := wrapped.(http.Flusher)
	if !ok {
		t.Fatal("metrics removed real Flusher")
	}
	f.Flush()
	if flushing.flushes != 1 || rec.status != http.StatusOK || !reflect.DeepEqual(flushing.codes, []int{200}) {
		t.Fatalf("flushes=%d status=%d codes=%v", flushing.flushes, rec.status, flushing.codes)
	}

	wantErr := errors.New("flush transport failed")
	errorFlushing := &metricsErrorFlushingWriter{metricsCapabilityWriter: &metricsCapabilityWriter{header: make(http.Header)}, err: wantErr}
	rec, wrapped = newStatusRecorder(errorFlushing)
	if _, ok := wrapped.(http.Flusher); ok {
		t.Fatal("FlushError-only writer was promoted to Flusher")
	}
	if err := http.NewResponseController(wrapped).Flush(); !errors.Is(err, wantErr) {
		t.Fatalf("transport flush error lost: %v", err)
	}
	if errorFlushing.flushes != 1 || rec.status != http.StatusOK {
		t.Fatalf("FlushError accounting flushes=%d status=%d", errorFlushing.flushes, rec.status)
	}
}
