package gateway

import (
	"net/http"
	"sync"
)

// syncResponseWriter wraps http.ResponseWriter and http.Flusher with a mutex
// so concurrent writes, flushes, and header operations are synchronized.
type syncResponseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	mu      sync.Mutex
}

var (
	_ http.ResponseWriter = (*syncResponseWriter)(nil)
	_ http.Flusher        = (*syncResponseWriter)(nil)
)

// newSyncResponseWriter wraps w in a *syncResponseWriter. If w is already
// a *syncResponseWriter, it is returned as-is (idempotent).
func newSyncResponseWriter(w http.ResponseWriter) *syncResponseWriter {
	if sw, ok := w.(*syncResponseWriter); ok {
		return sw
	}
	flusher, _ := w.(http.Flusher)
	return &syncResponseWriter{
		w:       w,
		flusher: flusher,
	}
}

func (s *syncResponseWriter) Header() http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Header()
}

func (s *syncResponseWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(p)
}

func (s *syncResponseWriter) WriteHeader(statusCode int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.w.WriteHeader(statusCode)
}

func (s *syncResponseWriter) Flush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

func (s *syncResponseWriter) Unwrap() http.ResponseWriter {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w
}
