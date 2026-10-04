package gateway

import (
	"bufio"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestStreamResponseOutlivesServerWriteTimeout pins the queued-subagent fix: a server-wide
// WriteTimeout (90s for proxy backends) bounds the WHOLE response, so an SSE turn that
// waited behind siblings was cut mid-stream. Through the metrics recorder every route
// shares, a 200 text/event-stream response clears that deadline while a buffered one
// keeps it.
//
// fak-test:runtime fast est=1s
func TestStreamResponseOutlivesServerWriteTimeout(t *testing.T) {
	const timeout = 200 * time.Millisecond
	handler := func(contentType string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, rw := newStatusRecorder(w)
			rw.Header().Set("Content-Type", contentType)
			rw.WriteHeader(http.StatusOK)
			_, _ = rw.Write([]byte("data: first\n\n"))
			rw.(http.Flusher).Flush()
			time.Sleep(3 * timeout)
			_, _ = rw.Write([]byte("data: late\n\n"))
		})
	}
	read := func(contentType string) (string, error) {
		ts := httptest.NewUnstartedServer(handler(contentType))
		ts.Config.WriteTimeout = timeout
		ts.Start()
		defer ts.Close()
		resp, err := http.Get(ts.URL)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()
		var b strings.Builder
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			b.WriteString(sc.Text() + "\n")
		}
		return b.String(), sc.Err()
	}
	got, err := read("text/event-stream")
	if err != nil || !strings.Contains(got, "data: late") {
		t.Fatalf("SSE stream cut by the server write timeout: err=%v body=%q", err, got)
	}
	got, _ = read("application/json")
	if strings.Contains(got, "data: late") {
		t.Fatalf("buffered response escaped the server write timeout: %q", got)
	}
}
