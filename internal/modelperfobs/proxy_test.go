package modelperfobs

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"time"
)

type failingRequestBody struct{ err error }

func (b failingRequestBody) Read([]byte) (int, error) { return 0, b.err }
func (failingRequestBody) Close() error               { return nil }

func TestProxyRecordsEarlyFailureInboundBodyRead(t *testing.T) {
	testProxyRecordsEarlyFailure(t, http.StatusBadRequest, func() *http.Request {
		r := httptest.NewRequest(http.MethodPost, "http://proxy.test/v1/chat/completions", nil)
		r.Body = failingRequestBody{err: errors.New("forced inbound body read failure")}
		return r
	})
}

func TestProxyRecordsEarlyFailureOutboundRequestConstruction(t *testing.T) {
	testProxyRecordsEarlyFailure(t, http.StatusBadGateway, func() *http.Request {
		r := httptest.NewRequest(http.MethodPost, "http://proxy.test/v1/chat/completions", strings.NewReader(`{"model":"test"}`))
		r.Method = "invalid method"
		return r
	})
}

func testProxyRecordsEarlyFailure(t *testing.T, wantStatus int, request func() *http.Request) {
	t.Helper()
	backend, err := ParseBackend("http://backend.test")
	if err != nil {
		t.Fatal(err)
	}
	ledger := t.TempDir() + "/observations.jsonl"
	proxy := &Proxy{Backend: backend, Ledger: ledger}
	w := httptest.NewRecorder()

	proxy.ServeHTTP(w, request())

	if w.Code != wantStatus {
		t.Fatalf("response status=%d, want %d; body=%q", w.Code, wantStatus, w.Body.String())
	}
	f, err := os.Open(ledger)
	if err != nil {
		t.Fatalf("early failure did not create observation ledger: %v", err)
	}
	rows, err := ReadObservations(bufio.NewReader(f))
	closeErr := f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if len(rows) != 1 {
		t.Fatalf("ledger rows=%d, want exactly 1: %+v", len(rows), rows)
	}
	got := rows[0]
	if got.RequestID == "" || got.CompletedAt.IsZero() || got.Error == "" {
		t.Fatalf("early-failure observation lacks id/completion/error: %+v", got)
	}
	proxy.mu.Lock()
	defer proxy.mu.Unlock()
	if len(proxy.active) != 0 || len(proxy.overlaps) != 0 {
		t.Fatalf("active request state leaked: active=%v overlaps=%v", proxy.active, proxy.overlaps)
	}
}

func TestProxyCapturesStreamingTimingAndUsage(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Fak-Observation-ID"); got == "" {
			t.Error("missing correlation header")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, line := range []string{
			`data: {"choices":[{"delta":{"content":"a"}}]}`,
			`data: {"choices":[{"delta":{"content":"b"}}]}`,
			`data: {"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`,
			`data: [DONE]`,
		} {
			_, _ = w.Write([]byte(line + "\n"))
			f.Flush()
		}
	}))
	defer backend.Close()
	u, _ := ParseBackend(backend.URL)
	ledger := t.TempDir() + "/observations.jsonl"
	times := []time.Time{time.Unix(0, 0), time.Unix(0, 100e6), time.Unix(0, 140e6), time.Unix(0, 200e6), time.Unix(0, 260e6), time.Unix(0, 300e6)}
	i := 0
	proxy := httptest.NewServer(&Proxy{Backend: u, Ledger: ledger, Now: func() time.Time { v := times[i]; i++; return v }})
	defer proxy.Close()

	resp, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"Qwen3.8-27B","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.Header.Get("X-Fak-Observation-ID") == "" {
		t.Fatal("response lacks observation ID")
	}

	f, err := os.Open(ledger)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := ReadObservations(bufio.NewReader(f))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows=%d", len(rows))
	}
	got := rows[0]
	if got.Model != "Qwen3.8-27B" || got.PromptTokens != 10 || got.CompletionTokens != 2 {
		t.Fatalf("observation=%+v", got)
	}
	if got.TTFTMS <= 0 || got.InterChunkCount != 1 || got.TPOTMS <= 0 || got.OutputTokensPerSec <= 0 {
		t.Fatalf("timing=%+v", got)
	}
}

func TestFallbackProxyClientBoundsHeadersWithoutCappingStreams(t *testing.T) {
	if fallbackProxyClient.Timeout != 0 {
		t.Fatalf("fallback client timeout=%v, want no whole-stream deadline", fallbackProxyClient.Timeout)
	}
	transport, ok := fallbackProxyClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("fallback transport=%T, want *http.Transport", fallbackProxyClient.Transport)
	}
	if transport.DialContext == nil || transport.TLSHandshakeTimeout <= 0 || transport.ResponseHeaderTimeout != proxyResponseHeaderTimeout {
		t.Fatalf("fallback transport lacks bounded connect/header deadlines: %+v", transport)
	}
}

func TestProxyRecordsConcurrentObservedRequests(t *testing.T) {
	arrived := make(chan string, 2)
	release := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		arrived <- r.Header.Get("X-Fak-Observation-ID")
		<-release
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer backend.Close()
	u, _ := ParseBackend(backend.URL)
	ledger := t.TempDir() + "/observations.jsonl"
	proxy := httptest.NewServer(&Proxy{Backend: u, Ledger: ledger})
	defer proxy.Close()

	errs := make(chan error, 2)
	for range 2 {
		go func() {
			resp, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test"}`))
			if err == nil {
				_, _ = io.Copy(io.Discard, resp.Body)
				err = resp.Body.Close()
			}
			errs <- err
		}()
	}
	for range 2 {
		select {
		case id := <-arrived:
			if id == "" {
				t.Fatal("backend request lacks observation ID")
			}
		case <-time.After(3 * time.Second):
			t.Fatal("requests did not overlap at backend")
		}
	}
	close(release)
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}

	f, err := os.Open(ledger)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := ReadObservations(bufio.NewReader(f))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows=%d, want 2", len(rows))
	}
	for i, row := range rows {
		peer := rows[1-i].RequestID
		if row.CompletedAt.IsZero() || row.OverlappingObservedRequests != 1 || !slices.Contains(row.OverlappingRequestIDs, peer) {
			t.Fatalf("row %s overlap provenance=%+v, want peer %s", row.RequestID, row, peer)
		}
	}
}

func TestSummarizeNamesPrefillOrQueueBottleneck(t *testing.T) {
	rows := []Observation{
		{Schema: Schema, Status: 200, PromptTokens: 4000, CompletionTokens: 20, DurationMS: 2500, TTFTMS: 1800, TPOTMS: 36, OutputTokensPerSec: 27},
		{Schema: Schema, Status: 200, PromptTokens: 4200, CompletionTokens: 20, DurationMS: 2700, TTFTMS: 2000, TPOTMS: 37, OutputTokensPerSec: 26},
	}
	s := Summarize(rows)
	if s.LikelyBottleneck != "prefill-or-queue" {
		b, _ := json.Marshal(s)
		t.Fatalf("summary=%s", b)
	}
	var b strings.Builder
	if err := WriteMarkdown(&b, s); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "Likely bottleneck: **prefill-or-queue**") {
		t.Fatal(b.String())
	}
}
