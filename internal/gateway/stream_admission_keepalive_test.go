package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func serveQueuedStreamGateway(t *testing.T, upstreamURL string, ctl *AdmissionController) string {
	t.Helper()
	srv := newQueuedStreamGateway(t, upstreamURL, ctl)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); _ = srv.Serve(ctx, ln) }()
	t.Cleanup(func() { cancel(); <-done })
	return "http://" + ln.Addr().String()
}

func newQueuedStreamGateway(t *testing.T, upstreamURL string, ctl *AdmissionController) *Server {
	t.Helper()
	t.Setenv("FAK_STREAM_PREFILL_KEEPALIVE_S", "")
	t.Setenv("FAK_STREAM_HEARTBEAT_S", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstreamURL + "/compat", Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	srv.SetAdmissionController(ctl)
	t.Cleanup(srv.Close)
	return srv
}

func queuedChatBody(stream bool) []byte {
	b, _ := json.Marshal(map[string]any{
		"model": "x:model", "stream": stream,
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
	})
	return b
}

func queuedStreamUpstream(t *testing.T, hit chan<- struct{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case hit <- struct{}{}:
		default:
		}
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
}

func waitAdmissionWaiting(t *testing.T, ctl *AdmissionController, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if ctl.Stats().Waiting == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("waiting=%d, want %d", ctl.Stats().Waiting, want)
}

type failingQueuedStreamWriter struct {
	header  http.Header
	written chan struct{}
}

type signalingQueuedStreamWriter struct {
	*httptest.ResponseRecorder
	written chan struct{}
}

func (w *signalingQueuedStreamWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseRecorder.Write(p)
	select {
	case w.written <- struct{}{}:
	default:
	}
	return n, err
}

func (w *failingQueuedStreamWriter) Header() http.Header { return w.header }
func (w *failingQueuedStreamWriter) WriteHeader(int)     {}
func (w *failingQueuedStreamWriter) Flush()              {}
func (w *failingQueuedStreamWriter) Write([]byte) (int, error) {
	select {
	case w.written <- struct{}{}:
	default:
	}
	return 0, errors.New("client write failed")
}

// fak-test:runtime medium est=1000ms lane=default
func TestQueuedChatStreamHeartbeatAfterGrace(t *testing.T) {
	previousCommit, previousEvery := prefillCommitAfter, prefillKeepaliveEvery
	prefillCommitAfter, prefillKeepaliveEvery = 20*time.Millisecond, 20*time.Millisecond
	t.Cleanup(func() { prefillCommitAfter, prefillKeepaliveEvery = previousCommit, previousEvery })

	ctl := NewAdmissionController(AdmissionPolicy{TokenBudget: 4096, MaxNumSeqs: 1, MaxWaiting: 4, AgingRounds: 1})
	blocker, err := ctl.Acquire(context.Background(), SeqRequest{TraceID: "blocker", Tokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(blocker.Release)
	hit := make(chan struct{}, 1)
	up := queuedStreamUpstream(t, hit)
	defer up.Close()

	resp, err := http.Post(serveQueuedStreamGateway(t, up.URL, ctl)+"/v1/chat/completions", "application/json", bytes.NewReader(queuedChatBody(true)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("queued response = %d %q, want 200 text/event-stream", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	scanner := bufio.NewScanner(resp.Body)
	if !scanner.Scan() || strings.TrimSpace(scanner.Text()) != ": keepalive" {
		t.Fatalf("first queued frame = %q, want keepalive", scanner.Text())
	}
	select {
	case <-hit:
		t.Fatal("upstream called before queued request was admitted")
	default:
	}
	blocker.Release()
	rest, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(rest, []byte("data: [DONE]")) {
		t.Fatalf("stream did not complete after admission: %s", rest)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestQueuedChatStreamImmediateRefusalRemainsHTTP(t *testing.T) {
	hit := make(chan struct{}, 1)
	up := queuedStreamUpstream(t, hit)
	defer up.Close()
	for _, stream := range []bool{true, false} {
		t.Run(map[bool]string{true: "stream", false: "buffered"}[stream], func(t *testing.T) {
			ctl := NewAdmissionController(AdmissionPolicy{TokenBudget: 1, MaxNumSeqs: 1, MaxWaiting: 4, AgingRounds: 1})
			resp, err := http.Post(serveQueuedStreamGateway(t, up.URL, ctl)+"/v1/chat/completions", "application/json", bytes.NewReader(queuedChatBody(stream)))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status=%d, want 400", resp.StatusCode)
			}
			if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
				t.Fatalf("immediate refusal committed SSE: %q", resp.Header.Get("Content-Type"))
			}
		})
	}
}

// fak-test:runtime medium est=1000ms lane=default
func TestQueuedChatStreamDisconnectCancelsAdmission(t *testing.T) {
	previousCommit, previousEvery := prefillCommitAfter, prefillKeepaliveEvery
	prefillCommitAfter, prefillKeepaliveEvery = 20*time.Millisecond, 20*time.Millisecond
	t.Cleanup(func() { prefillCommitAfter, prefillKeepaliveEvery = previousCommit, previousEvery })

	newFixture := func(t *testing.T) (*AdmissionController, *AdmissionLease, *httptest.Server, chan struct{}) {
		t.Helper()
		ctl := NewAdmissionController(AdmissionPolicy{TokenBudget: 4096, MaxNumSeqs: 1, MaxWaiting: 4, AgingRounds: 1})
		blocker, err := ctl.Acquire(context.Background(), SeqRequest{TraceID: "blocker", Tokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(blocker.Release)
		hit := make(chan struct{}, 1)
		up := queuedStreamUpstream(t, hit)
		t.Cleanup(up.Close)
		return ctl, blocker, up, hit
	}

	t.Run("client_context", func(t *testing.T) {
		ctl, _, up, hit := newFixture(t)
		base := serveQueuedStreamGateway(t, up.URL, ctl)
		reqCtx, cancel := context.WithCancel(context.Background())
		req, _ := http.NewRequestWithContext(reqCtx, http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(queuedChatBody(true)))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		waitAdmissionWaiting(t, ctl, 1)
		cancel()
		_ = resp.Body.Close()
		waitAdmissionWaiting(t, ctl, 0)
		assertUpstreamNotHit(t, hit)
	})

	t.Run("writer_failure", func(t *testing.T) {
		ctl, _, up, hit := newFixture(t)
		srv := newQueuedStreamGateway(t, up.URL, ctl)
		writer := &failingQueuedStreamWriter{header: make(http.Header), written: make(chan struct{}, 1)}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(queuedChatBody(true)))
		req.Header.Set("Content-Type", "application/json")
		done := make(chan struct{})
		go func() { defer close(done); srv.Handler().ServeHTTP(writer, req) }()
		select {
		case <-writer.written:
		case <-time.After(time.Second):
			t.Fatal("queued keepalive never attempted its write")
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("handler outlived queued keepalive write failure")
		}
		waitAdmissionWaiting(t, ctl, 0)
		assertUpstreamNotHit(t, hit)
	})

	t.Run("post_commit_error_is_terminal_sse", func(t *testing.T) {
		ctl, _, up, hit := newFixture(t)
		srv := newQueuedStreamGateway(t, up.URL, ctl)
		recorder := &signalingQueuedStreamWriter{ResponseRecorder: httptest.NewRecorder(), written: make(chan struct{}, 1)}
		reqCtx, cancel := context.WithCancel(context.Background())
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(queuedChatBody(true))).WithContext(reqCtx)
		req.Header.Set("Content-Type", "application/json")
		done := make(chan struct{})
		go func() { defer close(done); srv.Handler().ServeHTTP(recorder, req) }()
		select {
		case <-recorder.written:
		case <-time.After(time.Second):
			t.Fatal("queued stream was not committed")
		}
		cancel()
		<-done
		body := recorder.ResponseRecorder.Body.String()
		if !strings.Contains(body, `"error"`) || !strings.Contains(body, "data: [DONE]") {
			t.Fatalf("post-commit admission error was not terminal SSE: %q", body)
		}
		waitAdmissionWaiting(t, ctl, 0)
		assertUpstreamNotHit(t, hit)
	})
}

func assertUpstreamNotHit(t *testing.T, hit <-chan struct{}) {
	t.Helper()
	select {
	case <-hit:
		t.Fatal("cancelled queued request reached upstream")
	default:
	}
}
