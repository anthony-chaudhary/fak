package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// silentPrefillUpstream models a llama-server prefilling a long prompt: it accepts the
// streamed request and writes NOTHING for silence, then either streams a short answer or
// fails with failStatus (a 504 UPSTREAM_TIMEOUT, as the router logged in #13782).
func silentPrefillUpstream(t *testing.T, silence time.Duration, failStatus int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		select {
		case <-time.After(silence):
		case <-r.Context().Done():
			return
		}
		if failStatus != 0 {
			http.Error(w, `{"error":{"message":"upstream timed out","type":"server_error"}}`, failStatus)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for _, frame := range []string{
			`{"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"content":"after prefill"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}`,
			"[DONE]",
		} {
			_, _ = io.WriteString(w, "data: "+frame+"\n\n")
			fl.Flush()
		}
	}))
}

// servePrefillGateway runs the real http.Server (Serve) in front of upstreamURL with a
// 1s WriteTimeout, so an upstream silent for 3s is "longer than the write deadline" the
// way a 300s prefill is longer than the production 90s default.
func servePrefillGateway(t *testing.T, upstreamURL string) string {
	t.Helper()
	t.Setenv("FAK_HTTP_WRITE_TIMEOUT_S", "1")
	t.Setenv("FAK_STREAM_PREFILL_KEEPALIVE_S", "")
	t.Setenv("FAK_STREAM_HEARTBEAT_S", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	prevEvery := prefillKeepaliveEvery
	prefillKeepaliveEvery = 400 * time.Millisecond
	t.Cleanup(func() { prefillKeepaliveEvery = prevEvery })

	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstreamURL + "/compat", Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
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

type prefillWire struct {
	headAfter  time.Duration
	status     int
	ctype      string
	keepalives int
	data       []string
}

// readPrefillStream posts a streamed chat turn and records when the head arrived and
// every SSE line, split into keepalive comments and data frames.
func readPrefillStream(t *testing.T, base string) prefillWire {
	t.Helper()
	raw, _ := json.Marshal(map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "a very long prompt"}},
		"stream":   true,
	})
	began := time.Now()
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("post: %v (a lost write surfaces here as a client disconnect)", err)
	}
	defer resp.Body.Close()
	got := prefillWire{headAfter: time.Since(began), status: resp.StatusCode, ctype: resp.Header.Get("Content-Type")}
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		switch line := strings.TrimSpace(sc.Text()); {
		case line == ": keepalive":
			got.keepalives++
		case strings.HasPrefix(line, "data: "):
			got.data = append(got.data, strings.TrimPrefix(line, "data: "))
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("stream cut before it ended (write deadline not lifted?): %v; frames=%v", err, got.data)
	}
	return got
}

// TestLiveStreamCommitsHeadersAndKeepalivesDuringSilentPrefill is the #13783 witness: an
// upstream silent for longer than the server write deadline still gets its 200
// text/event-stream head to the client in well under 1s, `: keepalive` comments flow
// during the prefill, and the late answer arrives whole (#13782: deadline lifted).
// fak-test:runtime medium est=3500ms lane=default
func TestLiveStreamCommitsHeadersAndKeepalivesDuringSilentPrefill(t *testing.T) {
	up := silentPrefillUpstream(t, 3*time.Second, 0)
	defer up.Close()
	got := readPrefillStream(t, servePrefillGateway(t, up.URL))
	if got.headAfter >= time.Second {
		t.Fatalf("head arrived after %v, want < 1s during prefill", got.headAfter)
	}
	if got.status != http.StatusOK || !strings.HasPrefix(got.ctype, "text/event-stream") {
		t.Fatalf("head = %d %q, want 200 text/event-stream", got.status, got.ctype)
	}
	if got.keepalives < 3 {
		t.Fatalf("keepalives = %d during a 3s prefill at 400ms, want >= 3", got.keepalives)
	}
	var content strings.Builder
	for _, d := range got.data {
		if d == "[DONE]" {
			continue
		}
		var c ChatStreamResponse
		if err := json.Unmarshal([]byte(d), &c); err != nil {
			t.Fatalf("frame %q: %v", d, err)
		}
		if len(c.Choices) > 0 {
			content.WriteString(c.Choices[0].Delta.Content)
		}
	}
	if content.String() != "after prefill" || got.data[len(got.data)-1] != "[DONE]" {
		t.Fatalf("late answer not delivered whole past the write deadline: %v", got.data)
	}
}

// TestLiveStreamLateUpstreamErrorIsSSEErrorFrame is the #13782 witness: a 504 arriving
// after the write deadline reaches the client as an in-band SSE error frame + [DONE] on
// the early-committed 200, never a lost write the client sees as RemoteDisconnected.
// fak-test:runtime medium est=3500ms lane=default
func TestLiveStreamLateUpstreamErrorIsSSEErrorFrame(t *testing.T) {
	up := silentPrefillUpstream(t, 3*time.Second, http.StatusGatewayTimeout)
	defer up.Close()
	got := readPrefillStream(t, servePrefillGateway(t, up.URL))
	if got.status != http.StatusOK || got.headAfter >= time.Second {
		t.Fatalf("head = %d after %v, want early 200", got.status, got.headAfter)
	}
	if got.keepalives < 3 {
		t.Fatalf("keepalives = %d before the late failure, want >= 3", got.keepalives)
	}
	if len(got.data) != 2 || got.data[1] != "[DONE]" {
		t.Fatalf("data frames = %v, want [error, DONE]", got.data)
	}
	var frame struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(got.data[0]), &frame); err != nil || frame.Error.Type != "server_error" || frame.Error.Message == "" {
		t.Fatalf("first frame = %q (%v), want an SSE error frame", got.data[0], err)
	}
}

// TestLiveStreamFastUpstreamErrorKeepsHTTPStatus: a failure inside prefillCommitAfter
// still owns a real HTTP status rather than a 200 + SSE error.
// fak-test:runtime fast est=50ms lane=default
func TestLiveStreamFastUpstreamErrorKeepsHTTPStatus(t *testing.T) {
	up := silentPrefillUpstream(t, 0, http.StatusGatewayTimeout)
	defer up.Close()
	got := readPrefillStream(t, servePrefillGateway(t, up.URL))
	if got.status == http.StatusOK || got.keepalives != 0 {
		t.Fatalf("fast failure = %d with %d keepalives, want a non-200 status and no stream", got.status, got.keepalives)
	}
}
