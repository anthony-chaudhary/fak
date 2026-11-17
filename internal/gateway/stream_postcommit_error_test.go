package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime medium est=2000ms lane=default
func TestStreamUpstreamFailurePostCommitTypedEvent(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	up := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			return
		}
		_ = conn.Close()
	}))
	up.Start()
	t.Cleanup(up.Close)
	t.Cleanup(unblock)

	srv := newQueuedStreamGateway(t, up.URL, nil)
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	gw := httptest.NewServer(srv.Handler())
	t.Cleanup(gw.Close)
	// Unblock the upstream before Close waits for a request on an assertion failure.
	t.Cleanup(unblock)
	client := &http.Client{Timeout: 15 * time.Second}
	started := time.Now()
	resp, err := client.Post(gw.URL+"/v1/chat/completions", "application/json", bytes.NewReader(queuedChatBody(true)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("stream response = %d %q, want 200 text/event-stream", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	scanner := bufio.NewScanner(resp.Body)
	var frames []string
	for scanner.Scan() {
		line := scanner.Text()
		frames = append(frames, line)
		if line == ": keepalive" {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(frames) == 0 || frames[len(frames)-1] != ": keepalive" {
		t.Fatalf("no prefill keepalive before failure: %q", frames)
	}
	if elapsed := time.Since(started); elapsed < 250*time.Millisecond {
		t.Fatalf("upstream released after %s, want at least the 250ms commit grace", elapsed)
	}
	// The first request loses its connection; a transport retry must then dial
	// a closed listener and produce the stable unreachable classification.
	if err := up.Listener.Close(); err != nil {
		t.Fatal(err)
	}
	unblock()
	for scanner.Scan() {
		frames = append(frames, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("stream truncated: %v", err)
	}

	buffered, err := client.Post(gw.URL+"/v1/chat/completions", "application/json", bytes.NewReader(queuedChatBody(false)))
	if err != nil {
		t.Fatal(err)
	}
	defer buffered.Body.Close()
	var bufferedBody struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(buffered.Body).Decode(&bufferedBody); err != nil {
		t.Fatal(err)
	}
	if buffered.StatusCode != http.StatusBadGateway || bufferedBody.Error.Code != "upstream_unreachable" {
		t.Fatalf("buffered failure = %d %q, want 502 upstream_unreachable", buffered.StatusCode, bufferedBody.Error.Code)
	}

	var errorEvents, errorPayloads, done int
	lastData := ""
	eventType := ""
	for _, line := range frames {
		if line == "" {
			eventType = ""
			continue
		}
		if value, ok := strings.CutPrefix(line, "event: "); ok {
			eventType = value
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		lastData = data
		if data == "[DONE]" {
			if eventType != "" {
				t.Fatalf("[DONE] has event type %q, want default message", eventType)
			}
			done++
			continue
		}
		var payload struct {
			Error *struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			} `json:"error"`
			Choices []json.RawMessage `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			t.Fatalf("invalid SSE data: %v", err)
		}
		if len(payload.Choices) != 0 || payload.Error == nil {
			t.Fatalf("failure emitted a successful completion frame: %s", data)
		}
		if payload.Error.Type != "server_error" || strings.TrimSpace(payload.Error.Message) == "" {
			t.Fatalf("invalid terminal error payload: %s", data)
		}
		if eventType == "error" {
			errorEvents++
		} else if eventType != "" {
			t.Fatalf("error payload has event type %q, want error", eventType)
		}
		errorPayloads++
	}
	if errorPayloads != 1 || done != 1 || lastData != "[DONE]" {
		t.Fatalf("terminal stream = errors:%d done:%d last:%q, want one error then [DONE]", errorPayloads, done, lastData)
	}
	if errorEvents != 1 {
		t.Fatalf("typed event: error marker count = %d, want 1; frames=%q", errorEvents, frames)
	}
}
