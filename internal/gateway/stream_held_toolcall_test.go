package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// pacedToolCallUpstream streams a tool-call-only turn: the role frame, then each
// argument fragment after gap, then blocks on release before the finish frame.
func pacedToolCallUpstream(t *testing.T, gap time.Duration, fragments []string, release <-chan struct{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		send := func(frame string) {
			_, _ = io.WriteString(w, "data: "+frame+"\n\n")
			fl.Flush()
		}
		send(`{"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`)
		send(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"allow_x","arguments":""}}]}}]}`)
		for _, frag := range fragments {
			time.Sleep(gap)
			raw, _ := json.Marshal(frag)
			send(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":` + string(raw) + `}}]}}]}`)
		}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		send(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`)
		send("[DONE]")
	}))
}

func startToolStream(t *testing.T, url string) <-chan *http.Response {
	t.Helper()
	raw, _ := json.Marshal(map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "allow_x"}}},
		"stream":   true,
	})
	got := make(chan *http.Response, 1)
	go func() {
		resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
		if err != nil {
			t.Errorf("post: %v", err)
			close(got)
			return
		}
		got <- resp
	}()
	return got
}

// TestToolCallOnlyStreamCommitsBeforeTurnEnds: a turn whose only output is a tool call
// (the common agent step) must put the 200 and the opening role chunk on the wire as
// soon as the model starts decoding, not when the whole call has been generated.
func TestToolCallOnlyStreamCommitsBeforeTurnEnds(t *testing.T) {
	release := make(chan struct{})
	up := pacedToolCallUpstream(t, 0, []string{`{"a":1}`}, release)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()
	defer close(release)

	var resp *http.Response
	select {
	case resp = <-startToolStream(t, ts.URL):
	case <-time.After(10 * time.Second):
		t.Fatal("no response headers while the upstream was still decoding the tool call")
	}
	if resp == nil {
		t.FailNow()
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatalf("read first frame: %v", err)
	}
	var first ChatStreamResponse
	if err := json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSpace(line), "data: ")), &first); err != nil {
		t.Fatalf("first frame %q: %v", line, err)
	}
	if len(first.Choices) != 1 || first.Choices[0].Delta.Role != "assistant" || len(first.Choices[0].Delta.ToolCalls) != 0 {
		t.Fatalf("first frame = %+v, want the bare opening role chunk", first)
	}
}

// TestHeldToolCallStreamEmitsKeepalive: while a long tool call is held for adjudication
// the committed stream must not go silent, or a downstream stall watchdog cuts a healthy
// turn. The held call itself still arrives whole, after the keepalives.
func TestHeldToolCallStreamEmitsKeepalive(t *testing.T) {
	release := make(chan struct{})
	close(release)
	frags := []string{`{"a":`, `1`, `,"b":`, `2`, `}`}
	up := pacedToolCallUpstream(t, 600*time.Millisecond, frags, release)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	resp := <-startToolStream(t, ts.URL)
	if resp == nil {
		t.FailNow()
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)
	callAt := strings.Index(text, `"tool_calls"`)
	if callAt < 0 {
		t.Fatalf("no tool call delivered: %s", raw)
	}
	if !strings.Contains(text[:callAt], "\n: ") && !strings.HasPrefix(text, ": ") {
		t.Fatalf("no SSE comment keepalive before the held tool call: %s", raw)
	}
	var data []string
	for _, line := range strings.Split(text, "\n") {
		if !strings.HasPrefix(line, ":") {
			data = append(data, line)
		}
	}
	chunks, sawDone := parseSSEChunks(t, []byte(strings.Join(data, "\n")))
	if !sawDone {
		t.Fatalf("missing [DONE]: %s", raw)
	}
	var args string
	for _, c := range chunks {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			args += tc.Function.Arguments
		}
	}
	if args != `{"a":1,"b":2}` {
		t.Fatalf("tool call arguments = %q, want the whole held call", args)
	}
}
