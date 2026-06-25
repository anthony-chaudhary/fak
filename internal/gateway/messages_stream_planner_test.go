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

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestAnthropicMessagesPlannerStreamEmitsTextBeforeToolGate(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	contentSent := make(chan struct{})
	releaseTools := make(chan struct{})
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		var req struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(upstreamBody, &req)
		if !req.Stream {
			t.Errorf("gateway did not ask the upstream planner to stream: %s", upstreamBody)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		_, _ = io.WriteString(w,
			`data: {"model":"served-openai","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"checking"}}]}`+"\n\n")
		flusher.Flush()
		close(contentSent)
		select {
		case <-releaseTools:
		case <-time.After(2 * time.Second):
			t.Error("test timed out waiting to release tool-call deltas")
			return
		}
		_, _ = io.WriteString(w,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a1","type":"function","function":{"name":"allow_a","arguments":"{\"x\":1}"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"d1","type":"function","function":{"name":"deny_b","arguments":"{\"secret\":\"nope\"}"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13}}`+"\n\n"+
				"data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inbound := []byte(`{"model":"claude-client","max_tokens":256,"stream":true,` +
		`"tools":[{"name":"allow_a","input_schema":{"type":"object"}},{"name":"deny_b","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"call tools"}]}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "event-stream") {
		t.Fatalf("content-type = %q, want event-stream", ct)
	}

	lines := make(chan string, 128)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			lines <- sc.Text()
		}
		close(lines)
	}()

	select {
	case <-contentSent:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream never emitted the first content delta")
	}

	var beforeRelease []string
	sawText := false
	for !sawText {
		select {
		case line, ok := <-lines:
			if !ok {
				t.Fatalf("stream ended before live text_delta; lines=%q", strings.Join(beforeRelease, "\n"))
			}
			beforeRelease = append(beforeRelease, line)
			if strings.Contains(line, `"text_delta"`) && strings.Contains(line, "checking") {
				sawText = true
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("client did not receive text_delta before tool release; lines=%q", strings.Join(beforeRelease, "\n"))
		}
	}
	if early := strings.Join(beforeRelease, "\n"); strings.Contains(early, "allow_a") || strings.Contains(early, "input_json_delta") {
		t.Fatalf("tool-call bytes reached the Anthropic client before adjudication:\n%s", early)
	}

	close(releaseTools)
	allLines := append([]string{}, beforeRelease...)
	for line := range lines {
		allLines = append(allLines, line)
	}
	body := strings.Join(allLines, "\n")
	if !strings.Contains(body, `"allow_a"`) || !strings.Contains(body, `"input_json_delta"`) {
		t.Fatalf("adjudicated allowed tool call did not reach the Anthropic stream:\n%s", body)
	}
	for _, leak := range []string{`"name":"deny_b"`, "nope", `"secret"`} {
		if strings.Contains(body, leak) {
			t.Fatalf("denied tool-call bytes leaked into the Anthropic stream (%q):\n%s", leak, body)
		}
	}
	if !strings.Contains(body, `"stop_reason":"tool_use"`) {
		t.Fatalf("surviving tool call should keep stop_reason tool_use:\n%s", body)
	}
	if !strings.Contains(body, `"input_tokens":8`) || !strings.Contains(body, `"output_tokens":5`) {
		t.Fatalf("terminal usage from the streamed planner was not forwarded:\n%s", body)
	}
}
