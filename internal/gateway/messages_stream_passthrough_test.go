package gateway

import (
	"bufio"
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// sseFrame is one parsed event from the gateway's downstream SSE response.
type sseFrame struct {
	event string
	data  string
}

// readAnthropicSSE parses the gateway's SSE body into ordered frames.
func readAnthropicSSE(t *testing.T, r io.Reader) []sseFrame {
	t.Helper()
	var frames []sseFrame
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var ev, data string
	flush := func() {
		if data != "" {
			frames = append(frames, sseFrame{event: ev, data: data})
		}
		ev, data = "", ""
	}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "event:"):
			ev = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}
	flush()
	return frames
}

// TestAnthropicMessagesPassthroughStreamsLiveAndAdjudicates is the end-to-end proof of
// the flagship `fak guard -- claude` SPEEDUP: when fak fronts the real Anthropic API
// and the client asks to stream, fak relays the upstream's text deltas LIVE (the model
// dictates TTFT, not the whole-turn buffer) while still HOLDING every tool_use block
// for kernel adjudication — and the prompt-cache usage flows straight back. The text
// content_block is emitted to the client BEFORE any tool_use block (the streaming
// witness); the denied call is dropped and named in-band; the survivors are renumbered
// contiguously.
func TestAnthropicMessagesPassthroughStreamsLiveAndAdjudicates(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"model":"claude-test","max_tokens":4096,"stream":true,` +
		`"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],` +
		`"tools":[{"name":"allow_a","input_schema":{"type":"object"}},{"name":"deny_b","input_schema":{"type":"object"}},{"name":"transform_c","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"call tools"}]}`)

	// The upstream Anthropic SSE: a text block, then three tool_use blocks (one allowed,
	// one denied, one transformed), then the terminal delta carrying the real usage.
	upstreamSSE := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":3,"cache_read_input_tokens":4096,"cache_creation_input_tokens":0,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"checking"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"a1","name":"allow_a","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"a2","name":"deny_b","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":2}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"a3","name":"transform_c","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"secret\":\"y\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":3}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	var upstreamBody []byte
	var upstreamKey string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		upstreamKey = r.Header.Get("x-api-key")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, upstreamSSE)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode != 200 {
		t.Fatalf("status = %d", httpResp.StatusCode)
	}
	if ct := httpResp.Header.Get("Content-Type"); !strings.Contains(ct, "event-stream") {
		t.Fatalf("downstream is not an SSE stream: Content-Type=%q", ct)
	}
	frames := readAnthropicSSE(t, httpResp.Body)

	// The inbound bytes were forwarded verbatim (stream:true already present), so the
	// client's cache_control prefix lands intact upstream — a real cache hit.
	if string(upstreamBody) != string(inbound) {
		t.Errorf("upstream body not byte-identical (cache prefix would miss):\n got %q\nwant %q", upstreamBody, inbound)
	}
	if upstreamKey != "caller-key" {
		t.Errorf("upstream x-api-key = %q, want the forwarded caller-key", upstreamKey)
	}

	// The whole stream as one blob for content assertions, plus structural checks below.
	var blob strings.Builder
	for _, f := range frames {
		blob.WriteString(f.data)
		blob.WriteByte('\n')
	}
	all := blob.String()

	// Live text reached the client.
	if !strings.Contains(all, `"text_delta"`) || !strings.Contains(all, "checking") {
		t.Errorf("live text delta not relayed to the client:\n%s", all)
	}
	// The cache-read count flows back on message_start (verbatim relay).
	if !strings.Contains(all, `"cache_read_input_tokens":4096`) {
		t.Errorf("cache_read_input_tokens not forwarded to the client:\n%s", all)
	}
	// The denied call never reaches the client; the allowed + transformed survive.
	if strings.Contains(all, `"deny_b"`) {
		t.Errorf("denied tool call must NOT reach the client through the live stream:\n%s", all)
	}
	if !strings.Contains(all, `"allow_a"`) {
		t.Errorf("allowed tool call missing from the stream:\n%s", all)
	}
	if !strings.Contains(all, `"transform_c"`) {
		t.Errorf("transformed tool call missing from the stream:\n%s", all)
	}
	// The kernel rewrote transform_c's args; the repaired object (not the secret) ships.
	if strings.Contains(all, `"secret"`) {
		t.Errorf("repaired tool call leaked the original secret args:\n%s", all)
	}
	if !strings.Contains(all, `redacted`) {
		t.Errorf("repaired tool call did not carry the kernel's rewritten args:\n%s", all)
	}
	// The in-band note names the dropped + repaired calls.
	if !strings.Contains(all, "[fak]") || !strings.Contains(all, "deny_b") || !strings.Contains(all, "transform_c") {
		t.Errorf("in-band note must name the refused + repaired calls:\n%s", all)
	}

	// Streaming witness: the text content block is fully relayed to the client BEFORE
	// any tool_use block start — i.e. the model's prose was streamed live while the tool
	// calls were still being held for adjudication.
	firstToolUse, firstTextStop := -1, -1
	for i, f := range frames {
		if firstTextStop < 0 && f.event == "content_block_stop" && i >= 1 && strings.Contains(frames[i-1].data, "text_delta") {
			firstTextStop = i
		}
		if firstToolUse < 0 && strings.Contains(f.data, `"tool_use"`) {
			firstToolUse = i
		}
	}
	if firstTextStop < 0 {
		t.Fatalf("no text content block found in the stream")
	}
	if firstToolUse >= 0 && firstToolUse < firstTextStop {
		t.Errorf("a tool_use block was emitted before the live text finished (frame %d < %d) — text was not streamed ahead of adjudication", firstToolUse, firstTextStop)
	}

	// Contiguous renumbering: dropping deny_b must not leave an index gap. Collect the
	// indices of emitted content_block_start frames; they must be 0,1,2,... with no hole.
	var idxs []string
	for _, f := range frames {
		if f.event == "content_block_start" {
			if i := strings.Index(f.data, `"index":`); i >= 0 {
				rest := f.data[i+len(`"index":`):]
				end := strings.IndexAny(rest, ",}")
				idxs = append(idxs, strings.TrimSpace(rest[:end]))
			}
		}
	}
	want := []string{"0", "1", "2", "3"} // text, allow_a, transform_c, note
	if strings.Join(idxs, ",") != strings.Join(want, ",") {
		t.Errorf("content-block indices not contiguous after the drop: got %v, want %v", idxs, want)
	}

	// stop_reason stays tool_use because a call survived.
	if !strings.Contains(all, `"tool_use"`) {
		t.Errorf("message_delta stop_reason should remain tool_use (a call survived):\n%s", all)
	}
}

// TestAnthropicMessagesPassthroughStreamAllDeniedEndsTurn proves the all-denied case:
// when the kernel drops EVERY proposed call, the relayed message_delta rewrites
// stop_reason from "tool_use" to "end_turn" so the client does not hunt for a tool_use
// block that isn't there.
func TestAnthropicMessagesPassthroughStreamAllDeniedEndsTurn(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,` +
		`"tools":[{"name":"deny_b","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"go"}]}`)

	upstreamSSE := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":2,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"d1","name":"deny_b","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":2}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, upstreamSSE)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer httpResp.Body.Close()
	frames := readAnthropicSSE(t, httpResp.Body)

	var delta string
	for _, f := range frames {
		if f.event == "message_delta" {
			delta = f.data
		}
		if strings.Contains(f.data, `"deny_b"`) {
			t.Errorf("denied call leaked to the client: %s", f.data)
		}
	}
	if delta == "" {
		t.Fatalf("no message_delta frame")
	}
	if !strings.Contains(delta, `"end_turn"`) || strings.Contains(delta, `"tool_use"`) {
		t.Errorf("all-denied turn must rewrite stop_reason to end_turn, got: %s", delta)
	}
}
