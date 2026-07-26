package gateway

// messages_stream_warmcontinue_test.go — the #3353 contract witness. It kills the upstream
// worker MID-TURN (after the client's SSE has already received "Hel") and asserts that, with
// warm-continue armed, the client sees ONE unbroken turn whose text is prefix+continuation
// ("Hello world") with no duplication — the partial output is replayed as an assistant prefill
// on a fresh worker and the budget is decremented, instead of the caller getting a terminal
// error and a cold session restart. The OFF twin proves the behavior is gated: with the lever
// unset the same death ends the stream with a terminal error frame, no resume.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// warmContinueDeadFrames is the partial first segment: a message_start, an open text block,
// and one "Hel" delta — then the worker dies. No content_block_stop / message_delta: the turn
// is cut off mid-block, exactly as a real mid-stream death leaves it.
const warmContinueDeadFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_wc1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}

`

// warmContinueTailFrames is the continuation segment served to the resumed request: its own
// message_start (which the relay swallows), a fresh text block emitting "lo world", and the
// terminal message_delta/message_stop. The model does NOT re-emit the "Hel" prefill.
const warmContinueTailFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_wc2","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":14,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`

// writeDeadPartial hijacks the upstream connection and writes a chunked response whose body is
// closed MID-STREAM (no terminating 0-chunk), so the gateway's upstream read fails with
// io.ErrUnexpectedEOF — a deterministic, instant mid-stream death (no TCP-reset race, no
// idle-stall wait). Best-effort: a hijack failure just returns (the test then fails on the
// missing second call / assertions).
func writeDeadPartial(w http.ResponseWriter, frames string) {
	hj, ok := w.(http.Hijacker)
	if !ok {
		return
	}
	conn, bufrw, err := hj.Hijack()
	if err != nil {
		return
	}
	fmt.Fprint(bufrw, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
	fmt.Fprintf(bufrw, "%x\r\n%s\r\n", len(frames), frames) // one chunk, then a premature close
	_ = bufrw.Flush()
	_ = conn.Close()
}

// clientTextDeltas reconstructs the assistant text the client saw by concatenating every
// text_delta in the relayed SSE — the "single unbroken turn" the witness asserts on.
func clientTextDeltas(sse string) string {
	var sb strings.Builder
	for _, line := range strings.Split(sse, "\n") {
		v, ok := strings.CutPrefix(strings.TrimSpace(line), "data:")
		if !ok {
			continue
		}
		var d struct {
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(v)), &d) == nil && d.Delta.Type == "text_delta" {
			sb.WriteString(d.Delta.Text)
		}
	}
	return sb.String()
}

func warmContinueInbound() []byte {
	return []byte(`{"model":"claude-test","max_tokens":4096,"stream":true,` +
		`"messages":[{"role":"user","content":"say hello world"}]}`)
}

// postStreamCollect sends a streaming /v1/messages request and returns the full relayed SSE
// body the client saw (drained to completion), plus the HTTP status.
func postStreamCollect(t *testing.T, gatewayURL string, inbound []byte) (string, int) {
	t.Helper()
	req, _ := http.NewRequest("POST", gatewayURL+"/v1/messages", strings.NewReader(string(inbound)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return string(body), resp.StatusCode
}

// TestWarmContinueResumesLiveTurnAcrossWorkerDeath is the #3353 witness: with the lever armed,
// a worker that dies after "Hel" is warm-continued on a fresh worker that replays "Hel" as a
// prefill turn (budget decremented) and emits "lo world" — the client sees one unbroken
// "Hello world" turn, no duplication, no error frame.
func TestWarmContinueResumesLiveTurnAcrossWorkerDeath(t *testing.T) {
	t.Setenv("FAK_WARM_CONTINUE", "1")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var calls int32
	var contBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			writeDeadPartial(w, warmContinueDeadFrames) // die mid-turn after "Hel"
			return
		}
		contBody, _ = io.ReadAll(r.Body) // the continuation (replay-as-context) request
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, warmContinueTailFrames)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, warmContinueInbound())
	if status != 200 {
		t.Fatalf("status = %d, want 200: %s", status, body)
	}

	// The worker died once and the turn was continued on a fresh worker: exactly two upstream calls.
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("upstream calls = %d, want 2 (one death + one warm-continue)", got)
	}
	// ONE unbroken turn: prefix + continuation, no duplication of the delivered prefix.
	if txt := clientTextDeltas(body); txt != "Hello world" {
		t.Fatalf("client saw text %q, want %q (unbroken prefix+continuation)", txt, "Hello world")
	}
	if n := strings.Count(body, `"text":"Hel"`); n != 1 {
		t.Fatalf("delivered prefix \"Hel\" appears %d times, want exactly 1 (no duplication)", n)
	}
	// The client turn ended cleanly — a single message_stop, and NO terminal error frame.
	if strings.Contains(body, "event: error") {
		t.Fatalf("warm-continue must hide the death: client got a terminal error frame:\n%s", body)
	}
	if n := strings.Count(body, "event: message_stop"); n != 1 {
		t.Fatalf("client saw %d message_stop frames, want exactly 1 unbroken turn:\n%s", n, body)
	}

	// The continuation request replayed the delivered text as an assistant PREFILL turn and
	// decremented the budget (4096 - est(3 chars)=1 -> 4095): dynamo track_response, in Go.
	cb := string(contBody)
	if cb == "" {
		t.Fatal("continuation request had no body — the warm-continue re-issue did not run")
	}
	if !strings.Contains(cb, `"role":"assistant"`) || !strings.Contains(cb, `"content":"Hel"`) {
		t.Fatalf("continuation must append the delivered text as an assistant prefill turn:\n%s", cb)
	}
	if !strings.Contains(cb, `"max_tokens":4095`) {
		t.Fatalf("continuation must decrement max_tokens by the delivered estimate (want 4095):\n%s", cb)
	}
}

// warmContinueToolDeathFrames is a CONTINUATION segment that itself dies mid-turn — but only
// after opening a tool_use block. It finishes the replayed text block ("lo"), then starts a
// tool_use whose input streams in, then the worker dies with no content_block_stop and no
// message_delta. This is the shape that makes a further replay unsafe: the held tool_use is
// already in toolOrder, so a second continuation that re-emits the same call would flush the
// batch twice and the client would run the tool TWICE.
const warmContinueToolDeathFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_wc2","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":14,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_wc","name":"allow_run"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"ls\"}"}}

`

// warmContinueToolCleanFrames is what a THIRD upstream call would be served if the resume loop
// kept going: a clean turn that re-emits the same tool_use and terminates. It exists to make the
// double-emit concrete — if warm-continue re-attempts after a tool_use is already held, this
// segment's call joins the still-held one and the client sees the SAME tool call twice.
const warmContinueToolCleanFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_wc3","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":18,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_wc","name":"allow_run"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"ls\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`

// TestWarmContinueStandsDownWhenContinuationHoldsToolUse pins the per-attempt safety re-check
// (#3353). canWarmContinue only guards the FIRST re-issue; a continuation that dies after
// opening a tool_use block leaves that call held in toolOrder, so replaying again would flush
// the batch twice and make the client execute the tool TWICE. The resume loop must therefore
// re-check the turn's shape before EVERY attempt and stand down to the terminal-error path —
// the same recovery the gate-OFF twin gets — rather than risk a duplicated side effect.
func TestWarmContinueStandsDownWhenContinuationHoldsToolUse(t *testing.T) {
	t.Setenv("FAK_WARM_CONTINUE", "1")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var calls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch atomic.AddInt32(&calls, 1) {
		case 1:
			writeDeadPartial(w, warmContinueDeadFrames) // die mid-turn after "Hel"
		case 2:
			writeDeadPartial(w, warmContinueToolDeathFrames) // continuation dies holding a tool_use
		default:
			// A third call means the loop re-attempted with a tool_use already held.
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, warmContinueToolCleanFrames)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, warmContinueInbound())
	if status != 200 {
		t.Fatalf("status = %d, want 200: %s", status, body)
	}

	// Exactly two upstream calls: the death, the one warm-continue — and NO third attempt once
	// the continuation held a tool_use.
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("upstream calls = %d, want 2 (warm-continue must stand down once a tool_use is held)", got)
	}
	// The held tool_use is never flushed: the client must not be handed a call the kernel would
	// otherwise emit twice.
	if n := strings.Count(body, `"type":"tool_use"`); n != 0 {
		t.Fatalf("client saw %d tool_use blocks, want 0 (the held batch must not flush on a stand-down):\n%s", n, body)
	}
	// The delivered text still reached the client, once, and the turn ends on the terminal-error
	// path — the same recovery the gate-OFF twin gets.
	if txt := clientTextDeltas(body); txt != "Hello" {
		t.Fatalf("client saw text %q, want %q (delivered prefix + continuation, no duplication)", txt, "Hello")
	}
	if !strings.Contains(body, "event: error") {
		t.Fatalf("a stand-down must end the client stream with a terminal error frame:\n%s", body)
	}
}

// TestWarmContinueGatedOffEndsWithTerminalError is the gate witness: with the lever unset, the
// SAME mid-stream death ends the client stream with a terminal error frame and no resume — the
// #3353 behavior is opt-in (gen/next, dogfood-first), never on by default.
func TestWarmContinueGatedOffEndsWithTerminalError(t *testing.T) {
	t.Setenv("FAK_WARM_CONTINUE", "") // explicitly OFF
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var calls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		writeDeadPartial(w, warmContinueDeadFrames)
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, warmContinueInbound())
	if status != 200 {
		t.Fatalf("status = %d, want 200 (stream already opened): %s", status, body)
	}
	// OFF: no resume — a single upstream call, and the client sees the terminal error frame.
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("upstream calls = %d, want 1 (gate OFF: no warm-continue)", got)
	}
	if !strings.Contains(body, "event: error") {
		t.Fatalf("gate OFF: the mid-stream death must end the client stream with a terminal error frame:\n%s", body)
	}
	// The delivered prefix still reached the client; it is simply not continued.
	if !strings.Contains(body, `"text":"Hel"`) {
		t.Fatalf("gate OFF: the delivered prefix should still have been relayed before the error:\n%s", body)
	}
	if strings.Contains(body, "lo world") {
		t.Fatalf("gate OFF: no continuation should have been emitted:\n%s", body)
	}
}

// warmContinueNewlineDeadFrames is the common real shape the naive replay got wrong: the worker
// dies right after a delta that ENDS IN A NEWLINE (mid-markdown, mid-code block). Replaying it
// verbatim is a prefill ending in whitespace, which Anthropic refuses outright.
const warmContinueNewlineDeadFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_wc3","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"# Title\n"}}

`

// TestWarmContinueTrimsTrailingWhitespaceFromPrefill is the prefill-contract witness: a death on
// a newline boundary must still produce a LEGAL continuation — the replayed assistant turn is
// right-trimmed, even though the client already saw (and keeps) the newline.
func TestWarmContinueTrimsTrailingWhitespaceFromPrefill(t *testing.T) {
	t.Setenv("FAK_WARM_CONTINUE", "1")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var calls int32
	var contBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			writeDeadPartial(w, warmContinueNewlineDeadFrames)
			return
		}
		contBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, warmContinueTailFrames)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, warmContinueInbound())
	if status != 200 {
		t.Fatalf("status = %d, want 200: %s", status, body)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("upstream calls = %d, want 2 (one death + one warm-continue)", got)
	}
	cb := string(contBody)
	if !strings.Contains(cb, `"content":"# Title"`) {
		t.Fatalf("continuation prefill must be right-trimmed (want content \"# Title\"):\n%s", cb)
	}
	if strings.Contains(cb, `"content":"# Title\n"`) {
		t.Fatalf("continuation prefill still ends in whitespace — Anthropic refuses that turn:\n%s", cb)
	}
	// The client keeps every byte it was already handed; only the replayed copy is trimmed.
	if txt := clientTextDeltas(body); txt != "# Title\nlo world" {
		t.Fatalf("client saw text %q, want %q", txt, "# Title\nlo world")
	}
}

// TestWarmContinuePrefillMergesOntoTrailingAssistantTurn covers the caller-prefill case: when
// the inbound ALREADY ends with an assistant turn (the one the model was continuing when the
// worker died), the delivered text merges onto it instead of stacking a second assistant turn,
// which Anthropic rejects. Both wire shapes of `content` are covered.
func TestWarmContinuePrefillMergesOntoTrailingAssistantTurn(t *testing.T) {
	for _, tc := range []struct {
		name, inbound, wantMsgs string
	}{
		{
			name:     "string content",
			inbound:  `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Once upon "}]}`,
			wantMsgs: `[{"role":"user","content":"hi"},{"content":"Once upon a time","role":"assistant"}]`,
		},
		{
			name:     "block array content",
			inbound:  `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"Once upon "}]}]}`,
			wantMsgs: `[{"role":"user","content":"hi"},{"content":[{"text":"Once upon a time","type":"text"}],"role":"assistant"}]`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, ok := warmContinueBody([]byte(tc.inbound), "a time", 2)
			if !ok {
				t.Fatal("warmContinueBody refused a mergeable trailing assistant turn")
			}
			var m map[string]json.RawMessage
			if err := json.Unmarshal(out, &m); err != nil {
				t.Fatalf("continuation body is not an object: %v", err)
			}
			if got := string(m["messages"]); got != tc.wantMsgs {
				t.Fatalf("messages =\n  %s\nwant\n  %s", got, tc.wantMsgs)
			}
			// Exactly one assistant turn survives — never two stacked in a row.
			if n := strings.Count(string(m["messages"]), `"role":"assistant"`); n != 1 {
				t.Fatalf("assistant turns = %d, want 1 (merged, not stacked): %s", n, m["messages"])
			}
			if got := string(m["max_tokens"]); got != "98" {
				t.Fatalf("max_tokens = %s, want 98 (100 - delivered estimate)", got)
			}
		})
	}
}

// TestWarmContinueBodyRefusesUnreplayablePrefix pins the honest stand-downs: an all-whitespace
// delivered prefix is not a legal prefill, so the resume is refused before an upstream call is
// spent rather than trading a terminal error for a 400 plus a terminal error.
func TestWarmContinueBodyRefusesUnreplayablePrefix(t *testing.T) {
	inbound := []byte(`{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	if _, ok := warmContinueBody(inbound, "  \n\t ", 1); ok {
		t.Fatal("an all-whitespace prefix must not be replayed as a prefill turn")
	}
	if _, ok := warmContinueBody([]byte(`{"model":"m"}`), "text", 1); ok {
		t.Fatal("a body with no messages array must be refused")
	}
}

// TestCanWarmContinueStandsDownWhenThinkingEnabled pins the structural half of the thinking
// guard: extended thinking enabled on the REQUEST rules out an assistant prefill, even when no
// thinking block was relayed for p.sawThinking to catch.
func TestCanWarmContinueStandsDownWhenThinkingEnabled(t *testing.T) {
	t.Setenv("FAK_WARM_CONTINUE", "1")
	newPassthrough := func(raw string) *anthropicPassthrough {
		p := &anthropicPassthrough{req: &agent.AnthropicMessagesRequest{Raw: []byte(raw)}}
		p.asstText.WriteString("delivered")
		return p
	}
	death := io.ErrUnexpectedEOF

	plain := `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	if !newPassthrough(plain).canWarmContinue(death) {
		t.Fatal("a text-only turn with the gate armed should warm-continue")
	}
	thinking := `{"model":"m","max_tokens":100,"thinking":{"type":"enabled","budget_tokens":1024},` +
		`"messages":[{"role":"user","content":"hi"}]}`
	if newPassthrough(thinking).canWarmContinue(death) {
		t.Fatal("thinking enabled on the request must stand warm-continue down (prefill unsupported)")
	}
	off := `{"model":"m","max_tokens":100,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`
	if !newPassthrough(off).canWarmContinue(death) {
		t.Fatal("thinking explicitly disabled should still warm-continue")
	}
}
