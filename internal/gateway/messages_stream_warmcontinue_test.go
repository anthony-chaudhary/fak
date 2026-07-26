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
