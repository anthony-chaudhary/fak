package gateway

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// gateway_ctxview_stream_http_test.go — the STREAMING-path twin of
// gateway_ctxview_http_test.go (issue #1092). The existing ctxview HTTP witnesses all
// drive the BUFFERED Anthropic passthrough (no stream:true → streamAnthropicPending),
// so the byte-identity of the cache_control prefix and the resident messages on the
// TRUE-streaming flagship (streamAnthropicPassthroughLive, messages_stream_passthrough.go)
// was code-proven only — never asserted on the bytes that reach the wire.
//
// This file closes that gap. It drives `stream:true` so messages.go:244 takes the live
// streamAnthropicPassthroughLive branch, which forwards req.Raw verbatim through
// StreamAnthropicRaw. maybePlanAnthropicRaw (messages.go:189) runs BEFORE that branch
// and mutates req.Raw, so the same #927 ctxview transform reaches the streaming wire as
// the buffered one. We assert, on the body the upstream actually received, that with the
// budget ON the off-topic middle turn is stubbed while the cached system prefix bytes and
// every resident message stay byte-identical — and with the budget OFF (the default 0)
// the streaming passthrough forwards req.Raw byte-for-byte.
//
// To take the live branch the mock upstream MUST answer as a real Anthropic stream would:
// StreamAnthropicRaw returns ErrStreamingUnsupported (→ buffered fallback) unless the
// upstream replies with Content-Type: text/event-stream. So the mock both CAPTURES the
// forwarded body (the post-transform req.Raw under assertion) and emits a minimal-but-valid
// Anthropic Messages SSE turn, exercising the genuine streaming relay end-to-end.

// anthropicStreamFrames is a minimal, well-formed Anthropic Messages SSE turn: a
// message_start carrying the usage block streamAnthropicPassthroughLive reads, one text
// content block, and the terminal message_delta/message_stop. The same frame grammar the
// agent-layer stall test (internal/agent/stream_stall_test.go) drives StreamAnthropicRaw
// with, so the relay's onEvent state machine runs exactly as it does against the real API.
const anthropicStreamFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_strm","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":11,"cache_read_input_tokens":7,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

// anthropicStreamUpstream stands up a mock that answers like the real Anthropic streaming
// API: it records the forwarded request body (the post-ctxview-transform req.Raw under
// assertion) into *got, then writes a text/event-stream SSE turn. Returning the
// event-stream content type is what makes StreamAnthropicRaw take the live relay instead
// of returning ErrStreamingUnsupported (which would silently fall back to the buffered path
// and defeat the point of this test).
func anthropicStreamUpstream(t *testing.T, got *[]byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*got, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, anthropicStreamFrames)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
}

// postAnthropicStream sends a streaming /v1/messages request through the gateway handler,
// drains the SSE body, and fails on a non-200. The drained client stream is discarded —
// this test asserts on the UPSTREAM body the gateway forwarded, not the relayed events.
func postAnthropicStream(t *testing.T, gatewayURL string, inbound []byte) {
	t.Helper()
	req, _ := http.NewRequest("POST", gatewayURL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}
	// Drain the SSE stream so the relay runs to completion (message_stop) before we assert.
	_, _ = io.Copy(io.Discard, resp.Body)
}

// TestCtxViewStreamPassthroughOffForwardsRaw is the OFF live-path guard for the STREAMING
// flagship wire: with CtxViewBudget == 0 (the default) and stream:true, the true-streaming
// passthrough (streamAnthropicPassthroughLive → StreamAnthropicRaw) forwards req.Raw
// byte-for-byte. A deploy that does not opt in sees the caller's original body, unmodified,
// on the streaming wire too — the cache_control prefix and every message ride through intact.
func TestCtxViewStreamPassthroughOffForwardsRaw(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	// A real Claude-Code-shaped streaming body with a cache_control prefix. stream:true is
	// present, so forceAnthropicStreaming returns it byte-identical (its exact prefix kept).
	inbound := []byte(`{"model":"claude-test","max_tokens":4096,"stream":true,` +
		`"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"rotate the auth token and check the refund policy"},` +
		`{"role":"assistant","content":"weather sunny 22C light wind from the west, unrelated padding to exceed any small resident budget"},` +
		`{"role":"user","content":"what is the auth token rotation and refund window"}]}`)

	var upstreamBody []byte
	upstream := anthropicStreamUpstream(t, &upstreamBody)
	defer upstream.Close()

	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key", CtxViewBudget: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	postAnthropicStream(t, ts.URL, inbound)

	// OFF: the streaming passthrough forwarded the ORIGINAL bytes, byte-for-byte. If this
	// fails the upstream prompt-cache hit would be lost on the streaming path even at the
	// default budget — the regression a default flip must never introduce.
	if !bytes.Equal(upstreamBody, inbound) {
		t.Errorf("OFF: streaming passthrough must forward req.Raw byte-for-byte when CtxViewBudget==0:\n got %q\nwant %q", upstreamBody, inbound)
	}
}

// TestCtxViewStreamPassthroughPlansView is the #1092 streaming-path witness: with
// CtxViewBudget set AND stream:true, the ctxplan planned view reaches the TRUE-streaming
// Anthropic passthrough (maybePlanAnthropicRaw runs before the stream branch and mutates
// req.Raw, which StreamAnthropicRaw forwards verbatim). It asserts, on the body the
// upstream actually received over the streaming wire:
//   - the off-topic middle turn is stubbed out (the bounded view), replaced by a same-role
//     [fak] ctxview-elided stub, not forwarded verbatim;
//   - the two resident auth/refund turns survive byte-for-byte;
//   - the cached system prefix bytes are byte-IDENTICAL so the upstream cache hit survives;
//   - the message count / role alternation is preserved so Anthropic accepts the body.
//
// This is the streaming twin of TestCtxViewHTTPAnthropicPassthroughPlansView, which proved
// the SAME identity on the buffered wire — the gap issue #1092 names is that no test drove
// stream:true through streamAnthropicPassthroughLive.
func TestCtxViewStreamPassthroughPlansView(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	// Same transcript as the buffered witness, but stream:true so messages.go:244 takes the
	// live streamAnthropicPassthroughLive branch instead of the buffered streamAnthropicPending.
	inbound := []byte(`{"model":"claude-test","max_tokens":4096,"stream":true,` +
		`"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"rotate the auth token and check the refund policy"},` +
		`{"role":"assistant","content":"weather sunny 22C light wind from the west, unrelated padding to exceed any small resident budget"},` +
		`{"role":"user","content":"what is the auth token rotation and refund window"}]}`)

	var upstreamBody []byte
	upstream := anthropicStreamUpstream(t, &upstreamBody)
	defer upstream.Close()

	// A tight budget that forces the off-topic assistant turn (the forecast MISS) to be elided
	// beyond the cached system prefix — the same budget the buffered witness uses.
	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key", CtxViewBudget: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	postAnthropicStream(t, ts.URL, inbound)

	forwarded := string(upstreamBody)
	if forwarded == "" {
		t.Fatal("streaming upstream received no body — the live passthrough branch did not run (check that the mock returns text/event-stream so StreamAnthropicRaw does not fall back to buffered)")
	}
	// The forwarded body MUST differ from the inbound: if it is byte-equal, the planner did
	// NOT reach the streaming wire and this test would pass vacuously.
	if bytes.Equal(upstreamBody, inbound) {
		t.Fatalf("ON: the forwarded streaming body is byte-identical to the inbound — the ctxview transform did not reach the streaming wire:\n%s", forwarded)
	}

	// (a) BOUNDED VIEW: the off-topic weather span was elided from the forwarded body —
	// replaced by a same-role [fak] ctxview-elided stub, not forwarded verbatim. The two
	// resident turns (the auth/refund user turns) are still present, byte-for-byte.
	if strings.Contains(forwarded, "weather sunny 22C") {
		t.Errorf("(a): the off-topic span must be elided from the forwarded streaming body, still present:\n%s", forwarded)
	}
	if !strings.Contains(forwarded, "[fak] ctxview-elided") {
		t.Errorf("(a): the elided turn must be replaced by a [fak] ctxview-elided stub on the streaming wire:\n%s", forwarded)
	}
	if !strings.Contains(forwarded, "rotate the auth token") || !strings.Contains(forwarded, "what is the auth token rotation") {
		t.Errorf("(a): the resident turns must still be present verbatim in the forwarded streaming body:\n%s", forwarded)
	}

	// (b) PREFIX BYTE-IDENTITY: the cached system prefix bytes are unchanged so the upstream
	// cache hit (cache_read_input_tokens) survives on the streaming wire. The system field is
	// outside messages[], so it rides through untouched; assert it byte-for-byte.
	wantSys := `"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}]`
	if !strings.Contains(forwarded, wantSys) {
		t.Errorf("(b): the cached system prefix must be byte-identical in the forwarded streaming body:\nwant %s\ngot  %s", wantSys, forwarded)
	}
	// The stream flag rode through too (forceAnthropicStreaming kept it true, so the cached
	// prefix was not re-marshalled): the body the upstream saw still asked to stream.
	if !strings.Contains(forwarded, `"stream":true`) {
		t.Errorf("(b): the forwarded streaming body must still carry stream:true:\n%s", forwarded)
	}
	// The message COUNT and role alternation are preserved (same-role stubs), so Anthropic
	// accepts the body — verify three messages survive (one stubbed, two resident).
	if c := strings.Count(forwarded, `"role":`); c != 3 {
		t.Errorf("(b): the forwarded streaming body must keep all 3 messages (one stubbed in place), got %d role keys:\n%s", c, forwarded)
	}
}
