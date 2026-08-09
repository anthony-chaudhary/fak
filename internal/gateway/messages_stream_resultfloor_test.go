package gateway

// The #5525 witness. A result-floor rejection on the TRUE-streaming Anthropic passthrough
// answers the caller with a clean terminal 502 and then returns through the one terminal-switch
// arm that used to count and print nothing. "The client was told" and "the operator was told"
// are different questions: the client's 502 must stay exactly one clean body, AND the failure
// must land on /metrics and on the default debug stderr line — otherwise a whole class of 502s
// is missing from the upstream-error count and nobody notices a number that is too small.
//
// The failure injected here is the real one the arm's single producer surfaces: the inbound
// result floor QUARANTINES a poisoned tool result, the engine control plane it must then reset
// drops the connection, and the turn cannot be forwarded. The transport shape of that drop is
// what makes the assertion sharp — it classifies as `transport`, so observing the
// errPassthroughResponded sentinel instead of the real cause would show up as `other` and fail.

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

// resultFloorTurnFrames is a healthy upstream turn: the passthrough arms the result floor on
// message_start, so the upstream is never the thing that fails here. Whatever follows the first
// frame must NOT reach the client once the floor refuses the turn.
const resultFloorTurnFrames = `event: message_start
data: {"type":"message_start","message":{"id":"msg_rf1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":11,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"never relayed"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`

// droppedControlPlane stands in for the engine's cache-control endpoint dropping an already
// established connection mid-flight: a non-dial read failure, the transient-transport shape the
// error ladder classifies as `transport`. Deterministic and offline — no dial is attempted.
type droppedControlPlane struct{ hits int32 }

func (d *droppedControlPlane) RoundTrip(*http.Request) (*http.Response, error) {
	atomic.AddInt32(&d.hits, 1)
	return nil, &net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET}
}

// resultFloorInbound is a streaming Claude-Code-shaped turn whose last user message carries the
// tool_result the client just executed. The result body is an obviously fake placeholder
// credential — the same fixture the rest of this package uses — so the real context-MMU screen
// QUARANTINES it and the quarantine drives the engine-cache reset that then fails.
func resultFloorInbound() []byte {
	return []byte(`{"model":"claude-test","max_tokens":4096,"stream":true,"messages":[` +
		`{"role":"user","content":"look up the config"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_fake01","name":"fetch_url","input":{"url":"https://example.invalid/config"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_fake01",` +
		`"content":"config loaded. api_key=sk-abcdef0123456789abcdef0123 was found in env"}]}` +
		`]}`)
}

func TestMessagesStreamResultFloorErrorIsCountedAndRendered(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New()) // the REAL result-side screen, not a double

	var upstreamHits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamHits, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, resultFloorTurnFrames)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	var dbgMu sync.Mutex
	var dbg []string
	srv, err := New(Config{
		EngineID: "test", Model: "claude-test", BaseURL: upstream.URL,
		Provider: "anthropic", APIKey: "configured-key",
		DebugStatsf: func(format string, args ...any) {
			dbgMu.Lock()
			defer dbgMu.Unlock()
			dbg = append(dbg, fmt.Sprintf(format, args...))
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	// The engine whose prefix cache must be reset when a poisoned result is quarantined. Its
	// control plane drops every connection, so the reset fails and the floor refuses the turn.
	drop := &droppedControlPlane{}
	srv.engineCache = &enginecache.Client{
		Engine:     enginecache.EngineSGLang,
		BaseURL:    "http://engine-control.invalid",
		HTTPClient: &http.Client{Transport: drop},
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, resultFloorInbound())

	// The setup must actually have driven the result floor into the engine-cache reset;
	// otherwise every assertion below would be vacuous.
	if got := atomic.LoadInt32(&drop.hits); got != 1 {
		t.Fatalf("engine cache control plane hit %d times, want 1 — the quarantine did not drive the reset, so no result-floor rejection happened", got)
	}
	if got := atomic.LoadInt32(&upstreamHits); got != 1 {
		t.Fatalf("upstream hit %d times, want 1", got)
	}

	// (1) The client still gets exactly ONE clean terminal 502 — no second body, no SSE.
	if status != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", status, body)
	}
	if n := strings.Count(body, "upstream model error"); n != 1 {
		t.Fatalf("client body carries the 502 message %d times, want exactly 1 (a duplicated/corrupted body): %q", n, body)
	}
	if strings.Contains(body, "event:") || strings.Contains(body, "message_start") || strings.Contains(body, "never relayed") {
		t.Fatalf("client body leaked upstream SSE frames after the terminal error: %q", body)
	}

	// (2) The failure is COUNTED on /metrics — the whole point of #5525.
	srv.metrics.upstreamErrMu.Lock()
	counts := map[string]uint64{}
	var total uint64
	for k, v := range srv.metrics.upstreamErrors {
		counts[k] = v
		total += v
	}
	srv.metrics.upstreamErrMu.Unlock()
	if total != 1 {
		t.Fatalf("upstream-error counter total = %d, want exactly 1 — a result-floor 502 must not be invisible on /metrics (counts=%v)", total, counts)
	}
	// ... and counted under the REAL cause's kind. The errPassthroughResponded sentinel means
	// "a response was already written", not "the turn failed", and would land in `other`.
	if counts["transport"] != 1 {
		t.Fatalf("upstream-error kinds = %v, want transport=1: the counted error must be the real cause, not the control-flow sentinel", counts)
	}

	// (3) ... and PRINTED on the default debug line, so an operator watching stderr sees it.
	dbgMu.Lock()
	lines := append([]string(nil), dbg...)
	dbgMu.Unlock()
	failed := 0
	for _, l := range lines {
		if strings.Contains(l, "FAILED") && strings.Contains(l, "wire=anthropic_messages") {
			failed++
		}
	}
	if failed != 1 {
		t.Fatalf("debug FAILED lines = %d, want exactly 1: %v", failed, lines)
	}
}
