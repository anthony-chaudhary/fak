package gateway

// The #5536 witness — the PLANNER-LIVE arm of the defect #5525 closed on the true-streaming
// Anthropic passthrough. When fak fronts an OpenAI-compatible/vLLM/SGLang upstream, a /v1/messages
// stream is served by streamAnthropicPlannerLive, whose FIRST act is the inbound result floor. A
// floor rejection there answers the caller with a clean terminal 502 and returns — and used to
// count and print nothing at all, so a whole class of 502s was missing from the upstream-error
// count on a wire #5525's fix never touches. "The client was told" and "the operator was told" are
// different questions; both must hold.
//
// The failure injected is the real one this arm's single producer surfaces, and it is the same
// mechanism the #5525 witness drives: the inbound result floor QUARANTINES a poisoned tool result,
// the engine control plane it must then reset drops the connection, and admitInboundResults
// returns that error. Three things make the assertions sharp rather than decorative:
//
//   - the upstream planner is asserted NEVER hit, so the observation cannot be coming from
//     streamPlannerUpstreamError (which has always observed) — only the pre-stream floor can
//     have produced it;
//   - the route is asserted to BE the planner-live one (not a passthrough, not the buffered
//     fallback), so the test cannot silently re-measure the arm #5525 already fixed;
//   - the drop is a non-dial *net.OpError, which the kind ladder classifies as `transport`, so
//     counting some freshly-minted stand-in error instead of the real cause would show up as
//     `other` and fail.
//
// The poisoned inbound fixture (resultFloorInbound) is shared with the #5525 witness — one
// deliberately fake, obviously-placeholder credential shape, defined once.

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

func TestMessagesStreamPlannerResultFloorErrorIsCountedAndRendered(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New()) // the REAL result-side screen, not a double

	// The planner upstream. The result floor runs BEFORE the stream opens, so a healthy run of
	// this test never reaches it; any hit means the failure came from somewhere else.
	var upstreamHits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&upstreamHits, 1)
		http.Error(w, "the planner upstream must not be reached", http.StatusInternalServerError)
	}))
	defer upstream.Close()

	var dbgMu sync.Mutex
	var dbg []string
	srv, err := New(Config{
		EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat",
		Provider: "openai-compatible",
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

	// Route guards. Without these the test could pass while measuring the passthrough arm
	// #5525 already fixed, or the buffered fallback — neither of which is this ticket.
	if srv.anthropicPassthroughFor("claude-test") {
		t.Fatal("this server would take the Anthropic passthrough, not the planner-live route — the #5536 arm would never run")
	}
	sp, ok := srv.planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		t.Fatalf("planner %T does not stream, so streamAnthropicPlannerLive returns false and the request falls back to the buffered path", srv.planner)
	}

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

	// The setup must actually have driven the result floor into the engine-cache reset, and the
	// planner must never have run; otherwise every assertion below would be vacuous or would be
	// crediting streamPlannerUpstreamError's long-standing observation instead of this arm's.
	if got := atomic.LoadInt32(&drop.hits); got != 1 {
		t.Fatalf("engine cache control plane hit %d times, want 1 — the quarantine did not drive the reset, so no result-floor rejection happened", got)
	}
	if got := atomic.LoadInt32(&upstreamHits); got != 0 {
		t.Fatalf("planner upstream hit %d times, want 0 — the floor must refuse BEFORE the stream opens, or the counted error is streamPlannerUpstreamError's, not the floor's", got)
	}

	// (1) The client still gets exactly ONE clean terminal 502 — no second body, no SSE.
	if status != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", status, body)
	}
	if n := strings.Count(body, "upstream model error"); n != 1 {
		t.Fatalf("client body carries the 502 message %d times, want exactly 1 (a duplicated/corrupted body): %q", n, body)
	}
	if strings.Contains(body, "event:") || strings.Contains(body, "message_start") {
		t.Fatalf("client body leaked SSE frames alongside the terminal error: %q", body)
	}

	// (2) The failure is COUNTED on /metrics — the whole point of #5536.
	srv.metrics.upstreamErrMu.Lock()
	counts := map[string]uint64{}
	var total uint64
	for k, v := range srv.metrics.upstreamErrors {
		counts[k] = v
		total += v
	}
	srv.metrics.upstreamErrMu.Unlock()
	if total != 1 {
		t.Fatalf("upstream-error counter total = %d, want exactly 1 — a planner-live result-floor 502 must not be invisible on /metrics (counts=%v)", total, counts)
	}
	// ... and counted under the REAL cause's kind, the dropped engine control plane.
	if counts["transport"] != 1 {
		t.Fatalf("upstream-error kinds = %v, want transport=1: the counted error must be the real cause the floor returned, not a stand-in", counts)
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
