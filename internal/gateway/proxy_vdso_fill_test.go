package gateway

// proxy_vdso_fill_test.go — the loop that makes "vDSO live in the hot path" pay off on
// a PURE PROXY. The served-inline path (served_inline_test.go) can only fire if the
// cache is warm, and on a proxy fak never executes the tools — so nothing fills the
// vDSO. This is the missing half: admitInboundResults warms tier-2 from the ADMITTED
// inbound tool_result the client sends back, so the NEXT identical read is served
// locally. These tests prove the loop closes AND that every soundness/security guard
// holds (write-shaped / quarantined / empty-principal / Shareable never fill).

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// vdsoFills reports the vDSO's cumulative tier-2 fill count (the fill side of Stats).
func vdsoFills(v *vdso.VDSO) int64 {
	_, _, fills, _ := v.Stats()
	return fills
}

// newProxyFillServer is newSharingServer with the proxy-fill opt-in ON, returning the
// fresh vDSO so a test can read its fill counters directly.
func newProxyFillServer(t *testing.T) (*Server, *vdso.VDSO) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	// The result-side screen must be live so a poisoned inbound result is quarantined
	// (and therefore never fills) — that is the guard the quarantine case exercises.
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	v := vdso.New(vdso.DefaultCacheSize)
	v.SetGranularity(vdso.Global)
	abi.RegisterFastPath(1, v) // the served probe reads this via abi.FastPaths()
	abi.RegisterEmitter(v)     // fillVDSOFromResult emits EvComplete to this instance

	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true, VDSOProxyFill: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv, v
}

// inboundTurn is the message history a client sends back after running one tool: the
// assistant tool_use (carrying the args) + the user tool_result (carrying the answer),
// paired by id. This is exactly what DecodeAnthropicMessagesRequest produces.
func inboundTurn(id, tool, args, result string) []agent.Message {
	return []agent.Message{
		{Role: agent.RoleUser, Content: "go"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: id, Type: "function", Function: agent.Func{Name: tool, Arguments: args}},
		}},
		{Role: agent.RoleTool, ToolCallID: id, Name: tool, Content: result},
	}
}

// reproposeServed runs adjudicateProposedServed for a re-proposed call and reports
// whether it was served inline (servedHits) and whether it survived to the wire (kept).
func reproposeServed(srv *Server, ctx context.Context, tool, args string) (servedHits, kept int) {
	calls := []agent.ToolCall{{ID: "re", Type: "function", Function: agent.Func{Name: tool, Arguments: args}}}
	k, _, _, _, hits := srv.adjudicateProposedServed(ctx, calls, "trace-2")
	return hits, len(k)
}

// TestProxyFill_LoopCloses is the headline: turn 1 the client returns a read-only
// tool_result; the gateway warms the vDSO from it; turn 2 the model re-proposes the
// SAME read and it is SERVED INLINE — no tool_use survives, the client never re-runs it.
func TestProxyFill_LoopCloses(t *testing.T) {
	srv, v := newProxyFillServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")
	const tool, args, result = "get_config", `{"k":"v"}`, `{"answer":42}`

	fillsBefore := vdsoFills(v)
	if _, err := srv.admitInboundResults(ctx, inboundTurn("c1", tool, args, result), nil, "trace-1"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	fillsAfter := vdsoFills(v)
	if fillsAfter != fillsBefore+1 {
		t.Fatalf("turn 1 did not fill the vDSO from the inbound result: fills %d -> %d, want +1", fillsBefore, fillsAfter)
	}

	// Turn 2: the SAME read, re-proposed by the model, must be served inline.
	hits, kept := reproposeServed(srv, ctx, tool, args)
	if hits != 1 {
		t.Fatalf("turn 2: servedHits=%d, want 1 (the warmed read should be served inline)", hits)
	}
	if kept != 0 {
		t.Fatalf("turn 2: %d calls survived to the wire, want 0 (the client must NOT re-run a served read)", kept)
	}
}

// TestProxyFill_KeyCanonicalization proves the fill key and the probe key agree even
// when the re-proposed args differ only in whitespace/order — argHash canonicalizes JSON.
func TestProxyFill_KeyCanonicalization(t *testing.T) {
	srv, _ := newProxyFillServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")
	const tool = "get_config"

	if _, err := srv.admitInboundResults(ctx, inboundTurn("c1", tool, `{"a":1,"b":2}`, `{"ok":true}`), nil, "trace-1"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	// Re-propose with reordered keys + extra whitespace — must still hit.
	hits, kept := reproposeServed(srv, ctx, tool, `{ "b" : 2 , "a" : 1 }`)
	if hits != 1 || kept != 0 {
		t.Fatalf("canonicalized re-propose: servedHits=%d kept=%d, want 1/0 (JSON key-order/whitespace must not miss)", hits, kept)
	}
}

// TestProxyFill_WriteShapedNeverFills: a write-shaped tool's result must never warm the
// cache, and a write tool is never even probed on the serve side.
func TestProxyFill_WriteShapedNeverFills(t *testing.T) {
	srv, v := newProxyFillServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")
	const tool, args = "update_config", `{"k":"v"}` // "update" is write-shaped

	before := vdsoFills(v)
	if _, err := srv.admitInboundResults(ctx, inboundTurn("c1", tool, args, `{"ok":true}`), nil, "trace-1"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	after := vdsoFills(v)
	if after != before {
		t.Fatalf("a write-shaped result filled the vDSO: fills %d -> %d, want unchanged", before, after)
	}
	if hits, _ := reproposeServed(srv, ctx, tool, args); hits != 0 {
		t.Fatalf("a write-shaped tool was served inline (hits=%d), want 0", hits)
	}
}

// TestProxyFill_EmptyPrincipalNeverFills: a client fill with no principal would land in
// the shared global slice — refused.
func TestProxyFill_EmptyPrincipalNeverFills(t *testing.T) {
	srv, v := newProxyFillServer(t)
	ctx := context.Background() // NO principal
	const tool, args = "get_config", `{"k":"v"}`

	before := vdsoFills(v)
	if _, err := srv.admitInboundResults(ctx, inboundTurn("c1", tool, args, `{"ok":true}`), nil, "trace-1"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	after := vdsoFills(v)
	if after != before {
		t.Fatalf("an unattributed (empty-principal) result filled the vDSO: fills %d -> %d, want unchanged", before, after)
	}
}

// TestProxyFill_ShareableNeverFills: a Shareable tool drops the principal dimension, so a
// client fill into one would be cross-tenant — refused.
func TestProxyFill_ShareableNeverFills(t *testing.T) {
	srv, v := newProxyFillServer(t)
	v.RegisterShareable("get_config")
	ctx := WithPrincipal(context.Background(), "tenantA")
	const tool, args = "get_config", `{"k":"v"}`

	before := vdsoFills(v)
	if _, err := srv.admitInboundResults(ctx, inboundTurn("c1", tool, args, `{"ok":true}`), nil, "trace-1"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	after := vdsoFills(v)
	if after != before {
		t.Fatalf("a Shareable tool was filled from a client result: fills %d -> %d, want unchanged", before, after)
	}
}

// TestProxyFill_OffByDefault: with VDSOProxyFill off, an inbound result NEVER warms the
// cache — the default behavior is byte-for-byte unchanged.
func TestProxyFill_OffByDefault(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	v := vdso.New(vdso.DefaultCacheSize)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)
	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true}) // VDSOProxyFill defaults false
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	ctx := WithPrincipal(context.Background(), "tenantA")
	before := vdsoFills(v)
	if _, err := srv.admitInboundResults(ctx, inboundTurn("c1", "get_config", `{"k":"v"}`, `{"ok":true}`), nil, "trace-1"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	after := vdsoFills(v)
	if after != before {
		t.Fatalf("proxy-fill fired with the flag OFF: fills %d -> %d, want unchanged", before, after)
	}
}
