package gateway

// served_inline_test.go — vDSO live in the hot path.
//
// These tests prove the served-TURN vDSO fast path: when a model PROPOSES a read-only
// tool call on a buffered /v1/messages turn and the vDSO already holds a fresh result
// for it, the gateway serves it LOCALLY (folds the answer into assistant text, drops
// the tool_use so the client never re-runs it) instead of bouncing the call back to the
// client to execute. The vDSO was previously consulted only on the explicit
// /v1/fak/syscall path; this is the wiring that puts it on the served model turn.
//
// The three cases are the contract:
//   - FULL-SERVE: every proposed read is a vDSO hit -> no surviving tool_use, the answer
//     is assistant text, stop_reason=end_turn, fak_gateway_served_inline_total bumped,
//     and the engine is NOT re-dispatched (the round-trip is saved).
//   - PARTIAL: one warm + one cold read -> the cold call survives as tool_use
//     (stop_reason=tool_use) AND the served answer is present as text.
//   - MISS: a read the vDSO cannot answer -> byte-identical to today (the call survives
//     as tool_use, no served-text block). The regression guard for "MISS == today".

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// warmServedRead drives one read through the full k.Syscall boundary so the vDSO fills
// its tier-2 entry for (tool, args). The buffered served turn below proposes the SAME
// call, which must then be served from this fill.
func warmServedRead(t *testing.T, srv *Server, tool, args string) {
	t.Helper()
	// readOnly=true so buildCall stamps readOnlyHint+idempotentHint (the vDSO fill gate);
	// empty trace == empty principal, matching the served-turn probe's principal scoping.
	wv, _, err := srv.syscall(context.Background(), tool, args, true, "", "warm")
	if err != nil {
		t.Fatalf("warm read %s: %v", tool, err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("warm read %s: verdict=%q, want ALLOW", tool, wv.Kind)
	}
}

// proposeMessagesTurn posts a buffered /v1/messages turn whose scripted planner proposes
// the given tool calls, and returns the decoded response.
func proposeMessagesTurn(t *testing.T, srv *Server, calls []agent.ToolCall) anthropicMessageResponse {
	t.Helper()
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, ToolCalls: calls},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	body := `{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"go"}]}`
	return postAnthropic(t, ts.URL, "served-inline-trace", body)
}

func countToolUse(resp anthropicMessageResponse) (toolUse int, text string) {
	for _, b := range resp.Content {
		switch b.Type {
		case "tool_use":
			toolUse++
		case "text":
			text += b.Text
		}
	}
	return toolUse, text
}

// TestServedInline_FullServe is the headline: a re-proposed read the vDSO already holds
// fresh is served LOCALLY on the served turn — no tool_use survives, the answer is
// assistant text, the turn ends, the metric bumps, and the engine is not re-dispatched.
func TestServedInline_FullServe(t *testing.T) {
	srv, _ := newSharingServer(t, vdso.Global)
	const tool, args = "get_doc", `{"id":"refund-policy"}`

	warmServedRead(t, srv, tool, args) // fills tier-2; this is the only engine call.
	engineAfterWarm := srv.k.Counters().EngineCalls
	servedBefore := srv.metrics.servedInlineSnapshot()

	resp := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: args}},
	})

	toolUse, text := countToolUse(resp)
	if toolUse != 0 {
		t.Fatalf("full-serve: %d surviving tool_use blocks, want 0 (the read was served from the vDSO)", toolUse)
	}
	if !strings.Contains(text, tool) || !strings.Contains(text, "served from cache") {
		t.Fatalf("full-serve: served answer not folded into assistant text; got %q", text)
	}
	if resp.StopReason != "end_turn" {
		t.Fatalf("full-serve: stop_reason=%q, want end_turn (no surviving tool_use)", resp.StopReason)
	}
	if got := srv.metrics.servedInlineSnapshot(); got != servedBefore+1 {
		t.Fatalf("full-serve: served_inline counter %d -> %d, want +1", servedBefore, got)
	}
	if got := srv.k.Counters().EngineCalls; got != engineAfterWarm {
		t.Fatalf("full-serve: EngineCalls %d -> %d, the served turn re-dispatched to the engine (round-trip NOT saved)", engineAfterWarm, got)
	}
}

// TestServedInline_Partial: a warm read + a cold read in one turn. The warm read is
// served inline (text), the cold read survives as a tool_use the client still runs, so
// the turn keeps stop_reason=tool_use.
func TestServedInline_Partial(t *testing.T) {
	srv, _ := newSharingServer(t, vdso.Global)
	const warmTool, warmArgs = "get_doc", `{"id":"refund-policy"}`
	const coldTool, coldArgs = "get_other", `{"id":"never-warmed"}`

	warmServedRead(t, srv, warmTool, warmArgs)
	servedBefore := srv.metrics.servedInlineSnapshot()

	resp := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "warm", Type: "function", Function: agent.Func{Name: warmTool, Arguments: warmArgs}},
		{ID: "cold", Type: "function", Function: agent.Func{Name: coldTool, Arguments: coldArgs}},
	})

	toolUse, text := countToolUse(resp)
	if toolUse != 1 {
		t.Fatalf("partial: %d surviving tool_use, want 1 (the cold read the client must still run)", toolUse)
	}
	if !strings.Contains(text, warmTool) {
		t.Fatalf("partial: warm read not served inline as text; got %q", text)
	}
	if resp.StopReason != "tool_use" {
		t.Fatalf("partial: stop_reason=%q, want tool_use (a call survived)", resp.StopReason)
	}
	if got := srv.metrics.servedInlineSnapshot(); got != servedBefore+1 {
		t.Fatalf("partial: served_inline counter %d -> %d, want +1", servedBefore, got)
	}
}

// TestServedInline_Miss is the "MISS == today's behavior" regression guard: a read the
// vDSO has never seen is NOT served inline — it survives as a normal tool_use the client
// runs, no served-text block, the counter unchanged.
func TestServedInline_Miss(t *testing.T) {
	srv, _ := newSharingServer(t, vdso.Global)
	const tool, args = "get_cold", `{"id":"cold"}`
	servedBefore := srv.metrics.servedInlineSnapshot()

	resp := proposeMessagesTurn(t, srv, []agent.ToolCall{
		{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: args}},
	})

	toolUse, text := countToolUse(resp)
	if toolUse != 1 {
		t.Fatalf("miss: %d surviving tool_use, want 1 (an unserved read goes to the client)", toolUse)
	}
	if strings.Contains(text, "served from cache") {
		t.Fatalf("miss: a served-text block appeared for a cache miss; got %q", text)
	}
	if resp.StopReason != "tool_use" {
		t.Fatalf("miss: stop_reason=%q, want tool_use", resp.StopReason)
	}
	if got := srv.metrics.servedInlineSnapshot(); got != servedBefore {
		t.Fatalf("miss: served_inline counter moved %d -> %d on a cache miss", servedBefore, got)
	}
}
