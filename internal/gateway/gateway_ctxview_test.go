package gateway

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// issue #555: the live serve/guard loop wires the ctxplan planner in as a planned view that
// replaces appending the whole transcript. The two load-bearing properties a production
// deploy needs before an in-flight rewrite of turn history ships — both exercised through
// the gateway's own maybePlanMessages hook:
//   - OFF (the default) is byte-for-byte unchanged: the forwarded history is an identity.
//   - ON keeps the resident view under budget AND rewrites (a planned view, not a no-op).
//
// The exact-recall half (any elided span demand-pages back to its verbatim original) is
// witnessed at the seam the gateway calls — TestCtxSeamRenderTurnPlansViewAndKeepsExactRecall
// in internal/agent — since the gateway uses that seam unchanged.

// ctxViewSession is a multi-turn transcript whose full resident size exceeds a tight token
// budget, so an enabled planner must ELIDE at least one span (the forecast MISS). The last
// user turn's intents predict "auth"/"token"/"refund", so the weather span is the miss.
func ctxViewSession() []agent.Message {
	return []agent.Message{
		{Role: agent.RoleSystem, Content: "You are a support agent. Use the tools to help."},
		{Role: agent.RoleUser, Content: "rotate the auth token and check the refund policy"},
		{Role: agent.RoleTool, Name: "WebSearch", Content: "auth token rotation runbook: mint roll revoke"},
		{Role: agent.RoleTool, Name: "Read", Content: "refund policy: full refund within 30 days"},
		{Role: agent.RoleTool, Name: "Read", Content: "weather sunny 22C light wind from the west"},
	}
}

// messageTokens is the bytes/4 token estimate the planner charges and the render realizes,
// so a planned history's messageTokens equals the view's RenderedTokens (<= budget).
func messageTokens(ms []agent.Message) int {
	n := 0
	for _, m := range ms {
		n += (len(m.Content) + 3) / 4
	}
	return n
}

// TestCtxViewOffIsByteForByteUnchanged is the guard: with no budget configured the gateway
// forwards the EXACT history it received. maybePlanMessages is an inert identity until an
// operator opts in, so a deploy that leaves --ctx-view-budget at 0 sees no behavior change
// at all — the property the issue's honesty clause requires ("with the flag OFF the existing
// path is byte-for-byte unchanged").
func TestCtxViewOffIsByteForByteUnchanged(t *testing.T) {
	ctx := context.Background()
	in := ctxViewSession()

	// The default, disabled state: no planner wired at all.
	off := &Server{} // ctxView == nil
	if got := off.maybePlanMessages(ctx, in); !reflect.DeepEqual(got, in) {
		t.Fatalf("OFF (nil planner) must return the input byte-for-byte unchanged: got %+v want %+v", got, in)
	}
	// An explicitly disabled planner is the same inert identity.
	disabled := &Server{ctxView: &agent.CtxViewPlanner{Enabled: false, Budget: 1024}}
	if got := disabled.maybePlanMessages(ctx, in); !reflect.DeepEqual(got, in) {
		t.Fatalf("a disabled planner must be an identity: got %+v want %+v", got, in)
	}
}

// TestCtxViewOnRewritesHistoryUnderBudget is the ON witness: an enabled planner rewrites the
// forwarded history into an O(1) resident view that stays AT OR UNDER the configured token
// budget — bounded residency, the property that lets a long session hold a constant resident
// set instead of appending the whole transcript.
func TestCtxViewOnRewritesHistoryUnderBudget(t *testing.T) {
	ctx := context.Background()
	const budget = 28 // tight window: elides at least one span (the weather miss)
	on := &Server{
		ctxView: &agent.CtxViewPlanner{Enabled: true, Budget: budget},
		logf:    func(string, ...any) {},
	}
	session := ctxViewSession()

	planned := on.maybePlanMessages(ctx, session)

	// The rewrite happened: the planned history differs from the full transcript.
	if reflect.DeepEqual(planned, session) {
		t.Fatal("ON path must rewrite the history (planned must differ from the full transcript)")
	}
	// BOUNDED: the rendered view is at or under the configured token budget.
	if tokens := messageTokens(planned); tokens > budget {
		t.Errorf("planned view %d tokens must be <= budget %d", tokens, budget)
	}
	// Every rendered message carries its paged-in bytes (a faithful resident span is never
	// empty) — the view genuinely rendered as turn history, not a stub.
	if len(planned) == 0 {
		t.Fatal("the planned history must be non-empty")
	}
	for _, m := range planned {
		if len(m.Content) == 0 {
			t.Errorf("a rendered resident span must carry its bytes, got empty content role=%q", m.Role)
		}
	}
	// The full transcript exceeds the budget, so a faithful plan must have elided something:
	// the planned history is strictly smaller than the full session.
	if messageTokens(session) <= budget {
		t.Fatalf("test session must exceed the budget for the rewrite to be meaningful: full=%d budget=%d",
			messageTokens(session), budget)
	}
}
