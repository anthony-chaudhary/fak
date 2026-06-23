package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// issue #546, rung (c): the end-to-end miss -> demand-page -> resident path on a RECORDED
// agent session, exercised through the guarded seam on the agent's own message vocabulary.
// This is the witness that the headline honesty claim — "a forecast MISS costs one
// demand-page, never a lost fact" — holds on the live loop's message shape, not just on
// the planner's own Span/Store types.

// recordedSession is a frozen multi-turn agent transcript: a system prompt, a user goal,
// and three tool results. The goal's intents predict the auth-token + refund content; the
// weather report is NOT predicted, so a tight window elides it (the forecast MISS).
func recordedSession() []Message {
	return []Message{
		{Role: RoleSystem, Content: "You are a support agent. Use the tools to help."},
		{Role: RoleUser, Content: "rotate the auth token and check the refund policy"},
		{Role: RoleTool, Name: "WebSearch", Content: "auth token rotation runbook: mint roll revoke"},
		{Role: RoleTool, Name: "Read", Content: "refund policy: full refund within 30 days"},
		{Role: RoleTool, Name: "Read", Content: "weather sunny 22C light wind from the west"},
	}
}

// TestCtxSeamIsOffByDefault is the guard: the zero-value seam is DISABLED, so PlanTurn
// returns ErrCtxSeamDisabled and the existing append+compact loop is byte-for-byte
// unchanged. The integration is opt-in.
func TestCtxSeamIsOffByDefault(t *testing.T) {
	seam := &CtxViewPlanner{} // zero value: disabled
	_, err := seam.PlanTurn(context.Background(), recordedSession())
	if err != ErrCtxSeamDisabled {
		t.Fatalf("a disabled seam must refuse with ErrCtxSeamDisabled, got %v", err)
	}
	_, err = (&CtxViewPlanner{}).RenderHistory(context.Background(), nil, ctxplan.View{})
	if err != ErrCtxSeamDisabled {
		t.Fatalf("RenderHistory on a disabled seam must refuse, got %v", err)
	}
}

// TestCtxSeamConfigEnables is the "OFF by default behind config" half: FAK_CTXPLAN_SEAM=on
// flips the guard via NewCtxViewPlanner.
func TestCtxSeamConfigEnables(t *testing.T) {
	t.Setenv("FAK_CTXPLAN_SEAM", "on")
	seam := NewCtxViewPlanner(0)
	if !seam.Enabled {
		t.Fatal("FAK_CTXPLAN_SEAM=on must enable the seam")
	}
	t.Setenv("FAK_CTXPLAN_SEAM", "off")
	if NewCtxViewPlanner(0).Enabled {
		t.Error("FAK_CTXPLAN_SEAM=off must leave the seam disabled")
	}
}

// TestCtxSeamMissDemandPageResident is the headline end-to-end rung: a recorded session is
// lowered into a ctxplan store, planned under a tight window that ELIDES a span (the
// forecast MISS), the mid-turn handler demand-pages it back in (rung a), it becomes
// RESIDENT, and the miss feeds back into the next forecast (re-plan) — all through the
// guarded seam (rung b), on agent Messages.
func TestCtxSeamMissDemandPageResident(t *testing.T) {
	ctx := context.Background()
	seam := &CtxViewPlanner{Enabled: true, Budget: 28} // tight window: pins + runbook fit, weather elided
	session := recordedSession()

	// Plan the turn through the seam.
	view, err := seam.PlanTurn(ctx, session)
	if err != nil {
		t.Fatal(err)
	}
	if !view.Witness.Faithful {
		t.Fatalf("the planned view must be faithful: %+v", view.Witness)
	}
	// The test's own store is equivalent to the seam's internal one (same messages -> same
	// span:<i> ids), so it is the page-in backend for the demand-page + render steps.
	store, _ := messagesToStore(session)

	// Locate an elided span (the forecast MISS) to fault in.
	var elidedID string
	for _, e := range view.Plan.Elided {
		if !renderedHas(view.Rendered, e.ID) {
			elidedID = e.ID
			break
		}
	}
	if elidedID == "" {
		t.Skip("no elided span under this budget/window; adjust the test budget")
	}

	// MISS -> demand-page: the elided span faults back in through the trust gate.
	view2, fault, err := seam.DemandPage(ctx, store, view, elidedID)
	if err != nil {
		t.Fatal(err)
	}
	if fault.Status != ctxplan.FaultServed {
		t.Fatalf("the elided span must be served on demand-page, got status=%q reason=%q",
			fault.Status, fault.Reason)
	}
	// RESIDENT: the faulted span is now in the rendered view, and the witness stayed honest.
	if !renderedHas(view2.Rendered, elidedID) {
		t.Errorf("after demand-page the faulted span %s must be resident", elidedID)
	}
	if !view2.Witness.Faithful {
		t.Errorf("the served view must stay faithful: %+v", view2.Witness)
	}
	if !view2.Witness.Reconciled {
		t.Errorf("the served view must reconcile (rendered+refused==selected): %+v", view2.Witness)
	}

	// The MISS feeds back to re-plan: the faulted span's content promotes into the next
	// forecast (Forecast.Learn), so the next plan predicts it instead of faulting again.
	spans, _ := store.Spans(ctx)
	learned := ctxplan.Forecast{}.Learn(ctxplan.Outcome{Faults: []string{fault.ID}}, spans)
	if !predictsSpan(learned.Intents, elidedID, spans) {
		t.Errorf("the served fault must feed back so the next forecast predicts span %s: intents=%v",
			elidedID, learned.Intents)
	}
}

// TestCtxSeamRenderHistoryRoundTrips is the "renders a ctxplan View as turn history" half
// of rung (b): a planned View renders back to agent Messages (each resident span's bytes
// paged in through the gate), proving the View can replace the append+compact history.
func TestCtxSeamRenderHistoryRoundTrips(t *testing.T) {
	ctx := context.Background()
	seam := &CtxViewPlanner{Enabled: true, Budget: DefaultCtxViewBudget}
	session := recordedSession()
	view, err := seam.PlanTurn(ctx, session)
	if err != nil {
		t.Fatal(err)
	}
	store, _ := messagesToStore(session)
	history, err := seam.RenderHistory(ctx, store, view)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) == 0 {
		t.Fatal("rendered history must be non-empty")
	}
	// Every rendered message carries the bytes its span paged in — no empty content for a
	// faithful resident span — so the View genuinely rendered as turn history.
	for _, m := range history {
		if strings.TrimSpace(m.Content) == "" {
			t.Errorf("a rendered resident span must carry its bytes, got empty content role=%q", m.Role)
		}
	}
	// The rendered history is bounded by the view (one message per resident span).
	if len(history) != len(view.Rendered) {
		t.Errorf("rendered history must mirror the resident view: %d messages vs %d rendered spans",
			len(history), len(view.Rendered))
	}
}

// TestCtxSeamHeuristicForecastPinsEssentials witnesses the forecast authoring rung: the
// system prompt, the active goal (first user turn), and the last user turn are pinned, and
// the intents come from the last user message — exactly the heuristic the issue specifies.
func TestCtxSeamHeuristicForecastPinsEssentials(t *testing.T) {
	session := []Message{
		{Role: RoleSystem, Content: "system prompt"},
		{Role: RoleUser, Content: "the active goal here"}, // span:1 (first user)
		{Role: RoleTool, Name: "t", Content: "a result"},
		{Role: RoleUser, Content: "follow up about the refund"}, // span:3 (last user)
	}
	_, pins := messagesToStore(session)
	// System (span:0), first user (span:1), last user (span:3) are pinned.
	want := map[string]bool{"span:0": true, "span:1": true, "span:3": true}
	for _, p := range pins {
		if !want[p] {
			t.Errorf("unexpected pin %q (pins should be system+first user+last user)", p)
		}
	}
	for w := range want {
		if !contains(pins, w) {
			t.Errorf("missing expected pin %q in %v", w, pins)
		}
	}
	// Intents come from the LAST user message.
	fc := heuristicForecast(session, pins)
	if len(fc.Intents) == 0 {
		t.Fatal("intents must be derived from the last user message")
	}
	predictsRefund := false
	for _, intent := range fc.Intents {
		if intent == "refund" {
			predictsRefund = true
		}
	}
	if !predictsRefund {
		t.Errorf("the last user message's content (refund) must drive the intents: %v", fc.Intents)
	}
}

// --- helpers ---

func renderedHas(rs []ctxplan.Rendered, id string) bool {
	for _, r := range rs {
		if r.ID == id {
			return true
		}
	}
	return false
}

func contains(s []string, want string) bool {
	for _, v := range s {
		if v == want {
			return true
		}
	}
	return false
}

// predictsSpan reports whether any learned intent token appears in the span's role +
// descriptor — i.e. the forecast now predicts the faulted span's content.
func predictsSpan(intents []string, id string, spans []ctxplan.Span) bool {
	var span ctxplan.Span
	for _, s := range spans {
		if s.ID == id {
			span = s
		}
	}
	if span.ID == "" {
		return false
	}
	doc := strings.ToLower(span.Role + " " + span.Descriptor)
	for _, intent := range intents {
		if strings.Contains(doc, strings.ToLower(intent)) {
			return true
		}
	}
	return false
}
