package gateway

// spend_governor_test.go — the acceptance witnesses for the control-plane SPEND CAP
// (#3273, epic #3256): a session driven past a tiny budget is hard-stopped by the KERNEL
// (at the served boundary, not by asking the model), with a closed reason + a breach
// counter + a webhook; and budgets compose with tenancy — a tenant cap bounds the SUM of
// its sessions.

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestSpendGovernorHardStopsSessionAtBudget is the primary acceptance: drive a served
// session past a tiny per-session token budget and assert the gateway hard-refuses the
// NEXT turn (409) with the closed BUDGET_SPEND_EXCEEDED reason, a webhook fire, a breach
// counter, and the /metrics row — the whole "kill by the kernel, not by asking the model"
// contract on the live serve path.
func TestSpendGovernorHardStopsSessionAtBudget(t *testing.T) {
	srv := newTestServer(t)

	gov := NewSpendGovernor()
	// A 5-token session cap with the terminal action. One served turn below charges ~9
	// tokens (prompt+completion), so the first turn crosses it and the second is refused.
	gov.SetDefaultBudget(SpendScopeSession, SpendBudget{TokenCap: 5, Action: SpendActionKill})
	breaches := make(chan SpendBreach, 4)
	gov.OnBreach(func(b SpendBreach) { breaches <- b })
	srv.SetSpendGovernor(gov, nil) // nil resolver ⇒ session-only, Session=trace

	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 1, TotalTokens: 9},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Turn 1 is admitted (nothing spent yet) and its debit crosses the 5-token cap.
	resp := chatPostWithTrace(t, ts.URL, "spend-1")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first turn status = %d, want 200 (admitted, then crosses the cap on debit)", resp.StatusCode)
	}

	// The crossing fired the webhook exactly once, naming the closed action + reason.
	select {
	case b := <-breaches:
		if b.Scope != SpendScopeSession || b.ID != "spend-1" || b.Action != SpendActionKill || b.Reason != SpendReasonExceeded {
			t.Fatalf("webhook breach = %+v, want session/spend-1/kill/%s", b, SpendReasonExceeded)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("spend breach webhook never fired on the crossing turn")
	}

	// Turn 2 is HARD-STOPPED by the kernel: 409 carrying the closed reason and the kill
	// run-state token (session_stopped), before the model is ever consulted.
	resp = chatPostWithTrace(t, ts.URL, "spend-1")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("second turn status = %d, want 409 (hard spend stop); body=%s", resp.StatusCode, raw)
	}
	body := string(raw)
	for _, want := range []string{SpendReasonExceeded, "session_stopped"} {
		if !strings.Contains(body, want) {
			t.Fatalf("spend refusal body missing %q: %s", want, body)
		}
	}

	// The breach counter is on /metrics, keyed by scope + action, exactly once.
	if got := spendBreachCountIn(gov.Snapshot(), SpendScopeSession, SpendActionKill); got != 1 {
		t.Fatalf("session/kill breach count = %d, want 1", got)
	}
	if m := srv.renderMetrics(); !strings.Contains(m, `fak_gateway_spend_breaches_total{scope="session",action="kill"} 1`) {
		t.Fatalf("/metrics missing the spend-breach counter row:\n%s", m)
	}
}

// TestSpendGovernorTenantCapBoundsSumOfSessions is the tenancy-composition acceptance
// (#C5): two sessions under one tenant, neither of which alone crosses the tenant cap,
// together breach it — and then BOTH sessions are refused at the tenant scope, with the
// reversible pause action. Proven directly on the governor (the kernel), which is what the
// served boundary consults.
func TestSpendGovernorTenantCapBoundsSumOfSessions(t *testing.T) {
	gov := NewSpendGovernor()
	// Only a tenant cap; the sessions have no cap of their own, so any stop is the tenant
	// bounding the sum. Pause is the reversible action an operator can resume.
	gov.SetDefaultBudget(SpendScopeTenant, SpendBudget{TokenCap: 10, Action: SpendActionPause})
	breaches := make(chan SpendBreach, 4)
	gov.OnBreach(func(b SpendBreach) { breaches <- b })

	key1 := ScopeKey{Tenant: "acme", Session: "acme-s1"}
	key2 := ScopeKey{Tenant: "acme", Session: "acme-s2"}
	cost := SpendCost{InputTokens: 6} // 6 tokens each — neither alone reaches 10

	// Session 1 spends 6: tenant total 6 < 10, nobody breaches yet.
	gov.Charge(key1, cost)
	if br := gov.Evaluate(key1); br != nil {
		t.Fatalf("session 1 alone must not breach a 10-token tenant cap (spent 6); got %+v", br)
	}

	// Session 2 spends 6: tenant total 12 >= 10 — the tenant cap is crossed by the SUM.
	gov.Charge(key2, cost)
	select {
	case b := <-breaches:
		if b.Scope != SpendScopeTenant || b.ID != "acme" || b.Action != SpendActionPause {
			t.Fatalf("tenant breach = %+v, want tenant/acme/pause", b)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("tenant-sum breach webhook never fired")
	}

	// BOTH sessions are now refused — the breach is the shared tenant, not either session.
	for _, k := range []ScopeKey{key1, key2} {
		br := gov.Evaluate(k)
		if br == nil {
			t.Fatalf("session %q must be refused once its tenant cap is crossed by the sum", k.Session)
		}
		if br.Scope != SpendScopeTenant || br.Reason != SpendReasonExceeded {
			t.Fatalf("evaluate(%q) = %+v, want tenant-scope %s breach", k.Session, br, SpendReasonExceeded)
		}
	}

	// The counter records exactly one tenant/pause crossing (not one per refused session).
	if got := spendBreachCountIn(gov.Snapshot(), SpendScopeTenant, SpendActionPause); got != 1 {
		t.Fatalf("tenant/pause breach count = %d, want 1 (one crossing)", got)
	}
}

// TestSpendGovernorUnattachedIsHistorical is the no-regression guard: with no governor
// attached, the served path is byte-for-byte historical — spendBreach admits and
// chargeSpend is inert, so a session that would blow any cap is never refused.
func TestSpendGovernorUnattachedIsHistorical(t *testing.T) {
	srv := newTestServer(t)
	if br := srv.spendBreach("nobody"); br != nil {
		t.Fatalf("unattached governor must admit; got %+v", br)
	}
	// chargeSpend is a no-op (must not panic) when nothing is wired.
	srv.chargeSpend("nobody", agent.Usage{PromptTokens: 1_000_000, CompletionTokens: 1_000_000})
	if br := srv.spendBreach("nobody"); br != nil {
		t.Fatalf("charging an unattached governor must not create a breach; got %+v", br)
	}
}

// TestSpendGovernorPerScopeOverrideAndAxes covers the resolution ladder and the dollar
// axis: a per-id override beats the scope default, and a USD cap fires independently of
// the token axis.
func TestSpendGovernorPerScopeOverrideAndAxes(t *testing.T) {
	gov := NewSpendGovernor()
	gov.SetDefaultBudget(SpendScopeSession, SpendBudget{TokenCap: 100})
	gov.SetBudget(SpendScopeSession, "vip", SpendBudget{TokenCap: 1000}) // override: bigger cap

	// The default-capped session breaches at 100 tokens; the VIP override does not.
	gov.Charge(ScopeKey{Session: "plain"}, SpendCost{InputTokens: 100})
	gov.Charge(ScopeKey{Session: "vip"}, SpendCost{InputTokens: 100})
	if gov.Evaluate(ScopeKey{Session: "plain"}) == nil {
		t.Fatal("default-capped session must breach at its 100-token cap")
	}
	if br := gov.Evaluate(ScopeKey{Session: "vip"}); br != nil {
		t.Fatalf("VIP-override session (cap 1000) must not breach at 100 tokens; got %+v", br)
	}

	// A dollar-only cap fires on the USD axis with no token cap set.
	gov.SetBudget(SpendScopeAgent, "billing", SpendBudget{USDMicros: 5_000_000, Action: SpendActionKill})
	gov.Charge(ScopeKey{Agent: "billing"}, SpendCost{USDMicros: 5_000_000})
	br := gov.Evaluate(ScopeKey{Agent: "billing"})
	if br == nil || br.Action != SpendActionKill || br.USDCap != 5_000_000 {
		t.Fatalf("dollar-axis breach = %+v, want a kill at the $5 cap", br)
	}
}

func spendBreachCountIn(rows []SpendBreachCount, scope SpendScope, action SpendAction) uint64 {
	for _, r := range rows {
		if r.Scope == scope && r.Action == action {
			return r.Count
		}
	}
	return 0
}
