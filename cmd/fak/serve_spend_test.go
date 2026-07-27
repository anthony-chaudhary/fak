package main

// serve_spend_test.go — the #4859 acceptance witnesses for the `fak serve` spend-cap CLI.
//
// #3273 proved the GOVERNOR at the gateway library boundary. What was unproven — and what
// this issue is — is that an OPERATOR can reach it: that real `fak serve` flags construct
// and attach a governor, that a served session crossing the configured cap is hard-stopped,
// and that the breach lands on the --budget-webhook sink. So the primary test drives the
// flags through the actual serve flagset (never a hand-built struct), attaches to a real
// gateway.Server, and posts real HTTP turns at it.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// spendChatPost posts one /v1/chat/completions turn tagged with a trace id — the served
// boundary the spend gate adjudicates.
func spendChatPost(t *testing.T, base, trace string) *http.Response {
	t.Helper()
	body, _ := json.Marshal(map[string]any{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", trace)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// TestServeSpendCapHardStopsServedSessionAndWebhooks is the PRIMARY #4859 acceptance, and it
// covers both bullets in one live path: `fak serve` flags (--spend-cap + --budget-webhook,
// parsed by the real serve flagset) build and attach a SpendGovernor; a served session that
// crosses the configured cap is hard-stopped by the kernel with the closed reason; and the
// breach is delivered to the operator's --budget-webhook as a structured event.
func TestServeSpendCapHardStopsServedSessionAndWebhooks(t *testing.T) {
	// The operator's webhook sink.
	got := make(chan map[string]any, 4)
	hook := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("webhook got %s ct=%q, want POST application/json", r.Method, r.Header.Get("Content-Type"))
		}
		var ev map[string]any
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &ev); err != nil {
			t.Errorf("decode webhook body: %v (raw=%s)", err, raw)
		}
		got <- ev
		w.WriteHeader(http.StatusNoContent)
	}))
	defer hook.Close()

	// Drive the REAL serve flagset — this is the half #3273 left unreachable, so a
	// hand-built serveFlags would prove nothing.
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--spend-cap", "session=tokens=5,action=kill", "--budget-webhook", hook.URL}); err != nil {
		t.Fatalf("serve flags rejected the spend-cap configuration: %v", err)
	}
	gov, scopeOf, err := buildSpendGovernor(sf.spendCap.Values(), *sf.spendScopeTrace, *sf.budgetWebhook)
	if err != nil {
		t.Fatalf("buildSpendGovernor: %v", err)
	}
	if gov == nil {
		t.Fatal("--spend-cap was configured but buildSpendGovernor returned no governor")
	}

	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless"})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()
	srv.SetSpendGovernor(gov, scopeOf)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Turn 1 is admitted (nothing spent yet); its debit crosses the 5-token cap.
	resp := spendChatPost(t, ts.URL, "spend-cli-1")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first turn status = %d, want 200 (admitted, then crosses the cap on debit)", resp.StatusCode)
	}

	// Turn 2 is HARD-STOPPED by the kernel before the model is consulted.
	resp = spendChatPost(t, ts.URL, "spend-cli-1")
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("second turn status = %d, want 409 (hard spend stop); body=%s", resp.StatusCode, raw)
	}
	for _, want := range []string{"BUDGET_SPEND_EXCEEDED", "session_stopped"} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("spend refusal body missing %q: %s", want, raw)
		}
	}

	// The breach reached the operator through --budget-webhook, tagged so a monitor can
	// tell it apart from the #743 context-budget events sharing the same URL.
	select {
	case ev := <-got:
		if ev["kind"] != spendBreachEventKind {
			t.Fatalf("webhook event kind = %v, want %q", ev["kind"], spendBreachEventKind)
		}
		if ev["scope"] != "session" || ev["id"] != "spend-cli-1" || ev["action"] != "kill" {
			t.Fatalf("webhook event = %v, want session/spend-cli-1/kill", ev)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("spend breach never reached the --budget-webhook sink")
	}
}

// TestBuildSpendGovernorInertWithoutFlag pins the default posture: no --spend-cap ⇒ no
// governor, so buildGateway never attaches one and the served path stays historical.
func TestBuildSpendGovernorInertWithoutFlag(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	gov, scopeOf, err := buildSpendGovernor(sf.spendCap.Values(), *sf.spendScopeTrace, *sf.budgetWebhook)
	if err != nil || gov != nil || scopeOf != nil {
		t.Fatalf("default serve = (gov=%v, scopeOf-nil=%t, err=%v), want an inert (nil, nil, nil) seam",
			gov, scopeOf == nil, err)
	}
}

// TestParseSpendCapGrammar covers the flag grammar, including the fail-loud cases: a budget
// that can never breach, and a bad scope/action/field are all typos in a SAFETY control, so
// they must refuse rather than boot an uncapped server that reads as capped.
func TestParseSpendCapGrammar(t *testing.T) {
	for _, tc := range []struct {
		raw   string
		scope gateway.SpendScope
		id    string
		want  gateway.SpendBudget
	}{
		{"session=200000", gateway.SpendScopeSession, "", gateway.SpendBudget{TokenCap: 200000}},
		{"tenant:acme=tokens=5000000,action=kill", gateway.SpendScopeTenant, "acme",
			gateway.SpendBudget{TokenCap: 5000000, Action: gateway.SpendActionKill}},
		{"team:core=usd=250000", gateway.SpendScopeTeam, "core", gateway.SpendBudget{USDMicros: 250000}},
		{" agent:worker-1 = tokens=10 , action=pause ", gateway.SpendScopeAgent, "worker-1",
			gateway.SpendBudget{TokenCap: 10, Action: gateway.SpendActionPause}},
	} {
		got, err := parseSpendCap(tc.raw)
		if err != nil {
			t.Errorf("parseSpendCap(%q) = error %v, want ok", tc.raw, err)
			continue
		}
		if got.scope != tc.scope || got.id != tc.id || got.budget != tc.want {
			t.Errorf("parseSpendCap(%q) = %v/%q/%+v, want %v/%q/%+v",
				tc.raw, got.scope, got.id, got.budget, tc.scope, tc.id, tc.want)
		}
	}

	for _, bad := range []string{
		"",                     // empty
		"session",              // no spec
		"session=",             // empty spec
		"galaxy=100",           // unknown scope
		"session=0",            // a zero cap can never breach
		"session=-5",           // negative
		"session=action=kill",  // no ceiling on any axis
		"session=tokens=abc",   // non-numeric
		"session=tokens=5,x=1", // unknown field
		"session=tokens=5,action=explode",
	} {
		if got, err := parseSpendCap(bad); err == nil {
			t.Errorf("parseSpendCap(%q) = %+v, want a refusal", bad, got)
		}
	}
}

// TestSpendScopeResolverBindsTraceSegments proves the scope resolver half of the ask: a
// trace id is mapped onto its tenant/team/agent/session ScopeKey by the operator's template.
func TestSpendScopeResolverBindsTraceSegments(t *testing.T) {
	fields, err := parseSpendScopeTrace("tenant/team/session")
	if err != nil {
		t.Fatalf("parseSpendScopeTrace: %v", err)
	}
	resolve := spendScopeResolver(fields)
	if resolve == nil {
		t.Fatal("a non-empty template must yield a resolver")
	}
	if got := resolve("acme/core/s-17"); got != (gateway.ScopeKey{Tenant: "acme", Team: "core", Session: "s-17"}) {
		t.Fatalf("resolve(acme/core/s-17) = %+v, want tenant=acme team=core session=s-17", got)
	}
	// A trace shorter than the template leaves the unnamed scopes empty (never charged).
	if got := resolve("acme"); got != (gateway.ScopeKey{Tenant: "acme"}) {
		t.Fatalf("resolve(acme) = %+v, want tenant=acme only", got)
	}
	// Empty template ⇒ nil resolver ⇒ the gateway's documented session-only default.
	if empty, err := parseSpendScopeTrace(""); err != nil || spendScopeResolver(empty) != nil {
		t.Fatalf("empty template = (%v, %v), want the nil session-only resolver", empty, err)
	}
	for _, bad := range []string{"tenant/galaxy", "tenant//session", "tenant/tenant"} {
		if _, err := parseSpendScopeTrace(bad); err == nil {
			t.Errorf("parseSpendScopeTrace(%q) accepted a malformed template", bad)
		}
	}
}

// TestBuildSpendGovernorRefusesUnenforceableScope is the anti-footgun witness: a tenant cap
// with no tenant field in the trace template would accumulate against the empty id and never
// fire. Booting that server would present an uncapped deployment as a capped one, so the
// flag combination is refused at startup instead.
func TestBuildSpendGovernorRefusesUnenforceableScope(t *testing.T) {
	_, _, err := buildSpendGovernor([]string{"tenant:acme=100"}, "", "")
	if err == nil {
		t.Fatal("a tenant cap with a session-only trace mapping must fail startup, not boot silently inert")
	}
	if !strings.Contains(err.Error(), "--spend-scope-trace") {
		t.Fatalf("refusal should name the fix (--spend-scope-trace); got %v", err)
	}
	// Naming the field makes the very same cap enforceable.
	gov, scopeOf, err := buildSpendGovernor([]string{"tenant:acme=100"}, "tenant/session", "")
	if err != nil || gov == nil || scopeOf == nil {
		t.Fatalf("tenant cap with tenant/session = (gov=%v, scopeOf-nil=%t, err=%v), want an armed governor",
			gov, scopeOf == nil, err)
	}
}
