package main

// serve_spend.go — the `fak serve` CLI/policy path onto the control-plane SPEND CAP
// (#4859, closing the gap #3273 left open).
//
// #3273 shipped the governor itself (internal/gateway/spend_governor.go) plus its served
// enforcement boundary (session_admit.go: a breached scope's NEXT turn is refused 409 with
// BUDGET_SPEND_EXCEEDED, before the model is consulted). But the only way to ATTACH one was
// the programmatic Server.SetSpendGovernor(g, scopeOf) — so the hard spend cap, the whole
// point of which is that it lives OUTSIDE the agent, could not be turned on from the command
// line. This file is that missing half: it turns operator flags into a governor, a
// trace->ScopeKey resolver, and a breach webhook, and buildGateway attaches the result.
//
// INERT BY DEFAULT. No --spend-cap ⇒ buildSpendGovernor returns a nil governor and
// buildGateway never calls SetSpendGovernor, so the served request path stays byte-for-byte
// historical — the same posture the gateway seam documents.
//
// FAIL LOUD, NEVER SILENTLY INERT. A budget is a safety control: a typo that leaves it
// unenforced is worse than a refused boot. So a malformed --spend-cap, a cap with no ceiling
// on any axis, and a cap on a scope the configured resolver can never populate (e.g. a
// tenant cap with no tenant field in --spend-scope-trace, which would accumulate against the
// empty id and never fire) all fail startup instead of booting an uncapped server that
// LOOKS capped.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// spendScopeNames is the closed flag vocabulary for the budget-scope ladder, mapping the
// operator-typed token onto the gateway's scope constant. Anything else is a typo, refused.
var spendScopeNames = map[string]gateway.SpendScope{
	"tenant":  gateway.SpendScopeTenant,
	"team":    gateway.SpendScopeTeam,
	"agent":   gateway.SpendScopeAgent,
	"session": gateway.SpendScopeSession,
}

// spendActionNames is the closed flag vocabulary for the breach action. Empty defaults to
// pause in the gateway (the reversible action); naming anything else is refused.
var spendActionNames = map[string]gateway.SpendAction{
	"pause": gateway.SpendActionPause,
	"kill":  gateway.SpendActionKill,
}

// spendScopeTokens lists the accepted scope tokens in a stable order for error messages.
func spendScopeTokens() string {
	out := make([]string, 0, len(spendScopeNames))
	for k := range spendScopeNames {
		out = append(out, k)
	}
	sort.Strings(out)
	return strings.Join(out, "|")
}

// spendCap is one parsed --spend-cap value: which scope it caps, which id (empty = the
// scope's default, applied to every id with no explicit override), and the budget itself.
type spendCap struct {
	scope  gateway.SpendScope
	id     string
	budget gateway.SpendBudget
}

// parseSpendCap parses ONE --spend-cap value.
//
//	SCOPE[:ID]=SPEC
//	SCOPE := tenant|team|agent|session
//	ID    := an explicit instance id; omitted ⇒ the scope DEFAULT (every id)
//	SPEC  := a bare token count (shorthand for tokens=N), or comma-separated fields:
//	         tokens=N | usd=N (micro-USD, 1e-6 USD) | action=pause|kill
//
// e.g. "session=200000", "tenant:acme=tokens=5000000,action=kill", "team:core=usd=250000".
func parseSpendCap(raw string) (spendCap, error) {
	var out spendCap
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return out, fmt.Errorf("empty spend cap")
	}
	head, spec, found := strings.Cut(raw, "=")
	if !found {
		return out, fmt.Errorf("%q: want SCOPE[:ID]=SPEC (e.g. session=200000)", raw)
	}
	scopeTok, id, _ := strings.Cut(strings.TrimSpace(head), ":")
	scope, ok := spendScopeNames[strings.ToLower(strings.TrimSpace(scopeTok))]
	if !ok {
		return out, fmt.Errorf("%q: unknown scope %q (want %s)", raw, scopeTok, spendScopeTokens())
	}
	out.scope, out.id = scope, strings.TrimSpace(id)

	spec = strings.TrimSpace(spec)
	if spec == "" {
		return out, fmt.Errorf("%q: empty budget spec", raw)
	}
	// Bare-integer shorthand: "session=200000" means a 200000-token cap.
	if n, err := strconv.ParseInt(spec, 10, 64); err == nil {
		if n <= 0 {
			return out, fmt.Errorf("%q: token cap must be > 0", raw)
		}
		out.budget.TokenCap = n
		return out, nil
	}
	for _, field := range strings.Split(spec, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		key, val, found := strings.Cut(field, "=")
		if !found {
			return out, fmt.Errorf("%q: field %q wants key=value (tokens|usd|action)", raw, field)
		}
		key, val = strings.ToLower(strings.TrimSpace(key)), strings.TrimSpace(val)
		switch key {
		case "tokens", "usd":
			n, err := strconv.ParseInt(val, 10, 64)
			if err != nil || n <= 0 {
				return out, fmt.Errorf("%q: %s must be a positive integer, got %q", raw, key, val)
			}
			if key == "tokens" {
				out.budget.TokenCap = n
			} else {
				out.budget.USDMicros = n
			}
		case "action":
			action, ok := spendActionNames[strings.ToLower(val)]
			if !ok {
				return out, fmt.Errorf("%q: unknown action %q (want pause|kill)", raw, val)
			}
			out.budget.Action = action
		default:
			return out, fmt.Errorf("%q: unknown field %q (want tokens|usd|action)", raw, key)
		}
	}
	// A budget with no ceiling on any axis can never breach — an operator who typed a cap
	// meant to enforce one, so refuse rather than boot a server that looks capped.
	if out.budget.TokenCap <= 0 && out.budget.USDMicros <= 0 {
		return out, fmt.Errorf("%q: no ceiling set — give tokens=N and/or usd=N", raw)
	}
	return out, nil
}

// parseSpendScopeTrace parses --spend-scope-trace: a "/"-separated template naming which
// scope each segment of a request's trace id carries, e.g. "tenant/team/session" against
// the trace "acme/core/s-17". Empty ⇒ no template (the gateway's session-only default).
func parseSpendScopeTrace(format string) ([]gateway.SpendScope, error) {
	format = strings.TrimSpace(format)
	if format == "" {
		return nil, nil
	}
	var fields []gateway.SpendScope
	seen := map[gateway.SpendScope]bool{}
	for _, tok := range strings.Split(format, "/") {
		tok = strings.ToLower(strings.TrimSpace(tok))
		if tok == "" {
			return nil, fmt.Errorf("--spend-scope-trace %q: empty field", format)
		}
		scope, ok := spendScopeNames[tok]
		if !ok {
			return nil, fmt.Errorf("--spend-scope-trace %q: unknown field %q (want %s)", format, tok, spendScopeTokens())
		}
		if seen[scope] {
			return nil, fmt.Errorf("--spend-scope-trace %q: field %q repeated", format, tok)
		}
		seen[scope] = true
		fields = append(fields, scope)
	}
	return fields, nil
}

// spendScopeResolver builds the trace -> ScopeKey resolver for a parsed template: the trace
// id is split on "/" and each segment bound to the scope its position names. A trace with
// fewer segments than the template leaves the unnamed scopes empty (and an empty scope is
// never charged or evaluated, per ScopeKey). nil fields ⇒ a nil resolver, which the gateway
// documents as session-only (Session = the whole trace).
func spendScopeResolver(fields []gateway.SpendScope) func(string) gateway.ScopeKey {
	if len(fields) == 0 {
		return nil
	}
	return func(trace string) gateway.ScopeKey {
		var key gateway.ScopeKey
		parts := strings.Split(trace, "/")
		for i, scope := range fields {
			if i >= len(parts) {
				break
			}
			part := strings.TrimSpace(parts[i])
			if part == "" {
				continue
			}
			switch scope {
			case gateway.SpendScopeTenant:
				key.Tenant = part
			case gateway.SpendScopeTeam:
				key.Team = part
			case gateway.SpendScopeAgent:
				key.Agent = part
			case gateway.SpendScopeSession:
				key.Session = part
			}
		}
		return key
	}
}

// spendBreachEventKind tags the spend-breach payload on the shared --budget-webhook sink so
// a monitor can tell it apart from the #743 context-budget warn/exhaust events that ride the
// same URL. The breach body itself is gateway.SpendBreach's own JSON, embedded (and so
// flattened) rather than restated, so the wire shape can never drift from the kernel's.
const spendBreachEventKind = "spend_breach"

type spendBreachEvent struct {
	Kind string `json:"kind"`
	gateway.SpendBreach
}

// spendBreachWebhookObserver returns the OnBreach observer that POSTs each spend breach to
// the operator's --budget-webhook as JSON — the #4859 half of the ask: today that URL fires
// only on the CONTEXT budget (warn/exhaust), and a hard spend stop is the event an operator
// most needs pushed. Fire-and-forget and fail-open, exactly like budgetWebhookObserver: the
// POST runs on its own goroutine under a short timeout and a transport error is logged, never
// blocking or failing the served turn that crossed the cap. An empty URL returns nil (no-op).
func spendBreachWebhookObserver(rawURL string) func(gateway.SpendBreach) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil
	}
	return func(b gateway.SpendBreach) {
		body, err := json.Marshal(spendBreachEvent{Kind: spendBreachEventKind, SpendBreach: b})
		if err != nil {
			return
		}
		webhookPOST("spend breach webhook", rawURL, body, "application/json")
	}
}

// buildSpendGovernor turns the operator's --spend-cap set into an attached-ready governor
// plus the trace->ScopeKey resolver Server.SetSpendGovernor takes, wiring --budget-webhook
// as the breach sink. A (nil, nil, nil) return is the INERT default: no caps configured, so
// buildGateway never attaches and the request path stays historical.
func buildSpendGovernor(caps []string, scopeTrace, breachWebhookURL string) (*gateway.SpendGovernor, func(string) gateway.ScopeKey, error) {
	fields, err := parseSpendScopeTrace(scopeTrace)
	if err != nil {
		return nil, nil, err
	}
	if len(caps) == 0 {
		return nil, nil, nil
	}
	// Which scopes can the configured resolver actually populate? A nil template is the
	// gateway's session-only default; otherwise exactly the templated fields.
	reachable := map[gateway.SpendScope]bool{}
	if len(fields) == 0 {
		reachable[gateway.SpendScopeSession] = true
	}
	for _, f := range fields {
		reachable[f] = true
	}

	gov := gateway.NewSpendGovernor()
	for _, raw := range caps {
		sc, err := parseSpendCap(raw)
		if err != nil {
			return nil, nil, err
		}
		// A cap on a scope no trace can populate would accumulate against the empty id and
		// never fire — an uncapped server that reads as capped. Refuse it at boot.
		if !reachable[sc.scope] {
			return nil, nil, fmt.Errorf("--spend-cap %q: scope %q is never populated by the configured trace mapping (add it to --spend-scope-trace, e.g. --spend-scope-trace %s/session)", raw, sc.scope, sc.scope)
		}
		if sc.id == "" {
			gov.SetDefaultBudget(sc.scope, sc.budget)
		} else {
			gov.SetBudget(sc.scope, sc.id, sc.budget)
		}
	}
	if obs := spendBreachWebhookObserver(breachWebhookURL); obs != nil {
		gov.OnBreach(obs)
	}
	return gov, spendScopeResolver(fields), nil
}
