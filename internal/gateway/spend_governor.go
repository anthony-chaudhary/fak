package gateway

// spend_governor.go — the control-plane SPEND CAP (#3273, workstream C, epic #3256).
//
// The lesson FinOps practitioners drew from the runaway-agent-cost incidents of 2026 is
// that budget enforcement must live OUTSIDE the agent's own code — a hard kill/pause at
// the control plane, not a limit the model is asked to respect. That is precisely this
// seam. The gateway already accumulates every served turn's provider usage (prompt /
// completion / cache read / cache write tokens); the SpendGovernor folds that usage into
// per-SCOPE running totals (tenant / team / agent / session), and when a scope crosses
// its versioned budget it produces a CLOSED action — pause (the session drains, an
// operator may resume) or kill (terminal) — surfaced as a structured refusal reason, a
// breach counter, and a webhook event. The gateway enforces it at the served boundary
// (session_admit.go): a breached scope's NEXT turn is refused by the kernel, never by
// asking the model.
//
// DISTINCT FROM the context budget (BUDGET-TRIGGERED-SESSION-RESET): that budget triggers
// a human-like COMPACT/RESET so a long session keeps going; this one is a hard STOP on
// cumulative spend. A context drain wants to continue cheaper; a spend breach wants to
// halt.
//
// TENANCY COMPOSITION (#C5). One served turn charges EVERY scope level its ScopeKey names,
// so a tenant cap bounds the SUM of its sessions: a tenant may breach while no single
// session has, and every session under that tenant is then refused. Evaluate names the
// narrowest breached scope (the most actionable), so a session that blew its own cap is
// reported at the session scope even if its tenant is also over.
//
// INERT BY DEFAULT. A Server with no governor attached (SetSpendGovernor) leaves the
// request path byte-for-byte historical — the same inject-after-New posture as
// SetTokenRateGate / SetAdmissionController.

import (
	"fmt"
	"sync"
)

// SpendReasonExceeded is the closed refusal reason a breached spend budget surfaces. It
// is deliberately NOT one of the context/token drain reasons (session_admit.go
// isBudgetResetReason), so a spend breach hard-stops the session instead of triggering
// the human-like reset that continues past a context drain.
const SpendReasonExceeded = "BUDGET_SPEND_EXCEEDED"

// SpendAction is the closed set of actions a breached budget takes.
type SpendAction string

const (
	// SpendActionPause drains the session and leaves it resumable by an operator.
	SpendActionPause SpendAction = "pause"
	// SpendActionKill is terminal — the session is stopped for good.
	SpendActionKill SpendAction = "kill"
)

// SpendScope is the closed budget-scope ladder, widest (tenant) to narrowest (session).
type SpendScope string

const (
	SpendScopeTenant  SpendScope = "tenant"
	SpendScopeTeam    SpendScope = "team"
	SpendScopeAgent   SpendScope = "agent"
	SpendScopeSession SpendScope = "session"
)

// spendScopesNarrowToWide is the evaluation order: the narrowest breached scope is the
// most actionable, so Evaluate names it first. spendScopesWideToNarrow is only used for a
// stable metrics/debug render.
var (
	spendScopesNarrowToWide = []SpendScope{SpendScopeSession, SpendScopeAgent, SpendScopeTeam, SpendScopeTenant}
	spendScopesWideToNarrow = []SpendScope{SpendScopeTenant, SpendScopeTeam, SpendScopeAgent, SpendScopeSession}
)

// ScopeKey places one served turn in the scope hierarchy. An empty field means "this turn
// is not attributed to that scope" and no budget is charged or evaluated there — so a
// session-only deployment leaves Tenant/Team/Agent blank and only the session cap applies.
type ScopeKey struct {
	Tenant  string
	Team    string
	Agent   string
	Session string
}

// idFor returns the identifier this key carries for a given scope ("" ⇒ absent).
func (k ScopeKey) idFor(scope SpendScope) string {
	switch scope {
	case SpendScopeTenant:
		return k.Tenant
	case SpendScopeTeam:
		return k.Team
	case SpendScopeAgent:
		return k.Agent
	case SpendScopeSession:
		return k.Session
	}
	return ""
}

// SpendCost is one served turn's spend increment, taken from the counters the gateway has
// already accumulated. The token axes sum into the token budget; USDMicros (dollar cost in
// micro-USD, 1e-6 USD) feeds the dollar budget. All are non-negative; a negative field is
// treated as zero.
type SpendCost struct {
	InputTokens      int64
	OutputTokens     int64
	CacheReadTokens  int64
	CacheWriteTokens int64
	USDMicros        int64
}

// Tokens is the total token spend of one increment: every token axis summed. Negative
// fields (never expected) contribute zero, so an accumulator can only grow.
func (c SpendCost) Tokens() int64 {
	var t int64
	for _, v := range [...]int64{c.InputTokens, c.OutputTokens, c.CacheReadTokens, c.CacheWriteTokens} {
		if v > 0 {
			t += v
		}
	}
	return t
}

func (c SpendCost) add(o SpendCost) SpendCost {
	return SpendCost{
		InputTokens:      nonNeg(c.InputTokens) + nonNeg(o.InputTokens),
		OutputTokens:     nonNeg(c.OutputTokens) + nonNeg(o.OutputTokens),
		CacheReadTokens:  nonNeg(c.CacheReadTokens) + nonNeg(o.CacheReadTokens),
		CacheWriteTokens: nonNeg(c.CacheWriteTokens) + nonNeg(o.CacheWriteTokens),
		USDMicros:        nonNeg(c.USDMicros) + nonNeg(o.USDMicros),
	}
}

// empty reports whether the increment would change no accumulator (nothing to charge).
func (c SpendCost) empty() bool { return c.Tokens() == 0 && nonNeg(c.USDMicros) == 0 }

func nonNeg(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

// SpendBudget is one scope's cap plus the action to take when it is crossed. A zero cap on
// an axis means that axis is unlimited; a budget with both caps zero never breaches.
type SpendBudget struct {
	// TokenCap is the cumulative token ceiling (all token axes summed). 0 = unlimited.
	TokenCap int64
	// USDMicros is the cumulative dollar ceiling in micro-USD. 0 = unlimited.
	USDMicros int64
	// Action is the closed action taken on breach. Empty defaults to SpendActionPause
	// (the reversible action — an operator can always escalate a pause to a kill, but a
	// kill cannot be undone, so pause is the safe default).
	Action SpendAction
}

// exceeded reports whether accumulated spend has met or crossed this budget on any axis.
func (b SpendBudget) exceeded(spent SpendCost) bool {
	if b.TokenCap > 0 && spent.Tokens() >= b.TokenCap {
		return true
	}
	if b.USDMicros > 0 && nonNeg(spent.USDMicros) >= b.USDMicros {
		return true
	}
	return false
}

func (b SpendBudget) action() SpendAction {
	if b.Action == "" {
		return SpendActionPause
	}
	return b.Action
}

// SpendBreach is the closed refusal a crossed budget produces: which scope crossed, the
// action to take, the reason token, and the arithmetic (spent vs cap) that fired it.
type SpendBreach struct {
	Scope       SpendScope  `json:"scope"`
	ID          string      `json:"id"`
	Action      SpendAction `json:"action"`
	Reason      string      `json:"reason"`
	SpentTokens int64       `json:"spent_tokens"`
	TokenCap    int64       `json:"token_cap,omitempty"`
	SpentUSD    int64       `json:"spent_usd_micros,omitempty"`
	USDCap      int64       `json:"usd_micros_cap,omitempty"`
}

// String renders the breach arithmetic for a refusal message / log line.
func (b SpendBreach) String() string {
	if b.USDCap > 0 && b.TokenCap <= 0 {
		return fmt.Sprintf("%s spend budget: %s %q used $%d/1e6-USD reached cap $%d/1e6-USD (%s)",
			b.Scope, b.Scope, b.ID, b.SpentUSD, b.USDCap, b.Action)
	}
	return fmt.Sprintf("%s spend budget: %s %q used %d tokens reached cap %d (%s)",
		b.Scope, b.Scope, b.ID, b.SpentTokens, b.TokenCap, b.Action)
}

type spendInstance struct {
	scope SpendScope
	id    string
}

type spendBreachCount struct {
	scope  SpendScope
	action SpendAction
}

// scopeBudgets holds one scope's default budget plus per-id overrides.
type scopeBudgets struct {
	def  *SpendBudget
	byID map[string]SpendBudget
}

// SpendGovernor is the control-plane spend cap. Build with NewSpendGovernor; the zero
// value is not usable. Safe for concurrent use.
type SpendGovernor struct {
	mu       sync.Mutex
	budgets  map[SpendScope]*scopeBudgets
	spent    map[spendInstance]SpendCost
	breaches map[spendBreachCount]uint64
	onBreach func(SpendBreach)
}

// NewSpendGovernor builds an empty governor (no budgets, nothing spent).
func NewSpendGovernor() *SpendGovernor {
	return &SpendGovernor{
		budgets:  map[SpendScope]*scopeBudgets{},
		spent:    map[spendInstance]SpendCost{},
		breaches: map[spendBreachCount]uint64{},
	}
}

// SetDefaultBudget sets the budget applied to EVERY id in a scope that has no explicit
// override — e.g. "every session gets a 100k-token cap". A zero-cap budget is a no-op cap.
func (g *SpendGovernor) SetDefaultBudget(scope SpendScope, b SpendBudget) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	sb := g.budgets[scope]
	if sb == nil {
		sb = &scopeBudgets{byID: map[string]SpendBudget{}}
		g.budgets[scope] = sb
	}
	bb := b
	sb.def = &bb
}

// SetBudget sets an explicit per-id budget for one scope instance (e.g. a specific
// tenant), overriding that scope's default for that id.
func (g *SpendGovernor) SetBudget(scope SpendScope, id string, b SpendBudget) {
	if g == nil || id == "" {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	sb := g.budgets[scope]
	if sb == nil {
		sb = &scopeBudgets{byID: map[string]SpendBudget{}}
		g.budgets[scope] = sb
	}
	sb.byID[id] = b
}

// OnBreach registers the observer fired once each time a scope FIRST crosses its budget
// (the webhook seam). Replacing it is allowed; nil detaches it.
func (g *SpendGovernor) OnBreach(fn func(SpendBreach)) {
	if g == nil {
		return
	}
	g.mu.Lock()
	g.onBreach = fn
	g.mu.Unlock()
}

// budgetForLocked resolves the effective budget for a scope instance: the per-id override
// if present, else the scope default. ok=false ⇒ no budget configured for that instance.
func (g *SpendGovernor) budgetForLocked(scope SpendScope, id string) (SpendBudget, bool) {
	sb := g.budgets[scope]
	if sb == nil {
		return SpendBudget{}, false
	}
	if b, ok := sb.byID[id]; ok {
		return b, true
	}
	if sb.def != nil {
		return *sb.def, true
	}
	return SpendBudget{}, false
}

// Charge folds one served turn's spend into every scope level the key names. When a
// scope's accumulator FIRST crosses its budget on this charge, it records a breach (the
// counter) and fires the OnBreach observer once — so a webhook/counter fires per crossing,
// not per subsequent refused turn. A nil governor or an empty cost is a no-op.
func (g *SpendGovernor) Charge(key ScopeKey, cost SpendCost) {
	if g == nil || cost.empty() {
		return
	}
	var fired []SpendBreach
	g.mu.Lock()
	for _, scope := range spendScopesWideToNarrow {
		id := key.idFor(scope)
		if id == "" {
			continue
		}
		inst := spendInstance{scope: scope, id: id}
		prev := g.spent[inst]
		next := prev.add(cost)
		g.spent[inst] = next
		b, ok := g.budgetForLocked(scope, id)
		if !ok {
			continue
		}
		// A breach event is a fresh crossing: under before, at/over now.
		if !b.exceeded(prev) && b.exceeded(next) {
			br := spendBreachFor(scope, id, b, next)
			g.breaches[spendBreachCount{scope: scope, action: br.Action}]++
			fired = append(fired, br)
		}
	}
	obs := g.onBreach
	g.mu.Unlock()
	if obs != nil {
		for _, br := range fired {
			obs(br)
		}
	}
}

// Evaluate reports whether any scope the key names is currently over its budget, returning
// the breach for the NARROWEST such scope (the most actionable). It is a pure read — it
// neither charges nor re-fires the observer — so the served boundary can call it every turn
// to keep refusing a breached scope idempotently. nil ⇒ admit.
func (g *SpendGovernor) Evaluate(key ScopeKey) *SpendBreach {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, scope := range spendScopesNarrowToWide {
		id := key.idFor(scope)
		if id == "" {
			continue
		}
		b, ok := g.budgetForLocked(scope, id)
		if !ok {
			continue
		}
		spent := g.spent[spendInstance{scope: scope, id: id}]
		if b.exceeded(spent) {
			br := spendBreachFor(scope, id, b, spent)
			return &br
		}
	}
	return nil
}

func spendBreachFor(scope SpendScope, id string, b SpendBudget, spent SpendCost) SpendBreach {
	return SpendBreach{
		Scope:       scope,
		ID:          id,
		Action:      b.action(),
		Reason:      SpendReasonExceeded,
		SpentTokens: spent.Tokens(),
		TokenCap:    b.TokenCap,
		SpentUSD:    nonNeg(spent.USDMicros),
		USDCap:      b.USDMicros,
	}
}

// SpendBreachCount is one row of the breach tally: how many times a scope crossed a budget
// with a given action, for the /metrics counter.
type SpendBreachCount struct {
	Scope  SpendScope
	Action SpendAction
	Count  uint64
}

// Snapshot returns the cumulative breach counts, stable-sorted for a deterministic render.
func (g *SpendGovernor) Snapshot() []SpendBreachCount {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]SpendBreachCount, 0, len(g.breaches))
	for _, scope := range spendScopesWideToNarrow {
		for _, action := range []SpendAction{SpendActionPause, SpendActionKill} {
			if n := g.breaches[spendBreachCount{scope: scope, action: action}]; n > 0 {
				out = append(out, SpendBreachCount{Scope: scope, Action: action, Count: n})
			}
		}
	}
	return out
}
