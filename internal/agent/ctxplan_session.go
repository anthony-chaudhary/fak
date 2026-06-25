package agent

// ctxplan_session.go — the PERSISTENT per-session context planner (issue #558, half b): the
// stateful peer of CtxViewPlanner's stateless RenderTurn. CtxViewPlanner.RenderTurn rebuilds a
// fresh MemStore from the whole message list and FULL-SCANS it every turn (messagesToStore +
// ctxplan.Materialize) — O(N) work per turn, Θ(N²) cumulative, which is exactly the cost the
// candidate Index exists to flatten (ctxplan/index.go). That stateless path is correct but
// defeats the flatten: "Θ(c·N) cumulative planning" holds only if the index is MAINTAINED
// across turns, never rebuilt.
//
// SessionPlanner maintains ONE ctxplan.Index + one lossless MemStore for the whole session,
// Add-ing only each turn's NEW spans (O(tokens) per turn) and PROBING a bounded candidate set
// each turn (ctxplan.Index.PlanCells) instead of scoring all N — so the per-turn planning cost
// is bounded by the probe size c, not the turn count, and the cumulative cost is genuinely
// linear. It is the live-loop wiring the spine left unbuilt: a persistent per-session index on
// the agent seam's per-turn path.
//
// # Why it is a separate type, not a flag on CtxViewPlanner
//
// CtxViewPlanner is constructed ONCE per gateway Server and shared across every request, so it
// must stay stateless (a per-session index on a shared instance would cross-contaminate
// sessions). SessionPlanner holds the per-session state; CtxViewPlanner.NewSession() mints one
// from the shared config. This is the "agent seam first" rung: the mechanism + its witnesses
// ship here; threading a session id through the gateway and persisting the index alongside the
// recall core image (recall.PersistIndex) is the documented next rung, deferred because the
// flagship gateway route forwards req.Raw verbatim (the #555 req.Raw transform is its own step).
//
// # The honesty posture is unchanged
//
// The bounded probe changes the planner's COST, not its faithfulness. A span the probe PRUNES
// (irrelevant + old + non-durable) is not lost: it stays in the lossless MemStore and pages
// back in on demand (Materialize), exactly as a forecast miss does. The trust gate is
// untouched — a sealed/tombstoned span is still scored 0 and elided, never resident. And an
// incrementally-maintained index is structurally identical to a rebuilt one (ctxplan's
// incremental==batch witness), so SessionPlanner's per-turn output is identical to the
// stateless full-scan path whenever the bounded candidate set suffices — the common case.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// SessionPlanner is a persistent per-session context planner: a long-lived lossless store +
// candidate index that ingests each turn's new messages incrementally and probes a bounded
// candidate set per turn. Construct it with NewSessionPlanner or CtxViewPlanner.NewSession; the
// zero value is not usable (the store/index are nil).
type SessionPlanner struct {
	// Budget is the O(1) resident-token window each planned turn materializes under — the same
	// meaning as CtxViewPlanner.Budget.
	Budget int
	// Opts tunes the bounded probe (RecencyWindow, MaxCandidates, IncludeDurability). The zero
	// value is valid — ctxplan fills sensible defaults (DefaultRecencyWindow / DefaultMaxCandidates).
	Opts ctxplan.ProbeOptions
	// Layout optionally tunes the four-area context profile (base/current/recent/deep). nil keeps
	// the original ProbeOptions path; non-nil uses ctxplan.Index.PlanLayout.
	Layout *ctxplan.Layout

	store    *ctxplan.MemStore // the lossless byte store (id "span:<i>"); the demand-page backing
	index    *ctxplan.Index    // the persistent candidate index, maintained incrementally
	ingested int               // messages already lowered into store+index (the append cursor)

	// Incremental pins, maintained in O(1) per new message so a turn never re-scans all N to
	// re-derive them: the system prompt, the active goal (first user turn), and the last user
	// turn — the spans a turn cannot proceed without (the heuristicForecast contract).
	systemPin       string
	firstUserPin    string
	lastUserPin     string
	lastUserContent string // the last user message's content, for the forecast intents (O(content), not O(N))
}

// NewSessionPlanner mints a fresh per-session planner with an empty store and index. A
// non-positive budget falls back to DefaultCtxViewBudget — the same seed the stateless seam uses.
func NewSessionPlanner(budget int) *SessionPlanner {
	if budget <= 0 {
		budget = DefaultCtxViewBudget
	}
	return &SessionPlanner{
		Budget: budget,
		store:  ctxplan.NewMemStore(),
		index:  ctxplan.NewIndex(),
	}
}

// NewSession mints a per-session planner seeded from this CtxViewPlanner's Budget — the factory
// that turns the shared, stateless seam config into the stateful per-session index the live loop
// maintains across turns. The Budget is inherited; the per-session state lives on the returned
// SessionPlanner, never on the shared CtxViewPlanner.
func (p *CtxViewPlanner) NewSession() *SessionPlanner {
	sp := NewSessionPlanner(p.Budget)
	if p.Layout != nil {
		layout := *p.Layout
		sp.Layout = &layout
	}
	return sp
}

// ingest lowers any messages not yet seen into the persistent store + index, in turn order, and
// maintains the incremental pins — the O(new-spans) maintenance step that replaces the stateless
// path's O(N) per-turn store rebuild. It assigns the SAME ids ("span:<i>") and durability classes
// as messagesToStore, so a SessionPlanner and the stateless seam address spans identically.
//
// It assumes the message history is APPEND-ONLY across turns (the live-loop contract: a turn adds
// messages, it never rewrites earlier ones — the planner REPLACES compaction, so the prefix is
// stable). messages[:ingested] are taken as already-indexed and are not re-scanned.
func (sp *SessionPlanner) ingest(messages []Message) {
	for i := sp.ingested; i < len(messages); i++ {
		msg := messages[i]
		role := msg.Role
		if role == "" {
			role = RoleUser
		}
		span := sp.store.Add(role, messageDurability(role), []byte(msg.Content), false)
		sp.index.Add(span)
		// Pins track the NORMALIZED role (empty -> user), exactly as messagesToStore does, so
		// a SessionPlanner and the stateless seam pin the same spans.
		switch {
		case role == RoleSystem && sp.systemPin == "":
			sp.systemPin = span.ID
		case role == RoleUser && sp.firstUserPin == "":
			sp.firstUserPin = span.ID
		case role == RoleUser:
			sp.lastUserPin = span.ID
		}
		// Intents come from the last message whose RAW role is user — matching heuristicForecast
		// exactly (it scans messages[i].Role == RoleUser, WITHOUT the empty-role normalization the
		// store + pins use). Keeping the two role bases distinct is what makes forecast() identical
		// to heuristicForecast for every input, not just messages with explicit roles.
		if msg.Role == RoleUser {
			sp.lastUserContent = msg.Content
		}
	}
	sp.ingested = len(messages)
}

// pins returns the incremental pin id list in the same order messagesToStore produces — system
// prompt, first user turn, then the last user turn when it differs from the first. A pin id is
// resolved against the index in Probe, so a pin that names no span is simply skipped.
func (sp *SessionPlanner) pins() []string {
	var pins []string
	if sp.systemPin != "" {
		pins = append(pins, sp.systemPin)
	}
	if sp.firstUserPin != "" {
		pins = append(pins, sp.firstUserPin)
	}
	if sp.lastUserPin != "" && sp.lastUserPin != sp.firstUserPin {
		pins = append(pins, sp.lastUserPin)
	}
	return pins
}

// forecast authors the per-turn Forecast from the maintained state — intents from the last user
// message's content, Horizon 1, pins the incremental essentials. It is the exact heuristic
// heuristicForecast computes, but in O(content) instead of O(N): the pins and the last user
// content are maintained incrementally, so no per-turn re-scan of the whole history is needed.
func (sp *SessionPlanner) forecast() ctxplan.Forecast {
	return ctxplan.Forecast{
		Intents: contentIntents(sp.lastUserContent),
		Horizon: 1,
		Pins:    sp.pins(),
	}
}

// PlanTurn ingests any new messages incrementally, then PROBES the bounded candidate set and
// plans the O(1) resident view over it (ctxplan.Index.PlanCells) — the bounded-compute per-turn
// path. It is pure (no I/O): the result is the deterministic plan a caller can EXPLAIN and audit
// before rendering. RenderTurn is the I/O peer that pages the selected spans' bytes in.
func (sp *SessionPlanner) PlanTurn(messages []Message) ctxplan.Plan {
	sp.ingest(messages)
	if sp.Layout != nil {
		return sp.index.PlanLayout(sp.forecast(), ctxplan.Budget{Tokens: sp.Budget}, nil, *sp.Layout)
	}
	return sp.index.PlanCells(sp.forecast(), ctxplan.Budget{Tokens: sp.Budget}, nil, sp.Opts)
}

// RenderTurn ingests new messages, plans the bounded O(1) view, and renders the selected spans'
// bytes back to a message history — paging each in through the store's trust gate, in step
// order. It is the per-turn path the live loop calls in place of append+compact, now backed by a
// persistent bounded index instead of a fresh full-scan store every turn. A span the gate
// declines mid-render stays out of context (it is skipped, never emitted as poison).
func (sp *SessionPlanner) RenderTurn(ctx context.Context, messages []Message) []Message {
	plan := sp.PlanTurn(messages)
	out := make([]Message, 0, len(plan.Selected))
	for _, s := range plan.Selected {
		body, err := sp.store.Materialize(ctx, s.ID)
		if err != nil {
			continue
		}
		out = append(out, Message{Role: spanRoleToMessage(s.Role), Name: s.Role, Content: string(body)})
	}
	return out
}

// Index returns the persistent candidate index the planner maintains — the accessor a caller
// persists alongside the recall core image (recall.PersistIndex(dir, sp.Index())) so a resumed
// session re-attaches it. The returned index is the LIVE one (not a copy); it is exposed for
// persistence + audit, not for external mutation.
func (sp *SessionPlanner) Index() *ctxplan.Index { return sp.index }

// Materialize pages a span's bytes in through the store's trust gate — the demand-page backing
// for a pruned/elided span. A span the bounded probe left out of a turn's candidate set is not
// lost: it stays in the lossless store and Materialize recovers its VERBATIM bytes, exactly as a
// forecast miss is one demand-page away, never a lost fact.
func (sp *SessionPlanner) Materialize(ctx context.Context, id string) ([]byte, error) {
	return sp.store.Materialize(ctx, id)
}

// Len reports how many messages have been lowered into the store+index — the indexed span count
// N. After T appended turns it is exactly the message count (each message is Add-ed once), the
// witness that maintenance is O(total spans), not O(turns²).
func (sp *SessionPlanner) Len() int { return sp.index.Len() }
