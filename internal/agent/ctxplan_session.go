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
	"hash/fnv"
	"io"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// SessionPlanner is a persistent per-session context planner: a long-lived lossless store +
// candidate index that ingests each turn's new messages incrementally and probes a bounded
// candidate set per turn. Construct it with NewSessionPlanner or CtxViewPlanner.NewSession; the
// zero value is not usable (the store/index are nil).
type SessionPlanner struct {
	// Budget is the O(1) resident-token window each planned turn materializes under — the same
	// meaning as CtxViewPlanner.Budget.
	Budget int
	// baseBudget is the UNTHROTTLED window ApplyPace scales from (#628), captured at construction.
	// Holding the baseline separately makes ApplyPace idempotent (repeated calls with the same
	// pace land on the same Budget, never compounding) and restorable (clearing the pace restores
	// the full window), since Pace.ComposePlannerBudget always scales the baseline, not the
	// already-throttled value.
	baseBudget int
	// Opts tunes the bounded probe (RecencyWindow, MaxCandidates, IncludeDurability). The zero
	// value is valid — ctxplan fills sensible defaults (DefaultRecencyWindow / DefaultMaxCandidates).
	Opts ctxplan.ProbeOptions
	// Layout optionally tunes the four-area context profile (base/current/recent/deep). nil keeps
	// the original ProbeOptions path; non-nil uses ctxplan.Index.PlanLayout.
	Layout *ctxplan.Layout

	store    *ctxplan.MemStore // the lossless byte store (id "span:<i>"); the demand-page backing
	index    *ctxplan.Index    // the persistent candidate index, maintained incrementally
	ingested int               // messages already lowered into store+index (the append cursor)
	fprints  []uint64          // per-ingested-message fingerprints — the append-only contract's witness

	// Incremental pins, maintained in O(1) per new message so a turn never re-scans all N to
	// re-derive them: the active goal, the system prompt, the first user turn, and the last
	// user turn — the spans a turn cannot proceed without (the heuristicForecast contract).
	goalPin         string // the active-goal root (a RoleGoal message, #845); "" when no goal is set
	systemPin       string
	firstUserPin    string
	lastUserPin     string
	lastUserContent string // the last user message's content, for the forecast intents (O(content), not O(N))

	// tenure is the OPTIONAL generational-tenuring table (#848): a per-command recurrence
	// counter that promotes a long-lived recurring slash-command to a compact rollup so its
	// context stops being re-derived each turn, and demotes it on cold. nil is the default-off
	// path — a SessionPlanner with no tenure table behaves byte-for-byte as before. It is
	// orthogonal to the resident-span plan (object lifetime, not placement), so it lives
	// beside the pins rather than inside pins().
	tenure *tenureTable
}

// NewSessionPlanner mints a fresh per-session planner with an empty store and index. A
// non-positive budget falls back to DefaultCtxViewBudget — the same seed the stateless seam uses.
func NewSessionPlanner(budget int) *SessionPlanner {
	if budget <= 0 {
		budget = DefaultCtxViewBudget
	}
	return &SessionPlanner{
		Budget:     budget,
		baseBudget: budget,
		store:      ctxplan.NewMemStore(),
		index:      ctxplan.NewIndex(),
	}
}

// ApplyPace composes a session's Pace into this planner's resident-context Budget (#628,
// epic #620 track 5): a session paced BELOW its baseline per-turn output plans under a
// proportionally smaller window (floored, never starved), so "slow this session" drives its
// CONTEXT budget down — not just its output cap. This is the genuine wire of
// session.Pace.MaxTokensPerTurn into agent.SessionPlanner.Budget the design note (§4) named.
//
// baselineOutput is the session's unthrottled per-turn output target (the pace cap's
// reference). The scale is always taken from baseBudget (the window the planner was
// CONSTRUCTED with), not the current Budget, so ApplyPace is idempotent across turns and a
// cleared pace (MaxTokensPerTurn 0) restores the full baseline window. A pace that voices no
// opinion is a no-op — the planner keeps its full Budget, byte-for-byte the pre-compose path.
// It returns the new Budget.
//
// This composes ONLY the CONFIGURED cap (MaxTokensPerTurn). Use ApplyThroughput for the
// runtime-OBSERVED signal (#1585), or ApplyPaceAndThroughput to fold both in one call.
func (sp *SessionPlanner) ApplyPace(pace session.Pace, baselineOutput int) int {
	sp.Budget = pace.ComposePlannerBudget(sp.baseBudget, baselineOutput)
	return sp.Budget
}

// ApplyThroughput composes a session's OBSERVED runtime throughput into this planner's
// resident-context Budget (#1585, epic #1570 "managed context") — the measured-pace twin of
// ApplyPace. Where ApplyPace scales the window from a CONFIGURED cap set ahead of time,
// ApplyThroughput scales it from how fast the session is ACTUALLY moving right now
// (t.ObservedTokensPerSec against t.ExpectedTokensPerSec): a session measurably falling
// behind its expected rate — GPU contention, a slow upstream model, backpressure, none of
// it anyone's configured pace — sees its resident window shrink proportionally, floored at
// baseBudget/session.MinPlannerBudgetDivisor so the structural pins and a minimal recency
// tail always still fit (the "minimum resident context preserved" done condition).
//
// session.Throughput is a standalone type (compose.go), not fields on session.Pace, so this
// method takes it as its own parameter rather than reading it off pace.
//
// The scale is taken from baseBudget (never the current, possibly-already-throttled
// Budget), so repeated calls with the same observation are idempotent and a session that
// catches back up to its expected rate restores the full baseline window — the exact
// idempotent-and-restorable contract ApplyPace already established. A Throughput with no
// signal (either axis zero) is a no-op. It returns the new Budget.
func (sp *SessionPlanner) ApplyThroughput(t session.Throughput) int {
	sp.Budget = t.ComposePlannerBudgetForThroughput(sp.baseBudget)
	return sp.Budget
}

// ApplyPaceAndThroughput folds BOTH the configured cap and the observed throughput signal
// into this planner's resident-context Budget in one call (#1585): whichever constraint is
// tighter wins (session.Pace.ComposePace), so a session that is both configured-throttled
// AND running behind its expected rate gets the harder of the two shrinks, never one
// silently overriding the other. Like ApplyPace/ApplyThroughput, the scale is always taken
// from baseBudget, so this is idempotent and fully restorable when both signals clear. It
// returns the new Budget.
func (sp *SessionPlanner) ApplyPaceAndThroughput(pace session.Pace, t session.Throughput, baselineOutput int) int {
	sp.Budget = pace.ComposePace(t, sp.baseBudget, baselineOutput)
	return sp.Budget
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
// The append-only contract (the live-loop shape: a turn adds messages, it never rewrites earlier
// ones — the planner REPLACES compaction, so the prefix is stable) is VERIFIED per turn, not
// assumed: a stateless wire like OpenAI /v1/chat/completions sends an INDEPENDENT message list
// per request, and one gateway trace can carry many unrelated conversations back to back. An
// incoming history that is shorter than the ingested prefix, or whose prefix does not match it,
// resets the store/index/pins and re-ingests from scratch — without the reset, the count-only
// cursor made the planner render the FIRST conversation's spans forever (every later request
// with <= ingested messages ingested nothing), freezing the served prompt at conversation one.
// Verification hashes messages[:ingested] — O(prefix bytes), the same order the request already
// paid to parse its body — so the bounded-probe Θ(c) PLANNING cost is untouched; only a genuine
// divergence pays the one O(N) rebuild, which is the correctness price, never a replay.
func (sp *SessionPlanner) ingest(messages []Message) {
	if sp.divergesFromIngested(messages) {
		sp.resetConversation()
	}
	for i := sp.ingested; i < len(messages); i++ {
		msg := messages[i]
		role := msg.Role
		if role == "" {
			role = RoleUser
		}
		span := sp.store.Add(role, messageDurability(role), []byte(msg.Content), false)
		sp.index.Add(span)
		// Pins track the NORMALIZED role (empty -> user), exactly as messagesToStore does, so
		// a SessionPlanner and the stateless seam pin the same spans. The active goal (a
		// RoleGoal span, #845) is the intentional GC root — the FIRST goal span wins and is
		// charged ahead of the structural pins in pins().
		switch {
		case role == RoleGoal && sp.goalPin == "":
			sp.goalPin = span.ID
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
		sp.fprints = append(sp.fprints, messageFingerprint(msg))
	}
	sp.ingested = len(messages)
}

// divergesFromIngested reports whether the incoming history contradicts the already-ingested
// prefix: it is shorter than the cursor, or some message in messages[:ingested] no longer
// fingerprint-matches what was lowered into the store. A zero-ingested planner never diverges.
// Fingerprints (not stored bytes) keep the check allocation-free; a hash collision can only
// SUPPRESS a reset (never force one), and the 64-bit FNV space makes that vanishingly unlikely
// against the alternative the check replaces — trusting the count alone, which replayed a stale
// conversation on every same-trace stateless request.
func (sp *SessionPlanner) divergesFromIngested(messages []Message) bool {
	if len(messages) < sp.ingested {
		return true
	}
	for i, fp := range sp.fprints {
		if messageFingerprint(messages[i]) != fp {
			return true
		}
	}
	return false
}

// resetConversation drops every message-derived structure — store, index, cursor, fingerprints,
// pins, forecast content — so the next ingest rebuilds from the incoming history alone. The
// planner's CONFIG survives (Budget/baseBudget/Opts/Layout and the opt-in tenure table): those
// belong to the session's operator settings, not to the conversation that just diverged.
func (sp *SessionPlanner) resetConversation() {
	sp.store = ctxplan.NewMemStore()
	sp.index = ctxplan.NewIndex()
	sp.ingested = 0
	sp.fprints = sp.fprints[:0]
	sp.goalPin, sp.systemPin, sp.firstUserPin, sp.lastUserPin = "", "", "", ""
	sp.lastUserContent = ""
}

// messageFingerprint hashes the fields ingest lowers into planner state (role, name, content)
// — a divergence in any other field cannot change what the planner renders, so it does not
// force a rebuild.
func messageFingerprint(m Message) uint64 {
	h := fnv.New64a()
	io.WriteString(h, m.Role)
	h.Write([]byte{0})
	io.WriteString(h, m.Name)
	h.Write([]byte{0})
	io.WriteString(h, m.Content)
	return h.Sum64()
}

// pins returns the incremental pin id list in the same order messagesToStore produces — the
// active goal (the intentional GC root, charged first), then system prompt, first user turn,
// then the last user turn when it differs from the first. A pin id is resolved against the
// index in Probe, so a pin that names no span is simply skipped; with no goal set the list is
// byte-identical to before, so the plan is unchanged.
func (sp *SessionPlanner) pins() []string {
	var pins []string
	if sp.goalPin != "" {
		pins = append(pins, sp.goalPin)
	}
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

// EnableTenuring turns on generational tenuring of recurring slash-commands (#848) with the
// given promotion threshold and quiet-TTL (non-positive values fall back to the package
// defaults). It is opt-in: a SessionPlanner created without it has a nil tenure table and is
// behavior-preserving. Calling it again replaces the table (resetting recurrence history).
func (sp *SessionPlanner) EnableTenuring(threshold int, ttlMillis int64) {
	sp.tenure = newTenureTable(threshold, ttlMillis)
}

// RecordCommand observes one invocation of a recurring slash-command at nowMillis, advancing
// its recurrence counter and reviving its tenuring clock. It returns the command's compact
// rollup and whether it is currently tenured (promoted). With tenuring disabled (the default)
// it is a no-op returning (zero, false): a command is always safe to re-derive each turn.
func (sp *SessionPlanner) RecordCommand(command string, nowMillis int64) (Rollup, bool) {
	return sp.tenure.Record(command, nowMillis)
}

// CommandRollup returns a recurring command's compact rollup and whether it is tenured. A
// young / unknown / demoted command (or tenuring disabled) returns (zero, false) — the safe
// "re-derive this turn" signal. The caller uses the rollup as a CACHE of derived context; a
// false result means fall back to re-deriving (heuristicForecast), never a correctness change.
func (sp *SessionPlanner) CommandRollup(command string) (Rollup, bool) {
	return sp.tenure.Rollup(command)
}

// SweepTenure applies the time-driven demotion at nowMillis, demoting any tenured command that
// has gone quiet past its TTL back to young (dropping its rollup cache, keeping its history).
// It returns the commands demoted this sweep. A no-op with tenuring disabled.
func (sp *SessionPlanner) SweepTenure(nowMillis int64) []string {
	return sp.tenure.Sweep(nowMillis)
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
