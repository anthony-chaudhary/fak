// Package session is the per-session DRIVE state — the first-class, queryable,
// live-mutable control state of a served agent session: its run-state, planner
// budget, scheduling priority, and per-turn pace. It is the structural twin of
// internal/ifc's Ledger (TraceID-keyed, bounded-LRU, RWMutex), widened from the
// single taint high-water mark Ledger carries to a small drive-state struct.
//
// THE GAP IT CLOSES.
// A served session's drive changes while it runs — an operator drops its budget
// mid-flight, lowers its priority so an urgent one passes, pauses it, or stops it.
// Today none of that has a home: the turn loop's cap is frozen at entry, the
// matmul budget is resolved once at init, and "is this session still going, and
// how hard?" is RECONSTRUCTED after the fact from git commits + a process scan +
// a 0-byte-log heuristic (docs/dispatch-loop.md). Reconstruction is lossy, racy,
// and read-only — you can observe a guessed state, never SET it. This package
// makes the drive a value: written live, read each turn, so the current state is
// a lookup, never a re-derivation.
//
// THE SEAM IT GENERALIZES.
// internal/ifc.Ledger is already a TraceID-keyed, bounded-LRU, concurrent,
// live-mutable per-session store with a GET /v1/fak/trace/{id} read and a
// POST /v1/fak/trace/reset write — carrying exactly ONE value (the taint mark).
// Table is that exact mechanism with a wider value; the gateway session routes
// (GET/POST /v1/fak/session/...) are the trace routes with a wider payload.
//
// WRITE side (the control verbs): Transition / SetBudget / SetPace / SetPriority,
// each bumping a monotonic Rev so a stale operator write can be rejected (CAS).
// READ side: Get (one session), Snapshot (every live session, the SCHEDULER's
// data structure), and Decide (the one call the turn loop makes per boundary —
// it debits the turn and returns whether to proceed and, if not, why).
//
// SCHEDULING POSTURE. The table HOLDS Priority and exposes Snapshot; it never
// PICKS a winner. A multi-session scheduler reads the snapshot and decides who
// yields — keeping policy out of the table is what keeps the table a value. Budget
// exhaustion and pause/stop are scheduling EVENTS (a slot frees); a supervisor
// observes them through the table instead of re-deriving from a process scan.
//
// This package is a foundation leaf: stdlib-only (container/list + sync) plus the
// shared internal/lifecycle vocabulary leaf and the internal/dormancy clock (both
// tier-1 foundation leaves), off the request path, registers nothing. The zero Table
// is not usable — construct with NewTable / NewTableWithLimit.
package session

import (
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
)

// RunState is a served session's lifecycle position — a small, total state machine.
// The transitions are the control verbs the design names: throttle/pause/resume
// (reversible drive changes) and drain/stop (terminal). The zero value is Running,
// so an unseen trace is a live session at its defaults — never a phantom Stopped.
type RunState uint8

const (
	// Running is the default: the session advances each turn at its budget/pace.
	Running RunState = iota
	// Throttled means the session still advances but under a tightened pace
	// (lower MaxTokensPerTurn / a turn gap). It carries a Reason token for "why".
	Throttled
	// Paused holds the session at the next turn boundary without ending it; a
	// resume (Paused -> Running) is a state flip, not a cold re-attach.
	Paused
	// Draining means a stop was requested; the loop takes it at the NEXT turn
	// boundary (never mid-decode, so a stop never tears a half-emitted tool call).
	Draining
	// Stopped is terminal; it carries a closed Reason token so "why did it stop"
	// is a field, not an inference from an exit code.
	Stopped
	// Terminating means a FORCEFUL stop was requested (#2758): the loop takes it at
	// the next SAFE POINT — the in-flight model call's context is cancelled and no
	// further tool call is dispatched — unlike Draining, which lets the current turn
	// run to completion. Appended after Stopped so persisted numeric values of the
	// prior states never shift. Like Draining it advances exactly one more step (a
	// Decide finalizes it to Stopped with the TERMINATED reason), so it is not
	// terminal itself.
	Terminating
)

// String renders a RunState as its lowercase wire token (the form the
// /v1/fak/session routes emit and accept). The four shared-lifecycle tokens are
// SOURCED from internal/lifecycle (not re-spelled here) so the served session and
// the loop supervisor cannot drift apart; Throttled is the one session-only token.
// An out-of-range value renders "unknown" rather than panicking — a wire value is
// never trusted to be in range.
func (s RunState) String() string {
	if s == Throttled {
		return "throttled"
	}
	if s == Terminating {
		return "terminating"
	}
	if p, ok := s.Phase(); ok {
		return p.String()
	}
	return "unknown"
}

// ParseRunState maps a wire token back to a RunState. The bool is false for an
// unrecognized token, so a caller fails closed (the route returns 400) rather than
// defaulting an unknown verb to Running. The four shared tokens go through
// lifecycle.Parse — the single definition both layers share.
func ParseRunState(s string) (RunState, bool) {
	if s == "throttled" {
		return Throttled, true
	}
	if s == "terminating" {
		return Terminating, true
	}
	if p, ok := lifecycle.Parse(s); ok {
		return RunStateFromPhase(p)
	}
	return 0, false
}

// Phase projects a RunState onto the shared lifecycle skeleton. The bool is false
// for Throttled (a session-only pace modifier with no shared peer), for Terminating
// (the session-only forceful-stop staging state, #2758) and for any
// out-of-range value — the projection is explicit about the extras, never a silent
// default. This is the served-session half of the #912 "one machine" converter;
// internal/lifebridge composes it with the supervisor half.
func (s RunState) Phase() (lifecycle.Phase, bool) {
	switch s {
	case Running:
		return lifecycle.Running, true
	case Paused:
		return lifecycle.Paused, true
	case Draining:
		return lifecycle.Draining, true
	case Stopped:
		return lifecycle.Stopped, true
	}
	return 0, false
}

// RunStateFromPhase lifts a shared lifecycle Phase into a RunState. It is total
// over the four Phases (every shared state has a RunState peer); an out-of-range
// Phase yields (0, false).
func RunStateFromPhase(p lifecycle.Phase) (RunState, bool) {
	switch p {
	case lifecycle.Running:
		return Running, true
	case lifecycle.Paused:
		return Paused, true
	case lifecycle.Draining:
		return Draining, true
	case lifecycle.Stopped:
		return Stopped, true
	}
	return 0, false
}

// terminal reports whether a run-state can no longer advance. A Stopped session is
// terminal; everything else (including Draining, which advances exactly one more
// boundary) is not. Used by Decide and by the write guards (a terminal session
// rejects a resume — you cannot un-stop a stopped session, only start a new one).
func (s RunState) terminal() bool { return s == Stopped }

// Unbounded is the sentinel for a budget axis with no limit (the v0.1 default — a
// session runs until it ends on its own). A non-negative TurnsLeft/TokensLeft is a
// real remaining allotment that Decide/Debit debits toward zero; ContextTokensLeft
// uses 0 as "not configured" and a positive value as the long-window reset budget.
const Unbounded = -1

// Budget is a session's remaining work allotment. Decide debits TurnsLeft by one
// each turn and TokensLeft/ContextTokensLeft by the turn's reported usage; hitting
// a configured axis drives the session to Draining (the budget-exhausted stop). An
// operator RE-SETS any axis live — raising it (speed up / extend) or cutting it
// (slow down / the priority-queue "let an urgent one pass" move). Unbounded (-1)
// means no limit for the turn/output axes; context 0 means off.
type Budget struct {
	TurnsLeft                int `json:"turns_left"`                           // remaining model round-trips; Unbounded = no cap
	TokensLeft               int `json:"tokens_left"`                          // remaining output tokens; Unbounded = no cap
	ContextTokensLeft        int `json:"context_tokens_left,omitempty"`        // remaining prompt/context tokens; 0 = not configured
	ContextTokensCap         int `json:"context_tokens_cap,omitempty"`         // the configured context-budget size; the denominator the pre-exhaustion warning measures consumed-share against (0 = no context budget)
	ClarificationQueriesLeft int `json:"clarification_queries_left,omitempty"` // remaining clarification/self-query asks; 0 with no cap = not configured
	ClarificationQueriesCap  int `json:"clarification_queries_cap,omitempty"`  // configured clarification-query budget; positive cap with 0 left = exhausted
	// SpendMicroCentsLeft is the remaining PRICED-spend allotment in micro-cents
	// (1e-6 cent = 1e-8 USD — fine enough that every per-token price in the
	// published Anthropic tables, including the 0.1x cache-read multiplier, debits
	// as an exact integer). 0 = no spend budget configured, matching the context
	// axis's convention. The caller prices each turn (the table is deliberately
	// price-blind — see Usage.CostMicroCents); DebitUsage debits toward zero and
	// drains the session with ReasonBudgetSpend when the ceiling is crossed.
	SpendMicroCentsLeft int64 `json:"spend_micro_cents_left,omitempty"`
	// SpendMicroCentsCap is the configured spend-budget size (the denominator a
	// consumed-share display measures against); 0 = no spend budget.
	SpendMicroCentsCap int64 `json:"spend_micro_cents_cap,omitempty"`
	// ToolCallsLeft is the remaining DISPATCHED-tool-call allotment — the runaway
	// floor a scheduled/dispatched agent cannot extend (#2887, Hermes cron hardening).
	// Unlike TurnsLeft it is debited per tool call, not per model round-trip, so a
	// single turn that emits a long tool-call loop is cut at the budget too.
	// DebitToolCall spends one unit per call and drives the session to Draining with
	// ReasonBudgetToolCalls when the ceiling is crossed. It follows the spend/query
	// axis's 0=off convention (not the turns/tokens Unbounded=-1 sentinel), so a
	// Budget literal that never sets it stays permissive — the zero value is no cap.
	ToolCallsLeft int `json:"tool_calls_left,omitempty"`
	// ToolCallsCap is the configured tool-call-budget size, stamped from ToolCallsLeft
	// at set-time so "0 left with a positive cap = exhausted" is distinguishable from
	// "0 = unconfigured" once the remaining decrements to zero (the same disambiguation
	// ClarificationQueriesCap/SpendMicroCentsCap carry).
	ToolCallsCap int `json:"tool_calls_cap,omitempty"`
}

// MicroCentsPerCent converts a user-stated cent ceiling (SpendEnvelope.MaxCents)
// into the micro-cent unit the spend axis debits: 1 cent = 1_000_000 micro-cents.
const MicroCentsPerCent int64 = 1_000_000

// withContextCap stamps the context-budget capacity (the denominator the pre-exhaustion
// warning measures consumed-share against, #743) from the remaining when a budget is
// configured without an explicit cap. An explicit cap is preserved; an unbounded/zero
// context axis leaves the cap zero, so no warning is ever computed for a session that has
// no context budget. Decide/DebitUsage only ever decrement ContextTokensLeft, so the cap
// stamped here at set-time survives every later debit as the stable denominator.
func (b Budget) withContextCap() Budget {
	if b.ContextTokensCap <= 0 && b.ContextTokensLeft > 0 {
		b.ContextTokensCap = b.ContextTokensLeft
	}
	if b.ClarificationQueriesLeft < 0 {
		b.ClarificationQueriesCap = 0
	}
	if b.ClarificationQueriesCap <= 0 && b.ClarificationQueriesLeft > 0 {
		b.ClarificationQueriesCap = b.ClarificationQueriesLeft
	}
	if b.SpendMicroCentsCap <= 0 && b.SpendMicroCentsLeft > 0 {
		b.SpendMicroCentsCap = b.SpendMicroCentsLeft
	}
	if b.ToolCallsCap <= 0 && b.ToolCallsLeft > 0 {
		b.ToolCallsCap = b.ToolCallsLeft
	}
	return b
}

// unbounded reports whether an axis carries no limit. A negative value (canonically
// Unbounded) is treated as no-cap, so an operator clearing a budget with -1 is safe.
func (b Budget) turnsUnbounded() bool  { return b.TurnsLeft < 0 }
func (b Budget) tokensUnbounded() bool { return b.TokensLeft < 0 }
func (b Budget) contextBounded() bool  { return b.ContextTokensLeft > 0 }
func (b Budget) spendBounded() bool    { return b.SpendMicroCentsLeft > 0 }
func (b Budget) clarificationQueriesBounded() bool {
	return b.ClarificationQueriesCap > 0 || b.ClarificationQueriesLeft > 0
}

// toolCallsBounded reports whether a tool-call ceiling is configured. Like the
// clarification/spend axes it uses the 0=off convention (never the Unbounded=-1
// sentinel), so a Budget literal that leaves ToolCallsLeft/Cap zero is permissive;
// a positive cap keeps the axis bounded even after the remaining decrements to zero.
func (b Budget) toolCallsBounded() bool {
	return b.ToolCallsCap > 0 || b.ToolCallsLeft > 0
}

// Pace is the per-turn throttle — how to slow a session WITHOUT pausing it. It is
// admission control's cooperative twin: lowering MaxTokensPerTurn gives a shared
// GPU/CPU budget to an urgent session while the slow one keeps making progress.
// MaxTokensPerTurn caps THIS turn's output (lowered into the planner via
// agent.WithMaxTokens); MinTurnGapMs spaces turns apart. Zero on either axis means
// "no opinion" — the planner's own default stands, byte-identical to the pre-table
// path.
type Pace struct {
	MaxTokensPerTurn int `json:"max_tokens_per_turn"` // 0 = planner default
	MinTurnGapMs     int `json:"min_turn_gap_ms"`     // 0 = no inter-turn delay
}

// Assumption is one active fact-like row the session is relying on. It is data-only:
// the planner/self-query leaves decide whether a row is safe to use; the session
// table keeps the visible ledger so operator surfaces can show what is being
// assumed without reading hidden transcript text.
type Assumption struct {
	Key        string  `json:"key"`
	Statement  string  `json:"statement,omitempty"`
	Source     string  `json:"source"` // user_stated, inferred, queried, witnessed, stale, unknown
	Confidence float64 `json:"confidence,omitempty"`
	Expiry     string  `json:"expiry,omitempty"`
	SourceRef  string  `json:"source_ref,omitempty"`
}

// State is the full drive record for one session, keyed by TraceID. It carries its
// own TraceID so a Snapshot row is self-describing for a scheduler (which sorts a
// []State without re-keying). Rev is a monotonic revision bumped on every write —
// the optimistic-concurrency guard a stale operator UI is checked against, and the
// cursor a /v1/fak/changes stream of drive revisions would key on.
type State struct {
	TraceID        string   `json:"trace_id"`
	Run            RunState `json:"run"`
	Budget         Budget   `json:"budget"`
	Priority       int      `json:"priority"` // scheduling rank; lower yields first under contention
	Pace           Pace     `json:"pace"`
	Reason         string   `json:"reason,omitempty"`          // closed token on Throttled/Stopped; "" otherwise
	ContinuationID string   `json:"continuation_id,omitempty"` // fresh-window handoff id minted on context exhaustion
	ParentTrace    string   `json:"parent_trace,omitempty"`    // the trace this session was re-continued FROM (Recontinue lineage)
	Generation     int      `json:"generation,omitempty"`      // how many budget-reset re-continuations preceded this session (0 = original)
	// CacheAffinity is the provider/engine cache-affinity decision attached to a
	// continuation lineage (issue #1609). A context-budget reset must not silently
	// throw away a warm provider/engine cache route just because the visible trace
	// id changed, so the state carries an auditable decision: preserve the lineage's
	// opaque affinity key from parent trace to continuation trace. Advisory only:
	// correctness never depends on provider affinity landing.
	CacheAffinity CacheAffinityDecision `json:"cache_affinity,omitempty,omitzero"`
	// Intent is the ADVISORY, never-trust projection of what the kernel knows about
	// this session's next turn but the GPU cannot see (issue #807, the intent conduit
	// #805). A scheduler reading Snapshot MAY act on it to place KV / order prefill,
	// but MUST degrade to the GPU-visible decision when it is absent or stale — a hint
	// that gates correctness is a bug. The zero value is "no opinion".
	Intent TurnIntent `json:"intent,omitempty,omitzero"`
	// Goal is the session's active root descriptor (issue #849, the reachability-layer
	// epic #844). It is the cross-session bridge for the in-window goal pin
	// (internal/agent/ctxplan_session.go's goalPin, #845): a structural root a scheduler
	// reading Snapshot can rank a session by — an opaque id/digest plus an optional
	// Priority and Budget, NO transcript and NO model judgment. The zero value is "no
	// goal set", and a session with no goal behaves exactly as today. Advisory only: a
	// goal field that gated any decision would be a bug. Zero readers required — the
	// field is inert until a consumer (the scheduler, #627) acts on it.
	Goal Goal `json:"goal,omitempty,omitzero"`
	// Cost is the bounded per-session ring of the last CostRingSize turns' token cost
	// (issue #756, epic #748 Pillar 2), recorded by DebitUsage and carried out through
	// Snapshot so `fak ps` can render a true cost-PER-ITERATION column — the metric that
	// spikes ~200x on a runaway loop. Advisory/observability only, never trust: a renderer
	// reads it, no decision gates on it. The zero ring is the safe "no cost history yet"
	// default and, via omitzero, marshals byte-identically to a pre-ring State.
	Cost CostRing `json:"cost,omitempty,omitzero"`
	// LastActive is the durable dormancy clock (issue #1179, the random-time-horizons
	// epic #1178): a monotonic LastActiveAt stamp from which a session's dormancy band
	// (warm/cool/cold/frozen/ancient) is derivable without I/O via
	// LastActive.HorizonAt(now). It is the session's home for the "how long has this
	// been off?" measurement the rehydration rungs (#1181-#1186) will scale revalidation
	// to. ADVISORY / no-behavior-change in Phase 1: zero readers gate on it, and the zero
	// (never-stamped) Stamp marshals away via omitzero, so a pre-clock State is wire-
	// identical. A consumer that promotes it to a live field (resume's idle figure, the
	// scheduler's dormant-vs-stuck split #1180) lands in a later phase.
	LastActive dormancy.Stamp `json:"last_active,omitempty,omitzero"`
	// Time is the wall-clock budget tracker (issue #1584, epic #1570 "managed
	// context"): a persisted, timestamp-based allotment of REAL elapsed time,
	// independent of the token axes on Budget. It is carried forward across a
	// Recontinue re-arm exactly like Generation/ParentTrace (see (*Table) RecontinueAt in
	// table.go), so a hidden context reset does not zero the wall-clock accounting.
	// The zero value is unbounded/never-started — a State with no configured time
	// envelope behaves byte-identically to a pre-#1584 State (omitzero keeps the wire
	// shape unchanged when unused).
	Time TimeBudget `json:"time,omitempty,omitzero"`
	// Throughput is the throughput envelope axis as live drive state (issue #2762,
	// the out-of-band operator-control epic #2753): the soft expected pace-shaping
	// rate, the enforced minimum sustained-rate floor, and the accumulated
	// observation window DebitUsage judges the floor against (see throughput.go).
	// Carried forward across a Recontinue re-arm like Time/spend, so a hidden
	// context reset cannot launder a session running below its floor. The zero
	// value is unconfigured — a State with no throughput envelope behaves
	// byte-identically to a pre-#2762 State (omitzero keeps the wire shape
	// unchanged when unused).
	Throughput ThroughputBudget `json:"throughput,omitempty,omitzero"`
	// Assumptions is the live, visible ledger of facts the session is relying on.
	// It carries provenance, confidence, and expiry only; it never carries hidden
	// transcript bytes and it does not gate behavior by itself. Empty means the
	// session has no active assumptions to report.
	Assumptions []Assumption `json:"assumptions,omitempty"`
	// ResetTransaction is the latest context-budget reset row that minted this trace.
	// It binds old trace, new trace, seed digest, contributor list, omitted spans, and
	// fresh budget re-arm to the child state, so a continuation can be audited from
	// kernel data instead of a model self-report. The zero value means this state was
	// not produced by a reset.
	ResetTransaction ResetTransaction `json:"reset_transaction,omitempty,omitzero"`
	// ObjectivePin is the standing user objective's stable, addressable span (issue
	// #1583, the managed-context runtime-continuity epic #1570): a PinID that must
	// survive every replan/reset/migration unchanged, plus a content Digest that makes
	// "the objective was preserved" a checkable equality instead of a narrative claim.
	// It rides the drive record — and therefore this State's existing dump/restore and
	// sessionimage migration paths — so a session migrated to a new process (issue
	// #1589) reports the SAME pinned objective before continuing, not a silently reset
	// one. The zero value means no objective has been pinned yet; a State with no pin
	// behaves byte-identically to a pre-#1589 State (omitzero keeps the wire shape
	// unchanged when unused). session owns no pinning policy of its own — sessionreset's
	// PinObjective/RepinObjective/CarryObjective mint and reconcile the pin; this field
	// is only its durable home on the drive so a migration cannot drop it.
	ObjectivePin ctxplan.ObjectivePin `json:"objective_pin,omitempty,omitzero"`
	// Pins is the OPERATOR-declared keep-set for this session (issue #2211, the
	// out-of-band control-plane epic #2208): the span / fact ids an operator has asked
	// to stay resident through the planner's automatic shed. It is the cross-session
	// home of the in-window keep-set the planner applies via SessionPlanner
	// .SetOperatorPins (internal/agent) — a declared INTENT input, never a plan: the
	// table records the ids and nothing more, the planner forces them ahead of the
	// knapsack, and eviction stays automatic for everything else. An id naming no live
	// span is harmless (the planner skips it), so drive state may legitimately run
	// ahead of ingestion.
	//
	// The set is REPLACED wholesale by each write and cleared by a nil/empty one (see
	// Table.SetPins) — pin/unpin merge discipline lives with the CLI's
	// read-modify-write, so two operators racing cannot silently union their keep-sets.
	// SetPins copies the caller's slice in; readers must treat the slice a Get/Snapshot
	// hands back as read-only, exactly like every other reference field on this record.
	// The zero value is "no operator pins" and, via omitempty, marshals byte-identically
	// to a pre-#2211 State.
	Pins []string `json:"pins,omitempty"`
	// PendingTurn is the write-ahead checkpoint of an in-flight turn's retry/backoff
	// progress (issue #1363, epic #1352 Pillar 3 "durable turn"). The retry loop
	// (internal/agent's HTTPPlanner.Complete) tracks its attempt count and last
	// observed status purely in local Go variables — a kill -9 mid-retry loses that
	// progress even though the rest of this State survives via the Descriptor/
	// Registry (session_durable.go's persistServeSessionRevision already writes
	// through on every Decide/DebitUsage). Recording it here, through the SAME
	// already-wired persistence path, gives a restart a real pointer to how far the
	// lost turn had gotten instead of silently starting over with no memory of what
	// already failed. The zero value means "no turn in flight" (StartedAt is zero);
	// a caller clears it (SetPendingTurn with the zero value) once the turn
	// completes, so a session with no in-flight retry restores byte-identically to
	// a pre-#1363 State (omitzero keeps the wire shape unchanged when unused). This
	// field is the durable primitive only — wiring internal/agent's retry loop to
	// call SetPendingTurn is the follow-on that actually closes #1363.
	PendingTurn PendingTurn `json:"pending_turn,omitempty,omitzero"`
	// QualityEnvelope is the session-start QA origin record (issue #1964, QA-dogfood
	// spine #1961/QD-004): the single record saying which QA controls govern this
	// session — the budget axes it opened under, the witness policy that gates its
	// claims, the dogfood probes expected at origin, and the control-pane scorecards
	// it is a member of (see quality_envelope.go). It rides the drive record — and
	// therefore this State's existing dump/restore and sessionimage migration paths —
	// so a session dumped and restored (or migrated to a new process) exposes the SAME
	// QA envelope it started under, not a re-derived or silently reset one. The zero
	// value means no envelope was stamped; a State with none marshals byte-identically
	// to a pre-#1964 State (omitzero keeps the wire shape unchanged when unused).
	// Advisory/observability only: no decision gates on it — a renderer or inspect
	// surface reads it, nothing trusts it.
	QualityEnvelope QualityEnvelope `json:"quality_envelope,omitempty,omitzero"`
	Rev             uint64          `json:"rev"`
}

// Goal is the structural root descriptor carried on State (issue #849). It names the
// session's active goal so a scheduler reading Table.Snapshot can rank by it — the
// cross-session counterpart of the in-window goal pin (#845) that today lives only in
// SessionPlanner.pins(). It is deliberately data-only: an opaque ID (a digest or
// /goal id, never the goal text or a transcript), an optional scheduling Priority, and
// an optional token Budget. Every field defaults to the safe "no opinion" zero value.
//
// FENCE: advisory, never trust. A goal root affects RETENTION/ranking, never the
// answer — a scheduler MAY order a session by it but MUST behave identically when it
// is absent. The ranking reader is still pending — no scheduler orders by Goal today —
// but the field is NOT consumer-free: as of b588466054 cumulative_envelope.go copies
// State.Goal into SessionRecoveryCheckpoint, so a tripped envelope checkpoints the active
// root instead of losing it. Re-check both claims against the tree before trusting them.
type Goal struct {
	// ID is the opaque goal/root identifier — a digest or the /goal id, structural only.
	// "" means no goal is set (the zero value). NEVER the goal text or a transcript.
	ID string `json:"id,omitempty"`
	// Priority is the OPTIONAL scheduling rank this goal lends its session (lower yields
	// first, matching State.Priority's convention). 0 = no opinion; the scheduler falls
	// back to State.Priority.
	Priority int `json:"priority,omitempty"`
	// Budget is the OPTIONAL token budget the goal is granted. 0 = no opinion.
	Budget int `json:"budget,omitempty"`
}

// IsZero reports whether the goal carries no root — the safe default a scheduler reads
// as "this session has no active goal to rank by". A consumer checks this before acting
// on any field, so an unset goal is never mistaken for a positive root. It also drives
// the `omitzero` JSON tag so a goal-less State marshals byte-identically to today.
func (g Goal) IsZero() bool {
	return g.ID == "" && g.Priority == 0 && g.Budget == 0
}

// TurnIntent is the read-only, advisory hint set the adjudicator/session layer emits
// for a session's NEXT turn, folded into State so a scheduler reading Table.Snapshot
// can act on what the kernel already knows — the continuous-batching guesses it would
// otherwise have to reconstruct from sequence length, KV occupancy, and arrival order
// alone (issue #807). Every field defaults to the safe "no opinion" zero value.
//
// FENCE: advisory, never trust. A hint can be wrong (a turn expected to end keeps
// going); every consumer degrades to the GPU-visible decision when a hint is absent or
// stale. This is a cost/latency lever only — a hint must NEVER gate correctness. It is
// a pure projection over Table.Snapshot and adds nothing to the frozen ABI beyond this
// struct. The snapshot-reading scheduler HAS landed: as of 7ad164d0fa (#811)
// Scheduler.ReserveKnownComing walks Table.Snapshot and reservationFromState consumes
// Intent.ArrivingInMillis/Prefix/WillDiscard to mint advisory slot reservations. What is
// still absent is a live DRIVER — no non-test caller invokes ReserveKnownComing — so the
// hint changes no serving behavior yet. Re-check both halves before trusting them.
type TurnIntent struct {
	// EndsSoon: the agent is at a settle point — drain this turn, don't admit new
	// prefill behind it.
	EndsSoon bool `json:"ends_soon,omitempty"`
	// IsSpeculative: this turn is a branch that may be thrown away — prefer-not-to-prefill.
	IsSpeculative bool `json:"is_speculative,omitempty"`
	// WillDiscard: this turn's result is already known to be discarded — the strongest
	// prefer-not-to-prefill signal (ties to discard-aware admission, #808).
	WillDiscard bool `json:"will_discard,omitempty"`
	// SharesPrefixWith names another live session (by TraceID) this turn shares a
	// verbatim prompt prefix with — co-batch / pin the shared KV. "" means no known overlap.
	SharesPrefixWith string `json:"shares_prefix_with,omitempty"`
	// ArrivingInMillis is the deterministic forward-looking signal for a known-coming
	// follow-up turn (issue #811): a tool has been dispatched and the kernel expects this
	// session to re-enter after roughly this many milliseconds. It is advisory and
	// expires in the scheduler; <=0 means no forward reservation request.
	ArrivingInMillis int64 `json:"arriving_in,omitempty"`
	// Prefix is the known reusable prefix identity for that follow-up turn. It is an
	// opaque digest/key, never transcript text. A scheduler may pin matching KV residency
	// and promote the reservation when the real request arrives with the same prefix.
	Prefix string `json:"prefix,omitempty"`
	// ResultAlreadyKnown: the call's output is determined — route to the avoid-the-
	// forward-pass path (ties to vToolcall / vCache, #794/#795).
	ResultAlreadyKnown bool `json:"result_already_known,omitempty"`
}

// IsZero reports whether the intent carries no opinion — the safe default a scheduler
// reads as "fall back to the GPU-visible decision". A consumer checks this before
// acting on any field, so an unset (or never-emitted) intent is never mistaken for a
// positive hint.
func (ti TurnIntent) IsZero() bool {
	return !ti.EndsSoon && !ti.IsSpeculative && !ti.WillDiscard &&
		ti.SharesPrefixWith == "" && ti.ArrivingInMillis <= 0 && ti.Prefix == "" &&
		!ti.ResultAlreadyKnown
}

// PendingTurn is one in-flight turn's write-ahead checkpoint (issue #1363): how many
// retry attempts it has made, the last HTTP status observed, and when the turn
// started. It is deliberately narrow — just enough for a restart to tell "a turn was
// mid-retry, this far along" — not a replay log of the turn's messages (those stay in
// the provider's / Claude Code's own transcript store, matching the Descriptor's
// existing drive-state-not-transcript fence).
type PendingTurn struct {
	// Attempt is the retry attempt number in progress (1 = first attempt). 0 means
	// no attempt has been checkpointed yet.
	Attempt int `json:"attempt,omitempty"`
	// LastStatus is the last HTTP status the retry loop observed (e.g. 429, 503).
	// 0 means none observed yet.
	LastStatus int `json:"last_status,omitempty"`
	// StartedAtUnixNano is the wall-clock instant (unix nanoseconds) this turn
	// began, so a restart can tell how long the lost turn had been running. It is a
	// timestamp, not a monotonic clock reading (the same TimeBudget.StartedAtUnixNano
	// discipline: a value that must survive a JSON round-trip and a process restart
	// cannot depend on a monotonic reading that dies with the process). Zero means
	// no turn is checkpointed.
	StartedAtUnixNano int64 `json:"started_at_unix_nano,omitempty"`
}

// IsZero reports whether no turn is currently checkpointed — the safe default a
// restart reads as "nothing was in flight". A consumer checks this before treating
// the fields as a real in-progress attempt.
func (p PendingTurn) IsZero() bool {
	return p.Attempt == 0 && p.LastStatus == 0 && p.StartedAtUnixNano == 0
}

// DefaultState is the drive a fresh/unseen session reads: Running, unbounded budget,
// zero priority, no pace opinion. It is what Get returns for a trace the table has
// never seen and what an LRU-evicted trace reads on its next touch — the safe
// default (a live session at its defaults), never a phantom Stopped.
func DefaultState(traceID string) State {
	return State{
		TraceID: traceID,
		Run:     Running,
		Budget:  Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded},
	}
}
