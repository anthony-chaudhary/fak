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
// This package is a foundation leaf: stdlib-only (container/list + sync), off the
// request path, registers nothing. The zero Table is not usable — construct with
// NewTable / NewTableWithLimit.
package session

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
)

// String renders a RunState as its lowercase wire token (the form the
// /v1/fak/session routes emit and accept). An out-of-range value renders
// "unknown" rather than panicking — a wire value is never trusted to be in range.
func (s RunState) String() string {
	switch s {
	case Running:
		return "running"
	case Throttled:
		return "throttled"
	case Paused:
		return "paused"
	case Draining:
		return "draining"
	case Stopped:
		return "stopped"
	}
	return "unknown"
}

// ParseRunState maps a wire token back to a RunState. The bool is false for an
// unrecognized token, so a caller fails closed (the route returns 400) rather than
// defaulting an unknown verb to Running.
func ParseRunState(s string) (RunState, bool) {
	switch s {
	case "running":
		return Running, true
	case "throttled":
		return Throttled, true
	case "paused":
		return Paused, true
	case "draining":
		return Draining, true
	case "stopped":
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
	TurnsLeft         int `json:"turns_left"`                    // remaining model round-trips; Unbounded = no cap
	TokensLeft        int `json:"tokens_left"`                   // remaining output tokens; Unbounded = no cap
	ContextTokensLeft int `json:"context_tokens_left,omitempty"` // remaining prompt/context tokens; 0 = not configured
	ContextTokensCap  int `json:"context_tokens_cap,omitempty"`  // the configured context-budget size; the denominator the pre-exhaustion warning measures consumed-share against (0 = no context budget)
}

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
	return b
}

// unbounded reports whether an axis carries no limit. A negative value (canonically
// Unbounded) is treated as no-cap, so an operator clearing a budget with -1 is safe.
func (b Budget) turnsUnbounded() bool  { return b.TurnsLeft < 0 }
func (b Budget) tokensUnbounded() bool { return b.TokensLeft < 0 }
func (b Budget) contextBounded() bool  { return b.ContextTokensLeft > 0 }

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
	Goal Goal   `json:"goal,omitempty,omitzero"`
	Rev  uint64 `json:"rev"`
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
// is absent. No consumer exists until the snapshot-reading scheduler (#627) reads it;
// this carries the data structure so the root is defined ahead of its first reader.
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
// struct. No consumer exists until the snapshot-reading scheduler (#627) lands; this is
// filed now so the conduit is defined and sequenced ahead of its first reader.
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
