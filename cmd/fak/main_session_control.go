package main

// main_session_control.go — the served-session DRIVE control surface, lifted out of
// main.go along its concern seam (god-file ceiling, #2898). It owns the process-local
// session table and the closures the gateway holds by injection to observe, list,
// decide, debit, budget-reset, steer, and control one live session, so main.go stays
// the verb routing table. Pure code motion within package main: same declarations,
// same behavior, and session_control_test.go already exercises them from here.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionreset"
)

// serveSessions is the process-local per-session DRIVE-state table shared by the
// gateway session routes (observe/control) and any in-process agent loop. It is the
// structural twin of ifc.Default: TraceID-keyed, bounded-LRU, live-mutable  -  widened
// from the single taint bit to a small drive struct (run-state/budget/priority/pace).
// Constructed once at process start; the gateway holds it by injected closure, never
// by import, so the gateway stays session-internals-blind the way it stays
// IFC-internals-blind for the trace routes.
var serveSessions = session.NewTable()

// resetTransactions is the process-local, append-only audit trail of every
// budget-triggered reset resetServedSessionOnBudget performs (issue #1582): a
// ResetTransactionLog row is appended immediately after each successful
// Table.RecontinueWithTransaction call, so the full reset history for the process
// survives independent of any single trace's State (which only carries the LATEST
// transaction in its own lineage). Safe for concurrent resets across traces — see
// ResetTransactionLog's own locking.
var resetTransactions session.ResetTransactionLog

// observeSession is the read side of the /v1/fak/session control surface (#620): it
// returns one served session's current DRIVE state so an operator can read how hard
// a live session is running without reconstructing it from git + a process scan. An
// unseen trace reads its default  -  Running, unbounded budget  -  the table's own safe
// default, never a phantom Stopped.
func observeSession(_ context.Context, traceID string) gateway.SessionState {
	traceID = strings.TrimSpace(traceID)
	if serveSessionDurability != nil && !sessionTableHas(serveSessions, traceID) {
		if st, ok, err := serveSessionDurability.lookupState(traceID); err == nil && ok {
			return toGatewaySessionState(st)
		}
	}
	return toGatewaySessionState(serveSessions.Get(traceID))
}

// listSessions is the multi-session read side of the /v1/fak/session control surface:
// it projects the WHOLE live drive table (Snapshot order  -  by priority, lower yields
// first) into the gateway wire DTO so an operator can see what every session is doing
// right now in one read, instead of reconstructing liveness from git + a process scan
// (docs/dispatch-loop.md). Snapshot already returns a fresh, sorted copy.
func listSessions(_ context.Context) []gateway.SessionState {
	snap := serveSessions.Snapshot()
	out := make([]gateway.SessionState, 0, len(snap))
	seen := make(map[string]bool, len(snap))
	for _, s := range snap {
		out = append(out, toGatewaySessionState(s))
		seen[s.TraceID] = true
	}
	if serveSessionDurability != nil {
		if persisted, err := serveSessionDurability.snapshotStates(); err == nil {
			for _, s := range persisted {
				if seen[s.TraceID] {
					continue
				}
				out = append(out, toGatewaySessionState(s))
			}
		}
	}
	return out
}

// decideSession is the served request-boundary hook: it applies session.Table.Decide
// to the process-local DRIVE table so served model requests honor run-state,
// turn-budget, token-budget, and pace controls before the upstream request runs.
func decideSession(ctx context.Context, traceID string) gateway.SessionVerdict {
	traceID = strings.TrimSpace(traceID)
	v := serveSessions.Decide(traceID)
	// #5640: this is the one boundary EVERY served session crosses, so it is where the
	// broadcast tag registry is produced — tag the trace with this process's routing
	// identity on the way in, drop it once the session is terminal. Without a producer
	// here the registry stays empty and every --lane/--wave/--label fleet directive
	// resolves to zero sessions. See serve_session_tag.go.
	tagServedSessionAdmit(traceID, v.State)
	persistServeSessionRevision(ctx, traceID, v.State)
	return toGatewaySessionVerdict(v)
}

// debitSession records post-response token usage for a served request. The next
// Decide observes normal token-budget exhaustion at the following turn boundary;
// context-budget exhaustion drains immediately with a continuation id, and a
// priced spend-ceiling crossing drains immediately with BUDGET_SPEND_EXHAUSTED
// (the turn cost comes from the host spend meter — see session_spend.go).
func debitSession(ctx context.Context, traceID string, usage gateway.SessionUsage) gateway.SessionState {
	traceID = strings.TrimSpace(traceID)
	st := serveSessions.DebitUsage(traceID, session.Usage{
		OutputTokens:   usage.CompletionTokens,
		ContextTokens:  usage.ContextTokens,
		CostMicroCents: servedTurnSpendMicroCents(usage),
		// The turn's real duration feeds the throughput axis's sustained-rate observation
		// (#2762); dropping it here left an operator-set `min_throughput` floor inert on the
		// served path (ObservedNanos never accumulated, so BelowFloor never tripped).
		DurationNanos: usage.DurationNanos,
	})
	persistServeSessionRevision(ctx, traceID, st)
	return toGatewaySessionState(st)
}

// resetServedSessionOnBudget is the host-owned "human-like reset" hook the gateway
// calls after a context-budget drain. It distills the refused request transcript into
// a compact carryover seed, re-arms the continuation trace with a fresh context budget,
// and hands both back to the gateway so the live request can continue transparently.
func resetServedSessionOnBudget(freshContextTokens int) gateway.ResetOnBudgetFunc {
	if freshContextTokens <= 0 {
		return nil
	}
	return func(ctx context.Context, traceID string, messages []agent.Message) (string, []agent.Message, bool) {
		traceID = strings.TrimSpace(traceID)
		st := serveSessions.Get(traceID)
		child := strings.TrimSpace(st.ContinuationID)
		if traceID == "" || child == "" {
			return "", nil, false
		}
		rearmTokens := freshContextTokens
		if st.Budget.ContextTokensCap > 0 {
			rearmTokens = st.Budget.ContextTokensCap
		}
		resetInput := sessionreset.Input{
			Trace:          traceID,
			Messages:       resetMsgs(messages),
			FreshBudgetTok: rearmTokens,
		}
		seed := sessionreset.BuildSeed(resetInput)
		if strings.TrimSpace(seed.Recap) == "" {
			return "", nil, false
		}
		resetTx := sessionreset.BuildResetTransaction(resetInput, child, seed)
		childState := serveSessions.RecontinueWithTransaction(traceID, child, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: rearmTokens,
		}, resetTx)
		resetTransactions.Append(childState.ResetTransaction)
		persistServeSessionRevision(ctx, child, childState)
		return child, []agent.Message{{Role: agent.RoleSystem, Content: seed.Recap}}, true
	}
}

func resetMsgs(messages []agent.Message) []sessionreset.Msg {
	out := make([]sessionreset.Msg, 0, len(messages))
	for _, m := range messages {
		out = append(out, sessionreset.Msg{Role: m.Role, Content: m.Content})
	}
	return out
}

// budgetWebhookObserver returns the session.BudgetObserver that wires the operator
// webhook (#743): it POSTs each pre-exhaustion warning and each exhaustion (reset-trigger)
// event to rawURL as JSON, so an external monitor is notified BEFORE a served session
// drains, not only after. It is fire-and-forget and fail-open  -  the POST runs on its own
// goroutine under a short timeout, and any transport error is logged to stderr but never
// blocks or fails the served turn that produced the event. An empty URL returns nil (the
// no-op seam: behavior is byte-identical to today).
func budgetWebhookObserver(rawURL string) session.BudgetObserver {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil
	}
	return func(ev session.BudgetEvent) {
		body, err := json.Marshal(ev)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak: budget webhook encode failed: %v\n", err)
			return
		}
		webhookPOST("budget webhook", rawURL, body, "application/json")
	}
}

// controlSession is the write side of /v1/fak/session (#620): it applies one control
// verb (run|budget|pace|priority) to a live session's DRIVE state. The (ok=false,
// err=nil) return is a terminal/CAS refusal (the route maps it to 409); a non-nil
// err is a malformed verb/body (the route maps it to 400). if_rev, when non-zero,
// is an optimistic-concurrency guard: the write is taken only if the session's
// current Rev matches (read-then-CompareAndSet; a lost race returns ok=false).
func controlSession(ctx context.Context, traceID, verb string, req gateway.SessionControlRequest) (gateway.SessionState, bool, error) {
	traceID = strings.TrimSpace(traceID)
	st, ok, err := applySessionControlDurable(ctx, serveSessions, serveSessionDurability, traceID, verb, req)
	if err != nil {
		return gateway.SessionState{}, false, err
	}
	// #5640: an operator stop through this route is the one session end that produces no
	// following admission, so the broadcast tag would otherwise outlive the session it
	// names. Only on a TAKEN write — a refused CAS changed nothing.
	if ok {
		dropServedSessionTagIfEnded(traceID, st)
	}
	return toGatewaySessionState(st), ok, nil
}

// steerSession enqueues an operator steer onto the process-global a2achan bus so a RUNNING
// detached session can receive the input at its next turn boundary (#760). The serve
// process owns the bus, so the in-process Send happens HERE (the CLI is a separate process
// that POSTs HTTP; only the server can enqueue onto the bus it shares with the served loop).
//
// The body rides the a2achan floor: "operator" is a different principal from the target
// trace, so a Private (ScopeAgent) body would be refused — Shared (ScopeFleet) is the
// auditable cross-principal widening the operator must make, and it stays Tainted (operator
// input is untrusted, screened on ingress). A tainted/over-scoped/uncapped Send is refused
// by the SAME default-deny floor that gates a tool call; that deny-as-value becomes the
// error the route maps to 422 — "a tainted/over-scoped steer is refused", mechanically.
func steerSession(ctx context.Context, traceID, principal, text string) error {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		return errors.New("trace_id is required")
	}
	// `from` is the attribution of the steer source, not a trust grant: the a2achan floor
	// gates on caps+taint+scope (Shared ⇒ Tainted/ScopeFleet), never on the from-string, so
	// naming a machine principal (e.g. "doomloop-guard", #3529) truthfully labels the source
	// without buying it more authority than an operator steer. Empty ⇒ the human default.
	from := strings.TrimSpace(principal)
	if from == "" {
		from = "operator"
	}
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: traceID}
	v := a2achan.Default.Send(ctx, from, key, a2achan.Shared([]byte(text)), a2achan.CapA2ASend)
	if v.Kind != abi.VerdictAllow {
		return fmt.Errorf("a2a floor refused (%s)", abi.ReasonName(v.Reason))
	}
	return nil
}

// applySessionControl routes one control verb to the matching Table write. It is the
// single place that knows the verb set, so the verb vocabulary lives with the table
// owner (cmd/fak), not the gateway. CAS (if_rev>0) reads the current record, mutates
// the one field, and CompareAndSets; a concurrent write between read and CAS loses
// the race and returns ok=false for the client to retry.
func applySessionControl(tbl *session.Table, traceID, verb string, req gateway.SessionControlRequest) (session.State, bool, error) {
	switch verb {
	case "run":
		run, ok := session.ParseRunState(req.Run)
		if !ok {
			return session.State{}, false, fmt.Errorf("unknown run-state %q (want running|throttled|paused|draining|terminating|stopped)", req.Run)
		}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) {
				s.Run = run
				s.Reason = transitionReason(run, req.Reason)
			})
		}
		st, ok := tbl.Transition(traceID, run, req.Reason)
		return st, ok, nil
	case "pause", "resume", "throttle", "stop":
		var run session.RunState
		switch verb {
		case "pause":
			run = session.Paused
		case "resume":
			run = session.Running
		case "throttle":
			run = session.Throttled
		case "stop":
			run = session.Stopped
		}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) {
				s.Run = run
				s.Reason = transitionReason(run, req.Reason)
			})
		}
		st, ok := tbl.Transition(traceID, run, req.Reason)
		return st, ok, nil
	case "budget":
		if req.Budget == nil {
			return session.State{}, false, errors.New("budget is required")
		}
		// Spend rides the budget wire (#2762): the CLI projects `spend=$25` onto the
		// priced axis, and a partial `budget` write reads-then-preserves it — so the
		// two spend fields must be carried here, else the envelope's spend ceiling (or
		// a live one on a plain --turns edit) is silently cleared.
		b := session.Budget{
			TurnsLeft:           req.Budget.TurnsLeft,
			TokensLeft:          req.Budget.TokensLeft,
			ContextTokensLeft:   req.Budget.ContextTokensLeft,
			SpendMicroCentsLeft: req.Budget.SpendMicroCentsLeft,
			SpendMicroCentsCap:  req.Budget.SpendMicroCentsCap,
		}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) { s.Budget = b })
		}
		st, ok := tbl.SetBudget(traceID, b)
		return st, ok, nil
	case "pace":
		if req.Pace == nil {
			return session.State{}, false, errors.New("pace is required")
		}
		p := session.Pace{MaxTokensPerTurn: req.Pace.MaxTokensPerTurn, MinTurnGapMs: req.Pace.MinTurnGapMs}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) { s.Pace = p })
		}
		st, ok := tbl.SetPace(traceID, p)
		return st, ok, nil
	case "priority":
		if req.Priority == nil {
			return session.State{}, false, errors.New("priority is required")
		}
		pri := *req.Priority
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) { s.Priority = pri })
		}
		st, ok := tbl.SetPriority(traceID, pri)
		return st, ok, nil
	case "wall":
		// Wall-clock ceiling (#2762): set the envelope and arm the clock at now, mirroring
		// SetWallClockLimit. now is read at the process boundary like toGatewaySessionState;
		// Start is a no-op on an already-ticking clock, so a mid-run reset moves the ceiling
		// without rewinding elapsed time. A <=0 limit clears the envelope (WithLimit's rule).
		if req.Wall == nil {
			return session.State{}, false, errors.New("wall is required")
		}
		limit := time.Duration(req.Wall.LimitNanos)
		now := time.Now()
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) {
				s.Time = s.Time.WithLimit(limit).Start(now)
			})
		}
		st, ok := tbl.SetWallClockLimit(traceID, limit, now)
		return st, ok, nil
	case "throughput":
		// Throughput envelope (#2762): the soft expected pace-shaping rate plus the enforced
		// minimum sustained-rate floor. The accumulated observation window is preserved (only
		// the rates are re-stated), matching SetThroughputBudget — re-arming must not forget
		// what has already been measured under a live floor.
		if req.Throughput == nil {
			return session.State{}, false, errors.New("throughput is required")
		}
		tp := session.ThroughputBudget{
			ExpectedTokensPerSec: req.Throughput.ExpectedTokensPerSec,
			MinTokensPerSec:      req.Throughput.MinTokensPerSec,
		}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) {
				s.Throughput.ExpectedTokensPerSec = tp.ExpectedTokensPerSec
				s.Throughput.MinTokensPerSec = tp.MinTokensPerSec
			})
		}
		st, ok := tbl.SetThroughputBudget(traceID, tp)
		return st, ok, nil
	}
	return session.State{}, false, fmt.Errorf("unknown verb %q (want run|pause|resume|throttle|stop|budget|pace|priority|wall|throughput)", verb)
}

// casApply reads the current drive record, mutates it in place, and CompareAndSets
// it against expectRev. It is the optimistic-concurrency form of every verb. A lost
// race (the table moved between read and CAS) returns ok=false; the caller maps that
// to 409 and the client re-reads and retries.
func casApply(tbl *session.Table, traceID string, expectRev uint64, apply func(*session.State)) (session.State, bool, error) {
	cur := tbl.Get(traceID)
	apply(&cur)
	st, ok := tbl.CompareAndSet(traceID, expectRev, cur)
	return st, ok, nil
}

// transitionReason mirrors Table.Transition's reason bookkeeping so a CAS run-state
// change records/clears the reason token identically to the direct path: Throttled
// and Stopped carry the reason, Running clears it.
func transitionReason(to session.RunState, reason string) string {
	switch to {
	case session.Throttled, session.Stopped, session.Terminating, session.Paused, session.Draining:
		return reason
	case session.Running:
		return ""
	}
	return ""
}
