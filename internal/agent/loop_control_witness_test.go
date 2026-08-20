package agent

// loop_control_witness_test.go — the #2766 acceptance witness (epic #2753): ONE
// table across every live out-of-band control op, proving for each op BOTH halves
// of the epic's contract:
//
//   applied — the loop-side proof the op was CONSUMED (enqueue is not applied):
//             the table/mailbox write alone is never the witness; the witness is
//             the running arm observing it at a turn boundary.
//   refused — the illegal-for-state case surfaces a STRUCTURED refusal carrying
//             the right token from the closed vocabulary, never a bare boolean.
//
// Per-op boundary + witness + refusal (the #2766 op docs):
//
//	op        boundary     applied witness (loop-side)                       refusal tokens (closed)
//	steer     next turn    prose spliced into turn input; mailbox drained    DEFAULT_DENY (capability floor; TRUST_VIOLATION for taint/scope)
//	redirect  next turn    objective directive carried as SYSTEM message;    REDIRECT_MALFORMED,
//	                       mailbox drained; live objective updated           REDIRECT_NO_REDIRECTABLE_STATE
//	pause     next turn    arm holds at boundary: 0 turns run, PAUSED        CONTROL_SESSION_TERMINAL
//	resume    same turn    parked arm wakes and completes the HELD turn      CONTROL_SESSION_TERMINAL
//	cancel    next turn    Draining (enqueued) finalized to Stopped at the   CONTROL_SESSION_TERMINAL,
//	                       boundary; arm stops with DRAINING                 CONTROL_REV_STALE (--if-rev race)
//	terminate safe point   Terminating wakes the arm MID-TURN (#2758): the   CONTROL_SESSION_TERMINAL,
//	                       in-flight model call's context is cancelled, no   CONTROL_REV_STALE (--if-rev race)
//	                       further tool call dispatches; arm stops with
//	                       TERMINATED (a drain would run the turn out)
//	throttle  next turn    pace cap lowered into THIS turn's sampling        CONTROL_SESSION_TERMINAL
//	budget    next turn    exhausted allotment stops the arm with the        CONTROL_SESSION_TERMINAL
//	                       closed exhaustion reason; record finalized
//	priority  next pick    scheduler-consumed: Snapshot rank order reflects  CONTROL_SESSION_TERMINAL
//	                       the write (the loop itself never reads priority)
//
// The add-constraint op (#2756) registers its CONSTRAINT_* tokens when its own
// child lands; `fak session status`/`context` are pure reads (nothing to apply or
// refuse). The refused paths for the drive-state verbs go through
// session.ControlRefusalFor — the #2766 mapping from the bare ok=false every
// Table verb returns onto the closed control-refusal vocabulary.

import (
	"context"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// paceCapPlanner records the effective per-turn output cap each Complete call was
// sampled under (the resolved SampleParams.MaxTokens), then finishes the turn. It
// is how a test proves a pace write reached the SAMPLING of a turn — loop-side
// consumption — rather than merely sitting in the table.
type paceCapPlanner struct{ caps []int }

func (p *paceCapPlanner) Model() string { return "pace-cap" }
func (p *paceCapPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, opts ...SampleOpt) (*Completion, error) {
	var sp SampleParams
	for _, o := range opts {
		o(&sp)
	}
	cap := 0
	if sp.MaxTokens != nil {
		cap = *sp.MaxTokens
	}
	p.caps = append(p.caps, cap)
	return &Completion{
		Message:      Message{Role: RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        Usage{CompletionTokens: 1},
	}, nil
}

// controlOpWitness is one row of the #2766 contract table: the op, the closed
// refusal tokens its illegal-for-state cases may carry, the loop-side applied
// witness, and the refused path returning the tokens it actually observed.
type controlOpWitness struct {
	op      string
	tokens  []string
	applied func(t *testing.T)
	refused func(t *testing.T) []string
}

// stoppedTable returns a table whose trace is already terminal — the shared
// illegal state ("cannot <op> a Stopped session") the drive-state rows refuse on.
func stoppedTable(t *testing.T, trace string) *session.Table {
	t.Helper()
	tbl := session.NewTable()
	if _, ok := tbl.Transition(trace, session.Stopped, "done"); !ok {
		t.Fatalf("arranging terminal state: Transition(%q, Stopped) refused", trace)
	}
	return tbl
}

// refuseTerminal drives one Table verb against a Stopped session and returns the
// closed token of its structured refusal, failing if the write was applied or
// the refusal mis-typed.
func refuseTerminal(t *testing.T, op string, write func(tbl *session.Table, trace string) (session.State, bool)) []string {
	t.Helper()
	trace := "ctl-2766-" + op + "-terminal"
	tbl := stoppedTable(t, trace)
	st, ok := write(tbl, trace)
	r := session.ControlRefusalFor(op, st, ok)
	if r == nil {
		t.Fatalf("%s against a Stopped session was APPLIED (ok=%v) — a terminal session must refuse every control write", op, ok)
	}
	if r.Reason != session.ReasonControlSessionTerminal {
		t.Fatalf("%s on Stopped refused with %q, want the closed token %q", op, r.Reason, session.ReasonControlSessionTerminal)
	}
	return []string{r.Reason}
}

// controlOpWitnesses is the #2766 table: every live control op with both paths.
func controlOpWitnesses() []controlOpWitness {
	return []controlOpWitness{
		{
			op:     "steer",
			tokens: []string{abi.ReasonName(abi.ReasonDefaultDeny)},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-steer-applied"
				key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
				if v := a2achan.Default.Send(context.Background(), "operator", key, a2achan.Shared([]byte("steer marker 2766")), a2achan.CapA2ASend); v.Kind != abi.VerdictAllow {
					t.Fatalf("legal steer send refused: %v", v.Kind)
				}
				p := &recordingPlanner{}
				if _, err := RunArm(context.Background(), p, "task", false, 1, nil, WithSessionTable(nil, trace)); err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				spliced := false
				for _, m := range p.seen {
					if m.Role == RoleUser && strings.Contains(m.Content, "steer marker 2766") {
						spliced = true
					}
				}
				if !spliced {
					t.Fatal("steer enqueued but NOT consumed: the turn input never carried it (enqueue is not applied)")
				}
				if n := a2achan.Default.Len(key); n != 0 {
					t.Fatalf("steer mailbox not drained: %d queued", n)
				}
			},
			refused: func(t *testing.T) []string {
				const trace = "ctl-2766-steer-refused"
				key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
				// No CapA2ASend advertised: the capability floor refuses the send
				// with its closed reason and the op never enters the mailbox.
				v := a2achan.Default.Send(context.Background(), "operator", key, a2achan.Shared([]byte("no cap")))
				if v.Kind != abi.VerdictDeny {
					t.Fatalf("capless steer send: got %v, want VerdictDeny", v.Kind)
				}
				if n := a2achan.Default.Len(key); n != 0 {
					t.Fatalf("refused steer was still enqueued: %d queued", n)
				}
				return []string{abi.ReasonName(v.Reason)}
			},
		},
		{
			op: "redirect",
			tokens: []string{
				string(sessionctl.RedirectMalformed),
				string(sessionctl.RedirectNoRedirectableState),
			},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-redirect-applied"
				const goal = "pursue the 2766 witness goal"
				sessionctl.ClearObjective(trace)
				defer sessionctl.ClearObjective(trace)
				if r := sessionctl.EnqueueRedirect(trace, sessionctl.Redirect{Goal: goal}); r != nil {
					t.Fatalf("EnqueueRedirect: %v", r)
				}
				// Enqueued is NOT applied: before the boundary the live objective
				// is untouched and the op is still queued.
				if _, ok := sessionctl.CurrentObjective(trace); ok {
					t.Fatal("redirect applied at ENQUEUE time — must wait for the loop's turn boundary")
				}
				if n := sessionctl.RedirectPendingLen(trace); n != 1 {
					t.Fatalf("pending redirects = %d, want 1 (enqueued, awaiting the boundary)", n)
				}
				p := &recordingPlanner{}
				if _, err := RunArm(context.Background(), p, "task", false, 1, nil, WithSessionTable(nil, trace)); err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				carried := false
				for _, m := range p.seen {
					if m.Role == RoleSystem && strings.Contains(m.Content, goal) {
						carried = true
					}
				}
				if !carried {
					t.Fatal("redirect enqueued but NOT consumed: no objective directive reached the turn")
				}
				if n := sessionctl.RedirectPendingLen(trace); n != 0 {
					t.Fatalf("redirect mailbox not drained: %d queued", n)
				}
				if obj, ok := sessionctl.CurrentObjective(trace); !ok || obj.Goal != goal {
					t.Fatalf("live objective = %+v (ok=%v), want the redirected goal", obj, ok)
				}
			},
			refused: func(t *testing.T) []string {
				var got []string
				// Illegal shape: a redirect with no goal has nothing to redirect to.
				if r := sessionctl.EnqueueRedirect("ctl-2766-redirect-malformed", sessionctl.Redirect{}); r == nil {
					t.Fatal("empty-goal redirect was accepted, want REDIRECT_MALFORMED")
				} else {
					got = append(got, string(r.Reason))
				}
				// Illegal state: a terminal (met) objective is not redirectable.
				const trace = "ctl-2766-redirect-terminal"
				sessionctl.SetObjective(trace, sessionctl.Objective{Goal: "shipped", Status: sessionctl.ObjectiveMet})
				defer sessionctl.ClearObjective(trace)
				if r := sessionctl.EnqueueRedirect(trace, sessionctl.Redirect{Goal: "new goal"}); r == nil {
					t.Fatal("redirect against a met objective was accepted, want REDIRECT_NO_REDIRECTABLE_STATE")
				} else {
					got = append(got, string(r.Reason))
				}
				return got
			},
		},
		{
			op:     "pause",
			tokens: []string{session.ReasonControlSessionTerminal},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-pause-applied"
				tbl := session.NewTable()
				if _, ok := tbl.Transition(trace, session.Paused, "operator hold"); !ok {
					t.Fatal("pause write refused on a live session")
				}
				p := &recordingPlanner{}
				m, err := RunArm(context.Background(), p, "task", false, 1, nil, WithSessionTable(tbl, trace))
				if err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				// Consumed: the arm observed the hold at its boundary — zero turns
				// ran and the closed reason is recorded on the arm, not inferred.
				if m.StoppedBySession != session.ReasonPaused {
					t.Fatalf("StoppedBySession = %q, want %q (the pause was not consumed by the loop)", m.StoppedBySession, session.ReasonPaused)
				}
				if m.Turns != 0 || len(p.seen) != 0 {
					t.Fatalf("paused arm still ran %d turns (planner saw %d messages) — a hold must gate the boundary", m.Turns, len(p.seen))
				}
			},
			refused: func(t *testing.T) []string {
				return refuseTerminal(t, "pause", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.Transition(trace, session.Paused, "hold")
				})
			},
		},
		{
			op:     "resume",
			tokens: []string{session.ReasonControlSessionTerminal},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-resume-applied"
				tbl := session.NewTable()
				tbl.Decide(trace) // seed a live record
				if _, ok := tbl.Transition(trace, session.Paused, "operator hold"); !ok {
					t.Fatal("pause write refused on a live session")
				}
				firstDecide := make(chan struct{}, 1)
				gate := SessionGate{
					Decide: func(tr string) (int, bool, int, string) {
						v := tbl.Decide(tr)
						select {
						case firstDecide <- struct{}{}:
						default:
						}
						return v.MaxTokens, v.Proceed, v.MinGapMs, v.Reason
					},
					Wait: func(tr string) (bool, string) {
						v := tbl.WaitResume(context.Background(), tr)
						return v.Resumed, v.Reason
					},
				}
				p := &countingFinalPlanner{}
				done := make(chan ArmMetrics, 1)
				go func() {
					m, err := RunArm(context.Background(), p, "task", false, 3, nil, WithSessionGate(gate, trace))
					if err != nil {
						t.Errorf("RunArm: %v", err)
					}
					done <- m
				}()
				<-firstDecide // the loop observed the hold and is parking
				if _, ok := tbl.Transition(trace, session.Running, "operator resume"); !ok {
					t.Fatal("resume write refused on a paused session")
				}
				m := <-done
				// Consumed: the parked arm WOKE on the resume and completed the
				// held turn — no terminal stop, exactly one model round-trip.
				if m.StoppedBySession != "" {
					t.Fatalf("arm stopped (%q) instead of resuming the held turn", m.StoppedBySession)
				}
				if m.FinalAnswer == "" || p.calls != 1 {
					t.Fatalf("resume not consumed as a same-turn wake: final=%q calls=%d, want a final answer from exactly 1 turn", m.FinalAnswer, p.calls)
				}
			},
			refused: func(t *testing.T) []string {
				return refuseTerminal(t, "resume", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.Transition(trace, session.Running, "")
				})
			},
		},
		{
			op:     "cancel",
			tokens: []string{session.ReasonControlSessionTerminal, session.ReasonControlRevStale},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-cancel-applied"
				tbl := session.NewTable()
				if _, ok := tbl.Transition(trace, session.Draining, ""); !ok {
					t.Fatal("cancel (drain) write refused on a live session")
				}
				// Enqueued is NOT applied: the write parks the session at Draining;
				// the stop is TAKEN at the loop's next boundary, never mid-turn.
				if st := tbl.Get(trace); st.Run != session.Draining {
					t.Fatalf("pre-boundary run-state = %v, want Draining (stop must wait for the boundary)", st.Run)
				}
				p := &recordingPlanner{}
				m, err := RunArm(context.Background(), p, "task", false, 1, nil, WithSessionTable(tbl, trace))
				if err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				if m.StoppedBySession != session.ReasonDrained {
					t.Fatalf("StoppedBySession = %q, want %q (the cancel was not consumed at the boundary)", m.StoppedBySession, session.ReasonDrained)
				}
				// Consumed: the boundary finalized Draining -> Stopped.
				if st := tbl.Get(trace); st.Run != session.Stopped {
					t.Fatalf("post-boundary run-state = %v, want Stopped (boundary must finalize the drain)", st.Run)
				}
			},
			refused: func(t *testing.T) []string {
				// Cannot cancel a Stopped session — the case #2766 names.
				got := refuseTerminal(t, "cancel", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.Transition(trace, session.Draining, "")
				})
				// A stale --if-rev cancel refuses with the CAS token, not silence.
				const trace = "ctl-2766-cancel-stale-rev"
				tbl := session.NewTable()
				tbl.Decide(trace) // seed a live record at some Rev
				cur := tbl.Get(trace)
				want := cur
				want.Run = session.Draining
				st, ok := tbl.CompareAndSet(trace, cur.Rev+7, want)
				r := session.ControlRefusalFor("cancel", st, ok)
				if r == nil {
					t.Fatal("stale-rev cancel was APPLIED — a lost CAS race must refuse")
				}
				if r.Reason != session.ReasonControlRevStale {
					t.Fatalf("stale-rev cancel refused with %q, want %q", r.Reason, session.ReasonControlRevStale)
				}
				return append(got, r.Reason)
			},
		},
		{
			op:     "terminate",
			tokens: []string{session.ReasonControlSessionTerminal},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-terminate-applied"
				tbl := session.NewTable()
				tbl.Decide(trace) // seed a live record
				// The flip lands MID-TURN: the planner has already returned a turn
				// with two pending tool calls when the operator terminates. Consumed
				// means the safe point took it — NO pending tool call dispatches
				// (a drain would run them all; that contrast is the dedicated
				// drain-vs-terminate witness in loop_terminate_test.go).
				p := &midTurnFlipPlanner{flip: func() {
					if _, ok := tbl.Transition(trace, session.Terminating, ""); !ok {
						t.Error("terminate write refused on a live session")
					}
				}}
				m, err := RunArm(context.Background(), p, "task", false, 3, nil, WithSessionTable(tbl, trace))
				if err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				if m.StoppedBySession != session.ReasonTerminated {
					t.Fatalf("StoppedBySession = %q, want %q (the terminate was not consumed at the safe point)", m.StoppedBySession, session.ReasonTerminated)
				}
				if m.ToolCalls != 0 {
					t.Fatalf("terminated arm still dispatched %d tool calls — terminate must start no new work", m.ToolCalls)
				}
				// Consumed: the safe point finalized Terminating -> Stopped.
				if st := tbl.Get(trace); st.Run != session.Stopped || st.Reason != session.ReasonTerminated {
					t.Fatalf("post-terminate state = %v/%q, want Stopped/%q", st.Run, st.Reason, session.ReasonTerminated)
				}
			},
			refused: func(t *testing.T) []string {
				return refuseTerminal(t, "terminate", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.Transition(trace, session.Terminating, "")
				})
			},
		},
		{
			op:     "throttle",
			tokens: []string{session.ReasonControlSessionTerminal},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-throttle-applied"
				tbl := session.NewTable()
				if _, ok := tbl.Transition(trace, session.Throttled, "dial down"); !ok {
					t.Fatal("throttle write refused on a live session")
				}
				if _, ok := tbl.SetPace(trace, session.Pace{MaxTokensPerTurn: 7}); !ok {
					t.Fatal("pace write refused on a live session")
				}
				p := &paceCapPlanner{}
				m, err := RunArm(context.Background(), p, "task", false, 1, nil, WithSessionTable(tbl, trace))
				if err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				if m.StoppedBySession != "" {
					t.Fatalf("throttled arm stopped (%q) — throttle must slow, not stop", m.StoppedBySession)
				}
				// Consumed: the cap reached THIS turn's sampling, not just the table.
				if len(p.caps) != 1 || p.caps[0] != 7 {
					t.Fatalf("per-turn sampling caps = %v, want [7] (the pace write never reached the turn)", p.caps)
				}
			},
			refused: func(t *testing.T) []string {
				return refuseTerminal(t, "throttle", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.SetPace(trace, session.Pace{MaxTokensPerTurn: 7})
				})
			},
		},
		{
			op:     "budget",
			tokens: []string{session.ReasonControlSessionTerminal},
			applied: func(t *testing.T) {
				const trace = "ctl-2766-budget-applied"
				tbl := session.NewTable()
				if _, ok := tbl.SetBudget(trace, session.Budget{TurnsLeft: 0, TokensLeft: session.Unbounded, ContextTokensLeft: session.Unbounded}); !ok {
					t.Fatal("budget write refused on a live session")
				}
				p := &recordingPlanner{}
				m, err := RunArm(context.Background(), p, "task", false, 3, nil, WithSessionTable(tbl, trace))
				if err != nil {
					t.Fatalf("RunArm: %v", err)
				}
				// Consumed: the exhausted allotment gated the boundary with its
				// closed reason and the record finalized — no turn ran.
				if m.StoppedBySession != session.ReasonBudgetTurns {
					t.Fatalf("StoppedBySession = %q, want %q (the budget write was not consumed by the gate)", m.StoppedBySession, session.ReasonBudgetTurns)
				}
				if m.Turns != 0 || len(p.seen) != 0 {
					t.Fatalf("budget-exhausted arm still ran %d turns", m.Turns)
				}
				if st := tbl.Get(trace); st.Run != session.Stopped {
					t.Fatalf("post-exhaustion run-state = %v, want Stopped", st.Run)
				}
			},
			refused: func(t *testing.T) []string {
				return refuseTerminal(t, "budget", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.SetBudget(trace, session.Budget{TurnsLeft: 5})
				})
			},
		},
		{
			op:     "priority",
			tokens: []string{session.ReasonControlSessionTerminal},
			applied: func(t *testing.T) {
				// Priority's consumer is the SCHEDULER, not the turn loop: the
				// applied witness is the consumer read — Snapshot's yield order —
				// reflecting the write, per Table.SetPriority's contract.
				tbl := session.NewTable()
				const hi, lo = "ctl-2766-priority-hi", "ctl-2766-priority-lo"
				if _, ok := tbl.SetPriority(hi, 3); !ok {
					t.Fatal("priority write refused on a live session")
				}
				if _, ok := tbl.SetPriority(lo, 5); !ok {
					t.Fatal("priority write refused on a live session")
				}
				if snap := tbl.Snapshot(); snap[0].TraceID != hi {
					t.Fatalf("scheduler order head = %s, want %s (priority 3 yields before 5)", snap[0].TraceID, hi)
				}
				// The op is consumed live: a re-prioritize re-ranks the next read.
				if _, ok := tbl.SetPriority(hi, 9); !ok {
					t.Fatal("re-prioritize refused on a live session")
				}
				if snap := tbl.Snapshot(); snap[0].TraceID != lo {
					t.Fatalf("scheduler order head after re-prioritize = %s, want %s (the write was not consumed by the scheduler read)", snap[0].TraceID, lo)
				}
			},
			refused: func(t *testing.T) []string {
				return refuseTerminal(t, "priority", func(tbl *session.Table, trace string) (session.State, bool) {
					return tbl.SetPriority(trace, 1)
				})
			},
		},
	}
}

// TestControlOpsAppliedAndRefused is the #2766 table-driven witness: every live
// control op proves its applied (consumed, not merely enqueued) path AND at
// least one illegal-for-state refusal carrying the right closed token.
func TestControlOpsAppliedAndRefused(t *testing.T) {
	for _, w := range controlOpWitnesses() {
		w := w
		t.Run(w.op+"/applied", w.applied)
		t.Run(w.op+"/refused", func(t *testing.T) {
			got := w.refused(t)
			if len(got) == 0 {
				t.Fatalf("op %s observed no refusal token on its illegal path", w.op)
			}
			for _, token := range got {
				if !slices.Contains(w.tokens, token) {
					t.Fatalf("op %s refused with %q — not in its declared closed token set %v", w.op, token, w.tokens)
				}
			}
		})
	}
}

// TestControlRefusalVocabularyComplete is the #2766 completeness test over the
// closed refusal set: every op declares at least one refusal token and both
// contract paths; every token is a well-formed closed token; the drive-state
// control tokens are exactly session.ControlRefusalTokens and are DISJOINT from
// the stop-reason vocabulary and the per-tool abi refusal vocabulary, so a
// control-write refusal can never be misread as "why the session stopped" or
// "why a tool call was refused" (the #2633 category discipline).
func TestControlRefusalVocabularyComplete(t *testing.T) {
	rows := controlOpWitnesses()
	tokenShape := regexp.MustCompile(`^[A-Z][A-Z0-9_]+$`)
	seenOps := map[string]bool{}
	declared := map[string]bool{}
	for _, w := range rows {
		if w.op == "" || seenOps[w.op] {
			t.Fatalf("op name %q empty or duplicated — the op set must be closed and unambiguous", w.op)
		}
		seenOps[w.op] = true
		if w.applied == nil || w.refused == nil {
			t.Fatalf("op %s is missing a contract path (applied=%v refused=%v) — every op needs both", w.op, w.applied != nil, w.refused != nil)
		}
		if len(w.tokens) == 0 {
			t.Fatalf("op %s declares no refusal token — every op needs a closed refusal for its illegal states", w.op)
		}
		for _, token := range w.tokens {
			if !tokenShape.MatchString(token) {
				t.Fatalf("op %s token %q is not a closed SCREAMING_SNAKE token", w.op, token)
			}
			declared[token] = true
		}
	}
	// The epic's live op set at HEAD — a new control op must add its row here
	// (and its tokens to the closed vocabulary) or this test names the gap.
	for _, op := range []string{"steer", "redirect", "pause", "resume", "cancel", "terminate", "throttle", "budget", "priority"} {
		if !seenOps[op] {
			t.Fatalf("live control op %q has no witness row — applied/refused contract missing", op)
		}
	}
	// The drive-state refusal vocabulary is closed: everything the mapper can
	// emit is declared by some op row, and vice versa nothing dangles.
	for _, token := range session.ControlRefusalTokens() {
		if !declared[token] {
			t.Fatalf("closed control token %q is emitted by session.ControlRefusalFor but no op row declares it", token)
		}
	}
	// Category disjointness: a control-write refusal token must never collide
	// with a session stop reason or a per-tool abi refusal name.
	stopReasons := []string{
		session.ReasonBudgetTurns, session.ReasonBudgetTokens, session.ReasonBudgetContext,
		session.ReasonBudgetQueries, session.ReasonBudgetSpend, session.ReasonPaused,
		session.ReasonDrained, session.ReasonTerminated, session.ReasonStopped,
		session.ReasonBudgetReset, session.ReasonProviderSessionClear,
		session.ReasonResumeCancelled, session.ReasonTimeBudgetExhausted,
	}
	for _, token := range session.ControlRefusalTokens() {
		if slices.Contains(stopReasons, token) {
			t.Fatalf("control token %q collides with the stop-reason vocabulary", token)
		}
		if slices.Contains(abi.ReasonNames(), token) {
			t.Fatalf("control token %q collides with the per-tool abi refusal vocabulary", token)
		}
	}
}

// TestControlWitnessTableBindsVocabSpine binds this #2766 witness table (the loop-side
// proof of what each op DOES) to the #2754 vocabulary spine (internal/sessionctl, the
// declarative op contract). Without this, the two authoritative op tables live in
// separate packages and can silently drift — a 9th op could ship in one and be
// forgotten in the other while every per-package test stays green. Here the spine's
// op set, and each op's declared refusal tokens, are checked against what this table
// actually proves, so a newly-shipped op cannot land in one without the other going red.
func TestControlWitnessTableBindsVocabSpine(t *testing.T) {
	witnessed := map[string]controlOpWitness{}
	for _, w := range controlOpWitnesses() {
		witnessed[w.op] = w
	}
	spine := map[string]bool{}
	for _, op := range sessionctl.Ops() {
		spine[string(op)] = true
	}
	// 1. Op-set equality (as sets, in both directions).
	if len(witnessed) != len(spine) {
		t.Fatalf("op-set size drift: #2766 witnesses %d ops, sessionctl spine registers %d", len(witnessed), len(spine))
	}
	for op := range witnessed {
		if !spine[op] {
			t.Fatalf("op %q is proven by the #2766 table but not registered in the sessionctl spine", op)
		}
	}
	for op := range spine {
		if _, ok := witnessed[op]; !ok {
			t.Fatalf("op %q is registered in the sessionctl spine but has no #2766 witness row", op)
		}
	}
	// 2. Per-op token binding: every refusal token this table PROVES an op can emit
	//    must be declared by the spine's spec for that op (the spine may declare more —
	//    e.g. a race token an "always terminal" row does not exercise — but never fewer).
	for op, w := range witnessed {
		spec, ok := sessionctl.Spec(sessionctl.ControlOp(op))
		if !ok {
			t.Fatalf("no spine spec for proven op %q", op)
		}
		for _, tok := range w.tokens {
			if !slices.Contains(spec.RefusalReasons, tok) {
				t.Fatalf("op %q proves refusal token %q but the spine spec omits it (declares %v)", op, tok, spec.RefusalReasons)
			}
		}
	}
	// 3. steer's spine witness IS the splice kind — and this table's steer/applied path
	//    (RunArm observing "steer marker 2766" spliced into the turn input) is the proof
	//    that names. The #2754 acceptance ties steer's witness to that existing evidence.
	if s, _ := sessionctl.Spec(sessionctl.OpSteer); s.Witness != sessionctl.WitnessSplice {
		t.Fatalf("steer spine witness = %q, want the splice kind proven by steer/applied here", s.Witness)
	}
}
