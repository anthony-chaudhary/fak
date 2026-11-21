package agent

// loop_turn_intent_test.go — the producer witness for fak#13383: armRunner.dispatchToolCalls
// publishes the tool-lifecycle facts it actually knows into the session table's advisory
// TurnIntent while a call is outstanding (IsSpeculative for an unconfirmed streaming
// speculation, WillDiscard once cancellation decides the results go unconsumed) and
// withdraws exactly those facts on result, failure and cancellation. Observations are
// taken on the dispatch goroutine from the tool_started progress event, so every read
// happens while the call is outstanding and before any cleanup can run.

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// intentToolPlanner scripts one tool turn and a final answer. streamedArgs is the inline
// call the StreamingToolFSM speculates on; args is the authoritative call the turn admits,
// so a mismatch exercises the squash path.
func intentToolPlanner(tool, streamedArgs, args string) *scriptedStreamingPlanner {
	return &scriptedStreamingPlanner{turns: []*Completion{
		{
			Message: Message{
				Content:   `<tool_call>{"name": "` + tool + `", "arguments": ` + streamedArgs + `}</tool_call>`,
				ToolCalls: []ToolCall{{ID: "call_1", Type: "function", Function: Func{Name: tool, Arguments: args}}},
			},
			FinishReason: "tool_calls",
		},
		{Message: Message{Content: "done"}, FinishReason: "stop"},
	}}
}

// runSpeculativeIntentArm runs p as a streaming-speculation arm keyed on trace (no table
// when tbl is nil) and calls onStart on the dispatch goroutine as each tool call starts.
func runSpeculativeIntentArm(t *testing.T, p Planner, tbl *session.Table, trace string, onStart func()) (ArmMetrics, []traceEvent) {
	t.Helper()
	var log []traceEvent
	var fsms []*StreamingToolFSM
	opts := []RunOption{
		WithStreamingSpeculation(true),
		WithStreamingFSMHook(func(f *StreamingToolFSM) { fsms = append(fsms, f) }),
		WithProgressObserver(func(ev ProgressEvent) {
			if ev.Kind == ProgressToolStarted && onStart != nil {
				onStart()
			}
		}),
	}
	if tbl != nil {
		opts = append(opts, WithSessionTable(tbl, trace))
	}
	m, err := RunArmStream(context.Background(), p, "look up the policy", false, 3, nil, &log, opts...)
	if err != nil {
		t.Fatalf("RunArmStream: %v", err)
	}
	if len(fsms) == 0 || !fsms[0].IsDispatched() {
		t.Fatal("fixture: the streaming FSM never speculatively dispatched the turn's call")
	}
	return m, log
}

// waitIntent polls the record until want holds; the dispatch goroutine is parked in the
// tool_started observer meanwhile, so the cleanup cannot race the read.
func waitIntent(tbl *session.Table, trace string, want func(session.TurnIntent) bool) session.TurnIntent {
	deadline := time.Now().Add(5 * time.Second)
	for {
		in := tbl.Get(trace).Intent
		if want(in) || time.Now().After(deadline) {
			return in
		}
		time.Sleep(time.Millisecond)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestLoopPublishesTurnIntent(t *testing.T) {
	speculative := session.TurnIntent{IsSpeculative: true}

	t.Run("speculative_while_outstanding_cleared_on_result", func(t *testing.T) {
		const trace = "intent-13383-result"
		tbl := session.NewTable()
		var during session.TurnIntent
		m, _ := runSpeculativeIntentArm(t, intentToolPlanner(toolFetchDoc, `{"topic": "baggage"}`, `{"topic": "baggage"}`), tbl, trace, func() {
			during = tbl.Get(trace).Intent
		})
		if m.SpecCommitted < 1 {
			t.Fatalf("fixture: SpecCommitted = %d, want the call served from the speculation", m.SpecCommitted)
		}
		if during != speculative {
			t.Fatalf("intent while outstanding = %+v, want exactly %+v (no guessed timing, prefix or settle)", during, speculative)
		}
		if got := tbl.Get(trace).Intent; !got.IsZero() {
			t.Fatalf("intent after result = %+v, want zero", got)
		}
	})

	t.Run("speculative_squashed_failure_cleared", func(t *testing.T) {
		const trace = "intent-13383-failure"
		tbl := session.NewTable()
		var during session.TurnIntent
		m, _ := runSpeculativeIntentArm(t, intentToolPlanner(toolCalculate, `{"a": 1, "b": 2}`, `{"a": 1}`), tbl, trace, func() {
			during = tbl.Get(trace).Intent
		})
		if m.SpecSquashed < 1 || m.ToolErrors != 1 {
			t.Fatalf("fixture: SpecSquashed=%d ToolErrors=%d, want a squashed speculation and one failed call", m.SpecSquashed, m.ToolErrors)
		}
		if during != speculative {
			t.Fatalf("intent while outstanding = %+v, want %+v", during, speculative)
		}
		if got := tbl.Get(trace).Intent; !got.IsZero() {
			t.Fatalf("intent after failure = %+v, want zero", got)
		}
	})

	t.Run("cancellation_publishes_discard_then_clears", func(t *testing.T) {
		const trace = "intent-13383-cancel"
		tbl := session.NewTable()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		discard := session.TurnIntent{WillDiscard: true}
		var during session.TurnIntent
		p := intentToolPlanner(toolGetUser, `{"user_id": "u1"}`, `{"user_id": "u1"}`)
		_, err := RunArm(ctx, p, "look up the user", false, 3, nil,
			WithSessionTable(tbl, trace),
			WithProgressObserver(func(ev ProgressEvent) {
				if ev.Kind != ProgressToolStarted {
					return
				}
				if in := tbl.Get(trace).Intent; !in.IsZero() {
					t.Errorf("intent before cancellation = %+v, want zero (no known fact)", in)
				}
				cancel()
				during = waitIntent(tbl, trace, func(in session.TurnIntent) bool { return in.WillDiscard })
			}))
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("RunArm: %v", err)
		}
		if during != discard {
			t.Fatalf("intent after cancellation while outstanding = %+v, want exactly %+v", during, discard)
		}
		if got := tbl.Get(trace).Intent; !got.IsZero() {
			t.Fatalf("intent after the cancelled dispatch = %+v, want zero", got)
		}
	})

	t.Run("concurrent_unrelated_state_survives", func(t *testing.T) {
		const trace = "intent-13383-concurrent"
		tbl := session.NewTable()
		runSpeculativeIntentArm(t, intentToolPlanner(toolFetchDoc, `{"topic": "baggage"}`, `{"topic": "baggage"}`), tbl, trace, func() {
			peer := tbl.Get(trace).Intent
			peer.ArrivingInMillis = 50
			peer.Prefix = "sha256:peer"
			if _, ok := tbl.SetTurnIntent(trace, peer); !ok {
				t.Error("peer SetTurnIntent refused on a live session")
			}
			if _, ok := tbl.SetPriority(trace, 7); !ok {
				t.Error("peer SetPriority refused on a live session")
			}
		})
		st := tbl.Get(trace)
		want := session.TurnIntent{ArrivingInMillis: 50, Prefix: "sha256:peer"}
		if st.Intent != want || st.Priority != 7 {
			t.Fatalf("after cleanup intent=%+v priority=%d, want intent=%+v priority=7", st.Intent, st.Priority, want)
		}
	})

	t.Run("flags_set_by_another_writer_are_not_owned", func(t *testing.T) {
		const trace = "intent-13383-unowned"
		tbl := session.NewTable()
		tbl.Decide(trace)
		seeded := session.TurnIntent{IsSpeculative: true, ArrivingInMillis: 75, Prefix: "sha256:seed"}
		if _, ok := tbl.SetTurnIntent(trace, seeded); !ok {
			t.Fatal("seed SetTurnIntent refused on a live session")
		}
		runSpeculativeIntentArm(t, intentToolPlanner(toolFetchDoc, `{"topic": "baggage"}`, `{"topic": "baggage"}`), tbl, trace, nil)
		if got := tbl.Get(trace).Intent; got != seeded {
			t.Fatalf("intent after cleanup = %+v, want the other writer's %+v untouched", got, seeded)
		}
	})

	t.Run("terminal_session_not_revived", func(t *testing.T) {
		const trace = "intent-13383-terminal"
		tbl := session.NewTable()
		var stopped session.State
		m, _ := runSpeculativeIntentArm(t, intentToolPlanner(toolFetchDoc, `{"topic": "baggage"}`, `{"topic": "baggage"}`), tbl, trace, func() {
			st, ok := tbl.Transition(trace, session.Stopped, "")
			if !ok {
				t.Error("Stopped transition refused on a live session")
			}
			stopped = st
		})
		got := tbl.Get(trace)
		if got.Run != session.Stopped || got.Rev != stopped.Rev || got.Intent != stopped.Intent {
			t.Fatalf("after cleanup run=%v rev=%d intent=%+v, want the Stopped record untouched (rev=%d intent=%+v)", got.Run, got.Rev, got.Intent, stopped.Rev, stopped.Intent)
		}
		if m.StoppedBySession == "" {
			t.Fatal("the loop did not stop on the terminal session")
		}
	})
}

// TestLoopPublishesTurnIntentNeverChangesOutcome: an absent table, a wired table and a
// table carrying hostile pre-set hints yield the same verdicts, tool counts and answer,
// on the speculative path and on the kernel arm's policy-denied call.
// fak-test:runtime fast est=150ms lane=default
func TestLoopPublishesTurnIntentNeverChangesOutcome(t *testing.T) {
	hostile := session.TurnIntent{EndsSoon: true, IsSpeculative: true, WillDiscard: true, ArrivingInMillis: 1, Prefix: "sha256:hostile", ResultAlreadyKnown: true}
	hostileTable := func(trace string) *session.Table {
		tbl := session.NewTable()
		tbl.Decide(trace)
		if _, ok := tbl.SetTurnIntent(trace, hostile); !ok {
			t.Fatal("hostile SetTurnIntent refused on a live session")
		}
		return tbl
	}
	sameOutcome := func(t *testing.T, label string, base, got ArmMetrics, baseLog, gotLog []traceEvent) {
		t.Helper()
		if got.ToolCalls != base.ToolCalls || got.Turns != base.Turns || got.FinalAnswer != base.FinalAnswer ||
			got.SpecCommitted != base.SpecCommitted || got.DestructiveExecuted != base.DestructiveExecuted {
			t.Fatalf("%s metrics = calls %d turns %d answer %q spec %d destructive %v, want %d %d %q %d %v", label,
				got.ToolCalls, got.Turns, got.FinalAnswer, got.SpecCommitted, got.DestructiveExecuted,
				base.ToolCalls, base.Turns, base.FinalAnswer, base.SpecCommitted, base.DestructiveExecuted)
		}
		if len(gotLog) != len(baseLog) {
			t.Fatalf("%s trace has %d events, want %d", label, len(gotLog), len(baseLog))
		}
		for i := range baseLog {
			if gotLog[i].Tool != baseLog[i].Tool || gotLog[i].Verdict != baseLog[i].Verdict || gotLog[i].By != baseLog[i].By || gotLog[i].Reason != baseLog[i].Reason {
				t.Fatalf("%s event %d = %s/%s/%s/%s, want %s/%s/%s/%s", label, i,
					gotLog[i].Tool, gotLog[i].Verdict, gotLog[i].By, gotLog[i].Reason,
					baseLog[i].Tool, baseLog[i].Verdict, baseLog[i].By, baseLog[i].Reason)
			}
		}
	}

	t.Run("speculative_path", func(t *testing.T) {
		planner := func() Planner {
			return intentToolPlanner(toolFetchDoc, `{"topic": "baggage"}`, `{"topic": "baggage"}`)
		}
		base, baseLog := runSpeculativeIntentArm(t, planner(), nil, "", nil)
		wired, wiredLog := runSpeculativeIntentArm(t, planner(), session.NewTable(), "intent-13383-parity", nil)
		sameOutcome(t, "wired table", base, wired, baseLog, wiredLog)
		hostileM, hostileLog := runSpeculativeIntentArm(t, planner(), hostileTable("intent-13383-hostile"), "intent-13383-hostile", nil)
		sameOutcome(t, "hostile hints", base, hostileM, baseLog, hostileLog)
	})

	t.Run("kernel_authorization", func(t *testing.T) {
		const trace = "intent-13383-authz"
		run := func(opts ...RunOption) (ArmMetrics, []traceEvent) {
			var log []traceEvent
			p := intentToolPlanner(toolDelete, `{"user_id": "u1"}`, `{"user_id": "u1"}`)
			m, err := RunArm(context.Background(), p, "delete the account", true, 3, &log, opts...)
			if err != nil {
				t.Fatalf("RunArm: %v", err)
			}
			return m, log
		}
		base, baseLog := run()
		if len(baseLog) == 0 || baseLog[0].Verdict != "DENY" || base.DestructiveExecuted {
			t.Fatalf("fixture: the kernel arm must deny %s, got %+v", toolDelete, baseLog)
		}
		got, gotLog := run(WithSessionTable(hostileTable(trace), trace))
		sameOutcome(t, "hostile hints", base, got, baseLog, gotLog)
	})
}
