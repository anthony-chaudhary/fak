package agent

// session_control_native_test.go — the #1321 acceptance witness: session control wired
// END-TO-END on the owned loop (RunArm), across all three previously-dead seams. Each
// sub-test drives RunArm through the function-shaped SessionGate / context planner the
// native serve loop (internal/gateway/native_serve.go) wires in production:
//
//	(a) a Paused→Resumed session continues the SAME turn (turn-granular resume, not a
//	    fresh turn) — the loop PARKS on the hold and re-Decides, never terminating;
//	(b) an operator steer is spliced into the turn from the gate-driven loop (drainSteer
//	    reached via WithSessionGate's trace, the live native-loop path); and
//	(c) lowering Pace shrinks the SessionPlanner window via WithContextPlanner — the first
//	    production caller of SessionPlanner.ApplyPace on the owned loop.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// countingFinalPlanner ends each turn with a final answer and counts how many model
// turns ran, so a test can prove a pause/resume continued the SAME turn (exactly one
// model round-trip) rather than restarting the run.
type countingFinalPlanner struct{ calls int }

func (p *countingFinalPlanner) Model() string { return "counting" }
func (p *countingFinalPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.calls++
	return &Completion{Message: Message{Role: RoleAssistant, Content: "done"}, FinishReason: "stop", Usage: Usage{CompletionTokens: 1}}, nil
}

// TestNativeLoopPauseResumesSameTurn proves (a): a Paused session does NOT terminate the
// owned loop — it parks on the hold (WaitResume via SessionGate.Wait) and continues the
// SAME turn the instant an operator resumes it. The witness: the run ends with a final
// answer (not StoppedBySession), and exactly ONE model turn ran (the turn that was held,
// resumed — never a fresh restart).
func TestNativeLoopPauseResumesSameTurn(t *testing.T) {
	tbl := session.NewTable()
	const trace = "native-pause-resume"
	tbl.Decide(trace)                                  // seed a live (Running) record
	tbl.Transition(trace, session.Paused, "operator hold") // operator pauses it

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

	<-firstDecide // the loop Decided (paused) and is parking on Wait
	tbl.Transition(trace, session.Running, "operator resume")
	m := <-done

	if m.StoppedBySession != "" {
		t.Fatalf("loop TERMINATED on pause (%q) instead of resuming the turn — pause must be turn-granular, not a stop", m.StoppedBySession)
	}
	if m.FinalAnswer == "" {
		t.Fatal("the held turn never completed after resume")
	}
	if p.calls != 1 {
		t.Fatalf("model turns = %d, want exactly 1 (the SAME turn resumed, not a fresh run)", p.calls)
	}
}

// TestNativeLoopSplicesOperatorSteer proves (b): an operator steer enqueued on the
// Session-locale bus is spliced into the turn from the GATE-DRIVEN loop (the native
// serve path uses WithSessionGate, which sets the trace drainSteer keys on). This is
// drainSteer reached from the live native loop, not just the table path.
func TestNativeLoopSplicesOperatorSteer(t *testing.T) {
	const trace = "native-steer-trace"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	if v := a2achan.Default.Send(context.Background(), "operator", key, a2achan.Shared([]byte("switch to plan B")), a2achan.CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("producer send refused by the a2a floor: %v", v.Kind)
	}

	p := &recordingPlanner{}
	gate := SessionGate{Decide: func(string) (int, bool, int, string) { return 0, true, 0, "" }}
	if _, err := RunArm(context.Background(), p, "original task", false, 1, nil, WithSessionGate(gate, trace)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	var spliced bool
	for _, m := range p.seen {
		if m.Role == RoleUser && strings.Contains(m.Content, "switch to plan B") {
			spliced = true
		}
	}
	if !spliced {
		t.Fatalf("operator steer was NOT spliced into the turn from the gate-driven loop; planner saw %d messages", len(p.seen))
	}
	if n := a2achan.Default.Len(key); n != 0 {
		t.Fatalf("steer mailbox not drained: %d still queued", n)
	}
}

// TestNativeLoopApplyPaceShrinksWindow proves (c): lowering Pace shrinks the
// SessionPlanner's resident-context window through WithContextPlanner — the first
// production-shaped caller of SessionPlanner.ApplyPace on the owned loop. A turn paced far
// below its baseline output drives the planner Budget down from its constructed baseline.
func TestNativeLoopApplyPaceShrinksWindow(t *testing.T) {
	const (
		baseBudget     = 1000
		baselineOutput = 1000
		pacedTo        = 50 // paced to 1/20th of baseline → the window must shrink
	)
	sp := NewSessionPlanner(baseBudget)
	if sp.Budget != baseBudget {
		t.Fatalf("planner constructed with Budget %d, want %d", sp.Budget, baseBudget)
	}
	// The gate paces this session's per-turn output to `pacedTo`, far below baseline.
	gate := SessionGate{Decide: func(string) (int, bool, int, string) { return pacedTo, true, 0, "" }}
	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "task", false, 1, nil,
		WithSessionGate(gate, "native-pace-trace"), WithContextPlanner(sp, baselineOutput)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if sp.Budget >= baseBudget {
		t.Fatalf("ApplyPace did not shrink the planner window: Budget=%d, want < %d (pace %d << baseline %d)", sp.Budget, baseBudget, pacedTo, baselineOutput)
	}
}
