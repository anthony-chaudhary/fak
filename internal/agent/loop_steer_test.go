package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

// recordingPlanner captures the message list it is handed each turn and then ends
// the arm by emitting a final (no-tool-call) answer, so RunArm runs exactly one
// turn. It lets us assert WHICH messages reached the planner — the only way to
// prove a steer was spliced INTO the turn input, not merely enqueued on the bus.
type recordingPlanner struct {
	seen []Message
}

func (p *recordingPlanner) Model() string { return "recording" }
func (p *recordingPlanner) Complete(_ context.Context, msgs []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.seen = append(p.seen, msgs...)
	return &Completion{
		Message:      Message{Role: RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        Usage{PromptTokens: 5, CompletionTokens: 1, TotalTokens: 6},
	}, nil
}

// TestSteerSplicedAtTurnBoundary proves the #850 consumer half: a steer SENT onto
// the a2achan Session-locale bus (the exact producer shape steerSession uses) is
// CONSUMED by a running arm at its turn boundary — it appears in the planner's
// message list, not just in the bus queue. This is the live-consumption proof #760
// deferred; the enqueue half is already proven by the producer-side route test.
func TestSteerSplicedAtTurnBoundary(t *testing.T) {
	const trace = "steer-splice-trace"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}

	// Producer shape: operator principal, Shared (ScopeFleet, Tainted) body, CapA2ASend
	// — identical to cmd/fak/main.go steerSession.
	if v := a2achan.Default.Send(context.Background(), "operator", key, a2achan.Shared([]byte("switch to plan B")), a2achan.CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("producer send refused by a2a floor: %v", v.Kind)
	}

	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "original task", false, 1, nil, WithSessionTable(nil, trace)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	var spliced bool
	for _, m := range p.seen {
		if m.Role == RoleUser && strings.Contains(m.Content, "switch to plan B") {
			spliced = true
		}
	}
	if !spliced {
		t.Fatalf("steer was NOT spliced into the turn input; planner saw %d messages, none carried the steer (enqueue-only, consumer half missing)", len(p.seen))
	}
	// The mailbox must be drained, not left queued (a re-Recv would double-splice).
	if n := a2achan.Default.Len(key); n != 0 {
		t.Fatalf("steer mailbox not drained: %d messages still queued", n)
	}
}

// TestNoSteerNoSplice proves the no-op path: with nothing enqueued, the planner sees
// only the system + task messages — the historical loop is byte-for-byte unchanged.
func TestNoSteerNoSplice(t *testing.T) {
	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "lone task", false, 1, nil, WithSessionTable(nil, "no-steer-trace")); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	for _, m := range p.seen {
		if m.Role == RoleUser && m.Content != "lone task" {
			t.Fatalf("unexpected user message spliced with empty mailbox: %q", m.Content)
		}
	}
}

// TestDrainSteerNoTraceIsNoop proves a run with no trace wired drains nothing even
// when a steer happens to be queued under some other id — the channel is keyed by
// the run's own trace, so an untraced run cannot pick up another run's steer.
func TestDrainSteerNoTraceIsNoop(t *testing.T) {
	other := a2achan.ChannelKey{Locale: a2achan.Session, ID: "someone-elses-trace"}
	_ = a2achan.Default.Send(context.Background(), "operator", other, a2achan.Shared([]byte("not for you")), a2achan.CapA2ASend)
	defer func() {
		// drain the foreign mailbox so we don't leak state into other tests
		for {
			if _, _, ok := a2achan.Default.TryRecv(context.Background(), other, a2achan.CapA2ARecv); !ok {
				break
			}
		}
	}()

	c := runConfig{trace: ""}
	if got := c.drainSteer(); got != "" {
		t.Fatalf("untraced run drained a steer it should not see: %q", got)
	}
}
