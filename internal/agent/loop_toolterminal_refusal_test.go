package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

type retainedWakePlanner struct{ seen chan []Message }

func (p *retainedWakePlanner) Model() string { return "retained-wake" }
func (p *retainedWakePlanner) Complete(_ context.Context, msgs []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.seen <- append([]Message(nil), msgs...)
	return &Completion{Message: Message{Role: RoleAssistant, Content: "done"}, FinishReason: "stop"}, nil
}

func TestToolTerminalWakeSurvivesRefusedTurnAdmission(t *testing.T) {
	const trace = "terminal-refused-before-consume"
	sessionctl.ReadToolTerminalWakeNextRecords(trace)
	t.Cleanup(func() { sessionctl.ReadToolTerminalWakeNextRecords(trace) })

	wake := NewToolTerminalWakeQueue(trace)
	wake.Enqueue(toolproc.Proc{CallID: "call-refused", Session: trace, State: toolproc.StateDone})
	first := &terminalWakePlanner{first: make(chan struct{}), seen: make(chan []Message, 1)}
	decisions := 0
	_, err := RunArm(context.Background(), first, "task", false, 2, nil,
		WithSessionGate(SessionGate{Decide: func(string) (int, bool, int, string) {
			decisions++
			if decisions == 1 {
				return 0, true, 0, ""
			}
			return 0, false, 0, "stopped"
		}}, trace), WithToolTerminalWake(wake),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := wake.Journal(); len(got) != 1 || got[0].Status != "ENQUEUED" {
		t.Fatalf("refused turn consumed terminal wake: %+v", got)
	}
	if rows := sessionctl.ReadToolTerminalWakeNextRecords(trace); len(rows) != 0 {
		t.Fatalf("refused turn emitted Next: %+v", rows)
	}

	second := &terminalWakePlanner{first: make(chan struct{}), seen: make(chan []Message, 1)}
	_, err = RunArm(context.Background(), second, "task", false, 2, nil,
		WithSessionGate(SessionGate{Decide: func(string) (int, bool, int, string) { return 0, true, 0, "" }}, trace),
		WithToolTerminalWake(wake),
	)
	if err != nil {
		t.Fatal(err)
	}
	messages := <-second.seen
	if !strings.Contains(messages[len(messages)-1].Content, "call-refused") {
		t.Fatalf("admitted turn missing retained wake: %+v", messages)
	}
	if rows := sessionctl.ReadToolTerminalWakeNextRecords(trace); len(rows) != 1 || !rows[0].Applied {
		t.Fatalf("admitted retained wake Next=%+v", rows)
	}
	if got := wake.Journal(); len(got) != 2 || got[0].Status != "ENQUEUED" || got[1].Status != "DISPATCHED" {
		t.Fatalf("retained wake journal=%+v", got)
	}
}
