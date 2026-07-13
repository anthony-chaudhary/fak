package agent

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

type terminalWakePlanner struct {
	first chan struct{}
	seen  chan []Message
	calls atomic.Int32
}

func (p *terminalWakePlanner) Model() string { return "terminal-wake" }
func (p *terminalWakePlanner) Complete(_ context.Context, msgs []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	if p.calls.Add(1) == 1 {
		close(p.first)
		return &Completion{
			Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{
				ID: "foreground", Function: Func{Name: "search_kb", Arguments: `{}`},
			}}}, FinishReason: "tool_calls",
		}, nil
	}
	p.seen <- append([]Message(nil), msgs...)
	return &Completion{Message: Message{Role: RoleAssistant, Content: "done"}, FinishReason: "stop"}, nil
}

// TestLoopReentersOnToolTerminal witnesses the #2400 contract end to end: one
// supervisor Tick publishes an owned terminal verdict, the idle loop starts its
// next turn with that verdict, and PAUSED defers (rather than loses) the wake.
func TestLoopReentersOnToolTerminal(t *testing.T) {
	for _, paused := range []bool{false, true} {
		name := "running"
		if paused {
			name = "paused-defers"
		}
		t.Run(name, func(t *testing.T) {
			const trace, call = "owned-session", "background-trace"
			wake := NewToolTerminalWakeQueue(trace)
			sup := toolprocgate.NewSupervisor(toolproc.Config{})
			sup.SetTerminalSink(wake.Enqueue)
			if err := sup.Spawn(call, "shell", trace, 0, 0, 1, nil); err != nil {
				t.Fatal(err)
			}

			planner := &terminalWakePlanner{first: make(chan struct{}), seen: make(chan []Message, 1)}
			resume := make(chan struct{})
			var resumed atomic.Bool
			gate := SessionGate{Decide: func(string) (int, bool, int, string) {
				if paused && planner.calls.Load() >= 1 && !resumed.Load() {
					return 0, false, 0, session.ReasonPaused
				}
				return 0, true, 0, ""
			}, Wait: func(string) (bool, string) { <-resume; resumed.Store(true); return true, "" }}

			done := make(chan error, 1)
			go func() {
				_, err := RunArm(context.Background(), planner, "task", false, 2, nil,
					WithSessionGate(gate, trace), WithToolTerminalWake(wake))
				done <- err
			}()
			<-planner.first
			if err := sup.Exit(call, 2, "ok"); err != nil {
				t.Fatal(err)
			}
			if _, err := sup.Tick(2); err != nil {
				t.Fatal(err)
			}

			if paused {
				deadline := time.Now().Add(time.Second)
				for {
					journal := wake.Journal()
					if len(journal) >= 2 && journal[len(journal)-1].Status == "DEFERRED" {
						break
					}
					if time.Now().After(deadline) {
						t.Fatalf("paused wake was not journaled DEFERRED: %+v", journal)
					}
					time.Sleep(time.Millisecond)
				}
				select {
				case <-planner.seen:
					t.Fatal("paused wake dispatched before resume")
				default:
				}
				close(resume)
			}

			var seen []Message
			select {
			case seen = <-planner.seen:
			case <-time.After(time.Second):
				t.Fatal("next turn did not begin within the terminal Tick")
			}
			if err := <-done; err != nil {
				t.Fatal(err)
			}

			var seeded bool
			for _, m := range seen {
				if strings.Contains(m.Content, ToolTerminalWakeKind) && strings.Contains(m.Content, call) && strings.Contains(m.Content, `"state":"DONE"`) {
					seeded = true
				}
			}
			if !seeded {
				t.Fatalf("next turn missing terminal verdict: %+v", seen)
			}
			journal := wake.Journal()
			last := journal[len(journal)-1]
			if last.Status != "DISPATCHED" || last.Wake.TraceID != call || last.Wake.Kind != ToolTerminalWakeKind {
				t.Fatalf("uncorrelated wake journal: %+v", journal)
			}
		})
	}
}
