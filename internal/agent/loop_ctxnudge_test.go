package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// loop_ctxnudge_test.go — the agent-loop half of the #2197 context-spike nudge
// witness: contextNudge reads the advisory at the turn boundary from whichever
// session-control source is wired (table or function-shaped gate), and every
// unwired shape is a silent no-op so the historical loop stays byte-identical.

func TestContextNudgeUnwiredShapesAreSilent(t *testing.T) {
	cases := []struct {
		name string
		cfg  runConfig
	}{
		{"zero config", runConfig{}},
		{"trace without table or gate", runConfig{trace: "tr"}},
		{"table without trace", runConfig{table: session.NewTable()}},
		{"gate without Nudge hook", runConfig{trace: "tr", gate: &SessionGate{}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.contextNudge(); got != "" {
				t.Fatalf("want no nudge, got %q", got)
			}
		})
	}
}

func TestContextNudgeTablePath(t *testing.T) {
	tb := session.NewTable()
	cfg := resolveRunConfig([]RunOption{WithSessionTable(tb, "tr")})

	// Quiet before any spike: one turn is no baseline.
	tb.DebitUsage("tr", session.Usage{OutputTokens: 100, ContextTokens: 30000})
	if got := cfg.contextNudge(); got != "" {
		t.Fatalf("single-turn session must not nudge, got %q", got)
	}
	// The spiking turn debits; the NEXT boundary reads the nudge.
	tb.DebitUsage("tr", session.Usage{OutputTokens: 100, ContextTokens: 90000})
	got := cfg.contextNudge()
	if got == "" || !strings.Contains(got, "+60000") {
		t.Fatalf("spiked session must nudge with the delta, got %q", got)
	}
	// A quiet turn self-extinguishes the nudge.
	tb.DebitUsage("tr", session.Usage{OutputTokens: 100, ContextTokens: 91000})
	if after := cfg.contextNudge(); after != "" {
		t.Fatalf("plateau turn must silence the nudge, got %q", after)
	}
}

func TestContextNudgeGatePathAndPreference(t *testing.T) {
	// A wired function-shaped gate owns the boundary (mirroring gateTurn): its Nudge
	// hook answers even when a table is also present, so the gateway native loop and
	// the harness loop can never double-read one boundary.
	tb := session.NewTable()
	tb.DebitUsage("tr", session.Usage{OutputTokens: 100, ContextTokens: 30000})
	tb.DebitUsage("tr", session.Usage{OutputTokens: 100, ContextTokens: 90000})

	var asked string
	gate := SessionGate{Nudge: func(trace string) string {
		asked = trace
		return "gate advisory"
	}}
	cfg := resolveRunConfig([]RunOption{WithSessionTable(tb, "tr"), WithSessionGate(gate, "tr")})
	if got := cfg.contextNudge(); got != "gate advisory" || asked != "tr" {
		t.Fatalf("gate must own the boundary: got %q (asked=%q)", got, asked)
	}
}

func TestContextNudgeRunArmEmitsSharedNextWitness(t *testing.T) {
	const trace = "ctx-nudge-next-trace"
	sessionctl.ReadContextAdvisoryNextRecords(trace)
	gate := SessionGate{Nudge: func(string) string { return "summarize, then continue" }}
	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "original task", false, 1, nil, WithSessionGate(gate, trace)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	var spliced bool
	for _, m := range p.seen {
		if m.Role == RoleUser && m.Content == "summarize, then continue" {
			spliced = true
		}
	}
	if !spliced {
		t.Fatalf("context advisory was not spliced as the user turn the planner consumes: %+v", p.seen)
	}
	records := sessionctl.ReadContextAdvisoryNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("context advisory Next witnesses=%d, want 1", len(records))
	}
	r := records[0]
	if !r.Applied || r.Move.Kind != sessionctl.MoveAnnotate || r.Move.Render != sessionctl.RenderUserSplice || r.Move.Session != sessionctl.SessionInteractive || r.Move.Payload != "summarize, then continue" {
		t.Fatalf("context advisory Next witness=%+v", r)
	}
}

func TestContextNudgeRunArmNoAdvisoryNoNextWitness(t *testing.T) {
	const trace = "ctx-nudge-next-empty"
	sessionctl.ReadContextAdvisoryNextRecords(trace)
	p := &recordingPlanner{}
	if _, err := RunArm(context.Background(), p, "original task", false, 1, nil, WithSessionGate(SessionGate{Nudge: func(string) string { return "" }}, trace)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if records := sessionctl.ReadContextAdvisoryNextRecords(trace); len(records) != 0 {
		t.Fatalf("empty advisory emitted Next witnesses: %+v", records)
	}
}
