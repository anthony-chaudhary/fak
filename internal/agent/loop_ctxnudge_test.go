package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
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
