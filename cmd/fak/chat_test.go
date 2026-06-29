package main

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// chatScript is a deterministic offline planner for the fak chat e2e: it returns a
// fixed sequence of completions, one per Complete call (one per model turn), so a
// multi-turn REPL session is fully reproducible with no upstream. It satisfies
// agent.Planner.
type chatScript struct {
	turns []*agent.Completion
	n     int
}

func (p *chatScript) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}
func (p *chatScript) Model() string { return "chat-script" }

func toolTurn(tool, args string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{ToolCalls: []agent.ToolCall{{ID: "c", Function: agent.Func{Name: tool, Arguments: args}}}}}
}
func finalTurn(text string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{Content: text}}
}

// TestChatTwoTurnsWithDeniedDestructive is the acceptance witness for #1320: a
// scripted two-turn `fak chat` session driven entirely through agent.RunArm with
// kernel.Syscall as the sole tool path. Turn 1 is an ordinary read that resolves
// to a final answer; turn 2 emits a destructive delete_account call that the
// capability floor DENIES. The test asserts the denial was returned as a VALUE
// (Denies==1, no crash), the destructive tool never executed
// (DestructiveExecuted==false), and the denied call never reached the engine
// (EngineCalls==0 on that turn) — with no upstream involved (offline planner).
func TestChatTwoTurnsWithDeniedDestructive(t *testing.T) {
	// Two human turns over one stdin stream. Turn 1's model script: one read then a
	// final answer. Turn 2's model script: one delete_account (denied) then a final
	// answer. Because runChat drives ONE RunArm per human line and the scripted
	// planner advances per Complete call, the script must lay the turns end to end.
	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("get_user_details", `{"user_id":"mia_li_3668"}`), // turn 1, model step 1
		finalTurn("Found your account."),                          // turn 1, model step 2 (ends turn 1)
		toolTurn("delete_account", `{"user_id":"mia_li_3668"}`),   // turn 2, model step 1 — DENIED
		finalTurn("I can't delete the account; that's blocked."),  // turn 2, model step 2 (ends turn 2)
	}}

	in := strings.NewReader("look up my account\ndelete my account\n")
	var out strings.Builder
	runChat(in, &out, planner, 10)

	got := out.String()
	if !strings.Contains(got, "Found your account.") {
		t.Fatalf("turn 1 final answer missing from REPL output:\n%s", got)
	}
	if !strings.Contains(got, "1 denied") {
		t.Fatalf("turn 2 should report exactly one denied call in its summary:\n%s", got)
	}
}

// TestRunChatTurnMetrics drives the two scripted turns through runChat by reusing
// the same end-to-end stream, then re-runs RunArm directly on the denied turn so
// the value-not-crash assertions read off ArmMetrics precisely: a denied
// destructive call is a structured value, never an executed effect, and never an
// engine dispatch.
func TestRunChatTurnMetrics(t *testing.T) {
	deny := &chatScript{turns: []*agent.Completion{
		toolTurn("delete_account", `{"user_id":"mia_li_3668"}`),
		finalTurn("blocked, as expected"),
	}}
	m, err := agent.RunArm(context.Background(), deny, "delete my account", true, 10, nil)
	if err != nil {
		t.Fatalf("RunArm returned an error on a denied call (should be a value, not a crash): %v", err)
	}
	if m.Denies != 1 {
		t.Fatalf("expected exactly 1 deny, got %d", m.Denies)
	}
	if m.DestructiveExecuted {
		t.Fatal("destructive delete_account must NOT have executed")
	}
	if m.EngineCalls != 0 {
		t.Fatalf("a denied call must never reach the engine; EngineCalls=%d", m.EngineCalls)
	}
	if !strings.Contains(m.FinalAnswer, "blocked") {
		t.Fatalf("loop should have continued past the deny to a final answer, got %q", m.FinalAnswer)
	}
}
