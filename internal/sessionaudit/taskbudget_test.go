package sessionaudit

import (
	"strings"
	"testing"
)

// sampleSession builds a Session with the fields FoldTaskBudget reads, without
// going through Analyze — the fold is pure over the Session shape.
func sampleSession() Session {
	return Session{
		Session:        "abcdefgh1234",
		AssistantTurns: 12,
		NToolUse:       10,
		Tokens: TokenCounts{
			Input:       1000,
			Output:      2000,
			CacheRead:   5000,
			CacheCreate: 2000,
		},
		Tools: map[string]int64{
			"Read":  4, // read
			"Grep":  1, // read
			"Edit":  2, // edit
			"Write": 1, // edit
			"Bash":  2, // other
		},
		PerModel: map[string]ModelCounts{
			"claude-opus-4-8": {Turns: 10},
			"claude-haiku":    {Turns: 2},
		},
	}
}

func TestFoldTaskBudget_SpendAndBreakdown(t *testing.T) {
	b := FoldTaskBudget(sampleSession(), TaskBudgetTarget{Tokens: 20000, Turns: 20})

	if got, want := b.TotalTokens, int64(10000); got != want {
		t.Fatalf("total tokens = %d, want %d", got, want)
	}
	if got, want := b.CacheTokens, int64(7000); got != want {
		t.Fatalf("cache tokens = %d, want %d", got, want)
	}
	if b.TokenFrac == nil || *b.TokenFrac != 0.5 {
		t.Fatalf("token frac = %v, want 0.5", b.TokenFrac)
	}
	if b.OverTokens {
		t.Fatalf("should not be over token budget at 50%%")
	}
	if b.TurnFrac == nil || *b.TurnFrac != 0.6 {
		t.Fatalf("turn frac = %v, want 0.6", b.TurnFrac)
	}

	bd := b.Breakdown
	if bd.Reads != 5 {
		t.Errorf("reads = %d, want 5 (Read+Grep)", bd.Reads)
	}
	if bd.Edits != 3 {
		t.Errorf("edits = %d, want 3 (Edit+Write)", bd.Edits)
	}
	if bd.OtherTools != 2 {
		t.Errorf("other = %d, want 2 (Bash)", bd.OtherTools)
	}
	if bd.Reads+bd.Edits+bd.OtherTools != 10 {
		t.Errorf("categories %d+%d+%d must partition all 10 tool calls", bd.Reads, bd.Edits, bd.OtherTools)
	}
	if bd.Model != "opus" {
		t.Errorf("top model tier = %q, want opus (10 turns beats haiku 2)", bd.Model)
	}
}

func TestFoldTaskBudget_OverTarget(t *testing.T) {
	b := FoldTaskBudget(sampleSession(), TaskBudgetTarget{Tokens: 5000, Turns: 5})
	if !b.OverTokens {
		t.Errorf("10000 tok over a 5000 target should flag OverTokens")
	}
	if !b.OverTurns {
		t.Errorf("12 turns over a 5-turn target should flag OverTurns")
	}
	if b.TokenFrac == nil || *b.TokenFrac != 2.0 {
		t.Errorf("token frac = %v, want 2.0", b.TokenFrac)
	}
}

func TestFoldTaskBudget_NoTarget(t *testing.T) {
	b := FoldTaskBudget(sampleSession(), TaskBudgetTarget{})
	if b.TokenFrac != nil || b.TurnFrac != nil {
		t.Errorf("no target axes set → fractions must be nil, got token=%v turn=%v", b.TokenFrac, b.TurnFrac)
	}
	// Spend + breakdown are still reported without a target.
	if b.TotalTokens != 10000 || b.Breakdown.ToolCalls != 10 {
		t.Errorf("spend/breakdown must fold even with no target; got total=%d tools=%d", b.TotalTokens, b.Breakdown.ToolCalls)
	}
}

func TestRenderTaskBudget_Inline(t *testing.T) {
	line := RenderTaskBudget(FoldTaskBudget(sampleSession(), TaskBudgetTarget{Tokens: 20000, Turns: 20}))
	if strings.Contains(line, "\n") {
		t.Fatalf("readout must be a single inline line, got:\n%s", line)
	}
	for _, want := range []string{
		"task-budget",
		"abcdefgh", // truncated session id
		"10,000 tok",
		"50%",     // token fraction
		"12 turn", // turns spent
		"reads 5", // category breakdown
		"edits 3",
		"other 2",
		"model opus",
	} {
		if !strings.Contains(line, want) {
			t.Errorf("readout missing %q; line was:\n%s", want, line)
		}
	}
}

func TestRenderTaskBudget_OverFlag(t *testing.T) {
	line := RenderTaskBudget(FoldTaskBudget(sampleSession(), TaskBudgetTarget{Tokens: 5000}))
	if !strings.Contains(line, "OVER") {
		t.Errorf("over-budget readout must contain the OVER flag; line was:\n%s", line)
	}
}
