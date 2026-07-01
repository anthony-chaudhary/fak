package sessionaudit

import (
	"fmt"
	"strings"
)

// EditTools is the write-side complement of ReadOnlyTools: the tool calls that
// mutate the working tree (or a notebook). A call that is in neither set counts
// as a plain "tool call" (a command, a search-side effect, an MCP verb) so the
// three category buckets — reads, edits, other — partition every tool_use.
var EditTools = map[string]bool{
	"Edit":         true,
	"Write":        true,
	"MultiEdit":    true,
	"NotebookEdit": true,
}

// TaskBudgetTarget is the SOFT ceiling an operator (or the agent itself) sets
// for a single task: a token budget and a turn budget. A zero field means "no
// target on this axis" — the readout then reports spend without a fraction, so
// the agent still sees where it went even when no ceiling was declared.
type TaskBudgetTarget struct {
	Tokens int64 `json:"tokens,omitempty"`
	Turns  int64 `json:"turns,omitempty"`
}

// TaskCategoryBreakdown is the coarse "where did the effort go" split the issue
// asks for: read tool calls, edit (write) tool calls, and every other tool call,
// plus the assistant turns and the model tier the task ran on. It is derived
// purely from a Session's already-counted tool mix and per-model turns.
type TaskCategoryBreakdown struct {
	Reads      int64  `json:"reads"`
	Edits      int64  `json:"edits"`
	OtherTools int64  `json:"other_tools"`
	ToolCalls  int64  `json:"tool_calls"`
	Turns      int64  `json:"turns"`
	Model      string `json:"model,omitempty"`
}

// TaskBudget is the inline, per-task spend-vs-target readout the working agent
// can query as it goes. Tokens is the total context this task ingested (fresh
// input + cache read + cache create) plus output; the fractions are spent/target
// on each axis when a target was declared. It reuses the same token accounting
// the session audit already folds — no new usage plumbing.
type TaskBudget struct {
	Session string `json:"session,omitempty"`

	// Spend.
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	CacheTokens  int64 `json:"cache_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
	Turns        int64 `json:"turns"`

	// Target + fractions (nil when the corresponding target axis is unset).
	Target     TaskBudgetTarget `json:"target"`
	TokenFrac  *float64         `json:"token_frac,omitempty"`
	TurnFrac   *float64         `json:"turn_frac,omitempty"`
	OverTokens bool             `json:"over_tokens,omitempty"`
	OverTurns  bool             `json:"over_turns,omitempty"`

	Breakdown TaskCategoryBreakdown `json:"breakdown"`
}

// FoldTaskBudget builds the per-task readout from an already-Analyze'd Session
// and a soft target. It is a PURE fold — no I/O, deterministic — so a caller can
// render it inline at any point in a task without side effects.
func FoldTaskBudget(s Session, target TaskBudgetTarget) TaskBudget {
	total := s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate + s.Tokens.Output
	b := TaskBudget{
		Session:      s.Session,
		InputTokens:  s.Tokens.Input,
		OutputTokens: s.Tokens.Output,
		CacheTokens:  s.Tokens.CacheRead + s.Tokens.CacheCreate,
		TotalTokens:  total,
		Turns:        s.AssistantTurns,
		Target:       target,
		Breakdown:    foldBreakdown(s),
	}
	if target.Tokens > 0 {
		f := float64(total) / float64(target.Tokens)
		b.TokenFrac = &f
		b.OverTokens = total > target.Tokens
	}
	if target.Turns > 0 {
		f := float64(s.AssistantTurns) / float64(target.Turns)
		b.TurnFrac = &f
		b.OverTurns = s.AssistantTurns > target.Turns
	}
	return b
}

func foldBreakdown(s Session) TaskCategoryBreakdown {
	bd := TaskCategoryBreakdown{
		ToolCalls: s.NToolUse,
		Turns:     s.AssistantTurns,
		Model:     topModel(s),
	}
	for name, n := range s.Tools {
		switch {
		case ReadOnlyTools[name]:
			bd.Reads += n
		case EditTools[name]:
			bd.Edits += n
		default:
			bd.OtherTools += n
		}
	}
	return bd
}

// topModel returns the model tier that ran the most assistant turns for the
// task, so the readout names the model that dominated the spend. Ties break
// alphabetically for determinism.
func topModel(s Session) string {
	var top string
	var topTurns int64
	for model, c := range s.PerModel {
		if c.Turns > topTurns || (c.Turns == topTurns && (top == "" || model < top)) {
			top, topTurns = model, c.Turns
		}
	}
	if top == "" {
		return ""
	}
	return ModelTier(top)
}

// RenderTaskBudget renders the readout as one inline line the working agent can
// print or log mid-task. It leads with spend, shows the target fraction when a
// target exists (flagging OVER), then the coarse reads/edits/other breakdown.
func RenderTaskBudget(b TaskBudget) string {
	var sb strings.Builder
	sb.WriteString("task-budget")
	if b.Session != "" {
		id := b.Session
		if len(id) > 8 {
			id = id[:8]
		}
		fmt.Fprintf(&sb, " %s", id)
	}
	fmt.Fprintf(&sb, ": %s tok", fmtInt(b.TotalTokens))
	if b.TokenFrac != nil {
		fmt.Fprintf(&sb, " (%s of %s = %.0f%%%s)",
			fmtInt(b.TotalTokens), fmtInt(b.Target.Tokens), *b.TokenFrac*100, overMark(b.OverTokens))
	}
	fmt.Fprintf(&sb, "; %d turn", b.Turns)
	if b.TurnFrac != nil {
		fmt.Fprintf(&sb, " (of %d = %.0f%%%s)", b.Target.Turns, *b.TurnFrac*100, overMark(b.OverTurns))
	}
	fmt.Fprintf(&sb, "; %d tool call", b.Breakdown.ToolCalls)
	fmt.Fprintf(&sb, " [reads %d, edits %d, other %d]",
		b.Breakdown.Reads, b.Breakdown.Edits, b.Breakdown.OtherTools)
	fmt.Fprintf(&sb, "; out %s tok", fmtInt(b.OutputTokens))
	if b.Breakdown.Model != "" {
		fmt.Fprintf(&sb, "; model %s", b.Breakdown.Model)
	}
	return sb.String()
}

func overMark(over bool) string {
	if over {
		return " OVER"
	}
	return ""
}
