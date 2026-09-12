package gateway

// architect_editor.go — the architect/editor dual-model coding pattern (Aider's
// ArchitectCoder), expressed as an agent.Planner decorator.
//
// A high-reasoning (and expensive) ARCHITECT model is asked to plan a change in
// prose: what to change and how, in unambiguous direction. It never emits the
// final code. A cheap, syntax-exact EDITOR model then receives ONLY that
// instruction — a CLEAN SLATE, not the original conversation — and emits the
// final edit blocks. Splitting the roles lets the expensive model's reasoning be
// bought once while the bulk of the output comes from a fast/cheap model, and it
// keeps the editor's context small and unambiguous.

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// editorInstructionPrefix heads the clean-slate message the editor receives. It
// is package-private; tests recognize the envelope by containing the plan text.
const editorInstructionPrefix = "You are the editor. Apply the following plan exactly, emitting only the final edits.\n\n"

// ArchitectEditorPipeline is an agent.Planner decorator that runs a two-phase
// architect -> editor chain: the architect reasons and emits a plan/instruction,
// the editor receives ONLY that instruction (clean slate) and emits the final
// syntax-exact output. Mirrors Aider's ArchitectCoder.
type ArchitectEditorPipeline struct {
	architect agent.Planner
	editor    agent.Planner
	// hasEditor distinguishes a configured editor from the "one model does both" fallback.
	hasEditor bool
}

// NewArchitectEditorPipeline wires an architect planner and an optional editor
// planner. A nil editor means one model performs both phases (Aider's
// editor_model-or-main_model fallback) — in that case the architect doubles as
// the editor and hasEditor is false.
func NewArchitectEditorPipeline(architect, editor agent.Planner) (*ArchitectEditorPipeline, error) {
	if architect == nil {
		return nil, errors.New("gateway: architect/editor pipeline requires an architect planner")
	}
	// The editor field always holds the planner that runs phase two: the
	// configured editor, or the architect itself in the single-model fallback.
	// hasEditor is the separate "a DISTINCT editor was configured" signal that
	// keeps WalkPlanners from visiting the architect twice.
	p := &ArchitectEditorPipeline{architect: architect, editor: architect}
	if editor != nil && editor != architect {
		p.editor = editor
		p.hasEditor = true
	}
	return p, nil
}

// Model is the ARCHITECT's id — the primary/high-reasoning side.
func (p *ArchitectEditorPipeline) Model() string { return p.architect.Model() }

// Architect exposes the planning side (tests, wiring).
func (p *ArchitectEditorPipeline) Architect() agent.Planner { return p.architect }

// Editor exposes the editor side: the configured editor, or the architect when
// no distinct editor was supplied (the single-model fallback).
func (p *ArchitectEditorPipeline) Editor() agent.Planner { return p.editor }

// HasEditor reports whether a distinct editor planner was configured.
func (p *ArchitectEditorPipeline) HasEditor() bool { return p.hasEditor }

// WalkPlanners traverses each child planner. The gateway does a structural
// type-assert for this method (gateway.go) to recurse into a decorator's
// children, so retry/auth/observability hooks reach the wrapped planners; both
// phases must be walked. When the editor is not distinct from the architect, the
// architect is walked once and never double-visited. Nil-safe receiver.
func (p *ArchitectEditorPipeline) WalkPlanners(fn func(agent.Planner)) {
	if p == nil {
		return
	}
	if p.architect != nil {
		fn(p.architect)
	}
	if p.hasEditor && p.editor != nil {
		fn(p.editor)
	}
}

// planText extracts the architect's plan: its content verbatim when present, else
// its first tool-call arguments (a tool-driven planner may carry the plan there).
func planText(comp *agent.Completion) string {
	if comp == nil {
		return ""
	}
	if strings.TrimSpace(comp.Message.Content) != "" {
		return comp.Message.Content
	}
	for _, tc := range comp.Message.ToolCalls {
		if strings.TrimSpace(tc.Function.Arguments) != "" {
			return tc.Function.Arguments
		}
	}
	if comp.Message.FunctionCall != nil && strings.TrimSpace(comp.Message.FunctionCall.Arguments) != "" {
		return comp.Message.FunctionCall.Arguments
	}
	return ""
}

// formatEditorInstruction wraps the architect's plan in the stable instruction
// envelope the editor consumes as its sole user message. An empty plan yields a
// stable, distinguishable empty-plan marker rather than a silent blank prompt.
func formatEditorInstruction(plan string) string {
	plan = strings.TrimSpace(plan)
	if plan == "" {
		return editorInstructionPrefix + "(no plan was produced by the architect)"
	}
	return editorInstructionPrefix + plan
}

// Complete runs the architect on the ORIGINAL messages/tools/opts, then feeds the
// editor a CLEAN SLATE — a single user message carrying the architect's plan —
// and the SAME tools/opts forwarded unchanged. It returns the EDITOR's completion
// (the final edit blocks). The caller's opts are never rewritten: the
// account-binding path force-overrides params.Model LAST via appended opts, so
// rewriting them here would clobber it.
func (p *ArchitectEditorPipeline) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	arch, err := p.architect.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, fmt.Errorf("gateway: architect/editor: architect: %w", err)
	}
	editorMessages := []agent.Message{{
		Role:    agent.RoleUser,
		Content: formatEditorInstruction(planText(arch)),
	}}
	ed, err := p.editor.Complete(ctx, editorMessages, tools, opts...)
	if err != nil {
		return nil, fmt.Errorf("gateway: architect/editor: editor: %w", err)
	}
	return ed, nil
}
