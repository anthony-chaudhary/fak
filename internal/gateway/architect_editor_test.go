package gateway

// architect_editor_test.go — the architect/editor dual-model coding seam
// (architect_editor.go). The contract under test: the architect is asked to
// reason over the ORIGINAL conversation, the editor receives a CLEAN SLATE
// carrying only the architect's plan (never the original messages), the pipeline
// returns the EDITOR's completion, and caller opts reach both phases unchanged.

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// aeRecordingPlanner is a deterministic planner that records the exact messages
// and opts it was invoked with, returning a fixed completion.
type aeRecordingPlanner struct {
	id       string
	reply    string
	err      error
	calls    int
	messages []agent.Message
	opts     []agent.SampleOpt
}

func (p *aeRecordingPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	p.messages = append([]agent.Message(nil), messages...)
	p.opts = append([]agent.SampleOpt(nil), opts...)
	if p.err != nil {
		return nil, p.err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.reply}}, nil
}

func (p *aeRecordingPlanner) Model() string { return p.id }

func TestArchitectEditorChainOrderAndCleanSlate(t *testing.T) {
	arch := &aeRecordingPlanner{id: "big-architect", reply: "Rename Foo to Bar in baz.go."}
	edit := &aeRecordingPlanner{id: "small-editor", reply: "<<<<<<< SEARCH\nFoo\n=======\nBar\n>>>>>>> REPLACE"}
	p, err := NewArchitectEditorPipeline(arch, edit)
	if err != nil {
		t.Fatal(err)
	}

	original := []agent.Message{
		{Role: agent.RoleSystem, Content: "You are a coding assistant."},
		{Role: agent.RoleUser, Content: "Please rename Foo to Bar."},
	}
	comp, err := p.Complete(context.Background(), original, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Chain order: architect first, editor second, exactly once each.
	if arch.calls != 1 || edit.calls != 1 {
		t.Fatalf("call counts architect=%d editor=%d, want 1/1", arch.calls, edit.calls)
	}

	// The architect sees the ORIGINAL conversation.
	if len(arch.messages) != len(original) || arch.messages[1].Content != "Please rename Foo to Bar." {
		t.Errorf("architect messages = %+v, want the original conversation", arch.messages)
	}

	// The editor gets a CLEAN SLATE: one user message carrying the architect's plan.
	if len(edit.messages) != 1 {
		t.Fatalf("editor messages = %d, want exactly 1 (clean slate)", len(edit.messages))
	}
	if edit.messages[0].Role != agent.RoleUser {
		t.Errorf("editor message role = %q, want %q", edit.messages[0].Role, agent.RoleUser)
	}
	if !strings.Contains(edit.messages[0].Content, arch.reply) {
		t.Errorf("editor prompt %q does not contain the architect plan %q", edit.messages[0].Content, arch.reply)
	}
	// The editor must NOT see the original conversation.
	if strings.Contains(edit.messages[0].Content, "You are a coding assistant.") {
		t.Error("editor saw the original system message — the clean slate was violated")
	}

	// Result identity: the returned completion is the EDITOR's.
	if comp.Message.Content != edit.reply {
		t.Errorf("Complete returned %q, want the editor's output %q", comp.Message.Content, edit.reply)
	}
	if got := p.Model(); got != arch.id {
		t.Errorf("Model() = %q, want the architect id %q", got, arch.id)
	}
}

func TestArchitectEditorWalkPlanners(t *testing.T) {
	arch := &aeRecordingPlanner{id: "arch"}
	edit := &aeRecordingPlanner{id: "edit"}
	p, err := NewArchitectEditorPipeline(arch, edit)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[agent.Planner]int{}
	p.WalkPlanners(func(child agent.Planner) { seen[child]++ })
	if seen[arch] != 1 || seen[edit] != 1 {
		t.Errorf("WalkPlanners visits arch=%d edit=%d, want 1 each", seen[arch], seen[edit])
	}

	// Single-model fallback: the architect doubles as editor and must NOT be
	// double-visited.
	single, err := NewArchitectEditorPipeline(arch, nil)
	if err != nil {
		t.Fatal(err)
	}
	seen = map[agent.Planner]int{}
	single.WalkPlanners(func(child agent.Planner) { seen[child]++ })
	if seen[arch] != 1 || len(seen) != 1 {
		t.Errorf("fallback WalkPlanners visits %v, want the architect exactly once", seen)
	}

	// Nil-safe receiver.
	var nilP *ArchitectEditorPipeline
	nilP.WalkPlanners(func(agent.Planner) { t.Error("fn called on nil receiver") })
}

func TestArchitectEditorErrorWrapping(t *testing.T) {
	archErr := errors.New("boom-arch")
	arch := &aeRecordingPlanner{id: "arch", err: archErr}
	edit := &aeRecordingPlanner{id: "edit", reply: "x"}
	p, err := NewArchitectEditorPipeline(arch, edit)
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Complete(context.Background(), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "architect") || !errors.Is(err, archErr) {
		t.Errorf("architect error = %v, want non-nil, wrapped, naming the architect phase", err)
	}
	if edit.calls != 0 {
		t.Error("editor must not run when the architect fails")
	}

	// Editor error surfaces, wrapped and naming the editor phase.
	arch = &aeRecordingPlanner{id: "arch", reply: "plan"}
	editErr := errors.New("boom-edit")
	edit = &aeRecordingPlanner{id: "edit", err: editErr}
	p, err = NewArchitectEditorPipeline(arch, edit)
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Complete(context.Background(), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "editor") || !errors.Is(err, editErr) {
		t.Errorf("editor error = %v, want non-nil, wrapped, naming the editor phase", err)
	}
}

func TestArchitectEditorSingleModelFallback(t *testing.T) {
	m := &aeRecordingPlanner{id: "one-model", reply: "only output"}
	p, err := NewArchitectEditorPipeline(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if p.HasEditor() {
		t.Error("HasEditor() = true with a nil editor, want false")
	}
	if p.Editor() != m {
		t.Error("Editor() must fall back to the architect when none was configured")
	}
	comp, err := p.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Both phases run through the same planner: two calls, one clean-slate turn.
	if m.calls != 2 {
		t.Errorf("fallback calls = %d, want 2 (architect then editor)", m.calls)
	}
	if comp.Message.Content != "only output" {
		t.Errorf("Complete = %q, want the single model's output", comp.Message.Content)
	}
}

func TestArchitectEditorForwardsOptsUnchanged(t *testing.T) {
	arch := &aeRecordingPlanner{id: "arch", reply: "plan"}
	edit := &aeRecordingPlanner{id: "edit", reply: "out"}
	p, err := NewArchitectEditorPipeline(arch, edit)
	if err != nil {
		t.Fatal(err)
	}
	opt := agent.WithModel("caller-model")
	if _, err := p.Complete(context.Background(), nil, nil, opt); err != nil {
		t.Fatal(err)
	}
	if len(arch.opts) != 1 || len(edit.opts) != 1 {
		t.Fatalf("opts forwarded architect=%d editor=%d, want 1 each", len(arch.opts), len(edit.opts))
	}
	var sp agent.SampleParams
	edit.opts[0](&sp)
	if sp.Model != "caller-model" {
		t.Errorf("editor received model %q, want the caller's %q unchanged", sp.Model, "caller-model")
	}
}

// aeNilPlanner returns (nil, nil) — no error, no completion. The pipeline must
// still drive the editor with the stable empty-plan marker rather than panicking.
type aeNilPlanner struct{ id string }

func (p *aeNilPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, nil
}
func (p *aeNilPlanner) Model() string { return p.id }

func TestArchitectEditorNilCompletionDoesNotPanic(t *testing.T) {
	edit := &aeRecordingPlanner{id: "edit", reply: "out"}
	p, err := NewArchitectEditorPipeline(&aeNilPlanner{id: "arch"}, edit)
	if err != nil {
		t.Fatal(err)
	}
	comp, err := p.Complete(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("nil architect completion must not error, got %v", err)
	}
	if edit.calls != 1 {
		t.Fatalf("editor calls = %d, want 1 (the empty-plan marker still drives it)", edit.calls)
	}
	if !strings.Contains(edit.messages[0].Content, "no plan was produced") {
		t.Errorf("editor prompt %q lacks the empty-plan marker", edit.messages[0].Content)
	}
	if comp == nil || comp.Message.Content != "out" {
		t.Errorf("Complete = %+v, want the editor's completion", comp)
	}
}

func TestNewArchitectEditorPipelineValidates(t *testing.T) {
	if _, err := NewArchitectEditorPipeline(nil, &aeRecordingPlanner{id: "e"}); err == nil {
		t.Error("nil architect must be refused")
	}
	// A nil editor is allowed and doubles as the architect.
	if _, err := NewArchitectEditorPipeline(&aeRecordingPlanner{id: "a"}, nil); err != nil {
		t.Errorf("nil editor must be allowed (single-model fallback), got %v", err)
	}
	// Passing the same planner twice must collapse to the fallback (no double-walk).
	m := &aeRecordingPlanner{id: "same"}
	p, err := NewArchitectEditorPipeline(m, m)
	if err != nil {
		t.Fatal(err)
	}
	if p.HasEditor() {
		t.Error("the same planner for both phases must collapse to the fallback")
	}
}
