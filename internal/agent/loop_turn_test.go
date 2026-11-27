package agent

import (
	"context"
	"testing"
)

type loopTurnScriptedPlanner struct {
	replies []*Completion
	idx     int
}

func (p *loopTurnScriptedPlanner) Model() string { return "scripted-coding-test" }

func (p *loopTurnScriptedPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	if p.idx >= len(p.replies) {
		return &Completion{
			Message:      Message{Role: RoleAssistant, Content: "done"},
			FinishReason: "stop",
		}, nil
	}
	c := p.replies[p.idx]
	p.idx++
	return c, nil
}

func TestLoopTurnCodingTaskCompleted(t *testing.T) {
	codingTools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read"}},
		{Type: "function", Function: ToolDefFunction{Name: "Edit"}},
		{Type: "function", Function: ToolDefFunction{Name: "Bash"}},
	}

	t.Run("non-booking clean final answer sets TaskCompleted", func(t *testing.T) {
		p := &loopTurnScriptedPlanner{replies: []*Completion{
			{Message: Message{Role: RoleAssistant, Content: "All tests pass."}, FinishReason: "stop"},
		}}
		m, err := RunArm(t.Context(), p, "fix bug and run go test", false, 4, nil, WithToolCatalog(codingTools))
		if err != nil {
			t.Fatal(err)
		}
		if !m.TaskCompleted {
			t.Fatalf("TaskCompleted = false, want true for clean non-booking final answer: %+v", m)
		}
	})

	t.Run("truncated final answer leaves TaskCompleted false", func(t *testing.T) {
		p := &loopTurnScriptedPlanner{replies: []*Completion{
			{Message: Message{Role: RoleAssistant, Content: "partial answer"}, FinishReason: "length"},
		}}
		m, err := RunArm(t.Context(), p, "fix bug", false, 4, nil, WithToolCatalog(codingTools))
		if err != nil {
			t.Fatal(err)
		}
		if m.TaskCompleted || !m.FinalAnswerTruncated {
			t.Fatalf("TaskCompleted=%v FinalAnswerTruncated=%v, want false/true", m.TaskCompleted, m.FinalAnswerTruncated)
		}
	})

	t.Run("surrender note leaves TaskCompleted false", func(t *testing.T) {
		surrender := &Completion{Message: Message{Role: RoleAssistant, Content: "I cannot complete this task because access is blocked."}, FinishReason: "stop"}
		p := &loopTurnScriptedPlanner{replies: []*Completion{surrender, surrender, surrender, surrender}}
		m, err := RunArm(t.Context(), p, "fix bug", false, 4, nil, WithToolCatalog(codingTools))
		if err != nil {
			t.Fatal(err)
		}
		if m.TaskCompleted {
			t.Fatalf("TaskCompleted = true, want false for surrender note")
		}
	})

	t.Run("booking catalog still requires book_flight confirmation", func(t *testing.T) {
		p := &loopTurnScriptedPlanner{replies: []*Completion{
			{Message: Message{Role: RoleAssistant, Content: "I looked at the flights."}, FinishReason: "stop"},
		}}
		m, err := RunArm(t.Context(), p, DefaultTask, false, 4, nil)
		if err != nil {
			t.Fatal(err)
		}
		if m.TaskCompleted {
			t.Fatalf("TaskCompleted = true, want false when book_flight was never confirmed")
		}
	})
}
