package microagent_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type modePlanner struct{ mode microagent.ActionMode }

func (p modePlanner) Model() string { return "fixture" }
func (p modePlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	message := agent.Message{Role: agent.RoleAssistant, Content: `extract {"declaration":"func AdmitRequest() error"}`}
	usage := agent.Usage{PromptTokens: 12, CompletionTokens: 11}
	if p.mode == microagent.ActionModeTool {
		message = agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "one", Type: "function", Function: agent.Func{Name: "extract", Arguments: `{"declaration":"func AdmitRequest() error"}`}}}}
		usage.CompletionTokens = 7
	}
	return &agent.Completion{Message: message, Usage: usage, Model: "fixture"}, nil
}

type modeExecutor struct{}

func (modeExecutor) Execute(_ context.Context, name, arguments string) (string, error) {
	return name + ":AdmitRequest", nil
}

func TestRunActionTaskSameTaskAcrossModes(t *testing.T) {
	task := microagent.ActionTask{Prompt: "extract function", Tool: agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: "extract", Parameters: []byte(`{"type":"object"}`)}}}
	for _, mode := range []microagent.ActionMode{microagent.ActionModeString, microagent.ActionModeTool} {
		got, err := microagent.RunActionTask(context.Background(), modePlanner{mode: mode}, modeExecutor{}, mode, task)
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		if got.Output != "extract:AdmitRequest" {
			t.Fatalf("%s output=%q", mode, got.Output)
		}
	}
}
