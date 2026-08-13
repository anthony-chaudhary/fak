package microagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// ActionMode selects how one native loop asks a model to express an action.
type ActionMode string

const (
	ActionModeString ActionMode = "bash-string"
	ActionModeTool   ActionMode = "tool-calling"
)

// ActionExecutor is the shared effect seam used by both action modes.
type ActionExecutor interface {
	Execute(context.Context, string, string) (string, error)
}

// ActionTask describes one action-turn task. String mode expects
// "<tool> <json-arguments>" in assistant content; tool mode advertises Tool.
type ActionTask struct {
	Prompt string
	Tool   agent.ToolDef
}

// ActionResult records the action selected through either representation.
type ActionResult struct {
	Mode       ActionMode
	Tool       string
	Arguments  string
	Output     string
	Input      int
	OutputUsed int
}

// RunActionTask drives the same one-turn native loop in either portable string
// mode or provider-native typed-tool mode, then invokes one shared executor.
func RunActionTask(ctx context.Context, planner Gateway, executor ActionExecutor, mode ActionMode, task ActionTask) (ActionResult, error) {
	if planner == nil || executor == nil {
		return ActionResult{}, errors.New("microagent: action planner and executor are required")
	}
	messages := []agent.Message{{Role: agent.RoleUser, Content: task.Prompt}}
	var tools []agent.ToolDef
	if mode == ActionModeTool {
		tools = []agent.ToolDef{task.Tool}
	} else if mode != ActionModeString {
		return ActionResult{}, fmt.Errorf("microagent: unsupported action mode %q", mode)
	}
	completion, err := planner.Complete(ctx, messages, tools)
	if err != nil {
		return ActionResult{}, err
	}
	if completion == nil {
		return ActionResult{}, errors.New("microagent: nil action completion")
	}
	name, arguments, err := decodeAction(mode, completion.Message)
	if err != nil {
		return ActionResult{}, err
	}
	output, err := executor.Execute(ctx, name, arguments)
	if err != nil {
		return ActionResult{}, err
	}
	return ActionResult{Mode: mode, Tool: name, Arguments: arguments, Output: output, Input: completion.Usage.PromptTokens, OutputUsed: completion.Usage.CompletionTokens}, nil
}

func decodeAction(mode ActionMode, message agent.Message) (string, string, error) {
	if mode == ActionModeTool {
		if len(message.ToolCalls) != 1 || strings.TrimSpace(message.ToolCalls[0].Function.Name) == "" {
			return "", "", errors.New("microagent: tool mode requires exactly one named tool call")
		}
		return message.ToolCalls[0].Function.Name, message.ToolCalls[0].Function.Arguments, nil
	}
	parts := strings.SplitN(strings.TrimSpace(message.Content), " ", 2)
	if len(parts) != 2 || parts[0] == "" || !json.Valid([]byte(parts[1])) {
		return "", "", errors.New("microagent: string mode requires '<tool> <json-arguments>'")
	}
	return parts[0], parts[1], nil
}
