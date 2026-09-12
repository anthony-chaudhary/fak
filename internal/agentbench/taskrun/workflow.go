package taskrun

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type WorkflowReceipt struct {
	Kind             string   `json:"kind"`
	Area             string   `json:"area,omitempty"`
	PriorTaskID      string   `json:"prior_task_id,omitempty"`
	PriorStateSHA256 string   `json:"prior_state_sha256,omitempty"`
	PriorSteps       []string `json:"prior_steps,omitempty"`
	RequiredSteps    []string `json:"required_steps"`
	ObservedSteps    []string `json:"observed_steps"`
	Passed           bool     `json:"passed"`
	Error            string   `json:"error,omitempty"`
}

func taskfixtureByID(id string) taskfixture.Fixture {
	for _, fixture := range taskfixture.Cases(true) {
		if fixture.ID == id {
			return fixture
		}
	}
	return taskfixture.Fixture{}
}

func validateWorkflow(f taskfixture.Fixture, turns []PlannerTurn, testCommand string) (WorkflowReceipt, error) {
	receipt := WorkflowReceipt{Kind: f.Workflow.Kind, Area: f.Workflow.Area, PriorTaskID: f.Workflow.PriorTaskID, PriorStateSHA256: f.Workflow.PriorStateSHA256, PriorSteps: append([]string(nil), f.Workflow.PriorSteps...), RequiredSteps: append([]string(nil), f.Workflow.RequiredSteps...)}
	if receipt.Kind == "" || len(receipt.RequiredSteps) == 0 || strings.TrimSpace(testCommand) == "" {
		return workflowFailure(receipt, errors.New("task workflow contract is incomplete"))
	}
	seenCalls := make(map[string]bool)
	state := workflowState{}
	for turnIndex, turn := range turns {
		for _, call := range turn.ToolCalls {
			if call.ID == "" || seenCalls[call.ID] {
				continue
			}
			seenCalls[call.ID] = true
			result, ok := linkedWorkflowResult(turns, turnIndex, call.ID)
			if !ok {
				continue
			}
			step, qualifies := classifyWorkflowStep(f, call.Function.Name, call.Function.Arguments, result, testCommand, &state)
			if !qualifies {
				continue
			}
			receipt.ObservedSteps = append(receipt.ObservedSteps, step)
		}
	}
	if !orderedWorkflowSteps(receipt.ObservedSteps, receipt.RequiredSteps) {
		return workflowFailure(receipt, fmt.Errorf("required workflow %v not witnessed in order", receipt.RequiredSteps))
	}
	receipt.Passed = true
	return receipt, nil
}

func workflowFailure(receipt WorkflowReceipt, err error) (WorkflowReceipt, error) {
	receipt.Error = err.Error()
	return receipt, err
}

func linkedWorkflowResult(turns []PlannerTurn, callTurn int, id string) (agent.Message, bool) {
	for index := callTurn; index < len(turns); index++ {
		for _, message := range turns[index].Messages {
			if message.Role == agent.RoleTool && message.ToolCallID == id {
				return message, true
			}
		}
	}
	return agent.Message{}, false
}

type workflowState struct {
	edited         bool
	inspected      string
	candidate      string
	candidateKnown bool
}

func classifyWorkflowStep(f taskfixture.Fixture, name, rawArgs string, result agent.Message, testCommand string, state *workflowState) (string, bool) {
	var args struct {
		FilePath  string `json:"file_path"`
		Command   string `json:"command"`
		Content   string `json:"content"`
		OldString string `json:"old_string"`
		NewString string `json:"new_string"`
	}
	if json.Unmarshal([]byte(rawArgs), &args) != nil {
		return "", false
	}
	parsed := parseWorkflowResult(result.Content)
	switch name {
	case codetools.ToolRead:
		if parsed.failed || args.FilePath != f.TargetFile {
			return "", false
		}
		if state.edited {
			if !state.candidateKnown || parsed.content != state.candidate {
				return "", false
			}
			return "reread", true
		}
		state.inspected = parsed.content
		return "inspect", true
	case codetools.ToolGrep, codetools.ToolGlob:
		if !state.edited && !parsed.failed {
			return "inspect", true
		}
	case codetools.ToolWrite, codetools.ToolEdit, codetools.ToolApplyPatch:
		if !parsed.failed && (name == codetools.ToolApplyPatch || args.FilePath == f.TargetFile) {
			state.edited = true
			switch name {
			case codetools.ToolWrite:
				state.candidate, state.candidateKnown = args.Content, true
			case codetools.ToolEdit:
				if args.OldString != "" && strings.Count(state.inspected, args.OldString) == 1 {
					state.candidate = strings.Replace(state.inspected, args.OldString, args.NewString, 1)
					state.candidateKnown = true
				}
			}
			return "edit", true
		}
	case codetools.ToolBash:
		if args.Command != testCommand || parsed.exitCode == nil {
			return "", false
		}
		if *parsed.exitCode != 0 && !state.edited {
			return "test_fail", true
		}
		if *parsed.exitCode == 0 && state.edited {
			return "test_pass", true
		}
	}
	return "", false
}

type workflowResult struct {
	content  string
	exitCode *int
	failed   bool
}

func parseWorkflowResult(raw string) workflowResult {
	var value struct {
		Content  string          `json:"content"`
		ExitCode *int            `json:"exit_code"`
		Error    json.RawMessage `json:"error"`
		OK       *bool           `json:"ok"`
	}
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return workflowResult{failed: true}
	}
	failed := value.OK != nil && !*value.OK
	if len(value.Error) > 0 && string(value.Error) != "null" && string(value.Error) != `""` {
		failed = true
	}
	return workflowResult{content: value.Content, exitCode: value.ExitCode, failed: failed}
}

func orderedWorkflowSteps(observed, required []string) bool {
	next := 0
	for _, step := range observed {
		if next < len(required) && step == required[next] {
			next++
		}
	}
	return next == len(required)
}
