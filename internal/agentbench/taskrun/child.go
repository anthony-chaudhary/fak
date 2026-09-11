package taskrun

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type PlannerTurn struct {
	Messages  []agent.Message  `json:"messages"`
	ToolCalls []agent.ToolCall `json:"tool_calls,omitempty"`
	Content   string           `json:"content,omitempty"`
}

type ChildReceipt struct {
	Schema         string           `json:"schema"`
	Model          string           `json:"model"`
	PlannerCalls   int              `json:"planner_calls"`
	Inspected      bool             `json:"inspected"`
	Edited         bool             `json:"edited"`
	TestSucceeded  bool             `json:"test_succeeded"`
	DeniedAttempts int              `json:"denied_attempts"`
	Metrics        agent.ArmMetrics `json:"metrics"`
	Turns          []PlannerTurn    `json:"turns"`
	Workflow       WorkflowReceipt  `json:"workflow"`
	Error          string           `json:"error,omitempty"`
}

type recordingPlanner struct {
	inner agent.Planner
	turns []PlannerTurn
}

func (p *recordingPlanner) Model() string { return p.inner.Model() }
func (p *recordingPlanner) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if len(p.turns) >= maxTaskTurns {
		return nil, errors.New("agentbench planner call cap reached")
	}
	c, e := p.inner.Complete(ctx, msgs, tools, opts...)
	t := PlannerTurn{Messages: append([]agent.Message(nil), msgs...)}
	if c != nil {
		t.ToolCalls = append([]agent.ToolCall(nil), c.Message.ToolCalls...)
		t.Content = c.Message.Content
	}
	p.turns = append(p.turns, t)
	return c, e
}

func RunChild(ctx context.Context, configPath string, out io.Writer) error {
	var cfg childConfig
	if err := readConfig(configPath, &cfg); err != nil {
		return err
	}
	if cfg.Endpoint == "" || cfg.Model == "" || cfg.TrialRoot == "" || cfg.TestCommand == "" {
		return errors.New("task child: incomplete config")
	}
	catalog, err := agent.ArmCodeToolsWithOptions(agent.CodeToolsOptions{Root: cfg.TrialRoot, Focused: true, ExactCommandsOnly: true, ExactAllowedCommands: []string{cfg.TestCommand}})
	if err != nil {
		return err
	}
	defer agent.DisarmCodeTools()
	p := &recordingPlanner{inner: agent.NewHTTPPlanner(cfg.Endpoint, cfg.Model, "")}
	task := cfg.Prompt + "\n\nThe visible test below is immutable reference input outside the writable workspace:\n```go\n" + cfg.VisibleTest + "```\n\nAfter editing the target source, run the required test with the Bash tool using this byte-exact command:\n" + cfg.TestCommand
	metrics, runErr := agent.RunArm(ctx, p, task, true, maxTaskTurns, nil, agent.WithToolCatalog(catalog))
	r := ChildReceipt{Schema: "fak.agentbench.task-child.v1", Model: p.Model(), PlannerCalls: len(p.turns), Metrics: metrics, Turns: p.turns}
	for _, turn := range p.turns {
		for _, call := range turn.ToolCalls {
			switch call.Function.Name {
			case codetools.ToolRead, codetools.ToolGrep, codetools.ToolGlob:
				if followingToolSucceeded(p.turns, call.ID) {
					r.Inspected = true
				}
			case codetools.ToolWrite, codetools.ToolEdit, codetools.ToolApplyPatch:
				if followingToolSucceeded(p.turns, call.ID) {
					r.Edited = true
				}
			case codetools.ToolBash:
				if commandFromArgs(call.Function.Arguments) == cfg.TestCommand && followingToolSucceeded(p.turns, call.ID) {
					r.TestSucceeded = true
				}
			}
		}
	}
	r.DeniedAttempts = metrics.Denies
	workflow, workflowErr := validateWorkflow(taskfixtureByID(cfg.TaskID), p.turns, cfg.TestCommand)
	r.Workflow = workflow
	if runErr == nil && workflowErr != nil {
		runErr = workflowErr
	}
	if runErr != nil {
		r.Error = runErr.Error()
	}
	encErr := json.NewEncoder(out).Encode(r)
	if runErr != nil {
		return runErr
	}
	if encErr != nil {
		return encErr
	}
	return nil
}

func RunTestChild(ctx context.Context, configPath string, out io.Writer) error {
	var cfg testChildConfig
	if err := readConfig(configPath, &cfg); err != nil {
		return err
	}
	r, err := runCandidateTests(ctx, cfg.CandidateRoot, cfg.TargetFile, cfg.VisibleRoot, cfg.OracleRoot)
	if encErr := json.NewEncoder(out).Encode(r); encErr != nil {
		return encErr
	}
	if err != nil {
		return err
	}
	if r.ExitCode != 0 {
		return fmt.Errorf("sandboxed tests exited %d", r.ExitCode)
	}
	return nil
}

func followingToolSucceeded(turns []PlannerTurn, id string) bool {
	for _, t := range turns {
		for _, m := range t.Messages {
			if m.Role == agent.RoleTool && m.ToolCallID == id {
				return !strings.Contains(m.Content, `"error"`) && !strings.Contains(m.Content, "DENY") && !strings.Contains(m.Content, "FAIL")
			}
		}
	}
	return false
}

func commandFromArgs(raw string) string {
	var v struct {
		Command string `json:"command"`
	}
	_ = json.Unmarshal([]byte(raw), &v)
	return v.Command
}
func readConfig(path string, v any) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return errors.New("config must be a regular non-symlink file")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
