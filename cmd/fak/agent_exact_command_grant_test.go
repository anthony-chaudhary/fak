package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type bashCommandGrantPlanner struct {
	commands        []string
	results         []string
	seenToolResults int
}

func (*bashCommandGrantPlanner) Model() string { return "bash-command-grant-test" }

func (p *bashCommandGrantPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	var toolResults []string
	for _, message := range messages {
		if message.Role == agent.RoleTool {
			toolResults = append(toolResults, message.Content)
		}
	}
	if len(toolResults) > p.seenToolResults {
		p.results = append(p.results, toolResults[p.seenToolResults:]...)
		p.seenToolResults = len(toolResults)
	}
	if len(p.results) == len(p.commands) {
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}}, nil
	}
	args, err := json.Marshal(codetools.BashArgs{Command: p.commands[len(p.results)]})
	if err != nil {
		return nil, err
	}
	return &agent.Completion{Message: agent.Message{
		Role: agent.RoleAssistant,
		ToolCalls: []agent.ToolCall{{
			ID: "bash-command-grant",
			Function: agent.Func{
				Name:      codetools.ToolBash,
				Arguments: string(args),
			},
		}},
	}}, nil
}

func runAgentBashCommandGrants(t *testing.T, args []string, commands ...string) []string {
	t.Helper()
	fs, af := newAgentFlagSet()
	if err := fs.Parse(args); err != nil {
		t.Fatalf("parse agent flags: %v", err)
	}
	if err := validateAgentBashCommandGrants(*af.codeTools, af.allowBashCommands); err != nil {
		t.Fatalf("validate agent flags: %v", err)
	}
	catalog, err := armAgentCodeTools(t.TempDir(), af)
	if err != nil {
		t.Fatalf("arm CLI code tools: %v", err)
	}
	t.Cleanup(agent.DisarmCodeTools)
	t.Cleanup(agent.Configure)
	var sawBash bool
	for _, def := range catalog {
		if def.Function.Name == codetools.ToolBash {
			sawBash = true
			break
		}
	}
	if !sawBash {
		t.Fatalf("CLI arm catalog omitted %s: %+v", codetools.ToolBash, catalog)
	}

	planner := &bashCommandGrantPlanner{commands: commands}
	metrics, err := agent.RunArm(context.Background(), planner, "exercise CLI-armed Bash", true, len(commands)+1, nil,
		agent.WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("run native tool arm: %v", err)
	}
	if len(planner.results) != len(commands) || metrics.FinalAnswer != "done" {
		t.Fatalf("native arm incomplete: metrics=%+v results=%q", metrics, planner.results)
	}
	return planner.results
}

// fak-test:runtime fast est=3s
func TestAgentExactCommandGrantReachesNativeBash(t *testing.T) {
	const granted = "go version"

	defaults := runAgentBashCommandGrants(t, nil, granted, "git status --short")
	if !strings.Contains(defaults[0], string(codetools.CodeCommandDeny)) {
		t.Fatalf("default agent unexpectedly admitted %q: %s", granted, defaults[0])
	}
	if strings.Contains(defaults[1], string(codetools.CodeCommandDeny)) {
		t.Fatalf("empty grants changed the focused defaults: %s", defaults[1])
	}

	grantedResults := runAgentBashCommandGrants(t,
		[]string{"--allow-bash-command", granted, "--allow-bash-command", "git status --short"},
		granted, granted+" && echo suffix", "go env GOPATH",
	)
	if strings.Contains(grantedResults[0], string(codetools.CodeCommandDeny)) || !strings.Contains(grantedResults[0], "go version") {
		t.Fatalf("exact CLI grant did not execute through native Bash: %s", grantedResults[0])
	}
	for i, result := range grantedResults[1:] {
		if !strings.Contains(result, string(codetools.CodeCommandDeny)) {
			t.Fatalf("ungranted command %d escaped the exact grant: %s", i, result)
		}
	}

	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{"--allow-bash-command", granted, "--allow-bash-command", "git status --short"}); err != nil {
		t.Fatal(err)
	}
	if got := []string(af.allowBashCommands); len(got) != 2 || got[0] != granted || got[1] != "git status --short" {
		t.Fatalf("repeatable grants = %q, want both values in order", got)
	}
}

// fak-test:runtime fast est=1s
func TestAgentBashCommandGrantValidation(t *testing.T) {
	var grants exactBashCommandList
	if err := grants.Set(" \t "); err == nil {
		t.Fatal("blank --allow-bash-command was accepted")
	}
	if err := validateAgentBashCommandGrants(false, []string{"go version"}); err == nil {
		t.Fatal("grant with --code-tools=false was accepted")
	}
}
