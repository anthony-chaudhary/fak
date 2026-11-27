package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
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
	// go version is in the focused default allowlist, so the probe must be a
	// command the defaults do not admit.
	const granted = "go env GOPATH"

	defaults := runAgentBashCommandGrants(t, nil, granted, "git status --short")
	if !strings.Contains(defaults[0], `"exit_code":0`) {
		t.Fatalf("default-open Bash did not execute %q: %s", granted, defaults[0])
	}
	if strings.Contains(defaults[1], string(codetools.CodeCommandDeny)) {
		t.Fatalf("default-open Bash refused a normal Git command: %s", defaults[1])
	}

	focused := runAgentBashCommandGrants(t, []string{"--bash-policy", "focused"}, granted)
	if !strings.Contains(focused[0], string(codetools.CodeCommandDeny)) {
		t.Fatalf("focused Bash admitted an unlisted command: %s", focused[0])
	}

	grantedResults := runAgentBashCommandGrants(t,
		[]string{"--allow-bash-command", granted, "--allow-bash-command", "git status --short"},
		granted, granted+" && echo suffix", "go env GOROOT",
	)
	if strings.Contains(grantedResults[0], string(codetools.CodeCommandDeny)) || !strings.Contains(grantedResults[0], `"exit_code":0`) {
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

// fak-test:runtime fast est=3s
func TestAgentDefaultOpenBashPreservesKernelDenial(t *testing.T) {
	const blocked = "rm -rf blocked-test-fixture"
	results := runAgentBashCommandGrants(t,
		[]string{"--bash-policy", "default_open", "--allow-bash-command", blocked},
		"go env GOPATH 2>&1", blocked, "git diff --output=result.txt",
	)
	if !strings.Contains(results[0], `"exit_code":0`) {
		t.Fatalf("default-open Bash refused harmless test-output redirection: %s", results[0])
	}
	if !strings.Contains(results[1], "POLICY_BLOCK") {
		t.Fatalf("kernel did not deny the prohibited command with an exact grant: %s", results[1])
	}
	if !strings.Contains(results[2], string(codetools.CodeCommandDeny)) {
		t.Fatalf("default-open Bash admitted an explicitly denied Git option: %s", results[2])
	}
}

// fak-test:runtime fast est=3s
func TestAgentDefaultOpenBashHonorsPolicyManifest(t *testing.T) {
	root := t.TempDir()
	policy := filepath.Join(root, "policy.json")
	if err := os.WriteFile(policy, []byte(`{"posture":"default_open","deny":{"Bash":"POLICY_BLOCK"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_PROFILE", "")
	t.Setenv(guardAllowOverlayEnv, filepath.Join(root, "no-allow-overlay.json"))
	t.Setenv(guardDenyOverlayEnv, filepath.Join(root, "no-deny-overlay.json"))
	t.Cleanup(agent.Configure)
	planner := &bashCommandGrantPlanner{commands: []string{"go env GOPATH"}}
	runAgentWithPlanner([]string{
		"--native", "--task", "run a local command", "--bash-policy", "default_open",
		"--policy", policy, "--code-workspace", root, "--out", filepath.Join(root, "report.json"),
		"--sys-tools=false", "--mcp-tools=false", "--subagents=false", "--skills=false", "--memory=false",
	}, planner)
	if len(planner.results) != 1 || !strings.Contains(planner.results[0], "POLICY_BLOCK") {
		t.Fatalf("explicit manifest denial did not reach native Bash: %q", planner.results)
	}
}

// fak-test:runtime fast est=1s
func TestAgentDefaultOpenBashModeSelection(t *testing.T) {
	_, af := newAgentFlagSet()
	previous := agent.ConfiguredPosture()
	t.Cleanup(func() { agent.SetConfiguredPosture(previous) })
	agent.SetConfiguredPosture(adjudicator.PostureFailClosed)
	if !af.focusedBash() {
		t.Fatal("auto should keep focused Bash under fail_closed")
	}
	agent.SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	*af.policyPath = "policy.json"
	if !af.focusedBash() {
		t.Fatal("auto should keep focused Bash when an explicit manifest was supplied")
	}
	if err := af.setBashPolicy(agentBashDefaultOpen); err != nil {
		t.Fatal(err)
	}
	if err := af.constrainBashPolicy(true); err != nil || af.focusedBash() {
		t.Fatalf("governed default-open override selected focused Bash: %v", err)
	}
	if err := af.constrainBashPolicy(false); err == nil {
		t.Fatal("explicit default-open Bash was accepted for an unmediated arm")
	}
	af.bashPolicy = ""
	if err := af.constrainBashPolicy(false); err != nil || !af.focusedBash() {
		t.Fatalf("unmediated mode failed to keep focused Bash: %v", err)
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
