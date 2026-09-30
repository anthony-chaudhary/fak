package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type agentBashTimeoutPlanner struct {
	command string
	result  string
}

func (*agentBashTimeoutPlanner) Model() string { return "agent-bash-timeout-test" }

func (p *agentBashTimeoutPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	for _, message := range messages {
		if message.Role == agent.RoleTool {
			p.result = message.Content
		}
	}
	if p.result != "" {
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}}, nil
	}
	args, err := json.Marshal(codetools.BashArgs{Command: p.command})
	if err != nil {
		return nil, err
	}
	return &agent.Completion{Message: agent.Message{
		Role: agent.RoleAssistant,
		ToolCalls: []agent.ToolCall{{
			ID: "bash-timeout",
			Function: agent.Func{
				Name:      codetools.ToolBash,
				Arguments: string(args),
			},
		}},
	}}, nil
}

// fak-test:runtime fast est=1s
func TestAgentBashCommandTimeoutDefaultsAndBounds(t *testing.T) {
	fs, af := newAgentFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if *af.bashCommandTimeout != 2*time.Minute || af.bashCommandTimeoutSet {
		t.Fatalf("default timeout = %v explicit=%v, want 2m/false", *af.bashCommandTimeout, af.bashCommandTimeoutSet)
	}

	fs, af = newAgentFlagSet()
	if err := fs.Parse([]string{"--bash-command-timeout=10m"}); err != nil {
		t.Fatalf("parse maximum timeout: %v", err)
	}
	if *af.bashCommandTimeout != 10*time.Minute {
		t.Fatalf("maximum timeout = %v, want 10m", *af.bashCommandTimeout)
	}
	if _, err := armAgentCodeTools(t.TempDir(), af); err != nil {
		t.Fatalf("arm maximum timeout: %v", err)
	}
	t.Cleanup(agent.DisarmCodeTools)
}

// fak-test:runtime fast est=3s
func TestAgentBashCommandTimeoutRejections(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
	}{
		{name: "zero", args: []string{"--bash-command-timeout=0s"}},
		{name: "negative", args: []string{"--bash-command-timeout=-1s"}},
		{name: "above maximum", args: []string{"--bash-command-timeout=10m1ms"}},
		{name: "disabled tools", args: []string{"--code-tools=false", "--bash-command-timeout=3m"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := runAgentBashTimeoutCLI(t, tc.args...)
			if err == nil {
				t.Fatalf("invalid argv succeeded: %v\n%s", tc.args, out)
			}
			if !strings.Contains(out, "bash-command-timeout") {
				t.Fatalf("rejection omitted actionable flag name: %s", out)
			}
		})
	}
}

// fak-test:runtime fast est=3s
func TestAgentBashCommandTimeoutReachesNativeTool(t *testing.T) {
	const bound = 20 * time.Millisecond
	command := quoteAgentBashTimeoutCommand(os.Args[0]) + " -test.run=TestAgentBashTimeoutChild"
	t.Setenv("FAK_AGENT_BASH_TIMEOUT_CHILD", "1")

	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{
		"--bash-command-timeout=" + bound.String(),
		"--allow-bash-command", command,
	}); err != nil {
		t.Fatal(err)
	}
	catalog, err := armAgentCodeTools(t.TempDir(), af)
	if err != nil {
		t.Fatalf("arm CLI code tools: %v", err)
	}
	t.Cleanup(agent.DisarmCodeTools)
	t.Cleanup(agent.Configure)

	planner := &agentBashTimeoutPlanner{command: command}
	started := time.Now()
	_, err = agent.RunArm(context.Background(), planner, "exercise CLI Bash timeout", true, 2, nil,
		agent.WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("run native tool arm: %v", err)
	}
	if !strings.Contains(planner.result, `"timed_out":true`) {
		t.Fatalf("CLI timeout did not reach native Bash: %s", planner.result)
	}
	if elapsed := time.Since(started); elapsed >= 2*time.Second {
		t.Fatalf("20ms Bash bound took %v", elapsed)
	}
}

func quoteAgentBashTimeoutCommand(path string) string {
	if runtime.GOOS == "windows" {
		return `"` + strings.ReplaceAll(path, `"`, `""`) + `"`
	}
	return "'" + strings.ReplaceAll(path, "'", "'\"'\"'") + "'"
}

func runAgentBashTimeoutCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	childArgs := []string{"-test.run=TestAgentBashTimeoutCLIHelper", "--"}
	childArgs = append(childArgs, args...)
	cmd := exec.Command(os.Args[0], childArgs...)
	cmd.Env = append(os.Environ(), "FAK_AGENT_BASH_TIMEOUT_CLI_HELPER=1")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// fak-test:runtime fast est=1s
func TestAgentBashTimeoutCLIHelper(t *testing.T) {
	if os.Getenv("FAK_AGENT_BASH_TIMEOUT_CLI_HELPER") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			cmdAgent(os.Args[i+1:])
			return
		}
	}
	fmt.Fprintln(os.Stderr, "missing helper argv separator")
	os.Exit(2)
}

// fak-test:runtime fast est=1s
func TestAgentBashTimeoutChild(t *testing.T) {
	if os.Getenv("FAK_AGENT_BASH_TIMEOUT_CHILD") == "1" {
		time.Sleep(250 * time.Millisecond)
	}
}
