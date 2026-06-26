package main

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunCommandTau3(t *testing.T) {
	cmd := buildOfficialRunCommand("tau3", "retail", "gpt-4.1", "gpt-4.1", "http://localhost:8080/v1", 2)
	for _, want := range []string{
		"tau2 run",
		"--domain retail",
		"--agent-llm gpt-4.1",
		"--user-llm gpt-4.1",
		"--num-trials 2",
		"$env:TAU3_TASK_IDS",
		"$env:OPENAI_BASE_URL='http://localhost:8080/v1'",
	} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command missing %q:\n%s", want, cmd)
		}
	}
}

func TestBuildOfficialRunCommandToolSandbox(t *testing.T) {
	cmd := buildOfficialRunCommand("toolsandbox", "mobile", "GPT_4_o_2024_05_13", "GPT_4_o_2024_05_13", "", 1)
	for _, want := range []string{"tool_sandbox", "--scenario $env:TOOLSANDBOX_SCENARIO", "--agent GPT_4_o_2024_05_13"} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command missing %q:\n%s", want, cmd)
		}
	}
	if strings.Contains(cmd, "OPENAI_BASE_URL") {
		t.Fatalf("raw ToolSandbox command should not force a fak gateway:\n%s", cmd)
	}
}
