package main

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunCommandRaw(t *testing.T) {
	cmd := buildOfficialRunCommand("terminal-bench-core", "0.1.1", "terminus", "gpt-4.1", "", 2)
	for _, want := range []string{
		"tb run",
		"--dataset terminal-bench-core==0.1.1",
		"--agent terminus",
		"--model gpt-4.1",
		"--task-id $task",
		"--n-concurrent 2",
		"$env:TERMINAL_BENCH_TASK_IDS",
	} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command missing %q:\n%s", want, cmd)
		}
	}
	if strings.Contains(cmd, "OPENAI_BASE_URL") {
		t.Fatalf("raw command should not force a fak gateway:\n%s", cmd)
	}
}

func TestBuildOfficialRunCommandFakGateway(t *testing.T) {
	cmd := buildOfficialRunCommand("terminal-bench-core", "0.1.1", "terminus-through-fak", "gpt-4.1", "http://localhost:8080/v1", 1)
	for _, want := range []string{
		"$env:OPENAI_BASE_URL='http://localhost:8080/v1'",
		"$env:OPENAI_API_BASE='http://localhost:8080/v1'",
		"--agent terminus-through-fak",
		"--n-concurrent 1",
	} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command missing %q:\n%s", want, cmd)
		}
	}
}
