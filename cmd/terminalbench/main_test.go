package main

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunCommandRaw(t *testing.T) {
	cmd := buildOfficialRunCommand("terminal-bench/terminal-bench-2-1", "", "codex", "gpt-5.5", "", "", 2, false)
	for _, want := range []string{
		"harbor run",
		"-d terminal-bench/terminal-bench-2-1",
		"-a codex",
		"-m gpt-5.5",
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
	cmd := buildOfficialRunCommand("terminal-bench/terminal-bench-2-1", "", "codex", "gpt-5.5", "http://host.docker.internal:18080/v1", "{{FAK_GATEWAY_KEY}}", 1, false)
	for _, want := range []string{
		"harbor run",
		"-a codex",
		"-m gpt-5.5",
		"--agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1",
		"--agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1",
		"--agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}'",
		"--allow-agent-host host.docker.internal",
		"--n-concurrent 1",
	} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command missing %q:\n%s", want, cmd)
		}
	}
	if strings.Contains(cmd, "--allow-environment-host") {
		t.Fatalf("environment host allowlist should be opt-in:\n%s", cmd)
	}
}

func TestBuildOfficialRunCommandOptionalEnvironmentHost(t *testing.T) {
	cmd := buildOfficialRunCommand("terminal-bench/terminal-bench-2-1", "", "codex", "gpt-5.5", "http://host.docker.internal:18080/v1", "{{FAK_GATEWAY_KEY}}", 1, true)
	if !strings.Contains(cmd, "--allow-environment-host host.docker.internal") {
		t.Fatalf("command missing optional environment allowlist:\n%s", cmd)
	}
}

func TestGatewayAllowHostFollowsGatewayURL(t *testing.T) {
	if got := gatewayAllowHost("http://gateway.internal:18080/v1"); got != "gateway.internal" {
		t.Fatalf("gateway host = %q", got)
	}
}
