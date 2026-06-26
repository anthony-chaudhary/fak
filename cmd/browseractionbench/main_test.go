package main

import (
	"strings"
	"testing"
)

func TestBuildOfficialRunCommandRaw(t *testing.T) {
	cmd := buildOfficialRunCommand("webarena", "raw_agent_config", "gpt-4.1", "experiments/raw", "", 30)
	for _, want := range []string{
		"$env:BROWSERGYM_TASK_IDS",
		"$env:AGENTLAB_EXP_ROOT='experiments/raw'",
		"$env:FAK_BROWSERGYM_AGENT='raw_agent_config'",
		"$env:FAK_BROWSERGYM_MODEL='gpt-4.1'",
		"$env:FAK_BROWSERGYM_MAX_STEPS='30'",
		"make_study",
		"benchmark=''webarena''",
		"study.run(n_jobs=1)",
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
	cmd := buildOfficialRunCommand("webarena", "fak_agent_config", "gpt-4.1", "experiments/fak", "http://localhost:8080/v1", 20)
	for _, want := range []string{
		"$env:OPENAI_BASE_URL='http://localhost:8080/v1'",
		"$env:OPENAI_API_BASE='http://localhost:8080/v1'",
		"$env:FAK_BROWSERGYM_AGENT='fak_agent_config'",
		"$env:FAK_BROWSERGYM_MAX_STEPS='20'",
	} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command missing %q:\n%s", want, cmd)
		}
	}
}
