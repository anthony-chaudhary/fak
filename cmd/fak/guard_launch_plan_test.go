package main

import "testing"

func TestGuardLaunchPlanInteractive(t *testing.T) {
	tests := []struct {
		name        string
		command     []string
		interactive bool
	}{
		{name: "bare Codex", command: []string{"codex"}, interactive: true},
		{name: "Codex exec", command: []string{"codex", "exec", "resolve #1"}},
		{name: "Codex exec after config", command: []string{"codex", "-c", "model_auto_compact_token_limit=96000", "exec", "--skip-git-repo-check"}},
		{name: "Codex exec after long equals option", command: []string{"codex", "--model=gpt-5", "exec", "resolve #1"}},
		{name: "Claude short print", command: []string{"claude", "-p", "resolve #1"}},
		{name: "Claude long print", command: []string{"claude", "--print", "resolve #1"}},
		{name: "unrelated exec argument", command: []string{"custom-agent", "exec"}, interactive: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := newGuardLaunchPlan(test.command).interactive(); got != test.interactive {
				t.Fatalf("interactive() = %v, want %v for %v", got, test.interactive, test.command)
			}
		})
	}
}
