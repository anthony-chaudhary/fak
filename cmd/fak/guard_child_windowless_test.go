package main

import (
	"os"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestLaunchGuardChildConsoleUsabilityScope pins both sides of the launch gate:
// every headless harness keeps #3597's no-window posture, while every attended
// harness retains a console capable of initializing a TUI. Before #9656 this test
// expected the hidden-console seam for attended Codex and checked only os.File
// pointer identity; the live child kept those pointers but emitted no terminal-ready
// byte for 40s because CREATE_NO_WINDOW had removed the usable console semantics.
//
// The seam is asserted through the injectable configureManagedHiddenConsole var rather
// than the platform SysProcAttr fields, so the gate is observable on the Linux CI host
// where the underlying ConfigureBackgroundCommand is a no-op.
func TestLaunchGuardChildConsoleUsabilityScope(t *testing.T) {
	orig := configureManagedHiddenConsole
	t.Cleanup(func() { configureManagedHiddenConsole = orig })

	meta := guardChildSpawnMetadata{
		AgentRunID:   "agent-run-1",
		ToolCallID:   "guard-child:agent-run-1",
		PolicyDigest: "sha256:policy",
		Backend:      "anthropic",
		Envelope:     toolprocgate.CapabilityEnvelope{Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn}},
	}
	launcher := func(g toolprocgate.SpawnGrant) (*exec.Cmd, error) {
		cmd := exec.Command(g.Argv[0], g.Argv[1:]...)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		return cmd, nil
	}

	cases := []struct {
		name    string
		command []string
		want    int // times the hidden-console seam is applied
	}{
		{"headless Claude one-shot", []string{"claude", "-p", "resolve #1"}, 1},
		{"headless Claude print-eq", []string{"claude", "--print=resolve #1"}, 1},
		{"headless Codex one-shot", []string{"codex", "-p", "resolve #1"}, 1},
		{"headless Codex exec", []string{"codex", "-c", "model_auto_compact_token_limit=96000", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"}, 1},
		{"attended Codex keeps usable console", []string{"codex"}, 0},
		{"attended Claude keeps usable console", []string{"claude"}, 0},
		{"attended git keeps usable console", []string{"git", "status"}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			applied := 0
			configureManagedHiddenConsole = func(cmd *exec.Cmd) {
				applied++
				orig(cmd) // still exercise the real (no-op off-Windows) configuration
			}
			broker := toolprocgate.NewSpawnBroker()
			_, child, err := launchGuardChildWithBroker(tc.command, [][2]string{{"OPENAI_BASE_URL", "http://gw/v1"}}, false, meta, broker, launcher)
			if err != nil {
				t.Fatalf("launchGuardChildWithBroker(%v): %v", tc.command, err)
			}
			if child == nil {
				t.Fatalf("launchGuardChildWithBroker(%v): nil child", tc.command)
			}
			if applied != tc.want {
				t.Fatalf("hidden-console seam applied %d time(s), want %d for %v", applied, tc.want, tc.command)
			}
		})
	}
}
