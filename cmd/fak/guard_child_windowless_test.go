package main

import (
	"os"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestLaunchGuardChildHiddenConsoleScope pins both sides of the launch gate:
// headless harnesses keep #3597's no-window posture, attended Codex receives the
// #8853 hidden inherited-console posture, and unrelated attended commands remain
// untouched.
//
// The seam is asserted through the injectable configureManagedHiddenConsole var rather
// than the platform SysProcAttr fields, so the gate is observable on the Linux CI host
// where the underlying ConfigureBackgroundCommand is a no-op.
func TestLaunchGuardChildHiddenConsoleScope(t *testing.T) {
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
		name           string
		command        []string
		want           int // times the hidden-console seam is applied
		wantTerminalIO bool
	}{
		{"headless one-shot", []string{"claude", "-p", "resolve #1"}, 1, false},
		{"headless print-eq", []string{"claude", "--print=resolve #1"}, 1, false},
		{"attended managed Codex", []string{"codex"}, 1, true},
		{"attended Claude unchanged", []string{"claude"}, 0, false},
		{"attended git unchanged", []string{"git", "status"}, 0, false},
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
			if tc.wantTerminalIO &&
				(child.Stdin != os.Stdin || child.Stdout != os.Stdout || child.Stderr != os.Stderr) {
				t.Fatalf("attended Codex terminal handles changed: stdin=%v stdout=%v stderr=%v",
					child.Stdin, child.Stdout, child.Stderr)
			}
		})
	}
}
