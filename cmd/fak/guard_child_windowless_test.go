package main

import (
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestLaunchGuardChildWindowlessOnlyHeadless pins the #3597 gate: a headless
// (dispatched, non-attended) wrapped agent is launched windowless so no per-worker
// conhost/OpenConsole pane is allocated, while an attended/interactive session is
// byte-for-byte unchanged (the window-mode seam is NOT applied on that branch).
//
// The seam is asserted through the injectable guardHeadlessChildWindowMode var rather
// than the platform SysProcAttr fields, so the gate is observable on the Linux CI host
// where the underlying ConfigureBackgroundCommand is a no-op.
func TestLaunchGuardChildWindowlessOnlyHeadless(t *testing.T) {
	orig := guardHeadlessChildWindowMode
	t.Cleanup(func() { guardHeadlessChildWindowMode = orig })

	meta := guardChildSpawnMetadata{
		AgentRunID:   "agent-run-1",
		ToolCallID:   "guard-child:agent-run-1",
		PolicyDigest: "sha256:policy",
		Backend:      "anthropic",
		Envelope:     toolprocgate.CapabilityEnvelope{Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn}},
	}
	launcher := func(g toolprocgate.SpawnGrant) (*exec.Cmd, error) {
		return exec.Command(g.Argv[0], g.Argv[1:]...), nil
	}

	cases := []struct {
		name    string
		command []string
		want    int // times the windowless seam is applied
	}{
		{"headless one-shot", []string{"claude", "-p", "resolve #1"}, 1},
		{"headless print-eq", []string{"claude", "--print=resolve #1"}, 1},
		{"attended interactive", []string{"claude"}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			applied := 0
			guardHeadlessChildWindowMode = func(cmd *exec.Cmd) {
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
				t.Fatalf("windowless seam applied %d time(s), want %d for %v", applied, tc.want, tc.command)
			}
		})
	}
}
