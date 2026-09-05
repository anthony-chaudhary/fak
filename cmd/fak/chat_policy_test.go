package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestChatDefaultDeveloperPolicyFloor(t *testing.T) {
	t.Cleanup(func() {
		agent.SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	})

	ctx := context.Background()

	t.Run("DefaultDeveloperCapabilityFloorEnforcesFailClosedAndAllowedTools", func(t *testing.T) {
		initDevRules("")

		if got := agent.ConfiguredPosture(); got != adjudicator.PostureFailClosed {
			t.Fatalf("agent.ConfiguredPosture() = %v, want PostureFailClosed", got)
		}
		if got := adjudicator.Default.PolicySnapshot().Posture; got != adjudicator.PostureFailClosed {
			t.Fatalf("adjudicator.Default Posture = %v, want PostureFailClosed", got)
		}

		// Allowed standard developer tools:
		// Read, Write in-tree, Glob, Grep, Bash with git status or go test
		allowedCases := []struct {
			name string
			tool string
			args map[string]any
		}{
			{"Read allowed", "Read", map[string]any{"file_path": "README.md"}},
			{"Write in-tree allowed", "Write", map[string]any{"file_path": "notes.txt", "content": "hello"}},
			{"Glob allowed", "Glob", map[string]any{"pattern": "*.go"}},
			{"Grep allowed", "Grep", map[string]any{"pattern": "TODO"}},
			{"Bash git status allowed", "Bash", map[string]any{"command": "git status"}},
			{"Bash go test allowed", "Bash", map[string]any{"command": "go test ./..."}},
		}

		for _, tc := range allowedCases {
			t.Run(tc.name, func(t *testing.T) {
				call := guardToolCall(t, tc.tool, tc.args)
				v := adjudicator.Default.Adjudicate(ctx, call)
				if v.Kind == abi.VerdictDeny {
					t.Fatalf("%s (%s): got Kind=VerdictDeny (reason=%s), want allowed", tc.name, tc.tool, abi.ReasonName(v.Reason))
				}
			})
		}

		// Destructive commands: rm -rf, git push, .git/config write
		destructiveCases := []struct {
			name       string
			tool       string
			args       map[string]any
			wantReason abi.ReasonCode
		}{
			{"Bash rm -rf denied", "Bash", map[string]any{"command": "rm -rf /tmp/x"}, abi.ReasonPolicyBlock},
			{"Bash git push denied", "Bash", map[string]any{"command": "git push origin main"}, abi.ReasonPolicyBlock},
			{"PowerShell git push denied", "PowerShell", map[string]any{"command": "git push"}, abi.ReasonPolicyBlock},
			{"shell_command git push denied", "shell_command", map[string]any{"command": "git push"}, abi.ReasonPolicyBlock},
			{"Write .git/config denied", "Write", map[string]any{"file_path": ".git/config", "content": "evil"}, abi.ReasonSelfModify},
		}

		for _, tc := range destructiveCases {
			t.Run(tc.name, func(t *testing.T) {
				call := guardToolCall(t, tc.tool, tc.args)
				v := adjudicator.Default.Adjudicate(ctx, call)
				if v.Kind != abi.VerdictDeny {
					t.Fatalf("%s (%s): got Kind=%v, want VerdictDeny", tc.name, tc.tool, v.Kind)
				}
				if v.Reason != tc.wantReason {
					t.Fatalf("%s (%s): got Reason=%s, want %s", tc.name, tc.tool, abi.ReasonName(v.Reason), abi.ReasonName(tc.wantReason))
				}
			})
		}

		// Unlisted unknown tools are denied under default fail_closed
		unlistedCall := guardToolCall(t, "arbitrary_exec", map[string]any{})
		vUnlisted := adjudicator.Default.Adjudicate(ctx, unlistedCall)
		if vUnlisted.Kind != abi.VerdictDeny {
			t.Fatalf("unlisted tool arbitrary_exec: got Kind=%v, want VerdictDeny", vUnlisted.Kind)
		}
		if vUnlisted.Reason != abi.ReasonDefaultDeny {
			t.Fatalf("unlisted tool arbitrary_exec: got Reason=%s, want ReasonDefaultDeny", abi.ReasonName(vUnlisted.Reason))
		}
	})

	t.Run("DefaultOpenPermitsUnlistedBenignTools", func(t *testing.T) {
		initDevRules("default_open")

		if got := agent.ConfiguredPosture(); got != adjudicator.PostureDefaultOpen {
			t.Fatalf("agent.ConfiguredPosture() = %v, want PostureDefaultOpen", got)
		}
		if got := adjudicator.Default.PolicySnapshot().Posture; got != adjudicator.PostureDefaultOpen {
			t.Fatalf("adjudicator.Default Posture = %v, want PostureDefaultOpen", got)
		}

		// Unlisted benign tool should now be permitted
		unlistedCall := guardToolCall(t, "arbitrary_exec", map[string]any{})
		vUnlisted := adjudicator.Default.Adjudicate(ctx, unlistedCall)
		if vUnlisted.Kind != abi.VerdictAllow {
			t.Fatalf("unlisted tool arbitrary_exec under default_open: got Kind=%v, want VerdictAllow", vUnlisted.Kind)
		}

		// Destructive commands still denied under default_open
		bashPush := guardToolCall(t, "Bash", map[string]any{"command": "git push"})
		if v := adjudicator.Default.Adjudicate(ctx, bashPush); v.Kind != abi.VerdictDeny {
			t.Fatalf("git push under default_open: got Kind=%v, want VerdictDeny", v.Kind)
		}

		bashRm := guardToolCall(t, "Bash", map[string]any{"command": "rm -rf /tmp/x"})
		if v := adjudicator.Default.Adjudicate(ctx, bashRm); v.Kind != abi.VerdictDeny {
			t.Fatalf("rm -rf under default_open: got Kind=%v, want VerdictDeny", v.Kind)
		}
	})

	t.Run("CmdChatPostureResolution", func(t *testing.T) {
		t.Setenv("FAK_AGENT_POSTURE", "")
		t.Setenv("FAK_GUARD_POSTURE", "")

		captureAgentStdio(t, func() {
			cmdChat([]string{"--offline", "--task", "hello", "--tools=none"})
		})
		if got := agent.ConfiguredPosture(); got != adjudicator.PostureFailClosed {
			t.Fatalf("cmdChat default posture: got %v, want PostureFailClosed", got)
		}

		captureAgentStdio(t, func() {
			cmdChat([]string{"--offline", "--task", "hello", "--tools=none", "--posture", "default_open"})
		})
		if got := agent.ConfiguredPosture(); got != adjudicator.PostureDefaultOpen {
			t.Fatalf("cmdChat --posture default_open: got %v, want PostureDefaultOpen", got)
		}

		t.Setenv("FAK_AGENT_POSTURE", "fail_closed")
		captureAgentStdio(t, func() {
			cmdChat([]string{"--offline", "--task", "hello", "--tools=none"})
		})
		if got := agent.ConfiguredPosture(); got != adjudicator.PostureFailClosed {
			t.Fatalf("cmdChat FAK_AGENT_POSTURE=fail_closed: got %v, want PostureFailClosed", got)
		}

		t.Setenv("FAK_AGENT_POSTURE", "")
		t.Setenv("FAK_GUARD_POSTURE", "default_open")
		captureAgentStdio(t, func() {
			cmdChat([]string{"--offline", "--task", "hello", "--tools=none"})
		})
		if got := agent.ConfiguredPosture(); got != adjudicator.PostureDefaultOpen {
			t.Fatalf("cmdChat FAK_GUARD_POSTURE=default_open: got %v, want PostureDefaultOpen", got)
		}
	})
}
