package codetools

import (
	"context"
	"testing"
)

func TestExactCommandsOnly(t *testing.T) {
	const exact = "printf agentbench-exact-command"

	commandDenied := func(t *testing.T, ts *Toolset, command string) {
		t.Helper()
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: command}))
		if !bad || errCode(t, out) != CodeCommandDeny {
			t.Fatalf("command %q: want %s before execution, got bad=%v out=%s", command, CodeCommandDeny, bad, out)
		}
	}

	t.Run("admits only exact configured command", func(t *testing.T) {
		ts, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      true,
			ExactCommandsOnly:    true,
			ExactAllowedCommands: []string{exact},
		})
		if err != nil {
			t.Fatal(err)
		}

		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: exact}))
		if bad {
			t.Fatalf("exact configured command refused: %s", out)
		}
		for _, command := range []string{
			"go test ./... -count=1",
			"git status --short",
			exact + " ",
			exact + "; printf bypass",
			exact + " && printf bypass",
		} {
			commandDenied(t, ts, command)
		}
	})

	t.Run("empty exact allowlist denies", func(t *testing.T) {
		ts, err := New(Config{Root: t.TempDir(), FocusedCommands: true, ExactCommandsOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		commandDenied(t, ts, "go test ./... -count=1")
	})

	t.Run("zero value retains focused defaults", func(t *testing.T) {
		ts, err := New(Config{Root: t.TempDir(), FocusedCommands: true})
		if err != nil {
			t.Fatal(err)
		}
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "git status --short"}))
		if bad && errCode(t, out) == CodeCommandDeny {
			t.Fatalf("zero-value ExactCommandsOnly changed focused defaults: %s", out)
		}
	})
}
