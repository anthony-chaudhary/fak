package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
)

func TestArmCodeToolsExactCommandsOnly(t *testing.T) {
	const exact = "printf agentbench-exact-command"

	runBash := func(t *testing.T, command string) ([]byte, bool) {
		t.Helper()
		ts := armedCodeTools.Load()
		if ts == nil {
			t.Fatal("expected armed code tools")
		}
		args, err := json.Marshal(codetools.BashArgs{Command: command})
		if err != nil {
			t.Fatal(err)
		}
		return ts.Bash(context.Background(), args)
	}
	commandDenied := func(t *testing.T, command string) {
		t.Helper()
		out, bad := runBash(t, command)
		if !bad {
			t.Fatalf("command %q executed instead of being denied: %s", command, out)
		}
		var refusal struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(out, &refusal); err != nil {
			t.Fatalf("decode refusal for %q: %v: %s", command, err, out)
		}
		if refusal.Error.Code != string(codetools.CodeCommandDeny) {
			t.Fatalf("command %q: want %s, got %s", command, codetools.CodeCommandDeny, out)
		}
	}

	t.Run("option reaches toolset", func(t *testing.T) {
		_, err := ArmCodeToolsWithOptions(CodeToolsOptions{
			Root:                 t.TempDir(),
			Focused:              true,
			ExactCommandsOnly:    true,
			ExactAllowedCommands: []string{exact},
		})
		if err != nil {
			t.Fatal(err)
		}
		defer DisarmCodeTools()

		if out, bad := runBash(t, exact); bad {
			t.Fatalf("exact configured command refused: %s", out)
		}
		commandDenied(t, "go test ./... -count=1")
	})

	t.Run("false preserves focused defaults", func(t *testing.T) {
		_, err := ArmCodeToolsWithOptions(CodeToolsOptions{
			Root:              t.TempDir(),
			Focused:           true,
			ExactCommandsOnly: false,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer DisarmCodeTools()

		if out, bad := runBash(t, "git status --short"); bad {
			t.Fatalf("zero-value exact-only mode denied an ordinary focused command: %s", out)
		}
	})
}
