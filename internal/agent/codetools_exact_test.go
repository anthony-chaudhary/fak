package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
)

func TestArmCodeToolsExactAllowedCommands(t *testing.T) {
	root := t.TempDir()
	exactCmd := "powershell -NoProfile -Command Get-Date"

	catalog, err := ArmCodeToolsWithOptions(CodeToolsOptions{
		Root:                 root,
		Focused:              true,
		ExactAllowedCommands: []string{exactCmd},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer DisarmCodeTools()

	if len(catalog) == 0 {
		t.Fatal("expected non-empty tool catalog")
	}

	ts := armedCodeTools.Load()
	if ts == nil {
		t.Fatal("expected armedCodeTools to be set")
	}

	gotExact := ts.ExactAllowedCommands()
	if len(gotExact) != 1 || gotExact[0] != exactCmd {
		t.Fatalf("expected ExactAllowedCommands [%q], got %v", exactCmd, gotExact)
	}

	ctx := context.Background()

	// 1. Exact allowed command does not return CodeCommandDeny.
	argsBytes, _ := json.Marshal(codetools.BashArgs{Command: exactCmd})
	out, bad := ts.Bash(ctx, argsBytes)
	if bad {
		var r struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		_ = json.Unmarshal(out, &r)
		if r.Error.Code == string(codetools.CodeCommandDeny) {
			t.Fatalf("expected exact allowed command %q not to be denied with CodeCommandDeny, got: %s", exactCmd, out)
		}
	}

	// 2. Unlisted command is denied with CodeCommandDeny.
	unlistedArgs, _ := json.Marshal(codetools.BashArgs{Command: "powershell -NoProfile -Command Get-Process"})
	out, bad = ts.Bash(ctx, unlistedArgs)
	if !bad {
		t.Fatalf("expected unlisted command to be denied, but got bad=false")
	}
	var r struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(out, &r)
	if r.Error.Code != string(codetools.CodeCommandDeny) {
		t.Fatalf("expected CodeCommandDeny, got %s", out)
	}
}
