package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProjectClaudeHooksUseExecForm(t *testing.T) {
	path := filepath.Join(repoRoot(), ".claude", "settings.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	for event, groups := range settings.Hooks {
		for gi, group := range groups {
			for hi, hook := range group.Hooks {
				if hook.Type != "command" {
					continue
				}
				if len(hook.Args) == 0 {
					t.Fatalf("%s[%d].hooks[%d] uses shell-form command %q; use command+args exec form so Claude Code does not run hooks through Git Bash/profile startup output", event, gi, hi, hook.Command)
				}
				if strings.ContainsAny(hook.Command, " \t\r\n\"'") {
					t.Fatalf("%s[%d].hooks[%d] command %q is not an executable token; put flags and script text in args", event, gi, hi, hook.Command)
				}
			}
		}
	}
}
