package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestgateScopeRowTableCoversEveryGateFile(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	classified := map[string]bool{}
	for _, row := range gateScopes() {
		classified[row.File] = true
	}
	for name := range gateScopeFilesWithoutGates {
		classified[name] = true
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasPrefix(name, "gate_") || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if !classified[name] {
			t.Errorf("%s has no gate-scope classification", filepath.ToSlash(name))
		}
	}

	rows := map[string]bool{}
	for _, row := range gateScopes() {
		rows[row.Gate] = true
	}
	for _, gate := range PreCommitGates() {
		if !rows[gate.Name] {
			t.Errorf("pre-commit gate %s has no gate-scope classification", gate.Name)
		}
	}
}
