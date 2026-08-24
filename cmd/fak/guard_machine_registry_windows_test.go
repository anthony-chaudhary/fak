package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

func TestRecordInteractiveSessionRowsMirrorsMachineRegistry(t *testing.T) {
	user := t.TempDir()
	machineRoot := t.TempDir()
	t.Setenv("FLEET_REG_DIR", user)
	t.Setenv("ProgramData", machineRoot)
	machine := filepath.Join(machineRoot, "fak", "guard-control", "registry")
	if err := os.MkdirAll(machine, 0o755); err != nil {
		t.Fatal(err)
	}
	row := guardsessions.NewInteractiveRow("trace", "claude", 1, t.TempDir(), "", "", time.Now(), []string{"claude"}, false)
	if err := recordInteractiveSessionRows(row); err != nil {
		t.Fatal(err)
	}
	if got := guardsessions.LiveInteractive(guardsessions.Load(user)); len(got) != 1 {
		t.Fatalf("user rows=%+v", got)
	}
	if got := guardsessions.LiveInteractive(guardsessions.Load(machine)); len(got) != 1 {
		t.Fatalf("machine rows=%+v", got)
	}
}
