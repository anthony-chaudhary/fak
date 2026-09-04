package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAgentDefaultCodeToolsArmInNonFakWorkspace(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "benchmark.txt"), []byte("needle in non-fak repo\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.ArmFocusedCodeTools(root)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.DisarmCodeTools()
	want := map[string]bool{"Read": true, "Write": true, "Edit": true, "Bash": true, "Grep": true, "Glob": true, "skill": true}
	for _, def := range catalog {
		if !want[def.Function.Name] {
			t.Fatalf("unexpected or duplicate default code tool %q", def.Function.Name)
		}
		delete(want, def.Function.Name)
	}
	if len(want) != 0 {
		t.Fatalf("missing default code tools: %v", want)
	}
}

func TestAgentLaunchPostureReportsBoundedCodeToolsActive(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "agent", workspace: t.TempDir(), nativeCodeTools: true})
	if err != nil {
		t.Fatal(err)
	}
	got := postureByName(t, report, "bounded-code-tools")
	if got.State != "active" || got.Disable != "--code-tools=false" {
		t.Fatalf("agent code tools = %+v", got)
	}
}
