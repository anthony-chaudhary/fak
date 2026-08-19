package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestResolveNativeCodeWorkspaceDefaultsToNonFakRepository(t *testing.T) {
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "job.go"), []byte("package job\n\nconst DefaultOnSearchWitness = true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })

	workspace, err := resolveNativeCodeWorkspace(true, true, "")
	if err != nil {
		t.Fatalf("resolve default workspace: %v", err)
	}
	if workspace != repo {
		t.Fatalf("workspace = %q, want current non-fak repo %q", workspace, repo)
	}
	catalog, err := agent.ArmFocusedCodeTools(workspace)
	if err != nil {
		t.Fatalf("arm focused code tools: %v", err)
	}
	t.Cleanup(agent.DisarmCodeTools)
	var hasGrep bool
	for _, tool := range catalog {
		if tool.Function.Name == "Grep" {
			hasGrep = true
			break
		}
	}
	if !hasGrep {
		t.Fatalf("default catalog = %#v, want bounded Grep", catalog)
	}
}

func TestResolveNativeCodeWorkspaceControls(t *testing.T) {
	override := filepath.Join(t.TempDir(), "workspace")
	for _, tc := range []struct {
		name       string
		native     bool
		enabled    bool
		configured string
		want       string
	}{
		{name: "explicit override", native: true, enabled: true, configured: override, want: override},
		{name: "non-native stays off", native: false, enabled: true, configured: override},
		{name: "explicit opt-out", native: true, enabled: false, configured: override},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveNativeCodeWorkspace(tc.native, tc.enabled, tc.configured)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("workspace = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestNativeCodeToolsFlagDefaultsOn(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--native"}); err != nil {
		t.Fatal(err)
	}
	if !*sf.nativeCodeTools {
		t.Fatal("--native-code-tools defaulted off")
	}
	if err := fs.Parse([]string{"--native-code-tools=false"}); err != nil {
		t.Fatal(err)
	}
	if *sf.nativeCodeTools {
		t.Fatal("--native-code-tools=false did not opt out")
	}
}
