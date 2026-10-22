package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// fak-test:runtime integration est=5s lane=default
func TestCommitBuildCheckReleasesProspectiveSnapshotBeforeValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if runtime.GOOS == "windows" {
		t.Skip("native snapshot lifecycle; Windows validation uses a separate WSL filesystem")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 1 }\n")
	writeBuildCheckFile(t, repo, "p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() != 1 {\n\t\tt.Fatal(\"lifecycle gate witness\")\n\t}\n}\n")
	commitBuildCheckPlumbing(t, repo, git, "seed lifecycle fixture", "go.mod", "p/p.go", "p/p_test.go")
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 2 }\n")
	head, index := buildCheckGitState(t, git)
	tempRoot := t.TempDir()
	t.Setenv("TMPDIR", tempRoot)
	t.Setenv("GOWORK", "off")
	previous := validatePhaseHook
	t.Cleanup(func() { validatePhaseHook = previous })
	sawExtract, sawOverlay := false, false
	validatePhaseHook = func(_ context.Context, phase string) {
		want := 0
		switch phase {
		case "extract_tip":
			sawExtract = true
		case "overlay":
			sawOverlay = true
			want = 1
		default:
			return
		}
		trees, err := filepath.Glob(filepath.Join(tempRoot, "fak-committed-tree-*"))
		if err != nil || len(trees) != want {
			t.Fatalf("phase %s: active snapshots = %v, error = %v; want %d", phase, trees, err, want)
		}
	}
	var stderr bytes.Buffer
	outcome, detail := runCommitBuildCheck(&stderr, repo, []string{"p/p.go"}, defaultValidateTimeout)
	requireProspectiveRefusal(t, outcome, detail, "test", "lifecycle gate witness")
	if !sawExtract || !sawOverlay {
		t.Fatalf("outcome=%s detail=%q extract=%v overlay=%v stderr=%s", outcome, detail, sawExtract, sawOverlay, stderr.String())
	}
	entries, err := os.ReadDir(tempRoot)
	if err != nil || len(entries) != 0 {
		t.Fatalf("snapshot cleanup left entries=%v, error=%v", entries, err)
	}
	requireBuildCheckGitState(t, git, head, index)
}
