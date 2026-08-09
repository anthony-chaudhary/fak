package workflowlint

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCommentAndBlockScalarAreNotStructure(t *testing.T) {
	src := `name: test
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: |
          fakejob:
            needs: missing
      - run: echo ok # ] this comment is ignored
  test:
    needs: build # real edge survives trailing comment
    runs-on: ubuntu-latest
`
	got := Check("x.yml", src)
	if len(got) != 0 {
		t.Fatalf("findings = %+v", got)
	}
}

func TestStructuralAndDAGFailures(t *testing.T) {
	src := "jobs:\n  build:\n    runs-on: [ubuntu-latest\n  build:\n    needs: absent\n    bad:key\n\tsteps: []\n"
	got := Check("x.yml", src)
	want := []string{"unbalanced-delimiter", "duplicate-job", "unknown-needs", "missing-colon-space", "tab-indent"}
	for _, code := range want {
		found := false
		for _, f := range got {
			if f.Code == code {
				found = true
			}
		}
		if !found {
			t.Errorf("missing %s in %+v", code, got)
		}
	}
}

func TestWalkDiscoversEveryCurrentWorkflow(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	entries, err := os.ReadDir(filepath.Join(root, ".github", "workflows"))
	if err != nil {
		t.Fatal(err)
	}
	want := 0
	for _, e := range entries {
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if !e.IsDir() && (ext == ".yml" || ext == ".yaml") {
			want++
		}
	}
	seen := map[string]bool{}
	err = filepath.WalkDir(filepath.Join(root, ".github", "workflows"), func(p string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		ext := strings.ToLower(filepath.Ext(p))
		if !d.IsDir() && (ext == ".yml" || ext == ".yaml") {
			seen[p] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != want {
		t.Fatalf("walk found %d workflows, directory has %d", len(seen), want)
	}
}

func TestActionlintMissingIsCleanSkip(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	found, out, err := Actionlint(t.TempDir())
	if found || err != nil || len(out) != 0 {
		t.Fatalf("found=%v out=%q err=%v", found, out, err)
	}
}

func TestCurrentWorkflowsRatchet(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	got, err := CheckTree(root)
	if err != nil {
		t.Fatal(err)
	}
	// Named baseline: issue #5944 landed against this tree with zero structural findings.
	if len(got) > 0 {
		t.Fatalf("workflow structural ratchet: %d findings; first=%+v", len(got), got[0])
	}
}
