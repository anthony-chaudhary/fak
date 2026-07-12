package scdiff

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

func TestMatchGlob(t *testing.T) {
	cases := []struct {
		glob, path string
		want       bool
	}{
		// directory-prefix ("dir/") matches at any depth, and the bare dir itself.
		{"internal/uiquality/", "internal/uiquality/uiquality.go", true},
		{"internal/uiquality/", "internal/uiquality/sub/x.go", true},
		{"internal/uiquality/", "internal/uiquality", true},
		{"internal/uiquality/", "internal/uiqualityscore.go", false},
		{"internal/uiquality/", "cmd/fak/uiqualityscore.go", false},
		// segment-local "*" via path.Match (does not cross "/").
		{"tools/*.py", "tools/docs_scorecard.py", true},
		{"tools/*.py", "tools/sub/x.py", false},
		{"tools/*.py", "tools/docs_scorecard.go", false},
		// "**" crosses "/" boundaries.
		{"docs/**/*.md", "docs/a/b/c.md", true},
		{"docs/**/*.md", "docs/x.md", true},
		{"tools/**_test.py", "tools/code_slop_scorecard_test.py", true},
		{"tools/**_test.py", "tools/code_slop_scorecard.py", false},
		{"**/AGENTS.md", "internal/x/AGENTS.md", true},
		{"**/AGENTS.md", "AGENTS.md", true},
		// exact match.
		{"CLAUDE.md", "CLAUDE.md", true},
		{"CLAUDE.md", "AGENTS.md", false},
		// normalization: backslashes and "./" are canonicalized on both sides.
		{`internal\uiquality\`, "internal/uiquality/x.go", true},
		{"tools/*.py", "./tools/docs_scorecard.py", true},
		// empty glob never matches.
		{"", "anything", false},
	}
	for _, c := range cases {
		if got := MatchGlob(c.glob, c.path); got != c.want {
			t.Errorf("MatchGlob(%q, %q) = %v, want %v", c.glob, c.path, got, c.want)
		}
	}
}

func TestIntersect(t *testing.T) {
	corpus := []string{"a/x.go", "a/y.go", "b/z.go"}
	changed := []string{"b/z.go", "c/w.go", `a\x.go`} // backslash form still matches
	got := Intersect(corpus, changed)
	want := []string{"a/x.go", "b/z.go"} // preserves corpus order
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Intersect = %v, want %v", got, want)
	}
	if Intersect(nil, changed) != nil {
		t.Error("Intersect(nil, ...) should be nil")
	}
	if Intersect(corpus, nil) != nil {
		t.Error("Intersect(..., nil) should be nil")
	}
}

func TestFilter(t *testing.T) {
	changed := []string{"tools/docs_scorecard.py", "cmd/fak/main.go", "docs/a/b.md"}
	got := Filter(changed, []string{"tools/*.py", "docs/**/*.md"})
	want := []string{"tools/docs_scorecard.py", "docs/a/b.md"} // preserves changed order
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Filter = %v, want %v", got, want)
	}
	if Filter(changed, nil) != nil {
		t.Error("Filter(..., nil) should be nil")
	}
}

func TestChangedPathsBlankSince(t *testing.T) {
	got, err := ChangedPaths(".", "  ")
	if err != nil || got != nil {
		t.Errorf("ChangedPaths(blank) = (%v, %v), want (nil, nil)", got, err)
	}
}

// TestChangedPathsHermeticRepo drives ChangedPaths against a throwaway git repo so
// the tracked-diff + untracked union is verified end-to-end without depending on
// the state of the surrounding checkout.
func TestChangedPathsHermeticRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	dir := t.TempDir()
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	run("init", "-q")
	write("keep.go", "package a\n")
	write("edit.go", "package a\n")
	run("add", "-A")
	run("commit", "-qm", "base")
	base := "HEAD"

	// One tracked edit, one brand-new untracked file; keep.go is unchanged.
	write("edit.go", "package a\n// changed\n")
	write("new/added.go", "package b\n")

	got, err := ChangedPaths(dir, base)
	if err != nil {
		t.Fatalf("ChangedPaths: %v", err)
	}
	want := []string{"edit.go", "new/added.go"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ChangedPaths = %v, want %v", got, want)
	}
}

func TestChangedPathsBadRef(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	dir := t.TempDir()
	cmd := exec.Command("git", "-C", dir, "init", "-q")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	// An unresolvable ref must surface as an error (so the caller full-scans),
	// never as a silent empty "nothing changed".
	if _, err := ChangedPaths(dir, "nope-not-a-ref"); err == nil {
		t.Error("ChangedPaths(bad ref) = nil error, want error")
	}
}
