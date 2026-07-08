package safecommit

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// classifyFake is a Runner that answers `rev-parse --git-dir` plus the two `git ls-files`
// probes from a per-pathspec index/worktree truth table, so ClassifyPaths runs against no
// real git. It records every argv so a test can assert the exact probes issued.
type classifyFake struct {
	notRepo   bool
	tracked   map[string]bool // pathspec listed by `git ls-files --cached -- P`
	untracked map[string]bool // pathspec listed by `git ls-files --others --exclude-standard -- P`
	execErr   error
	calls     [][]string
}

func (f *classifyFake) run(_ context.Context, _ string, args ...string) (string, int, error) {
	if f.execErr != nil {
		return "", -1, f.execErr
	}
	f.calls = append(f.calls, append([]string(nil), args...))
	if len(args) == 0 {
		return "", 0, nil
	}
	switch args[0] {
	case "rev-parse":
		if f.notRepo {
			return "fatal: not a git repository", 128, nil
		}
		return ".git\n", 0, nil
	case "ls-files":
		p := args[len(args)-1] // the pathspec after "--"
		set := f.tracked
		for _, a := range args {
			if a == "--others" {
				set = f.untracked
			}
		}
		if set[p] {
			return p + "\x00", 0, nil // NUL-delimited, as -z emits
		}
		return "", 0, nil
	}
	return "", 0, nil
}

// TestClassifyPaths_allTrackedPassesClean is the DoD's clean witness: every requested pathspec
// is tracked, so the report is OK with no reason and every class is tracked.
func TestClassifyPaths_allTrackedPassesClean(t *testing.T) {
	g := &classifyFake{tracked: map[string]bool{"a.go": true, "internal/foo/b.go": true}}
	rep, err := ClassifyPaths(context.Background(), g.run, "/repo", []string{"a.go", "internal/foo/b.go"})
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if !rep.OK || rep.Reason != "" {
		t.Fatalf("all-tracked must pass clean, got OK=%v reason=%q detail=%q", rep.OK, rep.Reason, rep.Detail)
	}
	if len(rep.Classes) != 2 {
		t.Fatalf("want 2 classes, got %+v", rep.Classes)
	}
	for _, c := range rep.Classes {
		if c.State != PathTracked {
			t.Fatalf("path %q should be tracked, got %q", c.Path, c.State)
		}
		if c.Fix != "" {
			t.Fatalf("a tracked path carries no fix, got %q", c.Fix)
		}
	}
	if len(rep.Untracked) != 0 || len(rep.Unmatched) != 0 {
		t.Fatalf("clean report must have empty untracked/unmatched, got %+v / %+v", rep.Untracked, rep.Unmatched)
	}
}

// TestClassifyPaths_untrackedNamesGitAddFix is the DoD's refusal witness: a known-untracked
// path yields a structured refusal that names the path and the `git add` fix.
func TestClassifyPaths_untrackedNamesGitAddFix(t *testing.T) {
	g := &classifyFake{untracked: map[string]bool{"cmd/fak/new.go": true}}
	rep, err := ClassifyPaths(context.Background(), g.run, "/repo", []string{"cmd/fak/new.go"})
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if rep.OK {
		t.Fatalf("an untracked path must refuse, got %+v", rep)
	}
	if rep.Reason != ReasonPathUntracked {
		t.Fatalf("want reason %q, got %q", ReasonPathUntracked, rep.Reason)
	}
	if len(rep.Classes) != 1 || rep.Classes[0].State != PathUntracked {
		t.Fatalf("want a single untracked class, got %+v", rep.Classes)
	}
	fix := rep.Classes[0].Fix
	if !strings.Contains(fix, "git add") || !strings.Contains(fix, "cmd/fak/new.go") {
		t.Fatalf("fix must name `git add` and the path, got %q", fix)
	}
	if len(rep.Untracked) != 1 || rep.Untracked[0] != "cmd/fak/new.go" {
		t.Fatalf("untracked list should carry the path, got %+v", rep.Untracked)
	}
	if !strings.Contains(rep.Detail, "cmd/fak/new.go") {
		t.Fatalf("detail should name the path, got %q", rep.Detail)
	}
}

// TestClassifyPaths_unmatchedNamesTypoRenameStale: a pathspec matching neither the index nor
// the worktree is the raw `did not match any file(s)` case — a typo/rename/stale plan path.
func TestClassifyPaths_unmatchedNamesTypoRenameStale(t *testing.T) {
	g := &classifyFake{}
	rep, err := ClassifyPaths(context.Background(), g.run, "/repo", []string{"internal/gone.go"})
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if rep.OK || rep.Reason != ReasonPathUnmatched {
		t.Fatalf("want %q refusal, got OK=%v reason=%q", ReasonPathUnmatched, rep.OK, rep.Reason)
	}
	if len(rep.Classes) != 1 || rep.Classes[0].State != PathUnmatched {
		t.Fatalf("want a single unmatched class, got %+v", rep.Classes)
	}
	fix := rep.Classes[0].Fix
	if !strings.Contains(fix, "internal/gone.go") || !strings.Contains(fix, "typo") {
		t.Fatalf("fix must name the path and typo/rename/stale, got %q", fix)
	}
	if len(rep.Unmatched) != 1 || rep.Unmatched[0] != "internal/gone.go" {
		t.Fatalf("unmatched list should carry the path, got %+v", rep.Unmatched)
	}
}

// TestClassifyPaths_mixedPrefersUnmatchedReason: with both an untracked and an unmatched
// pathspec, the headline Reason is PATH_UNMATCHED (a wrong path outranks a real un-added one),
// yet BOTH buckets and every per-path class are still reported.
func TestClassifyPaths_mixedPrefersUnmatchedReason(t *testing.T) {
	g := &classifyFake{
		tracked:   map[string]bool{"ok.go": true},
		untracked: map[string]bool{"new.go": true},
	}
	rep, err := ClassifyPaths(context.Background(), g.run, "/repo", []string{"ok.go", "new.go", "gone.go"})
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if rep.OK {
		t.Fatalf("a mixed set with a bad path must refuse, got %+v", rep)
	}
	if rep.Reason != ReasonPathUnmatched {
		t.Fatalf("unmatched must outrank untracked in the headline reason, got %q", rep.Reason)
	}
	want := map[string]PathState{"ok.go": PathTracked, "new.go": PathUntracked, "gone.go": PathUnmatched}
	if len(rep.Classes) != 3 {
		t.Fatalf("want 3 classes in request order, got %+v", rep.Classes)
	}
	for _, c := range rep.Classes {
		if want[c.Path] != c.State {
			t.Fatalf("path %q classified %q, want %q", c.Path, c.State, want[c.Path])
		}
	}
	if len(rep.Untracked) != 1 || rep.Untracked[0] != "new.go" {
		t.Fatalf("untracked bucket wrong: %+v", rep.Untracked)
	}
	if len(rep.Unmatched) != 1 || rep.Unmatched[0] != "gone.go" {
		t.Fatalf("unmatched bucket wrong: %+v", rep.Unmatched)
	}
}

// TestClassifyPaths_notARepo surfaces a non-work-tree as a NOT_A_REPO value, not a crash.
func TestClassifyPaths_notARepo(t *testing.T) {
	g := &classifyFake{notRepo: true, tracked: map[string]bool{"a.go": true}}
	rep, err := ClassifyPaths(context.Background(), g.run, "/nope", []string{"a.go"})
	if err != nil {
		t.Fatalf("a non-repo is a value, not an infra error: %v", err)
	}
	if rep.OK || rep.Reason != ReasonNotARepo {
		t.Fatalf("want %q, got OK=%v reason=%q", ReasonNotARepo, rep.OK, rep.Reason)
	}
	if g.sawLsFiles() {
		t.Fatalf("must not probe ls-files outside a work tree, calls=%v", g.calls)
	}
}

// TestClassifyPaths_noValidPaths: an empty or unclean pathspec set is a NO_PATHS value.
func TestClassifyPaths_noValidPaths(t *testing.T) {
	for _, tc := range [][]string{nil, {"../escapes.go"}} {
		g := &classifyFake{}
		rep, err := ClassifyPaths(context.Background(), g.run, "/repo", tc)
		if err != nil {
			t.Fatalf("paths=%v unexpected error: %v", tc, err)
		}
		if rep.Reason != ReasonNoPath {
			t.Fatalf("paths=%v want %q, got %q", tc, ReasonNoPath, rep.Reason)
		}
		if g.sawLsFiles() {
			t.Fatalf("paths=%v must not touch git before a valid path, calls=%v", tc, g.calls)
		}
	}
}

// TestClassifyPaths_gitNotExecutable propagates a real exec failure as an infra error.
func TestClassifyPaths_gitNotExecutable(t *testing.T) {
	g := &classifyFake{execErr: errors.New("exec: git not found")}
	_, err := ClassifyPaths(context.Background(), g.run, "/repo", []string{"a.go"})
	if err == nil {
		t.Fatalf("git-not-executable must surface as an infra error, got nil")
	}
}

func (f *classifyFake) sawLsFiles() bool {
	for _, c := range f.calls {
		if len(c) > 0 && c[0] == "ls-files" {
			return true
		}
	}
	return false
}
