package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeDoc writes a doc under a temp root and returns the root.
func negframeTempRoot(t *testing.T, rel, content string) string {
	t.Helper()
	root := t.TempDir()
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// TestNegframeSuggestEmitsReframe proves --suggest surfaces the mechanical reframe for a
// negatively-framed line and exits 0.
func TestNegframeSuggestEmitsReframe(t *testing.T) {
	root := negframeTempRoot(t, "AGENTS.md", "Don't forget to stamp the commit.\n")
	var out, errb bytes.Buffer
	code := runNegframeScore(&out, &errb, []string{"--workspace", root, "--suggest"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "remember to stamp") {
		t.Errorf("--suggest output missing the reframe; got:\n%s", s)
	}
}

// TestNegframePerDocJSON proves --per-doc --json emits a machine breakdown with the finding.
func TestNegframePerDocJSON(t *testing.T) {
	root := negframeTempRoot(t, "CLAUDE.md", "No need to rebuild the tree.\n")
	var out, errb bytes.Buffer
	code := runNegframeScore(&out, &errb, []string{"--workspace", root, "--per-doc", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "you can skip rebuild") || !strings.Contains(s, "CLAUDE.md") {
		t.Errorf("--per-doc --json missing expected fields; got:\n%s", s)
	}
}

// TestNegframeDefaultScorecard proves the default (scorecard) surface runs and exit-codes on
// mechanical debt: a doc with a mechanical negative reds (exit 1), a clean doc passes (exit 0).
func TestNegframeDefaultScorecard(t *testing.T) {
	dirty := negframeTempRoot(t, "AGENTS.md", "Don't forget to sign off.\n")
	var out, errb bytes.Buffer
	if code := runNegframeScore(&out, &errb, []string{"--workspace", dirty}); code != 1 {
		t.Errorf("dirty corpus exit = %d, want 1 (mechanical debt); out=%s", code, out.String())
	}
	clean := negframeTempRoot(t, "AGENTS.md", "Remember to sign off. State the affordance first.\n")
	out.Reset()
	errb.Reset()
	if code := runNegframeScore(&out, &errb, []string{"--workspace", clean}); code != 0 {
		t.Errorf("clean corpus exit = %d, want 0; out=%s", code, out.String())
	}
}

// TestNegframeSinceNoGitPasses proves --since degrades to a clean pass when the ref is not a git
// object (git errors -> before == "" -> everything is "new", but there is no prior baseline so
// the working file is scanned fresh). With a non-repo temp dir, git fails and before is "" for
// every path; a NEW file's negatives DO count as introduced, so this asserts the exit contract
// on an explicitly clean file to isolate the git-unavailable degrade path.
func TestNegframeSinceCleanFilePasses(t *testing.T) {
	root := negframeTempRoot(t, "AGENTS.md", "Remember to sign off.\n")
	var out, errb bytes.Buffer
	code := runNegframeScore(&out, &errb, []string{"--workspace", root, "--since", "HEAD"})
	if code != 0 {
		t.Fatalf("clean file --since exit = %d, want 0; out=%s err=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "no new mechanical negatives") {
		t.Errorf("--since clean output unexpected:\n%s", out.String())
	}
}
