package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/uiquality"
)

// gitInitRepoWithFiles builds a throwaway repo with an initial commit of files,
// returning the repo dir. Used to drive the --since skip-gate hermetically.
func gitInitRepoWithFiles(t *testing.T, files map[string]string) string {
	t.Helper()
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
	writeFiles(t, dir, files)
	run("init", "-q")
	run("add", "-A")
	run("commit", "-qm", "base")
	return dir
}

func writeFiles(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	for rel, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestUIQualitySinceSkipsWhenCorpusUntouched: an edit that touches no render source
// must short-circuit to "unchanged" (done=true, exit 0) without a full rescan — the
// shift-left fast path an at-origin agent hits on the vast majority of edits.
func TestUIQualitySinceSkipsWhenCorpusUntouched(t *testing.T) {
	corpus := uiquality.Corpus()
	if len(corpus) == 0 {
		t.Fatal("empty ui-quality corpus")
	}
	dir := gitInitRepoWithFiles(t, map[string]string{
		corpus[0]:   "package main\n",
		"README.md": "hi\n",
	})
	// Change only the unrelated file.
	writeFiles(t, dir, map[string]string{"README.md": "hi there\n"})

	var out, errBuf bytes.Buffer
	done, code := uiQualitySinceSkip(&out, &errBuf, dir, "HEAD", false)
	if !done || code != 0 {
		t.Fatalf("expected skip (done=true, code=0), got done=%v code=%d", done, code)
	}
	if !strings.Contains(out.String(), "unchanged since HEAD") {
		t.Fatalf("expected 'unchanged since HEAD', got: %q", out.String())
	}
}

// TestUIQualitySinceRescansWhenCorpusTouched: an edit to a render source must NOT
// skip — it returns done=false so the caller falls through to a full, correct rescan
// (partial scans would corrupt the holistic KPIs).
func TestUIQualitySinceRescansWhenCorpusTouched(t *testing.T) {
	corpus := uiquality.Corpus()
	dir := gitInitRepoWithFiles(t, map[string]string{
		corpus[0]:   "package main\n",
		"README.md": "hi\n",
	})
	// Change a render source.
	writeFiles(t, dir, map[string]string{corpus[0]: "package main\n// edit\n"})

	var out, errBuf bytes.Buffer
	done, _ := uiQualitySinceSkip(&out, &errBuf, dir, "HEAD", false)
	if done {
		t.Fatalf("expected fall-through to full rescan (done=false), got done=true; stdout=%q", out.String())
	}
	if !strings.Contains(errBuf.String(), "changed since HEAD") {
		t.Fatalf("expected 'changed since HEAD' note, got stderr: %q", errBuf.String())
	}
}

// TestUIQualitySinceFallsThroughOnBadRef: an unresolvable ref must fall through to a
// full rescan (done=false), never falsely report "unchanged".
func TestUIQualitySinceFallsThroughOnBadRef(t *testing.T) {
	dir := gitInitRepoWithFiles(t, map[string]string{"README.md": "hi\n"})
	var out, errBuf bytes.Buffer
	done, _ := uiQualitySinceSkip(&out, &errBuf, dir, "nope-not-a-ref", false)
	if done {
		t.Fatal("bad ref must not report 'unchanged' (done should be false)")
	}
	if !strings.Contains(errBuf.String(), "diff failed") {
		t.Fatalf("expected 'diff failed' note, got stderr: %q", errBuf.String())
	}
}

// TestUIQualitySinceJSONSkip: the --json skip path emits an incremental_skip marker
// (so a machine consumer can tell a skip from a real clean scan).
func TestUIQualitySinceJSONSkip(t *testing.T) {
	corpus := uiquality.Corpus()
	dir := gitInitRepoWithFiles(t, map[string]string{
		corpus[0]:   "package main\n",
		"README.md": "hi\n",
	})
	writeFiles(t, dir, map[string]string{"README.md": "changed\n"})

	var out, errBuf bytes.Buffer
	done, code := uiQualitySinceSkip(&out, &errBuf, dir, "HEAD", true)
	if !done || code != 0 {
		t.Fatalf("expected json skip, got done=%v code=%d", done, code)
	}
	if !strings.Contains(out.String(), "\"incremental_skip\": true") {
		t.Fatalf("expected incremental_skip marker in JSON, got: %q", out.String())
	}
}
