package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// hooks_test.go — end-to-end CLI tests for `fak hooks` against a real temp git repo. Skipped if
// git is absent or under -short.

func gitHook(t *testing.T, repo string, args ...string) {
	t.Helper()
	c := exec.Command("git", append([]string{"-C", repo}, args...)...)
	c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := c.CombinedOutput(); err != nil {
		t.Skipf("git %v: %s", args, out)
	}
}

func newRepoWith(t *testing.T, files map[string]string) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitHook(t, repo, "init", "-q", "-b", "main")
	gitHook(t, repo, "config", "user.email", "t@t")
	gitHook(t, repo, "config", "user.name", "t")
	for p, content := range files {
		full := filepath.Join(repo, filepath.FromSlash(p))
		_ = os.MkdirAll(filepath.Dir(full), 0o755)
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		gitHook(t, repo, "add", "--", p)
	}
	return repo
}

func hookLeakIP() string { return "100" + ".64.0.10" }

func TestRunHooks_preCommitClean(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("clean staged set should pass (0), got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_preCommitBlocksLeak(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the host is " + hookLeakIP() + " here\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 1 {
		t.Fatalf("a leaked needle should block (1), got %d; stderr=%s", code, errb.String())
	}
}

func TestRunHooks_preCommitJSON(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the host is " + hookLeakIP() + " here\n"})
	var out, errb bytes.Buffer
	_ = runHooks(&out, &errb, []string{"pre-commit", "--root", repo, "--json"})
	if !bytes.Contains(out.Bytes(), []byte("PUBLIC_LEAK")) {
		t.Fatalf("--json should carry the gate name; got %s", out.String())
	}
}

func TestRunHooks_preCommitOffEnvSkips(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"docs/a.md": "the host is " + hookLeakIP() + " here\n"})
	t.Setenv("FLEET_SCRUB_GUARD", "off")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	if code != 0 {
		t.Fatalf("with the leak gate off, the commit should pass; got %d", code)
	}
}

func TestRunHooks_commitMsgVerbShape(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	dir := t.TempDir()
	good := filepath.Join(dir, "good.txt")
	bad := filepath.Join(dir, "bad.txt")
	_ = os.WriteFile(good, []byte("feat(x): add a thing\n"), 0o644)
	_ = os.WriteFile(bad, []byte("docs: clean up stuff\n"), 0o644) // 'clean' not a verb; warn-only

	var out, errb bytes.Buffer
	// default mode is warn -> exit 0 even on a noun-led subject.
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 0 {
		t.Fatalf("commit-msg defaults to warn; should not block, got %d", code)
	}
	// block mode -> a bad subject exits 1.
	t.Setenv("FLEET_MSG_GUARD", "block")
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 1 {
		t.Fatalf("FLEET_MSG_GUARD=block should block a noun-led subject, got %d", code)
	}
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", good}); code != 0 {
		t.Fatalf("a gradeable subject should pass even in block mode, got %d; %s", code, errb.String())
	}
}

func TestRunHooks_commitMsgBlocksHardwareTell(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.txt")
	_ = os.WriteFile(bad, []byte("docs(nightrun): add the dgx3 decode (fak nightrun)\n"), 0o644)

	var out, errb bytes.Buffer
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 1 {
		t.Fatalf("hardware tell should block, got %d; stderr=%s", code, errb.String())
	}
	if !bytes.Contains(errb.Bytes(), []byte("HARDWARE_TELL")) {
		t.Fatalf("stderr should name HARDWARE_TELL, got %s", errb.String())
	}

	t.Setenv("FLEET_ALLOW_HW", "1")
	errb.Reset()
	if code := runHooks(&out, &errb, []string{"commit-msg", bad}); code != 0 {
		t.Fatalf("FLEET_ALLOW_HW should escape the hardware gate, got %d; stderr=%s", code, errb.String())
	}
}
