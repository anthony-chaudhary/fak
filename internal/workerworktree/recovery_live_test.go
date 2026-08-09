package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRemoteRecoveryBareRemoteSurvivesFreshClone(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	source := filepath.Join(root, "source")
	fresh := filepath.Join(root, "fresh")
	runGitTest(t, root, "init", "--bare", remote)
	runGitTest(t, root, "init", source)
	runGitTest(t, source, "config", "user.name", "Test")
	runGitTest(t, source, "config", "user.email", "test-at-example.invalid")
	if err := os.WriteFile(filepath.Join(source, "a.txt"), []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitTest(t, source, "add", "a.txt")
	runGitTest(t, source, "commit", "-m", "base")
	runGitTest(t, source, "remote", "add", "origin", remote)
	runGitTest(t, source, "push", "-u", "origin", "HEAD")
	base := strings.TrimSpace(runGitTest(t, source, "rev-parse", "HEAD"))
	if err := os.WriteFile(filepath.Join(source, "a.txt"), []byte("two\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitTest(t, source, "add", "a.txt")
	tree := strings.TrimSpace(runGitTest(t, source, "write-tree"))
	candidate := strings.TrimSpace(runGitInputTest(t, source, "candidate\n", "commit-tree", tree, "-p", base))
	ref := RecoveryRefPrefix + "fak-worker-wt-live/" + candidate
	runGitTest(t, source, "update-ref", ref, candidate)
	receipt := PublishRecoveryRef(source, "origin", ref, candidate, nil)
	if !receipt.Witnessed {
		t.Fatalf("publish = %+v", receipt)
	}

	runGitTest(t, root, "clone", remote, fresh)
	if err := FetchRecoveryMirror(fresh, "origin", nil); err != nil {
		t.Fatal(err)
	}
	entries, err := RecoveryEntries(fresh, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Durability != DurabilityRemoteOnly || entries[0].SHA != candidate {
		t.Fatalf("fresh clone recovery = %+v", entries)
	}
	if got := strings.TrimSpace(runGitTest(t, fresh, "show", entries[0].MirrorRef+":a.txt")); got != "two" {
		t.Fatalf("recovered bytes = %q", got)
	}
}

func runGitTest(t *testing.T, dir string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	if runtime.GOOS == "windows" {
		c.SysProcAttr = nil
	}
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return string(out)
}
func runGitInputTest(t *testing.T, dir, input string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	c.Stdin = strings.NewReader(input)
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return string(out)
}
