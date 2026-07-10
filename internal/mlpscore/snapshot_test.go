package mlpscore

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestGitSnapshotIgnoresUncommittedProofs(t *testing.T) {
	root := t.TempDir()
	runTestGit(t, root, "init", "-q")
	runTestGit(t, root, "config", "user.email", "mlpscore@example.invalid")
	runTestGit(t, root, "config", "user.name", "MLP Score Test")

	tracked := filepath.Join(root, "tracked.txt")
	if err := os.WriteFile(tracked, []byte("tracked\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runTestGit(t, root, "add", "tracked.txt")
	runTestGit(t, root, "commit", "-q", "-m", "seed")

	untracked := filepath.Join(root, "proof.md")
	if err := os.WriteFile(untracked, []byte("not shipped\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := LoadGitSnapshot(root)
	if err != nil {
		t.Fatal(err)
	}
	if !snapshot.Exists("tracked.txt") {
		t.Fatal("committed path is absent from snapshot")
	}
	if snapshot.Exists("proof.md") {
		t.Fatal("untracked proof must not witness a criterion")
	}
	if _, err := snapshot.ReadFile("proof.md"); err == nil {
		t.Fatal("untracked proof read unexpectedly succeeded")
	}

	runTestGit(t, root, "add", "proof.md")
	runTestGit(t, root, "commit", "-q", "-m", "ship proof")
	shipped, err := LoadGitSnapshot(root)
	if err != nil {
		t.Fatal(err)
	}
	if !shipped.Exists("proof.md") {
		t.Fatal("committed proof is absent from refreshed snapshot")
	}
	if got, err := shipped.ReadFile("proof.md"); err != nil || string(got) != "not shipped\n" {
		t.Fatalf("committed proof read = %q, %v", got, err)
	}
}

func runTestGit(t *testing.T, root string, argv ...string) {
	t.Helper()
	cmd := exec.Command("git", argv...)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", argv, err, out)
	}
}
