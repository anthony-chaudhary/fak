package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestReferenceTransactionHookUsesConfiguredDevelopmentBranch(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	sh, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("sh not on PATH")
	}
	rootBytes, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skipf("not in a git work tree: %v", err)
	}
	hook := filepath.Join(strings.TrimSpace(string(rootBytes)), "tools", "githooks", "reference-transaction")

	repo := t.TempDir()
	if out, err := exec.Command("git", "-C", repo, "init", "-q").CombinedOutput(); err != nil {
		t.Skipf("git init failed: %v %s", err, out)
	}
	dos := `[branch_roles]
development_branch = "dev"
release_branch = "main"
release_source = "dev"
public_front_door = "main"
`
	if err := os.WriteFile(filepath.Join(repo, "dos.toml"), []byte(dos), 0o644); err != nil {
		t.Fatal(err)
	}

	if out, err := runReferenceTransactionHook(sh, hook, repo, "dev"); err != nil {
		t.Fatalf("configured development branch should pass: %v\n%s", err, out)
	}
	if out, err := runReferenceTransactionHook(sh, hook, repo, "main"); err == nil {
		t.Fatalf("release branch must not be implicitly allowed by the everyday branch hook; output=%s", out)
	} else if !strings.Contains(out, "configured development branch 'dev'") {
		t.Fatalf("refusal should name configured development branch, got %q", out)
	}
	if out, err := runReferenceTransactionHook(sh, hook, repo, "master"); err == nil {
		t.Fatalf("master must not be implicitly allowed; output=%s", out)
	}
}

func TestReferenceTransactionHookBlockModeCommittedPhaseFastExits(t *testing.T) {
	sh, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("sh not on PATH")
	}
	rootBytes, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skipf("not in a git work tree: %v", err)
	}
	hook := filepath.Join(strings.TrimSpace(string(rootBytes)), "tools", "githooks", "reference-transaction")

	repo := t.TempDir()
	out, err := runReferenceTransactionHookPhase(sh, hook, repo, "committed", "definitely-not-dev")
	if err != nil {
		t.Fatalf("block-mode committed phase must fast-exit instead of refusing: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "" {
		t.Fatalf("block-mode committed phase should be silent, got %q", out)
	}
}

func TestReferenceTransactionHookPreparedNormalUpdateFastExits(t *testing.T) {
	sh, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("sh not on PATH")
	}
	rootBytes, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skipf("not in a git work tree: %v", err)
	}
	hook := filepath.Join(strings.TrimSpace(string(rootBytes)), "tools", "githooks", "reference-transaction")

	repo := t.TempDir()
	oldRef := strings.Repeat("2", 40)
	newRef := strings.Repeat("3", 40)
	cmd := exec.Command(sh, hook, "prepared")
	cmd.Dir = repo
	cmd.Env = append(os.Environ(), "FLEET_TRUNK_GUARD=block", "FLEET_ALLOW_BRANCH=0")
	cmd.Stdin = strings.NewReader(oldRef + " " + newRef + " refs/heads/feature-x\n")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("normal branch update must fast-exit; only branch creation is guarded: %v\n%s", err, out)
	}
	if strings.TrimSpace(string(out)) != "" {
		t.Fatalf("normal branch update should be silent, got %q", out)
	}
}

func runReferenceTransactionHook(sh, hook, repo, branch string) (string, error) {
	return runReferenceTransactionHookPhase(sh, hook, repo, "prepared", branch)
}

func runReferenceTransactionHookPhase(sh, hook, repo, phase, branch string) (string, error) {
	const zero = "0000000000000000000000000000000000000000"
	newRef := strings.Repeat("1", 40)
	cmd := exec.Command(sh, hook, phase)
	cmd.Dir = repo
	cmd.Env = append(os.Environ(), "FLEET_TRUNK_GUARD=block", "FLEET_ALLOW_BRANCH=0")
	cmd.Stdin = strings.NewReader(zero + " " + newRef + " refs/heads/" + branch + "\n")
	out, err := cmd.CombinedOutput()
	return string(out), err
}
