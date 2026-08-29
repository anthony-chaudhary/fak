package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyInventoryCommittedAuthorityIgnoresDirtyAndUntrackedWIP(t *testing.T) {
	repo, git := seedStudySelfRepo(t)
	writeStudySelfFile(t, repo, "README.md", "committed\n")
	gitStudySelf(t, git, "add", "README.md")
	gitStudySelf(t, git, "commit", "-qm", "seed")

	var out, errOut bytes.Buffer
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo, "--json"}); code != 0 {
		t.Fatalf("refresh=%d stderr=%s", code, errOut.String())
	}
	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "manifest")

	writeStudySelfFile(t, repo, "README.md", "dirty peer edit\n")
	writeStudySelfFile(t, repo, "peer-untracked.go", "package peer\n")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo, "--json"}); code != 0 {
		t.Fatalf("verify=%d stderr=%s output=%s", code, errOut.String(), out.String())
	}
	var result studySelfInventoryOutput
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Verification == nil || !result.Verification.OK || len(result.Verification.Drift) != 0 {
		t.Fatalf("dirty/untracked WIP changed committed verdict: %+v", result)
	}
}

func TestStudyInventoryCommittedMutationFailsThenRefreshPasses(t *testing.T) {
	repo, git := seedStudySelfRepo(t)
	writeStudySelfFile(t, repo, "README.md", "one\n")
	gitStudySelf(t, git, "add", "README.md")
	gitStudySelf(t, git, "commit", "-qm", "seed")
	var out, errOut bytes.Buffer
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo}); code != 0 {
		t.Fatalf("refresh: %s", errOut.String())
	}
	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "manifest")

	writeStudySelfFile(t, repo, "README.md", "two\n")
	gitStudySelf(t, git, "add", "README.md")
	gitStudySelf(t, git, "commit", "-qm", "mutation")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo, "--json"}); code != 1 {
		t.Fatalf("stale verify=%d output=%s stderr=%s", code, out.String(), errOut.String())
	}
	var stale studySelfInventoryOutput
	if err := json.Unmarshal(out.Bytes(), &stale); err != nil {
		t.Fatal(err)
	}
	if stale.Verification == nil || len(stale.Verification.Drift) != 1 || stale.Verification.Drift[0].Kind != studymonitor.SelfDriftContentChanged || stale.Verification.Drift[0].Path != "README.md" {
		t.Fatalf("stale diagnostics = %+v", stale)
	}

	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo}); code != 0 {
		t.Fatalf("second refresh: %s", errOut.String())
	}
	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "refresh")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo}); code != 0 {
		t.Fatalf("verify after refresh=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
}

func seedStudySelfRepo(t *testing.T) (string, func(...string) ([]byte, error)) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	repo := t.TempDir()
	git := func(args ...string) ([]byte, error) {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.test", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.test")
		return cmd.CombinedOutput()
	}
	if out, err := git("init", "-q"); err != nil {
		t.Skipf("git init: %s", out)
	}
	return repo, git
}

func gitStudySelf(t *testing.T, git func(...string) ([]byte, error), args ...string) {
	t.Helper()
	if out, err := git(args...); err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
}

func writeStudySelfFile(t *testing.T, root, rel, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
