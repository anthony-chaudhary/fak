package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type recordingBackend struct {
	materialized bool
	released     bool
}

func (b *recordingBackend) Materialize(_, _, _, _, wtRoot string, _ GitRunner) Result {
	b.materialized = true
	return Result{OK: true, Path: filepath.Join(wtRoot, "fake")}
}

func (b *recordingBackend) Release(_, wtPath string, _ GitRunner) Result {
	b.released = true
	return Result{OK: true, Path: wtPath, Removed: true}
}

func TestIsolationBackendSelectionIsInjectable(t *testing.T) {
	backend := &recordingBackend{}
	got := PrepareWithBackend("repo", "lane", "key", "base", t.TempDir(), nil, backend)
	if !got.OK || !backend.materialized || filepath.Base(got.Path) != "fake" {
		t.Fatalf("prepare did not use injected backend: result=%+v backend=%+v", got, backend)
	}
	reaped := ReapWithBackend("repo", got.Path, nil, backend)
	if !reaped.OK || !backend.released || !reaped.Removed {
		t.Fatalf("reap did not use injected backend: result=%+v backend=%+v", reaped, backend)
	}
}

func TestIsolationBackendsPreserveBaseDiff(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	for _, backend := range IsolationBackends() {
		t.Run("git-worktree", func(t *testing.T) {
			repo := t.TempDir()
			git := func(dir string, args ...string) string {
				cmd := exec.Command("git", args...)
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
				}
				return string(out)
			}
			git(repo, "init", "-q", "-b", "main")
			git(repo, "config", "user.email", "backend@test")
			git(repo, "config", "user.name", "backend")
			git(repo, "config", "core.autocrlf", "false")
			if err := os.WriteFile(filepath.Join(repo, "known.txt"), []byte("before\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			git(repo, "add", "known.txt")
			git(repo, "commit", "-q", "-m", "base")
			base := strings.TrimSpace(git(repo, "rev-parse", "HEAD"))

			res := PrepareWithBackend(repo, "workerworktree", "5918", base, t.TempDir(), nil, backend)
			if !res.OK {
				t.Fatalf("materialize: %+v", res)
			}
			t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })
			if err := os.WriteFile(filepath.Join(res.Path, "known.txt"), []byte("after\n"), 0o644); err != nil {
				t.Fatal(err)
			}

			patch := git(res.Path, "diff", base, "--", "known.txt")
			if !strings.Contains(patch, "-before") || !strings.Contains(patch, "+after") {
				t.Fatalf("base diff lost worker edit:\n%s", patch)
			}
			names := strings.TrimSpace(git(res.Path, "diff", "--name-only", base))
			if names != "known.txt" {
				t.Fatalf("name-only diff = %q, want known.txt", names)
			}
		})
	}
}
