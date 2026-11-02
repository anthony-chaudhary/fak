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

// fak-test:runtime fast est=1s lane=default
func TestRecoveryRepeatedWitnessRefreshesUnchangedHead(t *testing.T) {
	for _, publish := range []bool{false, true} {
		name := "fetch"
		if publish {
			name = "publish"
		}
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			remote := filepath.Join(root, "remote.git")
			repo := filepath.Join(root, "repo")
			runGitTest(t, root, "init", "--bare", remote)
			runGitTest(t, root, "init", repo)
			t.Setenv("GIT_AUTHOR_NAME", "Recovery Test")
			t.Setenv("GIT_AUTHOR_EMAIL", "recovery@example.invalid")
			t.Setenv("GIT_COMMITTER_NAME", "Recovery Test")
			t.Setenv("GIT_COMMITTER_EMAIL", "recovery@example.invalid")
			t.Setenv("GIT_AUTHOR_DATE", "1000000000 +0000")
			t.Setenv("GIT_COMMITTER_DATE", "1000000000 +0000")
			runGitTest(t, repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "seed")
			head := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
			runGitTest(t, repo, "remote", "add", "origin", remote)
			ref := RecoveryRefPrefix + "repeat/" + head
			runGitTest(t, repo, "update-ref", ref, head)
			stamp, _ := mirrorStampRef("origin")
			peerStamp, _ := mirrorStampRef("unrelated")
			runGitTest(t, repo, "update-ref", "--create-reflog", peerStamp, head)
			witness := func(date string) {
				t.Setenv("GIT_COMMITTER_DATE", date)
				if publish {
					if got := PublishRecoveryRef(repo, "origin", ref, head, nil); !got.Witnessed {
						t.Fatalf("publish: %+v", got)
					}
				} else if err := FetchRecoveryMirror(repo, "origin", nil); err != nil {
					t.Fatal(err)
				}
			}
			witness("1000100000 +0000")
			first := strings.TrimSpace(runGitTest(t, repo, "rev-parse", stamp))
			witness("1000200000 +0000")
			age, known := mirrorAge(repo, "origin", nil, 1000200007)
			if !known || age != 7 {
				t.Fatalf("repeat same HEAD: age=%d known=%v; want 7,true", age, known)
			}
			second := strings.TrimSpace(runGitTest(t, repo, "rev-parse", stamp))
			if second == first {
				t.Fatal("repeat witness did not advance event")
			}
			if got := strings.TrimSpace(runGitTest(t, repo, "show", "-s", "--format=%ct", stamp)); got != "1000200000" {
				t.Fatalf("old reader timestamp=%s", got)
			}
			witness("1000200000 +0000")
			if got := strings.TrimSpace(runGitTest(t, repo, "rev-parse", stamp)); got == second {
				t.Fatal("same-second witness reused event")
			}
			if got := strings.TrimSpace(runGitTest(t, repo, "show", "-s", "--format=%P", stamp)); got != "" {
				t.Fatalf("event retained permanent parent chain: %s", got)
			}
			if got := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD")); got != head {
				t.Fatal("HEAD changed")
			}
			if got := strings.TrimSpace(runGitTest(t, repo, "rev-parse", peerStamp)); got != head {
				t.Fatal("unrelated remote stamp changed")
			}
		})
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
