package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func configureTestGitRepo(t *testing.T, dir, name, email string) {
	t.Helper()
	cfgPath := filepath.Join(dir, ".git", "config")
	appendStr := fmt.Sprintf("\n[user]\n\tname = %s\n\temail = %s\n[commit]\n\tgpgsign = false\n", name, email)
	f, err := os.OpenFile(cfgPath, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open .git/config in %s: %v", dir, err)
	}
	defer f.Close()
	if _, err := f.WriteString(appendStr); err != nil {
		t.Fatalf("write .git/config in %s: %v", dir, err)
	}
}

func TestCommitWith_PushAutoReconcilesDisjointDivergence(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	bare := filepath.Join(root, "origin.git")
	seed := filepath.Join(root, "seed")
	local := filepath.Join(root, "local")
	peer := filepath.Join(root, "peer")

	// 1. Set up bare origin repo + local clone with initial commit.
	tempRepoGit(t, root, "init", "-q", "-b", "main", seed)
	writeTempRepoFile(t, filepath.Join(seed, "initial.txt"), "initial\n")
	tempRepoGit(t, seed, "add", "--", "initial.txt")
	tempRepoGit(t, seed, "commit", "-qm", "initial commit")
	tempRepoGit(t, root, "clone", "-q", "--bare", seed, bare)

	tempRepoGit(t, root, "clone", "-q", bare, local)
	configureTestGitRepo(t, local, "fak test", "fak-test@example.invalid")

	// 2. Add a commit on origin touching peer.txt (creating disjoint divergence).
	tempRepoGit(t, root, "clone", "-q", bare, peer)
	configureTestGitRepo(t, peer, "peer test", "peer-test@example.invalid")

	writeTempRepoFile(t, filepath.Join(peer, "peer.txt"), "peer content\n")
	tempRepoGit(t, peer, "add", "--", "peer.txt")
	tempRepoGit(t, peer, "commit", "-qm", "peer commit")
	tempRepoGit(t, peer, "push", "-q", "origin", "main")

	// 3. In local repo, create mine.txt and invoke CommitWith(ctx, run, opts)
	// with Paths: []string{"mine.txt"}, Message: "test commit", Push: true.
	writeTempRepoFile(t, filepath.Join(local, "mine.txt"), "my content\n")

	opts := Options{
		Dir:     local,
		Paths:   []string{"mine.txt"},
		Message: "test commit",
		Push:    true,
		Trunk:   "main",
	}

	res, err := CommitWith(ctx, realRunner, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infrastructure error: %v", err)
	}

	// Verify res.Committed == true, res.Verified == true, res.Pushed == true, res.Reason == "".
	if !res.Committed {
		t.Fatalf("res.Committed = false, want true; reason=%q detail=%q", res.Reason, res.Detail)
	}
	if !res.Verified {
		t.Fatalf("res.Verified = false, want true; reason=%q detail=%q", res.Reason, res.Detail)
	}
	if !res.Pushed {
		t.Fatalf("res.Pushed = false, want true; reason=%q detail=%q", res.Reason, res.Detail)
	}
	if res.Reason != "" {
		t.Fatalf("res.Reason = %q, want empty; detail=%q", res.Reason, res.Detail)
	}

	// Verify origin contains both changes.
	originTree := tempRepoGit(t, bare, "ls-tree", "-r", "--name-only", "refs/heads/main")
	if !strings.Contains(originTree, "peer.txt") {
		t.Errorf("origin tree missing peer.txt; got:\n%s", originTree)
	}
	if !strings.Contains(originTree, "mine.txt") {
		t.Errorf("origin tree missing mine.txt; got:\n%s", originTree)
	}

	// Also test that when DisableAutoReconcile: true, it does not auto-reconcile
	// and returns res.Pushed == false, res.Reason == ReasonPushRejected.
	t.Run("DisableAutoReconcile", func(t *testing.T) {
		tempRepoGit(t, peer, "pull", "-q", "--ff-only", "origin", "main")
		writeTempRepoFile(t, filepath.Join(peer, "peer2.txt"), "peer2 content\n")
		tempRepoGit(t, peer, "add", "--", "peer2.txt")
		tempRepoGit(t, peer, "commit", "-qm", "peer2 commit")
		tempRepoGit(t, peer, "push", "-q", "origin", "main")

		writeTempRepoFile(t, filepath.Join(local, "mine2.txt"), "my content 2\n")

		optsDisabled := Options{
			Dir:                  local,
			Paths:                []string{"mine2.txt"},
			Message:              "test commit disabled",
			Push:                 true,
			DisableAutoReconcile: true,
			Trunk:                "main",
		}

		resDisabled, err := CommitWith(ctx, realRunner, okLock(nil), optsDisabled)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !resDisabled.Committed || !resDisabled.Verified {
			t.Fatalf("commit should be committed and verified, got: %+v", resDisabled)
		}
		if resDisabled.Pushed {
			t.Fatalf("resDisabled.Pushed = true, want false")
		}
		if resDisabled.Reason != ReasonPushRejected {
			t.Fatalf("resDisabled.Reason = %q, want %q", resDisabled.Reason, ReasonPushRejected)
		}
	})
}
