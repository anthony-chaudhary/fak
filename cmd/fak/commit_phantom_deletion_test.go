package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/patchcommit"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// setupDisjointMergeRepo creates a git repo where HEAD is a "disjoint integrate" merge
// commit that introduced peer_added.txt, but peer_added.txt is missing from the working tree
// (simulating desynchronized index/working tree following synthetic ODB merge).
func setupDisjointMergeRepo(t *testing.T, isDisjointIntegrate bool) (repoDir string, peerFile string, initialFile string) {
	t.Helper()
	d := t.TempDir()

	gitRun := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = d
		cmd.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1", "GIT_OPTIONAL_LOCKS=0")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	gitRun("init", "-b", "main")
	gitRun("config", "user.name", "Test Committer")
	gitRun("config", "user.email", "committer@example.test")

	// Initial commit on main
	initialFile = "initial.txt"
	if err := os.WriteFile(filepath.Join(d, initialFile), []byte("initial\n"), 0644); err != nil {
		t.Fatal(err)
	}
	gitRun("add", initialFile)
	gitRun("commit", "-m", "chore: initial commit")

	headSHA := gitRun("rev-parse", "HEAD")

	// Create a second commit with peer file on a temporary branch/ref
	gitRun("checkout", "-b", "peer-branch")
	peerFile = "peer_added.txt"
	if err := os.WriteFile(filepath.Join(d, peerFile), []byte("peer content\n"), 0644); err != nil {
		t.Fatal(err)
	}
	gitRun("add", peerFile)
	gitRun("commit", "-m", "feat: peer contribution")
	peerSHA := gitRun("rev-parse", "HEAD")

	// Switch back to main
	gitRun("checkout", "main")

	// Create merge commit directly using merge-tree and commit-tree (synthetic ODB merge)
	mergeMsg := "Merge origin/main (disjoint integrate) (fak safesync)"
	if !isDisjointIntegrate {
		mergeMsg = "Merge branch 'feature-normal' into main"
	}

	treeOID := gitRun("merge-tree", "--write-tree", headSHA, peerSHA)
	mergeSHA := gitRun("commit-tree", treeOID, "-p", headSHA, "-p", peerSHA, "-m", mergeMsg)
	gitRun("update-ref", "refs/heads/main", mergeSHA, headSHA)
	gitRun("update-ref", "HEAD", mergeSHA)

	// Note: working tree still only has initial.txt, peer_added.txt does NOT exist in working tree.
	if _, err := os.Stat(filepath.Join(d, peerFile)); !os.IsNotExist(err) {
		t.Fatalf("expected %s to be absent in working tree", peerFile)
	}

	return d, peerFile, initialFile
}

func TestCommitPhantomDeletionGuard(t *testing.T) {
	t.Run("RefusesRequestedDeletion", func(t *testing.T) {
		repo, peerFile, _ := setupDisjointMergeRepo(t, true)

		var stdout, stderr bytes.Buffer
		code := runCommit(&stdout, &stderr, []string{
			"--dir", repo,
			"--path", peerFile,
			"-m", "chore: delete peer added file (fak commit)",
		})

		if code != safecommit.ExitRefused {
			t.Fatalf("exit code = %d, want %d (ExitRefused); stdout=%s stderr=%s",
				code, safecommit.ExitRefused, stdout.String(), stderr.String())
		}

		combined := stderr.String() + "\n" + stdout.String()
		if !strings.Contains(combined, "PHANTOM_DELETION_RISK") {
			t.Fatalf("expected PHANTOM_DELETION_RISK in output, got:\n%s", combined)
		}
		if !strings.Contains(combined, "recovery") && !strings.Contains(combined, "git checkout") {
			t.Fatalf("expected recovery instructions in output, got:\n%s", combined)
		}
	})

	t.Run("RefusesWithJSON", func(t *testing.T) {
		repo, peerFile, _ := setupDisjointMergeRepo(t, true)

		var stdout, stderr bytes.Buffer
		code := runCommit(&stdout, &stderr, []string{
			"--dir", repo,
			"--path", peerFile,
			"-m", "chore: delete peer added file (fak commit)",
			"--json",
		})

		if code != safecommit.ExitRefused {
			t.Fatalf("exit code = %d, want %d; stderr=%s", code, safecommit.ExitRefused, stderr.String())
		}

		var res safecommit.Result
		if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
			t.Fatalf("json decode failed: %v; stdout=%q", err, stdout.String())
		}

		if res.Reason != "PHANTOM_DELETION_RISK" {
			t.Fatalf("res.Reason = %q, want PHANTOM_DELETION_RISK", res.Reason)
		}
		if !strings.Contains(res.Detail, peerFile) {
			t.Fatalf("res.Detail %q does not name %q", res.Detail, peerFile)
		}
		if !strings.Contains(res.Detail, "recovery") && !strings.Contains(res.Detail, "git checkout") {
			t.Fatalf("res.Detail %q lacks recovery advice", res.Detail)
		}
	})

	t.Run("RefusesStagedDeletionInSharedIndex", func(t *testing.T) {
		repo, peerFile, _ := setupDisjointMergeRepo(t, true)

		// Update index to match HEAD so peerFile is in the index
		cmdTree := exec.Command("git", "read-tree", "HEAD")
		cmdTree.Dir = repo
		if out, err := cmdTree.CombinedOutput(); err != nil {
			t.Fatalf("git read-tree HEAD failed: %v\n%s", err, string(out))
		}

		// Stage deletion of peerFile in index
		cmd := exec.Command("git", "rm", "--cached", peerFile)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git rm --cached failed: %v\n%s", err, string(out))
		}

		var stdout, stderr bytes.Buffer
		code := runCommit(&stdout, &stderr, []string{
			"--dir", repo,
			"--path", peerFile,
			"-m", "chore: commit staged deletion of peer file (fak commit)",
		})

		if code != safecommit.ExitRefused {
			t.Fatalf("exit code = %d, want %d; stdout=%s stderr=%s", code, safecommit.ExitRefused, stdout.String(), stderr.String())
		}
		combined := stderr.String() + "\n" + stdout.String()
		if !strings.Contains(combined, "PHANTOM_DELETION_RISK") {
			t.Fatalf("expected PHANTOM_DELETION_RISK in output, got:\n%s", combined)
		}
	})

	t.Run("AllowsLegitimateDeletionOfPreexistingFile", func(t *testing.T) {
		repo, _, initialFile := setupDisjointMergeRepo(t, true)

		// Delete initialFile which existed BEFORE the merge commit
		_ = os.Remove(filepath.Join(repo, initialFile))

		// Stub commitFn to verify that execution proceeds past phantom deletion guard to commitFn
		called := false
		withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
			called = true
			return safecommit.Result{
				SHA:       "abcdef123456",
				Paths:     o.Paths,
				Committed: true,
			}, nil
		})

		var stdout, stderr bytes.Buffer
		code := runCommit(&stdout, &stderr, []string{
			"--dir", repo,
			"--path", initialFile,
			"-m", "chore: delete initial preexisting file (fak commit)",
			"--no-build-check",
		})

		if !called {
			t.Fatalf("expected commitFn to be called for legitimate deletion, but it was not. code=%d stderr=%s", code, stderr.String())
		}
	})

	t.Run("AllowsWhenMergeNotDisjointIntegrate", func(t *testing.T) {
		repo, peerFile, _ := setupDisjointMergeRepo(t, false)

		called := false
		withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
			called = true
			return safecommit.Result{
				SHA:       "abcdef123456",
				Paths:     o.Paths,
				Committed: true,
			}, nil
		})

		var stdout, stderr bytes.Buffer
		code := runCommit(&stdout, &stderr, []string{
			"--dir", repo,
			"--path", peerFile,
			"-m", "chore: delete file from normal merge (fak commit)",
			"--no-build-check",
		})

		if !called {
			t.Fatalf("expected commitFn to be called when merge is not disjoint integrate. code=%d stderr=%s", code, stderr.String())
		}
	})

	t.Run("EnvOverrideAllowsBypass", func(t *testing.T) {
		repo, peerFile, _ := setupDisjointMergeRepo(t, true)
		t.Setenv("FAK_PHANTOM_DELETION_CHECK", "off")

		called := false
		withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
			called = true
			return safecommit.Result{
				SHA:       "abcdef123456",
				Paths:     o.Paths,
				Committed: true,
			}, nil
		})

		var stdout, stderr bytes.Buffer
		code := runCommit(&stdout, &stderr, []string{
			"--dir", repo,
			"--path", peerFile,
			"-m", "chore: force delete peer file with override (fak commit)",
			"--no-build-check",
		})

		if !called {
			t.Fatalf("expected commitFn to be called with FAK_PHANTOM_DELETION_GUARD=off. code=%d stderr=%s", code, stderr.String())
		}
	})

	t.Run("PatchCommitRefusesDeletion", func(t *testing.T) {
		repo, peerFile, _ := setupDisjointMergeRepo(t, true)

		// Create a patch that deletes peerFile
		patchContent := "diff --git a/" + peerFile + " b/" + peerFile + "\ndeleted file mode 100644\n--- a/" + peerFile + "\n+++ /dev/null\n@@ -1 +0,0 @@\n-peer content\n"
		patchPath := filepath.Join(repo, "delete.patch")
		if err := os.WriteFile(patchPath, []byte(patchContent), 0644); err != nil {
			t.Fatal(err)
		}

		res, err := patchcommit.Commit(context.Background(), patchcommit.Options{
			Dir:       repo,
			PatchFile: patchPath,
			Paths:     []string{peerFile},
			Message:   "chore: patch delete peer file",
		})
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if res.Reason != patchcommit.ReasonPhantomDeletionRisk {
			t.Fatalf("res.Reason = %q, want %q; detail=%s", res.Reason, patchcommit.ReasonPhantomDeletionRisk, res.Detail)
		}
		if !strings.Contains(res.Detail, "recovery") && !strings.Contains(res.Detail, "git checkout") {
			t.Fatalf("expected recovery advice in detail, got: %s", res.Detail)
		}
	})
}
