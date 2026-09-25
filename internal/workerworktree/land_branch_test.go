package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// setupPeerBranchRepo builds a real git repo whose shared root checkout sits on a
// PEER branch while `main` exists as the trunk. baseSHA is the tip of `main` at
// setup time (the commit every worker pins to); peerSHA is the peer branch tip
// after one peer-only commit, and is NOT an ancestor of `main`.
//
//	main:  base ────────────────┐ (still at base)
//	peer:  base ─ peer-commit   (root HEAD)
func setupPeerBranchRepo(t *testing.T) (root, baseSHA, peerSHA string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root = t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "tester@test")
	mustGit(t, root, "config", "user.name", "tester")
	mustGit(t, root, "config", "commit.gpgsign", "false")

	if err := os.WriteFile(filepath.Join(root, "init.txt"), []byte("init\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", "init.txt")
	mustGit(t, root, "commit", "-q", "-m", "init")
	baseSHA = strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	// Move the root checkout onto a peer branch and commit a peer-only change.
	// `main` stays pinned at baseSHA.
	mustGit(t, root, "checkout", "-q", "-b", "peer")
	if err := os.WriteFile(filepath.Join(root, "peer.txt"), []byte("peer content\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", "peer.txt")
	mustGit(t, root, "commit", "-q", "-m", "feat: peer-only commit (#1649) (fak peer)")
	peerSHA = strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
	if peerSHA == baseSHA {
		t.Fatalf("peer commit did not advance peer branch: %s", peerSHA)
	}
	return root, baseSHA, peerSHA
}

func revParse(t *testing.T, dir, ref string) string {
	t.Helper()
	return strings.TrimSpace(mustGit(t, dir, "rev-parse", "--verify", ref+"^{commit}"))
}

// TestLandExplicitBranchMovesNamedTrunkNotRootHead is the #1649 acceptance gate:
// with the root checkout on a PEER branch, an explicit WithLandBranch("main") must
// move MAIN forward and leave the root checkout's peer branch untouched.
func TestLandExplicitBranchMovesNamedTrunkNotRootHead(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1") // the isolated path is the one that CASes a ref
	root, baseSHA, peerSHA := setupPeerBranchRepo(t)

	mainBefore := revParse(t, root, "refs/heads/main")
	if mainBefore != baseSHA {
		t.Fatalf("precondition: main should sit at base %s, got %s", baseSHA, mainBefore)
	}

	// Worker is based on the ancestor of main; its own commit lives only in the worktree.
	wtRoot := t.TempDir()
	prep := Prepare(root, "lane", "key", baseSHA, wtRoot, nil)
	if !prep.OK {
		t.Fatalf("prepare failed: %+v", prep)
	}
	wtPath := prep.Path
	if err := os.WriteFile(filepath.Join(wtPath, "worker.txt"), []byte("worker content\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, wtPath, "add", "worker.txt")
	mustGit(t, wtPath, "commit", "-q", "-m", "feat: worker file (#1649) (fak workerworktree)")

	res := Land(root, wtPath, baseSHA, "", []string{"worker.txt"}, nil, nil,
		WithLandBranch("main"))
	if !res.OK {
		t.Fatalf("land with explicit --branch main must succeed, got: %+v", res)
	}
	if res.Code != LandResultSuccess {
		t.Fatalf("expected success code, got %q (%+v)", res.Code, res)
	}

	mainAfter := revParse(t, root, "refs/heads/main")
	if mainAfter == mainBefore {
		t.Fatalf("main did NOT move under explicit WithLandBranch(main): still %s", mainAfter)
	}

	// The peer branch the root was checked out on must NOT have moved: the land
	// targeted the explicit trunk, not the root's symbolic-ref HEAD.
	peerAfter := revParse(t, root, "refs/heads/peer")
	if peerAfter != peerSHA {
		t.Fatalf("peer branch moved but an explicit --branch main land must not touch it: before=%s after=%s", peerSHA, peerAfter)
	}
}

// TestLandExplicitBranchRefusesStaleBaseFailClosed pins a base NOT on main (a
// peer-only commit) and asserts the explicit-branch land refuses with the
// stale-base class and does NOT move main.
func TestLandExplicitBranchRefusesStaleBaseFailClosed(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	root, _, peerSHA := setupPeerBranchRepo(t)

	mainBefore := revParse(t, root, "refs/heads/main")

	// Pin the worker base to the peer-only commit — NOT an ancestor of main.
	wtRoot := t.TempDir()
	prep := Prepare(root, "lane", "key", peerSHA, wtRoot, nil)
	if !prep.OK {
		t.Fatalf("prepare failed: %+v", prep)
	}
	wtPath := prep.Path
	if err := os.WriteFile(filepath.Join(wtPath, "worker.txt"), []byte("worker content\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, wtPath, "add", "worker.txt")
	mustGit(t, wtPath, "commit", "-q", "-m", "feat: worker file (#1649) (fak workerworktree)")

	res := Land(root, wtPath, peerSHA, "", []string{"worker.txt"}, nil, nil,
		WithLandBranch("main"))
	if res.OK {
		t.Fatalf("land of a base not on main must refuse, got success: %+v", res)
	}
	if res.Code != LandResultStaleBase {
		t.Fatalf("expected stale-base refusal %q, got code %q (%+v)", LandResultStaleBase, res.Code, res)
	}
	mainAfter := revParse(t, root, "refs/heads/main")
	if mainAfter != mainBefore {
		t.Fatalf("main moved on a refused stale-base land: before=%s after=%s", mainBefore, mainAfter)
	}
}
