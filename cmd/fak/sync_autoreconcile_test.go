package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

const syncAutoReconcileSource = "0f1e2d3c4b5a60718293a4b5c6d7e8f90a1b2c3d"

// stubSyncDisjointPush wires the push seams so SafePush refuses with DIVERGED_DISJOINT
// and the reconciliation packet is a dispatchable safe-disjoint packet pinned to the
// captured source. It returns pointers to the build/execute call counters.
func stubSyncDisjointPush(t *testing.T) (builds, executes *int) {
	t.Helper()
	builds, executes = new(int), new(int)

	oldCapture := syncCaptureSource
	syncCaptureSource = func(string) (string, error) { return syncAutoReconcileSource, nil }
	oldPush := syncSafePush
	syncSafePush = func(context.Context, safesync.PushOptions) (safesync.PushResult, error) {
		return safesync.PushResult{
			Attempts:   1,
			Branch:     "main",
			Remote:     "origin",
			Reason:     safesync.ReasonDivergedDisjoint,
			Divergence: string(safesync.PushDiverged),
			Detail:     "diverged from origin/main with disjoint paths; run `fak sync check --fetch --remote origin --branch main` to preview integration",
		}, nil
	}
	oldBuild := syncBuildReconciliationPacket
	syncBuildReconciliationPacket = func(_ context.Context, opts safesync.PacketOptions) (*safesync.ReconciliationPacket, error) {
		*builds++
		if opts.Remote != "origin" || opts.Branch != "main" {
			t.Errorf("packet opts = %+v, want origin/main", opts)
		}
		return &safesync.ReconciliationPacket{
			Dispatchable: true,
			Disposition:  safesync.DispositionSafeDisjoint,
			LocalHead:    syncAutoReconcileSource,
		}, nil
	}
	oldExec := syncExecutePacket
	syncExecutePacket = func(_ context.Context, pkt *safesync.ReconciliationPacket, opts safesync.ExecuteOptions) (*safesync.ExecutionReceipt, error) {
		*executes++
		if pkt.Disposition != safesync.DispositionSafeDisjoint {
			t.Errorf("executed disposition %q, want safe-disjoint", pkt.Disposition)
		}
		return &safesync.ExecutionReceipt{
			Status: safesync.ExecuteStatusExecuted, Pushed: true,
			LocalCommitsContained: true, PeerBytesPreserved: true,
		}, nil
	}
	oldWorktree := syncWorktree
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) { return safesync.Worktree{}, false }
	t.Cleanup(func() {
		syncCaptureSource = oldCapture
		syncSafePush = oldPush
		syncBuildReconciliationPacket = oldBuild
		syncExecutePacket = oldExec
		syncWorktree = oldWorktree
	})
	return builds, executes
}

// TestRunSyncPushAutoReconcilesDisjointDivergence: a DIVERGED_DISJOINT push refusal is routed
// through the shared verified heal (packet build -> execute, whose SafePush re-pushes) and the
// verb exits OK with the heal evidence in the JSON.
func TestRunSyncPushAutoReconcilesDisjointDivergence(t *testing.T) {
	builds, executes := stubSyncDisjointPush(t)

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", t.TempDir(), "--remote", "origin", "--branch", "main", "--json"})
	if code != syncExitOK {
		t.Fatalf("exit=%d stderr=%q stdout=%s, want ok after auto-reconcile", code, errb.String(), out.String())
	}
	if *builds != 1 || *executes != 1 {
		t.Fatalf("builds=%d executes=%d, want the packet built and executed exactly once", *builds, *executes)
	}
	var got safesync.PushResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, out.String())
	}
	if !got.Pushed || got.Reason != "" || got.Attempts != 2 {
		t.Fatalf("result = %+v, want pushed with no reason after 2 attempts", got)
	}
	if got.AutoReconcile == nil || !got.AutoReconcile.Pushed || got.AutoReconcile.Disposition != safesync.DispositionSafeDisjoint {
		t.Fatalf("auto_reconcile = %+v, want a pushed safe-disjoint heal", got.AutoReconcile)
	}
	if !strings.Contains(got.Detail, "auto-reconciled") {
		t.Fatalf("detail = %q, want the auto-reconcile note", got.Detail)
	}
}

// TestRunSyncPushNoAutoReconcileKeepsRefusal: the opt-out keeps the pre-#3069 refusal byte for
// byte and never builds or executes a packet.
func TestRunSyncPushNoAutoReconcileKeepsRefusal(t *testing.T) {
	builds, executes := stubSyncDisjointPush(t)

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--no-auto-reconcile", "--repo", t.TempDir(), "--remote", "origin", "--branch", "main"})
	if code != syncExitRefused {
		t.Fatalf("exit=%d stderr=%q, want refused", code, errb.String())
	}
	if *builds != 0 || *executes != 0 {
		t.Fatalf("builds=%d executes=%d, want no heal under --no-auto-reconcile", *builds, *executes)
	}
	for _, want := range []string{"not pushed (" + safesync.ReasonDivergedDisjoint + ")", "fak sync check --fetch"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

// TestRunSyncPushKeepsRefusalWhenPacketNotSafeDisjoint: a packet that is not a dispatchable
// safe-disjoint packet (colliding dirty paths, semantic conflict) keeps the refusal and the
// executor is never invoked.
func TestRunSyncPushKeepsRefusalWhenPacketNotSafeDisjoint(t *testing.T) {
	_, executes := stubSyncDisjointPush(t)
	syncBuildReconciliationPacket = func(context.Context, safesync.PacketOptions) (*safesync.ReconciliationPacket, error) {
		return &safesync.ReconciliationPacket{Disposition: safesync.DispositionOwnerHandoffRequired, LocalHead: syncAutoReconcileSource}, nil
	}

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", t.TempDir(), "--remote", "origin", "--branch", "main"})
	if code != syncExitRefused {
		t.Fatalf("exit=%d stderr=%q, want refused", code, errb.String())
	}
	if *executes != 0 {
		t.Fatalf("executes=%d, want 0 for a non-safe-disjoint packet", *executes)
	}
	if !strings.Contains(out.String(), safesync.AutoReconcileReasonNotDispatched) {
		t.Fatalf("output should name why the heal declined:\n%s", out.String())
	}
}

// TestSyncPushAndCommitPushParityOnDisjointDivergence pins #3069's parity against real git:
// for the same disjoint divergence (a peer published peer.txt, we committed mine.txt),
// `fak commit --push` (safecommit.Commit) and `fak sync push` both publish, and both
// origins end with the same file set and a merge whose parents are our commit + the peer's.
func TestSyncPushAndCommitPushParityOnDisjointDivergence(t *testing.T) {
	ctx := context.Background()

	// commit --push side.
	commitBare, commitLocal := syncDisjointDivergenceFixture(t)
	syncWriteFile(t, filepath.Join(commitLocal, "mine.txt"), "mine\n")
	cres, err := safecommit.Commit(ctx, safecommit.Options{
		Dir: commitLocal, Paths: []string{"mine.txt"}, Message: "mine", Push: true, Trunk: "main",
	})
	if err != nil {
		t.Fatalf("safecommit.Commit: %v", err)
	}
	if !cres.Pushed {
		t.Fatalf("commit --push did not publish: reason=%q detail=%q", cres.Reason, cres.Detail)
	}

	// sync push side.
	syncBare, syncLocal := syncDisjointDivergenceFixture(t)
	syncWriteFile(t, filepath.Join(syncLocal, "mine.txt"), "mine\n")
	syncGit(t, syncLocal, "add", "--", "mine.txt")
	syncGit(t, syncLocal, "commit", "-q", "-m", "mine")
	oldWorktree := syncWorktree
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) { return safesync.Worktree{}, false }
	t.Cleanup(func() { syncWorktree = oldWorktree })
	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", syncLocal, "--remote", "origin", "--branch", "main", "--json"})
	if code != syncExitOK {
		t.Fatalf("sync push exit=%d stderr=%q stdout=%s, want ok (same outcome as commit --push)", code, errb.String(), out.String())
	}

	for name, bare := range map[string]string{"commit --push": commitBare, "sync push": syncBare} {
		tree := syncGitOut(t, bare, "ls-tree", "-r", "--name-only", "refs/heads/main")
		if strings.Join(strings.Fields(tree), ",") != "initial.txt,mine.txt,peer.txt" {
			t.Fatalf("%s origin tree = %q, want initial.txt,mine.txt,peer.txt", name, tree)
		}
		parents := strings.Fields(syncGitOut(t, bare, "rev-list", "--parents", "-n", "1", "refs/heads/main"))
		if len(parents) != 3 {
			t.Fatalf("%s origin tip %v, want a two-parent safe-disjoint merge", name, parents)
		}
	}
}

// syncDisjointDivergenceFixture builds bare origin + local clone, then has a peer clone
// publish peer.txt so local is behind by one disjoint commit.
func syncDisjointDivergenceFixture(t *testing.T) (bare, local string) {
	t.Helper()
	root := t.TempDir()
	seed := filepath.Join(root, "seed")
	bare = filepath.Join(root, "origin.git")
	local = filepath.Join(root, "local")
	peer := filepath.Join(root, "peer")
	syncGit(t, root, "init", "-q", "-b", "main", seed)
	syncGitIdentity(t, seed)
	syncWriteFile(t, filepath.Join(seed, "initial.txt"), "initial\n")
	syncGit(t, seed, "add", "--", "initial.txt")
	syncGit(t, seed, "commit", "-q", "-m", "initial")
	syncGit(t, root, "clone", "-q", "--bare", seed, bare)
	syncGit(t, root, "clone", "-q", bare, local)
	syncGitIdentity(t, local)
	syncGit(t, root, "clone", "-q", bare, peer)
	syncGitIdentity(t, peer)
	syncWriteFile(t, filepath.Join(peer, "peer.txt"), "peer\n")
	syncGit(t, peer, "add", "--", "peer.txt")
	syncGit(t, peer, "commit", "-q", "-m", "peer")
	syncGit(t, peer, "push", "-q", "origin", "main")
	return bare, local
}

func syncGitIdentity(t *testing.T, dir string) {
	t.Helper()
	syncGit(t, dir, "config", "user.name", "fak test")
	syncGit(t, dir, "config", "user.email", "fak-test@example.invalid")
	syncGit(t, dir, "config", "commit.gpgsign", "false")
	syncGit(t, dir, "config", "core.autocrlf", "false")
}

func syncGitOut(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git %v in %s: %v", args, dir, err)
	}
	return strings.TrimSpace(string(out))
}
