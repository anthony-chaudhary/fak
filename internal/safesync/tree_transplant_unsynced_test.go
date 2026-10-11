package safesync

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// divergeDisjoint builds the incident shape: origin gains incoming files, the
// clone gains a disjoint local commit.
func divergeDisjoint(t *testing.T) (origin, clone string, incoming []string) {
	t.Helper()
	origin, clone = setupTestOriginAndClone(t)
	incoming = []string{"in_a.txt", "nested/in_b.txt"}
	for _, p := range incoming {
		full := filepath.Join(origin, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		writeFile(t, full, "incoming "+p+"\n")
	}
	git(t, origin, append([]string{"add"}, incoming...)...)
	git(t, origin, "commit", "-m", "remote adds incoming files")

	writeFile(t, filepath.Join(clone, "local_only.txt"), "local\n")
	git(t, clone, "add", "local_only.txt")
	git(t, clone, "commit", "-m", "local adds local_only.txt")
	return origin, clone, incoming
}

func isIncomingCheckout(args []string) bool {
	return len(args) >= 3 && args[0] == "checkout" && args[2] == "--"
}

func TestReconcileHealsTransplantCheckoutFailureWithIndexAhead(t *testing.T) {
	_, clone, incoming := divergeDisjoint(t)

	failed := false
	merged := false
	runner := func(ctx context.Context, dir string, args ...string) RunResult {
		if len(args) > 0 && args[0] == "merge" {
			merged = true
		}
		if isIncomingCheckout(args) && !failed {
			failed = true
			// Incident shape: the index already holds the merge blobs, the files never landed.
			_ = RealRunner(ctx, dir, "read-tree", args[1])
			return RunResult{Code: 1, Stderr: []byte("simulated checkout failure under load")}
		}
		return RealRunner(ctx, dir, args...)
	}

	res, err := RouteReconciliation(context.Background(), ReconcileOptions{
		Repo: clone, Remote: "origin", Branch: "work", Goal: "integrate",
		Apply: true, Fetch: true, Runner: runner,
	})
	if err != nil {
		t.Fatalf("RouteReconciliation: %v", err)
	}
	if !failed {
		t.Fatal("step-4 checkout was never attempted; test did not exercise the failure")
	}
	if merged {
		t.Fatal("merge fallback ran after the transplant had already advanced the ref")
	}
	if res.Route != RouteDisjointIntegrate || !res.Applied || res.Execution == nil || !res.Execution.Success {
		t.Fatalf("route=%s applied=%t execution=%v; want healed disjoint integrate", res.Route, res.Applied, res.Execution != nil && res.Execution.Success)
	}
	for _, p := range incoming {
		if _, err := os.Stat(filepath.Join(clone, filepath.FromSlash(p))); err != nil {
			t.Fatalf("incoming %s not on disk after reported success: %v", p, err)
		}
	}
	for _, p := range incoming {
		if got, want := gitOutput(t, clone, "hash-object", "--", p), gitOutput(t, clone, "rev-parse", res.AppliedCommit+":"+p); got != want {
			t.Fatalf("%s on disk hashes %s, want merge blob %s", p, got, want)
		}
	}
}

func TestReconcileRefusesSuccessWhenIncomingPathsNeverLand(t *testing.T) {
	_, clone, _ := divergeDisjoint(t)

	merged := false
	runner := func(ctx context.Context, dir string, args ...string) RunResult {
		switch {
		case len(args) > 0 && args[0] == "merge":
			merged = true
		case isIncomingCheckout(args):
			return RunResult{Code: 1, Stderr: []byte("simulated checkout failure")}
		case len(args) > 2 && args[0] == "read-tree" && args[2] == "-u":
			return RunResult{Code: 1, Stderr: []byte("simulated read-tree failure")}
		}
		return RealRunner(ctx, dir, args...)
	}

	res, err := RouteReconciliation(context.Background(), ReconcileOptions{
		Repo: clone, Remote: "origin", Branch: "work", Goal: "integrate",
		Apply: true, Fetch: true, Runner: runner,
	})
	if err != nil {
		t.Fatalf("RouteReconciliation: %v", err)
	}
	if merged {
		t.Fatal("merge fallback ran after the transplant had already advanced the ref")
	}
	if res.Route != RouteDisjointIntegrate {
		t.Fatalf("route = %s, want %s", res.Route, RouteDisjointIntegrate)
	}
	if res.Applied || res.OK || res.Status == StateSynchronized {
		t.Fatalf("applied=%t ok=%t status=%q; want unapplied, not ok, not synchronized", res.Applied, res.OK, res.Status)
	}
	if res.Reason != "WORKTREE_UNSYNCED" {
		t.Fatalf("reason = %s, want WORKTREE_UNSYNCED", res.Reason)
	}
	if res.Execution == nil || res.Execution.Success || !res.Execution.Applied {
		t.Fatalf("execution = %+v; want ref-applied, unsuccessful", res.Execution)
	}
	if res.Execution.NewHead != res.AppliedCommit || res.AppliedCommit == "" {
		t.Fatalf("new head %s, applied commit %s; want HEAD at the transplant commit", res.Execution.NewHead, res.AppliedCommit)
	}
}

func TestTransplantReturnsTypedErrorAfterRefAdvance(t *testing.T) {
	_, clone, _ := divergeDisjoint(t)
	git(t, clone, "fetch", "origin")
	headSHA := revString(t, clone, "HEAD")
	targetSHA := revString(t, clone, "origin/work")

	runner := func(ctx context.Context, dir string, args ...string) RunResult {
		if isIncomingCheckout(args) {
			return RunResult{Code: 1}
		}
		return RealRunner(ctx, dir, args...)
	}
	commit, err := TransplantDisjointTreeWithRunner(context.Background(), runner, clone, "", headSHA, targetSHA, "origin/work")
	if err == nil {
		t.Fatal("transplant reported success with incoming paths unwritten")
	}
	// The caller must learn which commit the ref now names, so it heals instead of re-merging.
	if commit == "" || revString(t, clone, "HEAD") != commit {
		t.Fatalf("commit %q; want the advanced HEAD", commit)
	}
}
