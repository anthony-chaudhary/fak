package safesync

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2s lane=default
func TestReconcileFetchFailureStopsBeforeRouting(t *testing.T) {
	t.Parallel()
	origin, clone := setupTestOriginAndClone(t)
	writeFile(t, filepath.Join(origin, "remote_only.txt"), "new remote work\n")
	git(t, origin, "add", "remote_only.txt")
	git(t, origin, "commit", "-m", "advance remote")
	remoteHead := revString(t, origin, "HEAD")
	head := revString(t, clone, "HEAD")
	tracking := revString(t, clone, "refs/remotes/origin/work")
	if head != tracking || head == remoteHead {
		t.Fatal("fixture must have a stale tracking ref")
	}

	// A missing local transport is a real deterministic Git failure, not a
	// replacement Runner; the old tracking ref deliberately remains readable.
	git(t, clone, "remote", "set-url", "origin", filepath.Join(t.TempDir(), "absent-origin"))
	opts := ReconcileOptions{Repo: clone, Remote: "origin", Branch: "work", Goal: "integrate"}
	cached, err := RouteReconciliation(context.Background(), opts)
	if err != nil || cached.Route != RouteNoop {
		t.Fatalf("explicit no-fetch control: assessment=%+v error=%v", cached, err)
	}

	writeFile(t, filepath.Join(clone, "shared.txt"), "unrelated dirty sentinel\n")
	indexPath := filepath.Join(clone, ".git", "index")
	indexBefore, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	opts.Fetch, opts.Apply = true, true
	got, err := RouteReconciliation(context.Background(), opts)
	var gitErr *GitError
	if !errors.As(err, &gitErr) || gitErr.Code == 0 || len(gitErr.Args) == 0 || gitErr.Args[0] != "fetch" {
		t.Fatalf("requested failed fetch must return its Git error, got assessment=%+v error=%v", got, err)
	}
	if !strings.Contains(err.Error(), "reconcile: fetch origin work") {
		t.Fatalf("missing fetch context: %v", err)
	}
	if got.OK || got.Route != "" || got.Applied || got.Execution != nil || got.Packet != nil || got.ExecuteReceipt != nil {
		t.Fatalf("failed acquisition must stop before route/apply/packet: %+v", got)
	}
	if revString(t, clone, "HEAD") != head || revString(t, clone, "refs/remotes/origin/work") != tracking {
		t.Fatal("failed fetch changed local refs")
	}
	indexAfter, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(indexBefore, indexAfter) {
		t.Fatal("failed acquisition changed the index")
	}
	if readFile(t, filepath.Join(clone, "shared.txt")) != "unrelated dirty sentinel\n" {
		t.Fatal("failed acquisition changed unrelated dirty bytes")
	}

	// Restoring the actual local origin proves successful fetch still routes
	// against the newly acquired target; this control does not apply it.
	git(t, clone, "remote", "set-url", "origin", origin)
	opts.Apply = false
	fresh, err := RouteReconciliation(context.Background(), opts)
	if err != nil || fresh.Target != remoteHead || fresh.Route != RouteApply || fresh.Applied {
		t.Fatalf("successful fetch control: assessment=%+v error=%v", fresh, err)
	}
	if revString(t, clone, "HEAD") != head || readFile(t, filepath.Join(clone, "shared.txt")) != "unrelated dirty sentinel\n" {
		t.Fatal("assessment changed local work")
	}
}
