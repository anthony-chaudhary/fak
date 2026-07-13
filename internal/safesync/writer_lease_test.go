package safesync

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestApplyHoldsWriterLeaseAgainstConcurrentManagedWriter is the #4240 witness: a real
// temp-repo run with a barrier fired INSIDE Apply while it holds the worktree writer
// lease, and a genuinely concurrent second fak-managed writer that must be refused
// (WriterLeaseHeldError) for the whole assess+apply window — never entering the checkout
// window — while the fast-forward still lands and unrelated dirty work is preserved.
func TestApplyHoldsWriterLeaseAgainstConcurrentManagedWriter(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "mine.txt"), "local") // unrelated dirty work

	barrierReached := make(chan struct{})
	proceed := make(chan struct{})
	var secondErr error

	opts := Options{Repo: clone, Remote: "origin", Branch: "work", LeaseOwner: "sync-apply"}
	opts.barrier = func() {
		close(barrierReached) // apply now holds the lease
		<-proceed             // stay parked, still holding it, until the peer has tried
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		<-barrierReached
		// A second fak-managed writer honors the lease: it must be REFUSED while apply
		// holds it, so it can never overwrite a classified path mid-window.
		l, err := AcquireWriterLease(clone, "peer-writer", nil, 0)
		secondErr = err
		if l != nil {
			_ = l.Release()
		}
		close(proceed) // let apply finish and release the lease
	}()

	info, err := Apply(context.Background(), opts)
	<-done
	if err != nil {
		t.Fatal(err)
	}
	if !info.Applied || !info.OK {
		t.Fatalf("apply did not fast-forward while holding its own lease: %+v", info)
	}

	var held *WriterLeaseHeldError
	if !errors.As(secondErr, &held) {
		t.Fatalf("second managed writer was not refused while apply held the lease: err=%v", secondErr)
	}
	if held.Info.Owner != "sync-apply" {
		t.Fatalf("refusal names holder %q, want the apply lease owner sync-apply", held.Info.Owner)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q, want fast-forwarded v2", got)
	}
	if got := readFile(t, filepath.Join(clone, "mine.txt")); got != "local" {
		t.Fatalf("unrelated dirty work not preserved: %q", got)
	}
	// The lease is released once apply returns: a fresh writer can now take it.
	l, err := AcquireWriterLease(clone, "after", nil, 0)
	if err != nil {
		t.Fatalf("lease not released after apply returned: %v", err)
	}
	_ = l.Release()
}

// TestWriterLeaseReleaseFreesAndStaleIsReclaimed covers the crash/release recovery half
// of the witness: an explicit Release frees the lease, a live lease refuses peers, and a
// lease whose holder crashed (TTL expired) is reclaimed while a still-fresh lease is not.
func TestWriterLeaseReleaseFreesAndStaleIsReclaimed(t *testing.T) {
	clone := behindClone(t)
	var held *WriterLeaseHeldError

	// Release frees the lease.
	l1, err := AcquireWriterLease(clone, "w1", nil, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := AcquireWriterLease(clone, "w2", nil, time.Minute); !errors.As(err, &held) {
		t.Fatalf("acquire while held = %v, want WriterLeaseHeldError", err)
	}
	if err := l1.Release(); err != nil {
		t.Fatal(err)
	}
	l3, err := AcquireWriterLease(clone, "w3", nil, time.Minute)
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	if err := l3.Release(); err != nil {
		t.Fatal(err)
	}

	// Crash recovery: a holder that died without releasing left a lease whose acquire
	// time is far in the past, so its TTL is already expired at real "now" and a later
	// writer reclaims it.
	past := func() time.Time { return time.Unix(1000, 0) }
	crashed, err := AcquireWriterLease(clone, "crashed", past, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	_ = crashed // never released — crash residue on disk

	recovered, err := AcquireWriterLease(clone, "recovered", nil, time.Minute)
	if err != nil {
		t.Fatalf("stale (crashed) lease was not reclaimed: %v", err)
	}
	if recovered.Info().Owner != "recovered" {
		t.Fatalf("reclaimed lease owner = %q, want recovered", recovered.Info().Owner)
	}

	// A fresh, non-stale lease is NOT reclaimable — recovery reclaims only crash residue.
	if _, err := AcquireWriterLease(clone, "intruder", nil, time.Minute); !errors.As(err, &held) {
		t.Fatalf("fresh lease should not be reclaimable: %v", err)
	}
	if err := recovered.Release(); err != nil {
		t.Fatal(err)
	}
}

// TestApplyReturnsIndeterminateOnPartialFastForward proves a fast-forward that dies
// partway (leaving an in-progress MERGE_HEAD) is reported as a typed, content-preserving
// INDETERMINATE verdict — not a clean refusal and not a swallowed plain error — so a
// caller knows the worktree may be partially updated and must recover before re-syncing.
func TestApplyReturnsIndeterminateOnPartialFastForward(t *testing.T) {
	clone := behindClone(t)
	headBefore := revString(t, clone, "HEAD")
	injected := false
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && args[0] == "merge" {
			// A fast-forward that started mutating and died partway: leave MERGE_HEAD
			// behind and report a failure that is NOT a clean pre-merge refusal.
			writeFile(t, filepath.Join(clone, ".git", "MERGE_HEAD"), headBefore+"\n")
			injected = true
			return RunResult{Code: 128, Stderr: []byte("fatal: interrupted while updating worktree\n")}
		}
		return RealRunner(ctx, repo, args...)
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatalf("indeterminate partial ff should be a typed verdict, not an error: %v", err)
	}
	if !injected {
		t.Fatal("runner did not intercept the merge")
	}
	if !info.Indeterminate || info.Applied || info.OK {
		t.Fatalf("apply = %+v, want typed indeterminate verdict", info)
	}
	if !strings.Contains(info.Reason, "indeterminate") {
		t.Fatalf("reason should name the indeterminate recovery, got %q", info.Reason)
	}
	_ = os.Remove(filepath.Join(clone, ".git", "MERGE_HEAD"))
}
