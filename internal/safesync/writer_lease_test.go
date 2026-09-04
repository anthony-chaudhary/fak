package safesync

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

// TestApplyHeartbeatOutlivesTTLWithoutSelfReclaim is the #4612 barrier witness for the
// "apply fits inside the TTL" assumption: an apply window parked PAST its own lease TTL
// must not have its live lease reclaimed mid-window as crash residue — the keepAlive
// heartbeat renews the record, so a concurrent managed writer stays refused for the whole
// long window. The barrier first WITNESSES a moment where the wall clock has outlived
// acquire+ttl while the on-disk record is still fresh, then judges the peer at that
// pinned instant, so the refuse-vs-reclaim decision never races the next heartbeat tick.
func TestApplyHeartbeatOutlivesTTLWithoutSelfReclaim(t *testing.T) {
	clone := behindClone(t)
	const ttl = 2 * time.Second
	leasePath := filepath.Join(clone, ".git", writerLeaseFile)

	var (
		peerErr        error
		witnessedNow   time.Time
		witnessed      WriterLeaseInfo
		renewedPastTTL bool
	)
	opts := Options{Repo: clone, Remote: "origin", Branch: "work", LeaseOwner: "slow-apply", WriterLeaseTTL: ttl}
	opts.barrier = func() {
		start := time.Now()
		deadline := start.Add(15 * time.Second)
		for time.Now().Before(deadline) {
			if now := time.Now(); now.Sub(start) > ttl {
				rec, err := readLease(leasePath)
				if err == nil && !leaseStale(rec, now, ttl) {
					witnessedNow, witnessed, renewedPastTTL = now, rec, true
					break
				}
			}
			time.Sleep(25 * time.Millisecond)
		}
		if !renewedPastTTL {
			return // let apply finish; the assertion below reports the missing renewal
		}
		// The peer judges staleness at the witnessed instant (a pinned clock).
		l, err := AcquireWriterLease(clone, "peer", func() time.Time { return witnessedNow }, ttl)
		peerErr = err
		if l != nil {
			_ = l.Release()
		}
	}

	info, err := Apply(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !renewedPastTTL {
		rec, rerr := readLease(leasePath)
		t.Fatalf("lease was never observed fresh past its TTL — the heartbeat did not renew it: record=%+v readErr=%v", rec, rerr)
	}
	if !info.Applied || !info.OK {
		t.Fatalf("apply did not land after its long window: %+v", info)
	}
	if witnessed.RenewedUnix <= witnessed.AcquiredUnix {
		t.Fatalf("record was never renewed past its acquire stamp: %+v", witnessed)
	}
	// The counterfactual that makes this a real barrier: judged by its ACQUIRE stamp
	// alone (pre-#4612 behavior), the record was already reclaimable at the witnessed
	// instant — only the renew kept the window protected.
	if unrenewed := (WriterLeaseInfo{AcquiredUnix: witnessed.AcquiredUnix}); !leaseStale(unrenewed, witnessedNow, ttl) {
		t.Fatal("witness instant does not lie past the TTL; the barrier broke too early")
	}
	var held *WriterLeaseHeldError
	if !errors.As(peerErr, &held) {
		t.Fatalf("peer writer past the TTL = %v, want WriterLeaseHeldError (no self-reclaim mid-window)", peerErr)
	}
	if held.Info.Owner != "slow-apply" {
		t.Fatalf("refusal names holder %q, want slow-apply", held.Info.Owner)
	}
	// The heartbeat stopped with the window and the lease was released: it is free again.
	l, err := AcquireWriterLease(clone, "after", nil, 0)
	if err != nil {
		t.Fatalf("lease not free after apply returned (heartbeat still running?): %v", err)
	}
	_ = l.Release()
}

// TestWriterLeaseCrossHostPeerIsHonored is the #4612 cross-host witness: the lock file
// is exactly as cross-machine as the worktree it protects (a writer on another host
// reaches these bytes only through a shared mount, which carries the per-worktree git
// dir — hence this lock — with it), so honoring must key on the shared file + TTL
// ALONE. A foreign host's record carries a pid that means nothing locally; any local
// pid/hostname liveness probe would misjudge the live peer as dead and reclaim
// mid-window. And a crashed peer HOST must still not wedge the mount: TTL — the only
// liveness rule valid cross-host — reclaims its residue exactly as a local crash.
func TestWriterLeaseCrossHostPeerIsHonored(t *testing.T) {
	clone := behindClone(t)
	leasePath := filepath.Join(clone, ".git", writerLeaseFile)

	// A live peer-host writer on the same shared-mount checkout: foreign hostname, a pid
	// that exists on no local process table, fresh acquire stamp.
	foreign := WriterLeaseInfo{Owner: "peer-host-writer", PID: 1 << 30, Host: "other-machine", AcquiredUnix: time.Now().Unix()}
	enc, err := json.Marshal(foreign)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(leasePath, enc, 0o644); err != nil {
		t.Fatal(err)
	}

	var held *WriterLeaseHeldError
	if _, err := AcquireWriterLease(clone, "local-writer", nil, time.Minute); !errors.As(err, &held) {
		t.Fatalf("cross-host fak-managed writer was not refused while the peer host holds the lease: %v", err)
	}
	if held.Info.Host != "other-machine" || held.Info.Owner != "peer-host-writer" {
		t.Fatalf("refusal does not carry the peer host's record: %+v", held.Info)
	}
	if !strings.Contains(held.Error(), "on other-machine") {
		t.Fatalf("refusal is not diagnosable cross-machine (no holder host): %q", held.Error())
	}

	// Apply — the #4240 window itself — refuses the same way, as a value naming the holder.
	info, aerr := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if aerr != nil {
		t.Fatal(aerr)
	}
	if info.OK || info.Applied || info.Lease == nil || info.Lease.Host != "other-machine" {
		t.Fatalf("apply did not refuse on the peer host's live lease: %+v", info)
	}

	// A crashed peer host's residue: same foreign identity, acquire stamp far past TTL.
	stale := WriterLeaseInfo{Owner: "crashed-peer-host", PID: 1 << 30, Host: "other-machine", AcquiredUnix: 1000}
	enc, err = json.Marshal(stale)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(leasePath, enc, 0o644); err != nil {
		t.Fatal(err)
	}
	l, err := AcquireWriterLease(clone, "local-writer", nil, time.Minute)
	if err != nil {
		t.Fatalf("crashed peer host's stale lease was not reclaimed (cross-host wedge): %v", err)
	}
	_ = l.Release()
}

// TestWriterLeaseRefreshMovesWindowAndIsOwnerChecked pins Refresh's contract (#4612): a
// renew moves the staleness window forward for a record we still own (even one already
// past its TTL — identity, not freshness, decides), Release keeps working after a renew,
// and a holder that LOST the lease to a reclaiming peer gets ErrWriterLeaseLost and never
// clobbers the peer's live record — nor recreates a released one.
func TestWriterLeaseRefreshMovesWindowAndIsOwnerChecked(t *testing.T) {
	clone := behindClone(t)
	leasePath := filepath.Join(clone, ".git", writerLeaseFile)

	// Acquired far in the past: already reclaimable. A refresh with the real clock makes
	// it fresh again, so a peer is refused where it would have reclaimed.
	past := func() time.Time { return time.Unix(1000, 0) }
	l1, err := AcquireWriterLease(clone, "long-window", past, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if err := l1.Refresh(nil); err != nil {
		t.Fatalf("refresh of our own (expired but unreclaimed) record: %v", err)
	}
	var held *WriterLeaseHeldError
	if _, err := AcquireWriterLease(clone, "peer", nil, time.Minute); !errors.As(err, &held) {
		t.Fatalf("refreshed lease was reclaimed by a peer: %v", err)
	}
	if held.Info.RenewedUnix == 0 {
		t.Fatalf("peer's refusal does not carry the renewed record: %+v", held.Info)
	}

	// Release still works after a renew: the owner-check tracks the renewed record.
	if err := l1.Release(); err != nil {
		t.Fatal(err)
	}
	l2, err := AcquireWriterLease(clone, "next-holder", nil, time.Minute)
	if err != nil {
		t.Fatalf("lease not free after a renewed release: %v", err)
	}

	// The old holder lost the lease: its refresh reports the loss and leaves the live
	// peer record untouched.
	if err := l1.Refresh(nil); !errors.Is(err, ErrWriterLeaseLost) {
		t.Fatalf("refresh after losing the lease = %v, want ErrWriterLeaseLost", err)
	}
	cur, err := readLease(leasePath)
	if err != nil || cur.Owner != "next-holder" {
		t.Fatalf("lost holder's refresh clobbered the live peer record: %+v (err=%v)", cur, err)
	}

	// Refresh after release reports the loss too, and never recreates the lock file.
	if err := l2.Release(); err != nil {
		t.Fatal(err)
	}
	if err := l2.Refresh(nil); !errors.Is(err, ErrWriterLeaseLost) {
		t.Fatalf("refresh after release = %v, want ErrWriterLeaseLost", err)
	}
	if _, err := os.Stat(leasePath); !os.IsNotExist(err) {
		t.Fatalf("refresh recreated a released lease: stat err=%v", err)
	}
}

// TestApplySuppressesFastForwardAfterWriterLeaseLoss is the #5974 write-boundary
// witness: a displaced holder may finish its assessment, but it must not publish the
// fast-forward after the heartbeat observes that a peer reclaimed the lease.
func TestApplySuppressesFastForwardAfterWriterLeaseLoss(t *testing.T) {
	clone := behindClone(t)
	headBefore := revString(t, clone, "HEAD")
	gitDir, err := worktreeGitDir(clone)
	if err != nil {
		t.Fatal(err)
	}
	leasePath := filepath.Join(gitDir, writerLeaseFile)
	mergeCalls := 0
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && args[0] == "merge" {
			mergeCalls++
		}
		return RealRunner(ctx, repo, args...)
	}
	opts := Options{
		Repo: clone, Remote: "origin", Branch: "work", Runner: runner,
		LeaseOwner: "displaced-holder", WriterLeaseTTL: 40 * time.Millisecond,
	}
	opts.barrier = func() {
		if err := os.Remove(leasePath); err != nil {
			t.Fatal(err)
		}
		peer := WriterLeaseInfo{Owner: "reclaiming-peer", PID: os.Getpid(), AcquiredUnix: time.Now().Unix()}
		ok, err := writeLeaseExclusive(leasePath, peer)
		if err != nil || !ok {
			t.Fatalf("install reclaiming peer lease: ok=%v err=%v", ok, err)
		}
		deadline := time.Now().Add(time.Second)
		for {
			// The heartbeat stops only after observing the displacement. Waiting for its
			// goroutine count indirectly would be racy; the peer record remaining stable
			// across several heartbeat intervals plus a bounded delay gives it time to
			// close the holder's lost signal before Apply crosses the write boundary.
			if time.Now().After(deadline) {
				t.Fatal("heartbeat did not observe displaced lease in time")
			}
			time.Sleep(15 * time.Millisecond)
			cur, readErr := readLease(leasePath)
			if readErr == nil && cur.Owner == peer.Owner {
				// Two intervals are enough for the 10ms heartbeat; wait one more to avoid
				// scheduling the Apply goroutine ahead of the observation.
				time.Sleep(15 * time.Millisecond)
				return
			}
		}
	}

	info, err := Apply(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if mergeCalls != 0 {
		t.Fatalf("displaced holder published %d merge(s), want none", mergeCalls)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("displaced holder changed HEAD: got %s want %s", got, headBefore)
	}
	if info.OK || info.Applied || !strings.Contains(info.Reason, "lease lost") {
		t.Fatalf("apply = %+v, want refused lost-lease suppression", info)
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

// TestWriterLeaseReapsDeadProcessViaPIDCheck proves Issue #11234:
// When an on-disk lease was held by a dead process on the local machine,
// the OS PID check detects the dead process and reaps the stale lease immediately
// instead of waiting for the full TTL or declaring LEASE_OWNER_UNAVAILABLE.
func TestWriterLeaseReapsDeadProcessViaPIDCheck(t *testing.T) {
	clone := behindClone(t)
	gitDir, err := worktreeGitDir(clone)
	if err != nil {
		t.Fatal(err)
	}
	leasePath := filepath.Join(gitDir, writerLeaseFile)
	host, _ := os.Hostname()

	// Pick a PID that is provably dead on this system
	deadPID := 99999999
	for isProcessAlive(deadPID) {
		deadPID++
	}

	deadHolder := WriterLeaseInfo{
		Owner:        "dead-process-holder",
		PID:          deadPID,
		Host:         host,
		AcquiredUnix: time.Now().Unix(), // fresh timestamp, well within TTL
		Mode:         "exclusive-write",
	}
	enc, err := json.Marshal(deadHolder)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(leasePath, enc, 0o644); err != nil {
		t.Fatal(err)
	}

	// Direct acquireWriterLease must reap the dead process lease immediately
	l, err := AcquireWriterLease(clone, "reaping-writer", nil, time.Hour)
	if err != nil {
		t.Fatalf("expected dead process lease to be reaped immediately, got err: %v", err)
	}
	if l.Info().Owner != "reaping-writer" {
		t.Fatalf("expected reaping-writer owner, got %s", l.Info().Owner)
	}
	_ = l.Release()

	// AcquireQueuedWriterLease must also reap dead process lease without failing
	if err := os.WriteFile(leasePath, enc, 0o644); err != nil {
		t.Fatal(err)
	}
	ql, err := AcquireQueuedWriterLease(context.Background(), clone, "reaping-queued-writer", nil, time.Hour, time.Second)
	if err != nil {
		t.Fatalf("AcquireQueuedWriterLease expected dead process to be reaped, got err: %v", err)
	}
	if ql.Info().Owner != "reaping-queued-writer" {
		t.Fatalf("expected reaping-queued-writer owner, got %s", ql.Info().Owner)
	}
	_ = ql.Release()
}

// TestConcurrentSafesyncUnderContention verifies that concurrent sync routines
// complete without unhandled LEASE_OWNER_UNAVAILABLE timeouts under contention (#11234).
func TestConcurrentSafesyncUnderContention(t *testing.T) {
	clone := behindClone(t)
	const concurrent = 6
	errCh := make(chan error, concurrent)

	for i := 0; i < concurrent; i++ {
		go func(id int) {
			opts := Options{
				Repo:       clone,
				Remote:     "origin",
				Branch:     "work",
				LeaseOwner: fmt.Sprintf("routine-%d", id),
			}
			info, err := Apply(context.Background(), opts)
			if err != nil {
				errCh <- fmt.Errorf("routine %d error: %w", id, err)
				return
			}
			if errors.Is(err, ErrLeaseOwnerUnavailable) || info.Reason == ReasonLeaseOwnerUnavailable {
				errCh <- fmt.Errorf("routine %d hit LEASE_OWNER_UNAVAILABLE", id)
				return
			}
			errCh <- nil
		}(i)
	}

	for i := 0; i < concurrent; i++ {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}

// BenchmarkConcurrentSafesyncUnderContention proves Issue #11234:
// Proves concurrent sync routines complete quickly without unhandled LEASE_OWNER_UNAVAILABLE timeouts.
func BenchmarkConcurrentSafesyncUnderContention(b *testing.B) {
	clone := behindClone(b)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		routineID := fmt.Sprintf("bench-worker-%d-%d", os.Getpid(), time.Now().UnixNano())
		for pb.Next() {
			opts := Options{
				Repo:       clone,
				Remote:     "origin",
				Branch:     "work",
				LeaseOwner: routineID,
			}
			info, err := Apply(context.Background(), opts)
			if err != nil {
				b.Fatalf("Apply under contention error: %v", err)
			}
			if errors.Is(err, ErrLeaseOwnerUnavailable) || info.Reason == ReasonLeaseOwnerUnavailable {
				b.Fatalf("unhandled LEASE_OWNER_UNAVAILABLE timeout under contention")
			}
		}
	})
}
