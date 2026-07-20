package leaseref

import (
	"fmt"
	"os/exec"
	"sync"
	"testing"
	"time"
)

// concurrent_claim_test.go is the collision-safe-board witness for #2890 (Hermes-inspiration
// epic #2871). Hermes' kanban (`hermes_state.py` SQLite board) requires a SINGLE-DISPATCHER
// posture — exactly one gateway may keep `dispatch_in_gateway: true`; every other gateway must
// disable it to avoid SQLite/WAL write contention. That single-writer constraint is the scaling
// smell the issue targets.
//
// fak's lease-backed board has no such requirement: each claim is an `update-ref` OLD-VALUE
// compare-and-swap against a per-lane git ref (refs/fak/locks/<id>), so N workers on the same
// host claim concurrently with no dispatcher serializing them. These two tests are the "First
// slice" witness the issue asks for — concurrent claimers proving (1) no double-claim on a
// contended lane and (2) no single-writer requirement across disjoint lanes.
//
// Both exercise the REAL git binary (the CAS is git's, not a fake's) and skip when git is
// unavailable, matching TestRealGitRoundTrip / TestReapBatchRealGit; they run under the WSL
// suite and native where git is on PATH.

// initRealGitRepo initializes a throwaway git repo the lease store writes its refs into,
// mirroring the setup in TestRealGitRoundTrip. It skips the test when git is unavailable.
func initRealGitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

// TestConcurrentClaimNoDoubleClaim: N workers race AcquireFenced on the SAME lane id at the
// same instant. Exactly ONE is admitted; every other is refused with a lease-collision verdict
// (LEASE_CONTENDED when it lost the CAS, LEASE_HELD when it read the winner's live lease first)
// — never a second OK. The board cannot double-claim a contended lane even with no coordinating
// dispatcher, because git's update-ref old-value CAS is the arbiter.
func TestConcurrentClaimNoDoubleClaim(t *testing.T) {
	s := NewWithRunner(gitRunner, initRealGitRepo(t))
	now := time.Unix(1_000_000, 0) // fixed instant: TTL 300s never expires mid-race

	const workers = 12
	type outcome struct {
		rec Record
		v   FenceVerdict
		err error
	}
	results := make([]outcome, workers)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start // release all goroutines together to maximize the real race
			rec := Record{
				ID:         "contended-lane",
				TreeGlobs:  []string{"internal/kernel/**"},
				Holder:     fmt.Sprintf("worker-%d", i),
				TTLSeconds: 300,
			}
			r, v, err := s.AcquireFenced(ctx(), rec, now)
			results[i] = outcome{r, v, err}
		}(i)
	}
	close(start)
	wg.Wait()

	winners := 0
	var winner string
	for i, o := range results {
		if o.err != nil {
			t.Fatalf("worker %d AcquireFenced errored (infra failure, not a verdict): %v", i, o.err)
		}
		if o.v.OK {
			winners++
			winner = o.rec.Holder
			if o.rec.Generation != 1 {
				t.Fatalf("winner %d generation = %d, want 1 (a fresh acquire is generation 1)", i, o.rec.Generation)
			}
			continue
		}
		// Every non-winner must be refused by a lease-collision reason, not silently dropped.
		if o.v.Reason != ReasonLeaseContended && o.v.Reason != ReasonLeaseHeld {
			t.Fatalf("worker %d refusal reason = %q, want %s or %s (%+v)",
				i, o.v.Reason, ReasonLeaseContended, ReasonLeaseHeld, o.v)
		}
	}
	if winners != 1 {
		t.Fatalf("concurrent claim on one lane admitted %d winners, want exactly 1 (double-claim)", winners)
	}

	// The single admitted holder is the one actually recorded on the ref — no loser's write
	// clobbered it, and the fencing token did not double-increment.
	got, ok, err := s.Get(ctx(), "contended-lane")
	if err != nil || !ok {
		t.Fatalf("Get contended-lane: ok=%v err=%v", ok, err)
	}
	if got.Holder != winner {
		t.Fatalf("recorded holder = %q, want the sole winner %q", got.Holder, winner)
	}
	if got.Generation != 1 {
		t.Fatalf("recorded generation = %d, want 1 (no double-increment under the race)", got.Generation)
	}
}

// TestConcurrentClaimDisjointLanesNoSingleWriter: N workers each race AcquireFenced on a
// DISTINCT (disjoint) lane at the same instant. ALL are admitted — there is no single-writer
// bottleneck, no dispatcher any worker must funnel through. This is the property Hermes' board
// lacks: fak claims disjoint lanes fully concurrently because each lane is an independent git
// ref, not a row in one WAL-contended SQLite file.
func TestConcurrentClaimDisjointLanesNoSingleWriter(t *testing.T) {
	s := NewWithRunner(gitRunner, initRealGitRepo(t))
	now := time.Unix(1_000_000, 0)

	const workers = 12
	type outcome struct {
		v   FenceVerdict
		err error
	}
	results := make([]outcome, workers)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			rec := Record{
				ID:         fmt.Sprintf("lane-%d", i),
				TreeGlobs:  []string{fmt.Sprintf("internal/pkg%d/**", i)},
				Holder:     fmt.Sprintf("worker-%d", i),
				TTLSeconds: 300,
			}
			_, v, err := s.AcquireFenced(ctx(), rec, now)
			results[i] = outcome{v, err}
		}(i)
	}
	close(start)
	wg.Wait()

	for i, o := range results {
		if o.err != nil {
			t.Fatalf("worker %d AcquireFenced errored: %v", i, o.err)
		}
		if !o.v.OK {
			t.Fatalf("worker %d disjoint-lane claim refused %q — a single-writer bottleneck would do this (%+v)",
				i, o.v.Reason, o.v)
		}
	}

	// Every disjoint lane landed on its own ref, each held by its own worker: N concurrent
	// writers, N claimed lanes, zero coordination.
	recs, err := s.List(ctx())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(recs) != workers {
		t.Fatalf("List returned %d leases, want %d (one per disjoint lane)", len(recs), workers)
	}
	for i := 0; i < workers; i++ {
		id := fmt.Sprintf("lane-%d", i)
		got, ok, err := s.Get(ctx(), id)
		if err != nil || !ok {
			t.Fatalf("Get %s: ok=%v err=%v", id, ok, err)
		}
		if want := fmt.Sprintf("worker-%d", i); got.Holder != want {
			t.Fatalf("lane %s holder = %q, want %q", id, got.Holder, want)
		}
	}
}
