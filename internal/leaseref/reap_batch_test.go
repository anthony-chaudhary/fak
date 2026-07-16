package leaseref

// reap_batch_test.go is the witness for #4990 root cause #2: the reaper must drain an
// expired backlog in ONE `git update-ref --stdin` transaction, not one serial
// `git update-ref -d` process spawn per ref (the per-ref cost that made a full drain of
// the ~8k expired session descriptors on this host unbounded in practice — a bounded
// `fak leaseref reap` got guillotined mid-sweep). The fake git records every argv, so a
// test can prove the exact plumbing: one batched delete call, zero per-ref spawns.

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestReapSessionsBatchesDeleteIntoOneProcess is the headline #4990 witness at scale: a
// backlog of expired session descriptors reaps in a SINGLE update-ref --stdin call — zero
// per-ref update-ref -d spawns — while the live sessions survive untouched.
func TestReapSessionsBatchesDeleteIntoOneProcess(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	now := time.Now()

	const nExpired = 64 // representative of the ~8k backlog at test scale
	for i := 0; i < nExpired; i++ {
		id := fmt.Sprintf("dead-%d", i)
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: id, Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
			t.Fatalf("Publish %s: %v", id, err)
		}
	}
	// Two live sessions that must NOT be reaped.
	for _, id := range []string{"liveA", "liveB"} {
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: id, Host: "n", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
			t.Fatalf("Publish %s: %v", id, err)
		}
	}

	g.calls = nil // start the argv witness at the reap
	reaped, err := s.ReapSessions(ctx(), now)
	if err != nil {
		t.Fatalf("ReapSessions: %v", err)
	}
	if len(reaped) != nExpired {
		t.Fatalf("reaped %d sessions, want %d", len(reaped), nExpired)
	}
	// The fix: ONE process spawn for the whole backlog, not one per ref.
	if got := g.stdinDeleteCalls(); got != 1 {
		t.Fatalf("batched update-ref --stdin calls = %d, want exactly 1", got)
	}
	if got := g.perRefDeleteCalls(); got != 0 {
		t.Fatalf("per-ref update-ref -d spawns = %d, want 0 (the #4990 root cause #2 cost)", got)
	}
	// The live sessions survive; a sampling of the expired ones is gone.
	for _, id := range []string{"liveA", "liveB"} {
		if _, ok, _ := s.GetSession(ctx(), id); !ok {
			t.Fatalf("batched reap wrongly removed live session %s", id)
		}
	}
	if _, ok, _ := s.GetSession(ctx(), "dead-0"); ok {
		t.Fatalf("expired session dead-0 still present after batched reap")
	}
}

// TestReapBatchCoversAllThreeKinds: each reaper (leases, sessions, intents) batches its
// OWN namespace in one update-ref --stdin call and never cross-deletes another kind, even
// though all three ride the shared refs/fak/locks/ prefix.
func TestReapBatchCoversAllThreeKinds(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	now := time.Now()
	t0 := time.Unix(100, 0)

	// One expired + one live of each kind, sharing the id stem "x".
	if _, err := s.Acquire(ctx(), Record{ID: "x", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire lease: %v", err)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "liveLease", TreeGlobs: []string{"b"}, Holder: "h", AcquiredAt: now.Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatalf("Acquire live lease: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "x", Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish session: %v", err)
	}
	if _, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#7", Holder: "h", TTLSeconds: 10}, t0); err != nil || !v.OK {
		t.Fatalf("Claim intent: ok=%v err=%v", v.OK, err)
	}

	// Reap leases: batched, and the session + intent named "x"/"#7" survive.
	g.calls = nil
	lreaped, err := s.Reap(ctx(), now)
	if err != nil || len(lreaped) != 1 || lreaped[0] != "x" {
		t.Fatalf("Reap leases = %v err=%v, want [x]", lreaped, err)
	}
	if g.stdinDeleteCalls() != 1 || g.perRefDeleteCalls() != 0 {
		t.Fatalf("lease reap not batched: stdin=%d perRef=%d", g.stdinDeleteCalls(), g.perRefDeleteCalls())
	}
	if _, ok, _ := s.GetSession(ctx(), "x"); !ok {
		t.Fatalf("lease reap cross-deleted session x — namespace split violated")
	}
	if _, ok, _ := s.GetIntent(ctx(), "#7"); !ok {
		t.Fatalf("lease reap cross-deleted intent #7 — namespace split violated")
	}
	if _, ok, _ := s.Get(ctx(), "liveLease"); !ok {
		t.Fatalf("lease reap wrongly removed the live lease")
	}

	// Reap sessions: batched, only the session goes.
	g.calls = nil
	sreaped, err := s.ReapSessions(ctx(), now)
	if err != nil || len(sreaped) != 1 || sreaped[0] != "x" {
		t.Fatalf("ReapSessions = %v err=%v, want [x]", sreaped, err)
	}
	if g.stdinDeleteCalls() != 1 || g.perRefDeleteCalls() != 0 {
		t.Fatalf("session reap not batched: stdin=%d perRef=%d", g.stdinDeleteCalls(), g.perRefDeleteCalls())
	}

	// Reap intents: batched, the intent key issue-7 goes.
	g.calls = nil
	ireaped, err := s.ReapIntents(ctx(), now)
	if err != nil || len(ireaped) != 1 || ireaped[0] != "issue-7" {
		t.Fatalf("ReapIntents = %v err=%v, want [issue-7]", ireaped, err)
	}
	if g.stdinDeleteCalls() != 1 || g.perRefDeleteCalls() != 0 {
		t.Fatalf("intent reap not batched: stdin=%d perRef=%d", g.stdinDeleteCalls(), g.perRefDeleteCalls())
	}
}

// TestReapBatchEmptyIssuesNoDelete: with nothing expired, the reaper issues NO delete
// process at all (neither batched nor per-ref) — the idempotent second-pass no-op that
// makes racing reaps cheap.
func TestReapBatchEmptyIssuesNoDelete(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	now := time.Now()
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "live", Host: "n", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	g.calls = nil
	reaped, err := s.ReapSessions(ctx(), now)
	if err != nil || len(reaped) != 0 {
		t.Fatalf("ReapSessions over an all-live namespace = %v err=%v, want none", reaped, err)
	}
	if g.stdinDeleteCalls() != 0 || g.perRefDeleteCalls() != 0 {
		t.Fatalf("a no-op reap issued a delete: stdin=%d perRef=%d", g.stdinDeleteCalls(), g.perRefDeleteCalls())
	}
}

// TestReapFallsBackToPerRefWithoutStdinRunner: a Store built with NewWithRunner (no stdin
// seam) still reaps correctly, via the per-ref update-ref -d loop and ZERO batched calls —
// the graceful degradation that keeps every existing caller working unchanged.
func TestReapFallsBackToPerRefWithoutStdinRunner(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "") // no stdin runner wired
	now := time.Now()
	for i := 0; i < 3; i++ {
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: fmt.Sprintf("dead-%d", i), Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
			t.Fatalf("Publish: %v", err)
		}
	}
	g.calls = nil
	reaped, err := s.ReapSessions(ctx(), now)
	if err != nil || len(reaped) != 3 {
		t.Fatalf("fallback ReapSessions = %v err=%v, want 3 reaped", reaped, err)
	}
	if g.stdinDeleteCalls() != 0 {
		t.Fatalf("fallback path issued a batched delete: %d", g.stdinDeleteCalls())
	}
	if g.perRefDeleteCalls() != 3 {
		t.Fatalf("fallback per-ref delete spawns = %d, want 3", g.perRefDeleteCalls())
	}
}

// TestReapBatchFallsBackOnBatchFailure: when the batched transaction returns non-zero
// (git lock contention, a broken --stdin), the reaper degrades to the idempotent per-ref
// loop and still drains the backlog — the batch is an optimization, never a correctness
// dependency.
func TestReapBatchFallsBackOnBatchFailure(t *testing.T) {
	g := newFakeGit()
	// A stdin runner that FAILS the batch (exit 1) but leaves the plain runner's per-ref
	// -d path working, so the fallback must complete the reap.
	failBatch := func(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
		g.calls = append(g.calls, args)
		return "", 1, nil // non-zero: transaction refused
	}
	s := NewWithStdinRunner(g.run, failBatch, "")
	now := time.Now()
	for i := 0; i < 4; i++ {
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: fmt.Sprintf("dead-%d", i), Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
			t.Fatalf("Publish: %v", err)
		}
	}
	g.calls = nil
	reaped, err := s.ReapSessions(ctx(), now)
	if err != nil || len(reaped) != 4 {
		t.Fatalf("ReapSessions after batch failure = %v err=%v, want 4 reaped via fallback", reaped, err)
	}
	if g.stdinDeleteCalls() != 1 {
		t.Fatalf("expected exactly one attempted batch call, got %d", g.stdinDeleteCalls())
	}
	if g.perRefDeleteCalls() != 4 {
		t.Fatalf("fallback per-ref spawns = %d, want 4", g.perRefDeleteCalls())
	}
}

// TestReapBatchRealGit exercises the batched delete against the REAL git binary in a temp
// repo (skipped when git is unavailable — it runs under the WSL suite). It proves the
// actual `git update-ref --stdin` plumbing drains a mixed backlog of all three ref kinds,
// leaves the live records intact, and is idempotent on a second pass (the racing-reap
// property: a no-<oldvalue> delete of an already-gone ref never fails the transaction).
func TestReapBatchRealGit(t *testing.T) {
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

	s := NewInDir(dir) // wires the real gitStdinRunner
	now := time.Now()
	past := int64(100)

	// A backlog: 5 expired + 1 live of each kind.
	for i := 0; i < 5; i++ {
		if _, err := s.Acquire(ctx(), Record{ID: fmt.Sprintf("lease-%d", i), TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: past, TTLSeconds: 10}); err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: fmt.Sprintf("sess-%d", i), Host: "n", PCBState: "RUNNING", UpdatedAt: past, TTLSecs: 10}); err != nil {
			t.Fatalf("PublishSession: %v", err)
		}
		if _, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: fmt.Sprintf("#%d", 100+i), Holder: "h", TTLSeconds: 10}, time.Unix(past, 0)); err != nil || !v.OK {
			t.Fatalf("ClaimIntent: ok=%v err=%v", v.OK, err)
		}
	}
	if _, err := s.Acquire(ctx(), Record{ID: "live-lease", TreeGlobs: []string{"b"}, Holder: "h", AcquiredAt: now.Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatalf("Acquire live: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "live-sess", Host: "n", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("PublishSession live: %v", err)
	}

	lreaped, lerr := s.Reap(ctx(), now)
	sreaped, serr := s.ReapSessions(ctx(), now)
	ireaped, ierr := s.ReapIntents(ctx(), now)
	if lerr != nil || serr != nil || ierr != nil {
		t.Fatalf("reap errors: lease=%v session=%v intent=%v", lerr, serr, ierr)
	}
	if len(lreaped) != 5 || len(sreaped) != 5 || len(ireaped) != 5 {
		t.Fatalf("reaped counts lease=%d session=%d intent=%d, want 5/5/5", len(lreaped), len(sreaped), len(ireaped))
	}

	// The live records survive.
	if _, ok, _ := s.Get(ctx(), "live-lease"); !ok {
		t.Fatalf("batched reap removed the live lease")
	}
	if _, ok, _ := s.GetSession(ctx(), "live-sess"); !ok {
		t.Fatalf("batched reap removed the live session")
	}
	// The expired refs are gone from the real ref store.
	listing, _, _ := gitRunner(ctx(), dir, "for-each-ref", "--format=%(refname)", "refs/fak/locks/")
	for _, dead := range []string{"refs/fak/locks/lease-0", "refs/fak/locks/session-sess-0", "refs/fak/locks/intent-issue-100"} {
		if strings.Contains(listing, dead) {
			t.Fatalf("expired ref %s still present after batched reap:\n%s", dead, listing)
		}
	}

	// Idempotent second pass: nothing expired remains, so each reaper is a clean no-op.
	if r, err := s.ReapSessions(ctx(), now); err != nil || len(r) != 0 {
		t.Fatalf("second ReapSessions = %v err=%v, want a clean no-op", r, err)
	}
}
