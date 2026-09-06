package leaseref

import (
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestGenerationSurvivesReleaseReacquire verifies issue #11850:
// A durable monotonic generation floor is preserved across a successful fenced release
// and later acquire of the same lease ID using the existing Git store.
//
// Done condition:
// One real-Git acquire/release/reopen-store/reacquire sequence assigns a strictly higher
// generation to the replacement and the former generation fails the existing fence check,
// while a released lease is absent from live admission.
//
// Witness:
// go test ./internal/leaseref -run '^TestGenerationSurvivesReleaseReacquire$' -count=1
func TestGenerationSurvivesReleaseReacquire(t *testing.T) {
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

	s := NewInDir(dir)
	t0 := time.Unix(1000, 0)
	leaseID := "lease-hist-test"

	// 1. Acquire lease "lease-hist-test" with holder "agent-1". Generation is 1.
	req1 := Record{
		ID:         leaseID,
		TreeGlobs:  []string{"internal/leaseref/**"},
		Holder:     "agent-1",
		TTLSeconds: 300,
	}
	rec1, v1, err := s.AcquireFenced(ctx(), req1, t0)
	if err != nil {
		t.Fatalf("agent-1 AcquireFenced: %v", err)
	}
	if !v1.OK || rec1.Generation != 1 {
		t.Fatalf("agent-1 acquire: ok=%v gen=%d, want ok gen=1 (%+v)", v1.OK, rec1.Generation, v1)
	}

	// 2. Verify fence check passes for generation 1.
	fv1, err := s.Fence(ctx(), rec1, t0.Add(time.Second))
	if err != nil {
		t.Fatalf("agent-1 Fence: %v", err)
	}
	if !fv1.OK {
		t.Fatalf("agent-1 initial fence check failed: %+v", fv1)
	}

	// 3. Release lease with holder "agent-1", generation 1. Release succeeds.
	rv, err := s.ReleaseFenced(ctx(), leaseID, "agent-1", rec1.Generation, t0.Add(2*time.Second))
	if err != nil {
		t.Fatalf("agent-1 ReleaseFenced: %v", err)
	}
	if !rv.OK {
		t.Fatalf("agent-1 release failed: %+v", rv)
	}

	// 4. Verify lease is absent from live admission.
	if got, ok, err := s.Get(ctx(), leaseID); err != nil || ok {
		t.Fatalf("Get after release: ok=%v err=%v got=%+v, want absent", ok, err, got)
	}
	live, _, err := s.Live(ctx(), t0.Add(3*time.Second))
	if err != nil {
		t.Fatalf("Live: %v", err)
	}
	for _, r := range live {
		if r.ID == leaseID {
			t.Fatalf("released lease %q must not appear in Live", leaseID)
		}
	}
	arbLeases, err := s.LiveLeases(ctx(), t0.Add(3*time.Second))
	if err != nil {
		t.Fatalf("LiveLeases: %v", err)
	}
	for _, al := range arbLeases {
		if al.Lane == leaseID {
			t.Fatalf("released lease %q must not appear in LiveLeases", leaseID)
		}
	}
	// Fence check for absent lease returns ReasonNoLease.
	postReleaseFence, err := s.Fence(ctx(), rec1, t0.Add(3*time.Second))
	if err != nil {
		t.Fatalf("postReleaseFence: %v", err)
	}
	if postReleaseFence.OK || postReleaseFence.Reason != ReasonNoLease {
		t.Fatalf("fence check on released lease: ok=%v reason=%q, want refused NO_LEASE", postReleaseFence.OK, postReleaseFence.Reason)
	}

	// 5. Re-open store pointing to the same git repo to verify persistence on disk.
	s2 := NewInDir(dir)

	// 6. Acquire lease "lease-hist-test" with holder "agent-2".
	t1 := t0.Add(10 * time.Second)
	req2 := Record{
		ID:         leaseID,
		TreeGlobs:  []string{"internal/leaseref/**"},
		Holder:     "agent-2",
		TTLSeconds: 300,
	}
	rec2, v2, err := s2.AcquireFenced(ctx(), req2, t1)
	if err != nil {
		t.Fatalf("agent-2 AcquireFenced: %v", err)
	}
	if !v2.OK {
		t.Fatalf("agent-2 AcquireFenced verdict not OK: %+v", v2)
	}

	// 7. Assert new generation is strictly higher than 1 (specifically 2).
	if rec2.Generation <= rec1.Generation {
		t.Fatalf("agent-2 generation %d is not strictly higher than former generation %d", rec2.Generation, rec1.Generation)
	}
	if rec2.Generation != 2 {
		t.Fatalf("agent-2 generation = %d, want 2", rec2.Generation)
	}

	// 8. Verify that a fence check presenting the former generation (generation 1) fails with ReasonStaleLease.
	staleCheck, err := s2.Fence(ctx(), rec1, t1.Add(time.Second))
	if err != nil {
		t.Fatalf("Fence check with former generation: %v", err)
	}
	if staleCheck.OK {
		t.Fatalf("Fence admitted former generation %d on replacement lease (gen %d): %+v", rec1.Generation, rec2.Generation, staleCheck)
	}
	if staleCheck.Reason != ReasonStaleLease {
		t.Fatalf("Fence reason = %q, want %s (%+v)", staleCheck.Reason, ReasonStaleLease, staleCheck)
	}

	// 9. Verify fence check presenting generation 2 succeeds.
	validCheck, err := s2.Fence(ctx(), rec2, t1.Add(time.Second))
	if err != nil {
		t.Fatalf("Fence check with current generation: %v", err)
	}
	if !validCheck.OK {
		t.Fatalf("Fence refused current generation %d: %+v", rec2.Generation, validCheck)
	}

	// 10. Verify history ref is durable under refs/fak/history/ and not in refs/fak/locks/.
	histListing, _, err := gitRunner(ctx(), dir, "for-each-ref", "--format=%(refname)", "refs/fak/history/")
	if err != nil || !strings.Contains(histListing, "refs/fak/history/"+leaseID) {
		t.Fatalf("history ref not found in git: listing=%q err=%v", histListing, err)
	}
	lockListing, _, err := gitRunner(ctx(), dir, "for-each-ref", "--format=%(refname)", "refs/fak/locks/")
	if err != nil || !strings.Contains(lockListing, "refs/fak/locks/"+leaseID) {
		t.Fatalf("live lock ref not found in git: listing=%q err=%v", lockListing, err)
	}
	if strings.Contains(lockListing, "history") {
		t.Fatalf("history ref leaked into refs/fak/locks/ namespace: %q", lockListing)
	}
}

func TestGenerationSurvivesMultipleReleases(t *testing.T) {
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

	s := NewInDir(dir)
	leaseID := "multi-release-lane"
	now := time.Unix(1000, 0)

	for expectedGen := int64(1); expectedGen <= 4; expectedGen++ {
		holder := "agent-" + strings.Repeat("x", int(expectedGen))
		rec, v, err := s.AcquireFenced(ctx(), Record{ID: leaseID, Holder: holder, TTLSeconds: 300}, now)
		if err != nil || !v.OK {
			t.Fatalf("cycle %d AcquireFenced: %+v %v", expectedGen, v, err)
		}
		if rec.Generation != expectedGen {
			t.Fatalf("cycle %d generation = %d, want %d", expectedGen, rec.Generation, expectedGen)
		}

		fv, err := s.Fence(ctx(), rec, now.Add(time.Second))
		if err != nil || !fv.OK {
			t.Fatalf("cycle %d Fence: %+v %v", expectedGen, fv, err)
		}

		rv, err := s.ReleaseFenced(ctx(), leaseID, holder, rec.Generation, now.Add(2*time.Second))
		if err != nil || !rv.OK {
			t.Fatalf("cycle %d ReleaseFenced: %+v %v", expectedGen, rv, err)
		}

		hist, ok, err := s.ReadHistory(ctx(), leaseID)
		if err != nil || !ok {
			t.Fatalf("cycle %d ReadHistory ok=%v err=%v", expectedGen, ok, err)
		}
		if hist.Generation != expectedGen {
			t.Fatalf("cycle %d ReadHistory generation = %d, want %d", expectedGen, hist.Generation, expectedGen)
		}

		now = now.Add(10 * time.Second)
	}
}

func TestGenerationHistoryFakeGit(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(5000, 0)
	leaseID := "fakegit-lane"

	rec1, v1, err := s.AcquireFenced(ctx(), Record{ID: leaseID, Holder: "w1", TTLSeconds: 60}, now)
	if err != nil || !v1.OK {
		t.Fatalf("w1 acquire: %+v %v", v1, err)
	}
	if rec1.Generation != 1 {
		t.Fatalf("w1 generation = %d, want 1", rec1.Generation)
	}

	rv, err := s.ReleaseFenced(ctx(), leaseID, "w1", rec1.Generation, now.Add(time.Second))
	if err != nil || !rv.OK {
		t.Fatalf("w1 release: %+v %v", rv, err)
	}

	rec2, v2, err := s.AcquireFenced(ctx(), Record{ID: leaseID, Holder: "w2", TTLSeconds: 60}, now.Add(2*time.Second))
	if err != nil || !v2.OK {
		t.Fatalf("w2 acquire: %+v %v", v2, err)
	}
	if rec2.Generation != 2 {
		t.Fatalf("w2 generation = %d, want 2", rec2.Generation)
	}

	fv1, err := s.Fence(ctx(), rec1, now.Add(3*time.Second))
	if err != nil {
		t.Fatalf("Fence w1: %v", err)
	}
	if fv1.OK || fv1.Reason != ReasonStaleLease {
		t.Fatalf("Fence w1 ok=%v reason=%q, want refused STALE_LEASE", fv1.OK, fv1.Reason)
	}
}

func TestGenerationFloorNeverDecreases(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(5000, 0)
	leaseID := "monotonic-lane"

	// Pre-seed history at generation 5.
	if err := s.writeHistoryRecord(ctx(), leaseID, 5, now); err != nil {
		t.Fatalf("writeHistoryRecord: %v", err)
	}

	// Acquire when history is 5: next generation must be 6.
	rec, v, err := s.AcquireFenced(ctx(), Record{ID: leaseID, Holder: "w1", TTLSeconds: 60}, now)
	if err != nil || !v.OK {
		t.Fatalf("AcquireFenced: %+v %v", v, err)
	}
	if rec.Generation != 6 {
		t.Fatalf("Generation = %d, want 6", rec.Generation)
	}

	// Release with generation 6: history becomes 6.
	rv, err := s.ReleaseFenced(ctx(), leaseID, "w1", 6, now.Add(time.Second))
	if err != nil || !rv.OK {
		t.Fatalf("ReleaseFenced: %+v %v", rv, err)
	}

	hist, ok, err := s.ResolveHistoryRef(ctx(), leaseID)
	if err != nil || !ok {
		t.Fatalf("ResolveHistoryRef ok=%v err=%v", ok, err)
	}
	if hist.Generation != 6 {
		t.Fatalf("hist.Generation = %d, want 6", hist.Generation)
	}
}
