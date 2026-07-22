package workerworktree

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestColdEligibleDecisionTable is the >=4-case sweep-eligibility table (#5351 DoD):
// the PURE reap decision under every combination of lease liveness and age-vs-floor.
func TestColdEligibleDecisionTable(t *testing.T) {
	floor := 30 * time.Minute
	cases := []struct {
		name      string
		leaseLive bool
		age       time.Duration
		want      bool
	}{
		{"live lease past floor is KEPT", true, 2 * time.Hour, false},
		{"live lease under floor is KEPT", true, time.Minute, false},
		{"dead lease past floor is COLD", false, 2 * time.Hour, true},
		{"dead lease under floor is KEPT", false, time.Minute, false},
		{"dead lease exactly at floor is COLD", false, floor, true},
		{"dead lease unknown age (0) is KEPT", false, 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, reason := coldEligible(c.leaseLive, c.age, floor)
			if got != c.want {
				t.Fatalf("coldEligible(live=%v, age=%s) = %v (%q), want %v", c.leaseLive, c.age, got, reason, c.want)
			}
			if got && reason == "" {
				t.Fatal("an eligible verdict must carry a why-eligible reason")
			}
		})
	}
}

// TestLaneOfRecoversLaneFromDirName proves LaneOf is the inverse of DirName's lane
// segment — the binding the production lease-liveness oracle keys on.
func TestLaneOfRecoversLaneFromDirName(t *testing.T) {
	for _, lane := range []string{"cmd", "gateway", "docs", "some-lane"} {
		dir := DirName(lane, "12345")
		if got := LaneOf(dir); got != lane {
			t.Fatalf("LaneOf(%q) = %q, want %q", dir, got, lane)
		}
		// Also inverts a full path, not just a bare name.
		if got := LaneOf(filepath.Join("/wt/root", dir)); got != lane {
			t.Fatalf("LaneOf(path %q) = %q, want %q", dir, got, lane)
		}
	}
	// A non-worker path and the bare marker have no recoverable lane.
	if got := LaneOf("/work/fak"); got != "" {
		t.Fatalf("LaneOf(non-worker) = %q, want empty", got)
	}
	if got := LaneOf(WorktreeMarker); got != "" {
		t.Fatalf("LaneOf(bare marker) = %q, want empty", got)
	}
}

// coldTempWorktree makes a real dir named like one of OUR worktrees and back-dates its
// mtime so WorktreeAge reads it as `age` old. Returns the absolute path.
func coldTempWorktree(t *testing.T, parent, lane, key string, age time.Duration, now time.Time) string {
	t.Helper()
	dir := filepath.Join(parent, DirName(lane, key))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir worktree: %v", err)
	}
	stamp := now.Add(-age)
	if err := os.Chtimes(dir, stamp, stamp); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	return dir
}

// TestColdReapListEnumeratesAndDecides drives the enumerate-and-plan path over three
// real worktrees (live lease / dead-old / dead-young) via a fake `git worktree list`.
// It proves: the live-lease worktree is never eligible (#5351 safety case), the
// dead-lease worktree past the floor IS eligible, the dead-lease worktree under the
// floor is kept, and the plan DELETES NOTHING (never calls `worktree remove`).
func TestColdReapListEnumeratesAndDecides(t *testing.T) {
	parent := t.TempDir()
	now := time.Now()
	floor := 30 * time.Minute

	liveWT := coldTempWorktree(t, parent, "cmd", "111", 2*time.Hour, now) // dead? no — lease live
	deadOldWT := coldTempWorktree(t, parent, "docs", "222", 2*time.Hour, now)
	deadYoungWT := coldTempWorktree(t, parent, "docs", "333", time.Minute, now)

	porcelain := "worktree " + liveWT + "\n\nworktree " + deadOldWT + "\n\nworktree " + deadYoungWT + "\n"
	g := newFakeGit().reply("worktree", 0, porcelain)

	// Oracle: only the "cmd" lane worktree has a live lease.
	oracle := func(p string) bool { return LaneOf(p) == "cmd" }

	plan := ColdReapList("/repo", g.run, now, floor, oracle)
	if len(plan) != 3 {
		t.Fatalf("want 3 enumerated worktrees, got %d: %+v", len(plan), plan)
	}
	byPath := map[string]ColdWorktree{}
	for _, c := range plan {
		byPath[c.Path] = c
	}
	if c := byPath[liveWT]; c.Eligible || !c.LeaseLive {
		t.Fatalf("live-lease worktree must never be eligible: %+v", c)
	}
	if c := byPath[deadOldWT]; !c.Eligible {
		t.Fatalf("dead-lease worktree past the floor must be eligible: %+v", c)
	}
	if c := byPath[deadYoungWT]; c.Eligible {
		t.Fatalf("dead-lease worktree under the floor must be kept: %+v", c)
	}
	// The plan is pure: it must never issue a removal.
	if removes := g.callsWithPrefix("worktree", "remove"); len(removes) != 0 {
		t.Fatalf("ColdReapList must delete nothing, saw removes: %v", removes)
	}
}

// TestColdReapListNilOracleKeepsEverything proves the fail-toward-keeping contract:
// with no liveness oracle, every worktree is kept even when old, so an unreadable lease
// store can never cause a reap.
func TestColdReapListNilOracleKeepsEverything(t *testing.T) {
	parent := t.TempDir()
	now := time.Now()
	old := coldTempWorktree(t, parent, "cmd", "999", 5*time.Hour, now)
	g := newFakeGit().reply("worktree", 0, "worktree "+old+"\n")
	plan := ColdReapList("/repo", g.run, now, 30*time.Minute, nil)
	if len(plan) != 1 || plan[0].Eligible {
		t.Fatalf("nil oracle must keep everything, got %+v", plan)
	}
}
