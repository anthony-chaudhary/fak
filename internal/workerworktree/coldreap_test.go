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
		unlanded  int
		want      bool
		wantHeld  bool
	}{
		{"live lease past floor is KEPT", true, 2 * time.Hour, 0, false, false},
		{"live lease under floor is KEPT", true, time.Minute, 0, false, false},
		{"dead lease past floor and CLEAN is COLD", false, 2 * time.Hour, 0, true, false},
		{"dead lease under floor is KEPT", false, time.Minute, 0, false, false},
		{"dead lease exactly at floor and clean is COLD", false, floor, 0, true, false},
		{"dead lease unknown age (0) is KEPT", false, 0, 0, false, false},
		// The third gate: coldness proves nobody is WATCHING, not that nothing is HELD.
		{"dead lease past floor but 1 uncommitted entry is KEPT", false, 2 * time.Hour, 1, false, true},
		{"dead lease past floor but many uncommitted entries is KEPT", false, 2 * time.Hour, 15, false, true},
		{"unreadable working tree (-1) is KEPT", false, 2 * time.Hour, -1, false, true},
		// A live or young worktree is kept for its OWN reason — being dirty as well
		// must not relabel it as held-by-work, or the triage queue fills with
		// worktrees whose real answer is "wait", not "decide".
		{"live lease AND dirty is kept, not held-by-work", true, 2 * time.Hour, 9, false, false},
		{"young AND dirty is kept, not held-by-work", false, time.Minute, 9, false, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, held, reason := coldEligible(c.leaseLive, c.age, floor, c.unlanded)
			if got != c.want {
				t.Fatalf("coldEligible(live=%v, age=%s, unlanded=%d) = %v (%q), want %v",
					c.leaseLive, c.age, c.unlanded, got, reason, c.want)
			}
			if held != c.wantHeld {
				t.Fatalf("coldEligible(live=%v, age=%s, unlanded=%d) heldByWork = %v (%q), want %v",
					c.leaseLive, c.age, c.unlanded, held, reason, c.wantHeld)
			}
			if reason == "" {
				t.Fatal("every verdict must carry a reason for the ledger")
			}
			if got && held {
				t.Fatalf("a reapable worktree cannot also be held by work: %q", reason)
			}
		})
	}
}

// TestUnlandedCountFailsTowardCarryingWork pins the probe's fail-safe direction: a git
// that cannot answer yields -1 ("assume work"), never 0 ("assume clean"). Reading a
// broken worktree as clean is the one error mode that destroys an unrecoverable diff.
func TestUnlandedCountFailsTowardCarryingWork(t *testing.T) {
	broken := newFakeGit().reply("status", 128, "fatal: not a git repository")
	if got := UnlandedCount("/wt/gone", broken.run); got != -1 {
		t.Fatalf("UnlandedCount over a failing git = %d, want -1 (cannot tell => assume work)", got)
	}
	// Both tracked modifications and untracked files count: a worker that died before
	// landing may hold its only new file as untracked.
	dirty := newFakeGit().reply("status", 0, " M internal/journal/journal.go\n?? newfile.go\n")
	if got := UnlandedCount("/wt/dirty", dirty.run); got != 2 {
		t.Fatalf("UnlandedCount over 1 tracked + 1 untracked = %d, want 2", got)
	}
	clean := newFakeGit().reply("status", 0, "\n")
	if got := UnlandedCount("/wt/clean", clean.run); got != 0 {
		t.Fatalf("UnlandedCount over a clean tree = %d, want 0", got)
	}
}

// TestColdReapListKeepsWorktreeHoldingUnlandedWork is the regression fence for the
// measured hole: a worktree with a dead lease, well past the age floor, and carrying
// uncommitted work read as textbook cold under the two-gate rule. Witnessed 2026-07-26:
// 20/20 worktrees eligible, 17 of them holding uncommitted entries.
func TestColdReapListKeepsWorktreeHoldingUnlandedWork(t *testing.T) {
	parent := t.TempDir()
	now := time.Now()
	floor := 30 * time.Minute
	old := coldTempWorktree(t, parent, "docs", "444", 253*time.Hour, now)

	g := newFakeGit().
		reply("worktree", 0, "worktree "+old+"\n").
		reply("status", 0, " M cmd/fak/guard.go\n M internal/journal/journal.go\n")

	plan := ColdReapList("/repo", g.run, now, floor, func(string) bool { return false })
	if len(plan) != 1 {
		t.Fatalf("want 1 enumerated worktree, got %d: %+v", len(plan), plan)
	}
	c := plan[0]
	if c.Eligible {
		t.Fatalf("a worktree holding unlanded work must NEVER be eligible: %+v", c)
	}
	if !c.HeldByWork {
		t.Fatalf("it must be reported as held_by_work so an operator triages it: %+v", c)
	}
	if c.Unlanded != 2 {
		t.Fatalf("unlanded count = %d, want 2: %+v", c.Unlanded, c)
	}
	if removes := g.callsWithPrefix("worktree", "remove"); len(removes) != 0 {
		t.Fatalf("ColdReapList must delete nothing, saw removes: %v", removes)
	}
}

// TestColdReapListSkipsStatusProbeForKeptWorktrees proves the probe is not paid for a
// worktree the earlier gates already keep. The sweep runs over every worktree in the
// namespace, so a per-worktree subprocess that cannot change the verdict is pure cost.
func TestColdReapListSkipsStatusProbeForKeptWorktrees(t *testing.T) {
	parent := t.TempDir()
	now := time.Now()
	live := coldTempWorktree(t, parent, "cmd", "555", 2*time.Hour, now)
	young := coldTempWorktree(t, parent, "docs", "666", time.Minute, now)

	g := newFakeGit().reply("worktree", 0, "worktree "+live+"\n\nworktree "+young+"\n")
	// Only the "cmd" lane holds a live lease; "docs" is dead but under the floor.
	ColdReapList("/repo", g.run, now, 30*time.Minute, func(p string) bool { return LaneOf(p) == "cmd" })

	if probes := g.callsWithPrefix("status"); len(probes) != 0 {
		t.Fatalf("no status probe should run for already-kept worktrees, saw: %v", probes)
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
