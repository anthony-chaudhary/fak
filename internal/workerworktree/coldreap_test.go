package workerworktree

import (
	"archive/zip"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
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

func TestUnregisteredResidueArchivesBeforeRemoval(t *testing.T) {
	repo := newResidueRepo(t)
	base := t.TempDir()
	now := time.Now().UTC()
	old := now.Add(-2 * time.Hour)
	stale := filepath.Join(base, "fak-worker-wt-stale-deadbeef0001")
	if err := os.MkdirAll(stale, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stale, "valuable.txt"), []byte("preserve me"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(filepath.Join(stale, "valuable.txt"), old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatal(err)
	}

	items, err := CollectUnregisteredResidue(repo, ResidueOptions{BaseDir: base, Now: now, AgeFloor: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || !items[0].Eligible || items[0].Entries != 1 {
		t.Fatalf("unexpected plan: %+v", items)
	}
	if _, err := os.Stat(stale); err != nil {
		t.Fatalf("dry-run mutated source: %v", err)
	}

	archiveDir := filepath.Join(t.TempDir(), "archives")
	items, err = ApplyUnregisteredResidue(items, ResidueOptions{BaseDir: base, ArchiveDir: archiveDir, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if !items[0].Removed || items[0].Archive == "" {
		t.Fatalf("unexpected apply: %+v", items[0])
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("source not removed: %v", err)
	}
	zr, err := zip.OpenReader(filepath.FromSlash(items[0].Archive))
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	if len(zr.File) != 1 || zr.File[0].Name != "valuable.txt" {
		t.Fatalf("bad archive: %+v", zr.File)
	}
	if _, err := os.Stat(filepath.FromSlash(items[0].Archive) + ".json"); err != nil {
		t.Fatalf("receipt missing: %v", err)
	}
}

func TestUnregisteredResidueKeepsFreshOwnedAndForeign(t *testing.T) {
	repo := newResidueRepo(t)
	base := t.TempDir()
	now := time.Now().UTC()
	old := now.Add(-2 * time.Hour)
	for _, name := range []string{"fak-worker-wt-fresh-deadbeef0002", "fak-worker-wt-owned-deadbeef0003", "fak-worker-wt-foreign-deadbeef0004"} {
		if err := os.MkdirAll(filepath.Join(base, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	owned := filepath.Join(base, "fak-worker-wt-owned-deadbeef0003")
	_ = os.Chtimes(owned, old, old)
	if err := os.MkdirAll(filepath.Join(base, ".fak-worker-intents"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, ".fak-worker-intents", filepath.Base(owned)+".json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	foreign := filepath.Join(base, "fak-worker-wt-foreign-deadbeef0004")
	if err := os.WriteFile(filepath.Join(foreign, ".git"), []byte("gitdir: elsewhere"), 0o600); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(foreign, old, old)
	items, err := CollectUnregisteredResidue(repo, ResidueOptions{BaseDir: base, Now: now, AgeFloor: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d: %+v", len(items), items)
	}
	for _, item := range items {
		if item.Eligible {
			t.Fatalf("must retain %s: %+v", item.Path, item)
		}
		if !strings.HasPrefix(item.Reason, "kept:") {
			t.Fatalf("missing typed keep: %+v", item)
		}
	}
}

func TestApplyUnregisteredResidueRejectsEscapingPath(t *testing.T) {
	base := t.TempDir()
	outside := filepath.Join(t.TempDir(), "fak-worker-wt-outside-deadbeef0005")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	items, err := ApplyUnregisteredResidue([]ResidueItem{{Path: filepath.ToSlash(outside), Eligible: true}}, ResidueOptions{BaseDir: base, ArchiveDir: t.TempDir()})
	if err == nil || items[0].Removed {
		t.Fatalf("escape accepted: err=%v item=%+v", err, items[0])
	}
	if _, statErr := os.Stat(outside); statErr != nil {
		t.Fatalf("outside path changed: %v", statErr)
	}
}

func newResidueRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	cmd := exec.Command("git", "init", "-q", repo)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	return repo
}

func TestUnregisteredResidueRemovesEmptyAndPreservesOnArchiveFailure(t *testing.T) {
	repo := newResidueRepo(t)
	base := t.TempDir()
	now := time.Now().UTC()
	old := now.Add(-2 * time.Hour)
	empty := filepath.Join(base, "fak-worker-wt-empty-deadbeef0006")
	full := filepath.Join(base, "fak-worker-wt-full-deadbeef0007")
	for _, path := range []string{empty, full} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(full, "work.txt"), []byte("work"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{empty, full, filepath.Join(full, "work.txt")} {
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatal(err)
		}
	}
	items, err := CollectUnregisteredResidue(repo, ResidueOptions{BaseDir: base, Now: now, AgeFloor: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	archiveBlocker := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(archiveBlocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	items, err = ApplyUnregisteredResidue(items, ResidueOptions{BaseDir: base, ArchiveDir: archiveBlocker, Now: now, AgeFloor: time.Hour})
	if err == nil {
		t.Fatal("expected archive failure")
	}
	byPath := map[string]ResidueItem{}
	for _, item := range items {
		byPath[filepath.FromSlash(item.Path)] = item
	}
	if !byPath[empty].Removed {
		t.Fatalf("empty residue not removed: %+v", byPath[empty])
	}
	if byPath[full].Removed {
		t.Fatalf("non-empty residue removed after archive failure: %+v", byPath[full])
	}
	if _, statErr := os.Stat(filepath.Join(full, "work.txt")); statErr != nil {
		t.Fatalf("valuable source lost: %v", statErr)
	}
}

func TestApplyUnregisteredResidueRevalidatesFreshness(t *testing.T) {
	repo := newResidueRepo(t)
	base := t.TempDir()
	now := time.Now().UTC()
	old := now.Add(-2 * time.Hour)
	path := filepath.Join(base, "fak-worker-wt-race-deadbeef0008")
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(path, old, old)
	items, err := CollectUnregisteredResidue(repo, ResidueOptions{Repo: repo, BaseDir: base, Now: now, AgeFloor: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || !items[0].Eligible {
		t.Fatalf("bad plan: %+v", items)
	}
	if err := os.WriteFile(filepath.Join(path, "new-work.txt"), []byte("live"), 0o644); err != nil {
		t.Fatal(err)
	}
	items, err = ApplyUnregisteredResidue(items, ResidueOptions{Repo: repo, BaseDir: base, ArchiveDir: t.TempDir(), Now: now, AgeFloor: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if items[0].Removed || items[0].Eligible || !strings.Contains(items[0].Reason, "changed after planning") {
		t.Fatalf("race not refused: %+v", items[0])
	}
	if _, err := os.Stat(filepath.Join(path, "new-work.txt")); err != nil {
		t.Fatalf("new work lost: %v", err)
	}
}

// TestColdReapListWithOptionsBoundsLargeCensus is the scale regression for
// #10383: 240 delayed worktree probes make incremental progress without exceeding
// the configured ceiling, preserve every live or unlanded tree, and distinguish
// measured reclaim bytes from unknown estimates.
func TestColdReapListWithOptionsBoundsLargeCensus(t *testing.T) {
	const (
		entries = 240
		ceiling = 6
	)
	now := time.Now()
	floor := 30 * time.Minute
	root := t.TempDir()
	paths := make([]string, entries)
	index := make(map[string]int, entries)
	var porcelain strings.Builder
	for i := range entries {
		path := coldTempWorktree(t, root, "scale", fmt.Sprintf("%03d", i), 2*time.Hour, now)
		paths[i] = path
		index[path] = i
		fmt.Fprintf(&porcelain, "worktree %s\n\n", path)
	}

	var active atomic.Int64
	var maximum atomic.Int64
	probe := func() func() {
		current := active.Add(1)
		for {
			seen := maximum.Load()
			if current <= seen || maximum.CompareAndSwap(seen, current) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		return func() { active.Add(-1) }
	}
	git := func(workdir string, args []string) (int, string) {
		if len(args) >= 2 && args[0] == "worktree" && args[1] == "list" {
			return 0, porcelain.String()
		}
		if len(args) >= 1 && args[0] == "status" {
			done := probe()
			defer done()
			i := index[workdir]
			switch {
			case i >= 20 && i < 40:
				return 0, " M retained.go\n"
			case i >= 40 && i < 50:
				return 1, "status unavailable"
			default:
				return 0, ""
			}
		}
		return 0, ""
	}

	var progress []ColdReapProgress
	plan := ColdReapListWithOptions(root, git, now, floor, func(path string) bool {
		return index[path] < 20
	}, ColdReapOptions{
		Concurrency: ceiling,
		Size: func(path string) (int64, error) {
			done := probe()
			defer done()
			i := index[path]
			if i >= 50 && i < 60 {
				return 0, os.ErrPermission
			}
			return int64(i + 1), nil
		},
		Progress: func(event ColdReapProgress) {
			progress = append(progress, event)
		},
	})

	if got := maximum.Load(); got > ceiling || got < 2 {
		t.Fatalf("max concurrent probes = %d, want within [2,%d]", got, ceiling)
	}
	if len(progress) != entries {
		t.Fatalf("progress events = %d, want %d", len(progress), entries)
	}
	if first := progress[0]; first.Completed != 1 || first.Total != entries {
		t.Fatalf("first progress = %+v, want completed=1 total=%d", first, entries)
	}
	last := progress[len(progress)-1]
	if last.Completed != entries || last.Eligible != 190 || last.EligibleBytesKnown != 180 || last.EligibleBytesUnknown != 10 || last.EligibleBytes != 27090 {
		t.Fatalf("final progress = %+v", last)
	}

	for i, item := range plan {
		switch {
		case i < 20:
			if item.Eligible || !item.LeaseLive || item.HeldByWork {
				t.Fatalf("live item %d was not preserved: %+v", i, item)
			}
		case i < 40:
			if item.Eligible || !item.HeldByWork || item.Unlanded != 1 {
				t.Fatalf("dirty item %d was not preserved: %+v", i, item)
			}
		case i < 50:
			if item.Eligible || !item.HeldByWork || item.Unlanded != -1 {
				t.Fatalf("unreadable item %d was not preserved: %+v", i, item)
			}
		case i < 60:
			if !item.Eligible || item.ReclaimBytesKnown || item.ReclaimBytes != 0 {
				t.Fatalf("unknown-size item %d lost provenance: %+v", i, item)
			}
		default:
			if !item.Eligible || !item.ReclaimBytesKnown || item.ReclaimBytes != int64(i+1) {
				t.Fatalf("known-size item %d = %+v", i, item)
			}
		}
	}
}
