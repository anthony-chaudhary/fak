package workerworktree

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPrepareWritesOwnerStamp(t *testing.T) {
	now := time.Date(2026, time.August, 13, 12, 34, 56, 789, time.UTC)
	owner := OwnerStamp{PID: 4242, LeaseID: "resolve-workerworktree-3573", CreatedAt: now}
	g := newFakeGit().reply("rev-parse", 0, "deadbeef\n").reply("worktree", 0, "")

	res := PrepareOwned("/repo", "workerworktree", "3573", "", t.TempDir(), g.run, owner)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	raw, err := os.ReadFile(OwnerStampPath(res.Path))
	if err != nil {
		t.Fatalf("owner stamp: %v", err)
	}
	var got OwnerStamp
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode owner stamp: %v\n%s", err, raw)
	}
	if got.Schema != ownerStampSchema || got.PID != owner.PID || got.LeaseID != owner.LeaseID || !got.CreatedAt.Equal(now) {
		t.Fatalf("owner stamp = %+v, want pid=%d lease=%q created_at=%s schema=%q",
			got, owner.PID, owner.LeaseID, now, ownerStampSchema)
	}
}

func TestGCListSelectsOnlyOldDeadOwnerReleasedLease(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 13, 15, 0, 0, 0, time.UTC)
	maxAge := time.Hour
	type fixture struct {
		lane, key string
		pid       int
		lease     string
		created   time.Time
	}
	fixtures := []fixture{
		{"dead-old", "1", 101, "resolve-dead-old", now.Add(-2 * time.Hour)},
		{"live-owner", "2", 202, "resolve-live-owner", now.Add(-2 * time.Hour)},
		{"live-lease", "3", 303, "resolve-live-lease", now.Add(-2 * time.Hour)},
		{"dead-young", "4", 404, "resolve-dead-young", now.Add(-10 * time.Minute)},
	}
	var paths []string
	for _, f := range fixtures {
		wt := filepath.Join(root, DirName(f.lane, f.key))
		if err := os.MkdirAll(wt, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := writeOwnerStamp(wt, OwnerStamp{PID: f.pid, LeaseID: f.lease, CreatedAt: f.created}); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, wt)
	}
	porcelain := ""
	for _, p := range paths {
		porcelain += "worktree " + p + "\n\n"
	}
	g := newFakeGit().reply("worktree", 0, porcelain).reply("status", 0, "")
	processAlive := func(pid int) bool { return pid == 202 }
	leaseLive := func(id string) bool { return id == "resolve-live-lease" }

	rows := GCList("/repo", g.run, now, maxAge, processAlive, leaseLive)
	if len(rows) != 1 {
		t.Fatalf("GCList rows=%d, want only one candidate: %+v", len(rows), rows)
	}
	if !rows[0].Eligible || filepath.Clean(rows[0].Path) != filepath.Clean(paths[0]) {
		t.Fatalf("candidate=%+v, want only old + dead owner + released lease path %q", rows[0], paths[0])
	}
	for _, excluded := range paths[1:] {
		for _, row := range rows {
			if filepath.Clean(row.Path) == filepath.Clean(excluded) {
				t.Fatalf("kept worktree %q must never be listed: %+v", excluded, rows)
			}
		}
	}
}

func TestGarbageCollectApplyRemovesAndPrunes(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 13, 15, 0, 0, 0, time.UTC)
	wt := filepath.Join(root, DirName("workerworktree", "3573"))
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeOwnerStamp(wt, OwnerStamp{
		PID:       999999,
		LeaseID:   "resolve-workerworktree-3573",
		CreatedAt: now.Add(-24 * time.Hour),
	}); err != nil {
		t.Fatal(err)
	}

	g := newFakeGit().reply("worktree", 0, "worktree "+wt+"\n").reply("status", 0, "")
	git := func(repo string, args []string) (int, string) {
		rc, out := g.run(repo, args)
		if len(args) == 4 && args[0] == "worktree" && args[1] == "remove" && args[2] == "--force" {
			if err := os.RemoveAll(args[3]); err != nil {
				t.Fatalf("simulate worktree removal: %v", err)
			}
		}
		return rc, out
	}
	report := GarbageCollect("/repo", git, GCOptions{
		Now:          now,
		MaxAge:       time.Hour,
		Apply:        true,
		ProcessAlive: func(int) bool { return false },
		LeaseLive:    func(string) bool { return false },
	})
	if report.Mode != "apply" || report.WouldReap != 1 || report.Reaped != 1 {
		t.Fatalf("report = %+v, want one applied reap", report)
	}
	if len(report.Worktrees) != 1 || filepath.Clean(report.Worktrees[0].Path) != filepath.Clean(wt) {
		t.Fatalf("apply listed non-candidates: %+v", report.Worktrees)
	}
	if _, err := os.Stat(wt); !os.IsNotExist(err) {
		t.Fatalf("worktree dir still exists after apply: %v", err)
	}
	if got := len(g.callsWithPrefix("worktree", "remove", "--force")); got != 1 {
		t.Fatalf("worktree remove calls=%d, want 1: %v", got, g.calls)
	}
	if got := len(g.callsWithPrefix("worktree", "prune")); got != 1 {
		t.Fatalf("worktree prune calls=%d, want 1: %v", got, g.calls)
	}
	if _, err := os.Stat(OwnerStampPath(wt)); !os.IsNotExist(err) {
		t.Fatalf("owner stamp still exists after reap: %v", err)
	}
}

func TestGarbageCollectDefaultsToDryRun(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 13, 15, 0, 0, 0, time.UTC)
	wt := filepath.Join(root, DirName("workerworktree", "dry"))
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeOwnerStamp(wt, OwnerStamp{
		PID:       808,
		LeaseID:   "resolve-workerworktree",
		CreatedAt: now.Add(-2 * time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	g := newFakeGit().reply("worktree", 0, "worktree "+wt+"\n").reply("status", 0, "")
	report := GarbageCollect("/repo", g.run, GCOptions{
		Now:          now,
		MaxAge:       time.Hour,
		ProcessAlive: func(int) bool { return false },
		LeaseLive:    func(string) bool { return false },
	})
	if report.Mode != "dry-run" || report.WouldReap != 1 || report.Reaped != 0 {
		t.Fatalf("dry-run report = %+v", report)
	}
	if len(report.Worktrees) != 1 || filepath.Clean(report.Worktrees[0].Path) != filepath.Clean(wt) {
		t.Fatalf("dry-run listed non-candidates: %+v", report.Worktrees)
	}
	if len(g.callsWithPrefix("worktree", "remove")) != 0 || len(g.callsWithPrefix("worktree", "prune")) != 0 {
		t.Fatalf("dry-run touched deletion verbs: %v", g.calls)
	}
	if _, err := os.Stat(wt); err != nil {
		t.Fatalf("dry-run removed worktree: %v", err)
	}
}
