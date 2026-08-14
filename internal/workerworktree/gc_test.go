package workerworktree

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPrepareWritesOwnerStamp(t *testing.T) {
	now := time.Date(2026, time.August, 14, 12, 34, 56, 789, time.UTC)
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

type stampFailBackend struct {
	path     string
	released bool
}

func (b *stampFailBackend) Materialize(_, _, _, _, _ string, _ GitRunner) Result {
	_ = os.MkdirAll(b.path, 0o755)
	return Result{OK: true, Path: b.path}
}

func (b *stampFailBackend) Release(_, wtPath string, _ GitRunner) Result {
	b.released = true
	_ = os.RemoveAll(wtPath)
	return Result{OK: true, Path: wtPath, Removed: true}
}

func TestPrepareCleansNewWorktreeWhenOwnerStampFails(t *testing.T) {
	root := t.TempDir()
	// Block creation of the sidecar directory with a regular file.
	if err := os.WriteFile(filepath.Join(root, ownerStateDir), []byte("blocked"), 0o600); err != nil {
		t.Fatal(err)
	}
	wt := filepath.Join(root, DirName("workerworktree", "stamp-fail"))
	backend := &stampFailBackend{path: wt}
	res := PrepareOwnedWithBackend("/repo", "workerworktree", "stamp-fail", "base", root, nil, backend, OwnerStamp{
		PID:       123,
		LeaseID:   "resolve-workerworktree-stamp-fail",
		CreatedAt: time.Now(),
	})
	if res.OK {
		t.Fatalf("stamp failure must fail open: %+v", res)
	}
	if !backend.released {
		t.Fatalf("new unowned worktree was not source-cleaned: %+v", res)
	}
	if _, err := os.Stat(wt); !os.IsNotExist(err) {
		t.Fatalf("new unowned worktree leaked: %v", err)
	}
}

type gcGit struct {
	list       string
	status     map[string]string
	heads      map[string]string
	ancestors  map[string]int
	pruneCheck func() bool
	calls      [][]string
}

func (g *gcGit) run(root string, args []string) (int, string) {
	g.calls = append(g.calls, append([]string{}, args...))
	if len(args) == 0 {
		return 0, ""
	}
	switch args[0] {
	case "worktree":
		if len(args) > 1 && args[1] == "list" {
			return 0, g.list
		}
		if len(args) > 1 && args[1] == "prune" {
			if g.pruneCheck != nil && !g.pruneCheck() {
				return 1, "directory remains"
			}
			return 0, ""
		}
	case "status":
		return 0, g.status[filepath.Clean(root)]
	case "rev-parse":
		head, ok := g.heads[filepath.Clean(root)]
		if !ok {
			return 1, ""
		}
		return 0, head + "\n"
	case "merge-base":
		if len(args) >= 4 {
			if rc, ok := g.ancestors[args[2]]; ok {
				return rc, ""
			}
		}
		return 2, "unknown ancestry"
	}
	return 0, ""
}

func (g *gcGit) callsWithPrefix(prefix ...string) int {
	n := 0
	for _, call := range g.calls {
		if len(call) < len(prefix) {
			continue
		}
		ok := true
		for i := range prefix {
			if call[i] != prefix[i] {
				ok = false
				break
			}
		}
		if ok {
			n++
		}
	}
	return n
}

func gcPorcelain(paths ...string) string {
	var b strings.Builder
	for _, p := range paths {
		b.WriteString("worktree " + p + "\nHEAD deadbeef\ndetached\n\n")
	}
	return b.String()
}

func makeGCWorktree(t *testing.T, root, lane, key string, owner OwnerStamp) string {
	t.Helper()
	wt := filepath.Join(root, DirName(lane, key))
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeOwnerStamp(wt, owner); err != nil {
		t.Fatal(err)
	}
	return wt
}

func baseGCOptions(now time.Time, allowedRoot string) GCOptions {
	return GCOptions{
		Now:          now,
		MaxAge:       time.Hour,
		ProcessAlive: func(pid int) bool { return pid == 202 },
		LeaseLive:    func(id string) bool { return id == "resolve-live-lease" },
		PathActive:   func(string) (bool, error) { return false, nil },
		AllowedRoots: []string{allowedRoot},
	}
}

func TestGCListRequiresRootOwnerLeaseAgeProcessCleanAndNoUnpushedCommit(t *testing.T) {
	root := t.TempDir()
	outsideRoot := t.TempDir()
	now := time.Date(2026, time.August, 14, 15, 0, 0, 0, time.UTC)
	old := now.Add(-2 * time.Hour)

	safe := makeGCWorktree(t, root, "safe", "1", OwnerStamp{PID: 101, LeaseID: "resolve-safe", CreatedAt: old})
	liveOwner := makeGCWorktree(t, root, "live-owner", "2", OwnerStamp{PID: 202, LeaseID: "resolve-live-owner", CreatedAt: old})
	liveLease := makeGCWorktree(t, root, "live-lease", "3", OwnerStamp{PID: 303, LeaseID: "resolve-live-lease", CreatedAt: old})
	young := makeGCWorktree(t, root, "young", "4", OwnerStamp{PID: 404, LeaseID: "resolve-young", CreatedAt: now.Add(-10 * time.Minute)})
	active := makeGCWorktree(t, root, "active", "5", OwnerStamp{PID: 505, LeaseID: "resolve-active", CreatedAt: old})
	dirty := makeGCWorktree(t, root, "dirty", "6", OwnerStamp{PID: 606, LeaseID: "resolve-dirty", CreatedAt: old})
	unpushed := makeGCWorktree(t, root, "unpushed", "7", OwnerStamp{PID: 707, LeaseID: "resolve-unpushed", CreatedAt: old})
	outside := makeGCWorktree(t, outsideRoot, "outside", "8", OwnerStamp{PID: 808, LeaseID: "resolve-outside", CreatedAt: old})

	g := &gcGit{
		list: gcPorcelain(safe, liveOwner, liveLease, young, active, dirty, unpushed, outside),
		status: map[string]string{
			filepath.Clean(dirty): " M internal/workerworktree/gc.go\n",
		},
		heads: map[string]string{
			filepath.Clean(safe):     "safe-head",
			filepath.Clean(active):   "active-head",
			filepath.Clean(dirty):    "dirty-head",
			filepath.Clean(unpushed): "unpushed-head",
		},
		ancestors: map[string]int{
			"safe-head":     0,
			"active-head":   0,
			"dirty-head":    0,
			"unpushed-head": 1,
		},
	}
	opts := baseGCOptions(now, root)
	opts.PathActive = func(path string) (bool, error) {
		return filepath.Clean(path) == filepath.Clean(active), nil
	}

	rows := GCList("/repo", g.run, opts)
	if len(rows) != 1 || filepath.Clean(rows[0].Path) != filepath.Clean(safe) {
		t.Fatalf("GCList = %+v, want only %q", rows, safe)
	}
	if !rows[0].Eligible || rows[0].OwnerLive || rows[0].LeaseLive || rows[0].ProcessActive ||
		rows[0].Unlanded != 0 || rows[0].Unpushed != 0 {
		t.Fatalf("candidate evidence = %+v", rows[0])
	}
}

func TestGarbageCollectDefaultsToDryRun(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 14, 15, 0, 0, 0, time.UTC)
	wt := makeGCWorktree(t, root, "workerworktree", "dry", OwnerStamp{
		PID:       808,
		LeaseID:   "resolve-workerworktree",
		CreatedAt: now.Add(-2 * time.Hour),
	})
	g := &gcGit{
		list:      gcPorcelain(wt),
		status:    map[string]string{},
		heads:     map[string]string{filepath.Clean(wt): "safe-head"},
		ancestors: map[string]int{"safe-head": 0},
	}

	report := GarbageCollect("/repo", g.run, baseGCOptions(now, root))
	if report.Mode != "dry-run" || report.WouldReap != 1 || report.Reaped != 0 {
		t.Fatalf("dry-run report = %+v", report)
	}
	if len(g.calls) == 0 || g.callsWithPrefix("worktree", "prune") != 0 {
		t.Fatalf("dry-run touched prune/removal: %v", g.calls)
	}
	if _, err := os.Stat(wt); err != nil {
		t.Fatalf("dry-run removed worktree: %v", err)
	}
}

func TestGarbageCollectApplyRechecksExternalProcess(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 14, 15, 0, 0, 0, time.UTC)
	wt := makeGCWorktree(t, root, "workerworktree", "process-flip", OwnerStamp{
		PID:       909,
		LeaseID:   "resolve-workerworktree-process-flip",
		CreatedAt: now.Add(-2 * time.Hour),
	})
	g := &gcGit{
		list:      gcPorcelain(wt),
		status:    map[string]string{},
		heads:     map[string]string{filepath.Clean(wt): "safe-head"},
		ancestors: map[string]int{"safe-head": 0},
	}
	probes := 0
	opts := baseGCOptions(now, root)
	opts.Apply = true
	opts.PathActive = func(string) (bool, error) {
		probes++
		return probes >= 2, nil
	}

	report := GarbageCollect("/repo", g.run, opts)
	if report.WouldReap != 1 || report.Reaped != 0 {
		t.Fatalf("apply-time process activation must refuse: %+v", report)
	}
	if len(report.Failures) != 1 || report.Failures[0].Reason != "process_command_line_active" {
		t.Fatalf("failures = %+v", report.Failures)
	}
	if _, err := os.Stat(wt); err != nil {
		t.Fatalf("process-active worktree was removed: %v", err)
	}
	if g.callsWithPrefix("worktree", "prune") != 0 {
		t.Fatalf("process-active worktree was unregistered: %v", g.calls)
	}
}

func TestGarbageCollectApplyRemovesDirectoryBeforePrune(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 14, 15, 0, 0, 0, time.UTC)
	wt := makeGCWorktree(t, root, "workerworktree", "apply", OwnerStamp{
		PID:       999999,
		LeaseID:   "resolve-workerworktree-apply",
		CreatedAt: now.Add(-24 * time.Hour),
	})
	g := &gcGit{
		list:      gcPorcelain(wt),
		status:    map[string]string{},
		heads:     map[string]string{filepath.Clean(wt): "safe-head"},
		ancestors: map[string]int{"safe-head": 0},
		pruneCheck: func() bool {
			_, err := os.Stat(wt)
			return os.IsNotExist(err)
		},
	}
	opts := baseGCOptions(now, root)
	opts.Apply = true

	report := GarbageCollect("/repo", g.run, opts)
	if report.Mode != "apply" || report.WouldReap != 1 || report.Reaped != 1 || len(report.Failures) != 0 {
		t.Fatalf("report = %+v, want one applied reap", report)
	}
	if _, err := os.Stat(wt); !os.IsNotExist(err) {
		t.Fatalf("worktree dir still exists after apply: %v", err)
	}
	if g.callsWithPrefix("worktree", "prune") != 1 {
		t.Fatalf("worktree prune calls=%d, want 1: %v", g.callsWithPrefix("worktree", "prune"), g.calls)
	}
	if g.callsWithPrefix("worktree", "remove") != 0 {
		t.Fatalf("GC must not unregister before filesystem removal: %v", g.calls)
	}
	if _, err := os.Stat(OwnerStampPath(wt)); !os.IsNotExist(err) {
		t.Fatalf("owner stamp still exists after reap: %v", err)
	}
}
