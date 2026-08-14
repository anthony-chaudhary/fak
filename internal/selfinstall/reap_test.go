package selfinstall

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type buildGCRunner struct {
	list       string
	status     map[string]string
	heads      map[string]string
	ancestors  map[string]bool
	pruneOK    bool
	pruneCheck func() bool
	ran        [][]string
}

func (r *buildGCRunner) run(_ context.Context, dir, name string, args ...string) (string, bool) {
	r.ran = append(r.ran, append([]string{name}, args...))
	if name != "git" || len(args) == 0 {
		return "", true
	}
	switch args[0] {
	case "worktree":
		if len(args) >= 2 && args[1] == "list" {
			return r.list, true
		}
		if len(args) >= 2 && args[1] == "prune" {
			if r.pruneCheck != nil && !r.pruneCheck() {
				return "directory still present", false
			}
			if !r.pruneOK {
				return "prune failed", false
			}
			return "", true
		}
	case "status":
		out, ok := r.status[filepath.Clean(dir)]
		if !ok {
			return "", true
		}
		return out, true
	case "rev-parse":
		head, ok := r.heads[filepath.Clean(dir)]
		if !ok {
			return "", false
		}
		return head + "\n", true
	case "merge-base":
		if len(args) >= 4 && r.ancestors[args[2]] {
			return "", true
		}
		return "", false
	}
	return "", true
}

func (r *buildGCRunner) prunes() int {
	n := 0
	for _, c := range r.ran {
		if len(c) == 3 && c[0] == "git" && c[1] == "worktree" && c[2] == "prune" {
			n++
		}
	}
	return n
}

// porcelain builds a `git worktree list --porcelain` body from worktree paths.
func porcelain(paths ...string) string {
	var b strings.Builder
	for i, p := range paths {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString("worktree " + p + "\nHEAD deadbeef\ndetached\n")
	}
	return b.String()
}

func makeBuildWorktree(t *testing.T, tempRoot string, pid int, created time.Time, stamped bool) string {
	t.Helper()
	path := filepath.Join(tempRoot, BuildDirName(pid))
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, created, created); err != nil {
		t.Fatal(err)
	}
	if stamped {
		if err := writeBuildOwnerStamp(path, BuildOwnerStamp{
			PID:       pid,
			LeaseID:   "self-update-single-flight",
			CreatedAt: created,
		}); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func buildGCOptions(now time.Time, tempRoot string) BuildGCOptions {
	return BuildGCOptions{
		Now:          now,
		MinAge:       time.Hour,
		SelfPID:      1000,
		TempRoot:     tempRoot,
		BaseRef:      "origin/main",
		ProcessAlive: func(pid int) bool { return pid == 2000 },
		PathActive:   func(string) (bool, error) { return false, nil },
	}
}

func TestStaleBuildGCDryRunRequiresEverySafetyGate(t *testing.T) {
	tempRoot := t.TempDir()
	outsideRoot := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)

	safe := makeBuildWorktree(t, tempRoot, 3000, now.Add(-2*time.Hour), true)
	liveOwner := makeBuildWorktree(t, tempRoot, 2000, now.Add(-2*time.Hour), true)
	self := makeBuildWorktree(t, tempRoot, 1000, now.Add(-2*time.Hour), true)
	young := makeBuildWorktree(t, tempRoot, 4000, now.Add(-10*time.Minute), true)
	active := makeBuildWorktree(t, tempRoot, 5000, now.Add(-2*time.Hour), true)
	dirty := makeBuildWorktree(t, tempRoot, 6000, now.Add(-2*time.Hour), true)
	unpushed := makeBuildWorktree(t, tempRoot, 7000, now.Add(-2*time.Hour), true)
	outside := makeBuildWorktree(t, outsideRoot, 8000, now.Add(-2*time.Hour), true)

	r := &buildGCRunner{
		list: porcelain(
			tempRoot, // the temp root itself is never a candidate
			safe, liveOwner, self, young, active, dirty, unpushed, outside,
		),
		status: map[string]string{
			filepath.Clean(dirty): " M cmd/fak/selfupdate.go\n",
		},
		heads: map[string]string{
			filepath.Clean(safe):     "safe-head",
			filepath.Clean(active):   "active-head",
			filepath.Clean(dirty):    "dirty-head",
			filepath.Clean(unpushed): "unpushed-head",
		},
		ancestors: map[string]bool{
			"safe-head":   true,
			"active-head": true,
			"dirty-head":  true,
			// unpushed-head deliberately absent.
		},
		pruneOK: true,
	}
	opts := buildGCOptions(now, tempRoot)
	opts.PathActive = func(path string) (bool, error) {
		return filepath.Clean(path) == filepath.Clean(active), nil
	}

	report := GarbageCollectStaleBuilds(context.Background(), r.run, "/repo", opts)
	if report.Mode != "dry-run" || report.WouldReap != 1 || report.Reaped != 0 {
		t.Fatalf("dry-run report = %+v, want exactly one candidate and no deletion", report)
	}
	if len(report.Worktrees) != 1 || filepath.Clean(report.Worktrees[0].Path) != filepath.Clean(safe) {
		t.Fatalf("candidates = %+v, want only %q", report.Worktrees, safe)
	}
	if report.Worktrees[0].Owner.PID != 3000 || !report.Worktrees[0].OwnerStamped {
		t.Fatalf("candidate owner evidence = %+v", report.Worktrees[0])
	}
	for _, path := range []string{safe, liveOwner, self, young, active, dirty, unpushed, outside} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("dry-run removed %q: %v", path, err)
		}
	}
	if r.prunes() != 0 {
		t.Fatalf("dry-run ran prune: %v", r.ran)
	}
}

func TestStaleBuildGCApplyRechecksProcessCommandLine(t *testing.T) {
	tempRoot := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)
	wt := makeBuildWorktree(t, tempRoot, 3000, now.Add(-2*time.Hour), true)
	r := &buildGCRunner{
		list:      porcelain(wt),
		status:    map[string]string{},
		heads:     map[string]string{filepath.Clean(wt): "safe-head"},
		ancestors: map[string]bool{"safe-head": true},
		pruneOK:   true,
	}
	probes := 0
	opts := buildGCOptions(now, tempRoot)
	opts.Apply = true
	opts.PathActive = func(string) (bool, error) {
		probes++
		return probes >= 2, nil
	}

	report := GarbageCollectStaleBuilds(context.Background(), r.run, "/repo", opts)
	if report.WouldReap != 1 || report.Reaped != 0 {
		t.Fatalf("apply-time active process must refuse deletion: %+v", report)
	}
	if len(report.Failures) != 1 || report.Failures[0].Reason != "process_command_line_active" {
		t.Fatalf("failures = %+v, want process_command_line_active", report.Failures)
	}
	if _, err := os.Stat(wt); err != nil {
		t.Fatalf("apply-time refusal removed worktree: %v", err)
	}
	if r.prunes() != 0 {
		t.Fatalf("apply-time refusal unregistered the tree: %v", r.ran)
	}
}

func TestStaleBuildGCApplyRemovesDirectoryBeforePruneAndOwnerStamp(t *testing.T) {
	tempRoot := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)
	wt := makeBuildWorktree(t, tempRoot, 3000, now.Add(-2*time.Hour), true)
	r := &buildGCRunner{
		list:      porcelain(wt),
		status:    map[string]string{},
		heads:     map[string]string{filepath.Clean(wt): "safe-head"},
		ancestors: map[string]bool{"safe-head": true},
		pruneOK:   true,
		pruneCheck: func() bool {
			_, err := os.Stat(wt)
			return os.IsNotExist(err)
		},
	}
	opts := buildGCOptions(now, tempRoot)
	opts.Apply = true

	report := GarbageCollectStaleBuilds(context.Background(), r.run, "/repo", opts)
	if report.WouldReap != 1 || report.Reaped != 1 || len(report.Failures) != 0 {
		t.Fatalf("apply report = %+v, want one verified reap", report)
	}
	if _, err := os.Stat(wt); !os.IsNotExist(err) {
		t.Fatalf("worktree directory remains: %v", err)
	}
	if _, err := os.Stat(BuildOwnerStampPath(wt)); !os.IsNotExist(err) {
		t.Fatalf("owner stamp remains: %v", err)
	}
	if r.prunes() != 1 {
		t.Fatalf("prune count = %d, want 1: %v", r.prunes(), r.ran)
	}
	for _, c := range r.ran {
		if strings.Join(c, " ") == "git worktree remove --force "+wt {
			t.Fatalf("GC unregistered before proving filesystem removal: %v", r.ran)
		}
	}
}

func TestStaleBuildGCLegacyNameOwnerStillRequiresRootAgeProcessAndContent(t *testing.T) {
	tempRoot := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)
	wt := makeBuildWorktree(t, tempRoot, 4242, now.Add(-2*time.Hour), false)
	r := &buildGCRunner{
		list:      porcelain(wt),
		status:    map[string]string{},
		heads:     map[string]string{filepath.Clean(wt): "legacy-head"},
		ancestors: map[string]bool{"legacy-head": true},
		pruneOK:   true,
	}

	report := GarbageCollectStaleBuilds(context.Background(), r.run, "/repo", buildGCOptions(now, tempRoot))
	if report.WouldReap != 1 || len(report.Worktrees) != 1 {
		t.Fatalf("legacy safe candidate = %+v", report)
	}
	if report.Worktrees[0].OwnerStamped || report.Worktrees[0].Owner.PID != 4242 {
		t.Fatalf("legacy ownership evidence = %+v", report.Worktrees[0])
	}
}

func TestBuildDirNameRoundTrips(t *testing.T) {
	for _, pid := range []int{1, 42, 67260} {
		name := BuildDirName(pid)
		if !strings.HasPrefix(name, buildDirPrefix) {
			t.Fatalf("BuildDirName(%d)=%q lacks prefix %q", pid, name, buildDirPrefix)
		}
		got, ok := pidFromBuildDir(name)
		if !ok || got != pid {
			t.Fatalf("pidFromBuildDir(%q) = (%d,%v), want (%d,true)", name, got, ok, pid)
		}
	}
	for _, bad := range []string{"fak-build", "peer-worktree", "fak-selfupdate-build-", "fak-selfupdate-build-abc", "fak-selfupdate-build--1"} {
		if pid, ok := pidFromBuildDir(bad); ok {
			t.Fatalf("pidFromBuildDir(%q) = (%d,true), want not-ok", bad, pid)
		}
	}
}
