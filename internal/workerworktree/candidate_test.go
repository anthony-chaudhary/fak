package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// realSweepTopologyCandidatesBeforeCreate keeps the production hook: package

// variables initialize before init, so this captures it before init stubs it.

var realSweepTopologyCandidatesBeforeCreate = sweepTopologyCandidatesBeforeCreate

// Land-flow tests that reach verifyTopologyCandidate with a non-sibling root would

// otherwise sweep the host's real temp directory. Tests that exercise the sweep

// restore the real hook explicitly.

func init() {
	sweepTopologyCandidatesBeforeCreate = func(string, string, GitRunner) {}
}

// newSiblingTopologyRepo returns a committed repository whose go.work escapes it,

// so its topology candidates are created beside it, and that parent directory.

func newSiblingTopologyRepo(t *testing.T) (root, parent string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	parent = t.TempDir()
	root = filepath.Join(parent, "repo")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "trunk@test")
	mustGit(t, root, "config", "user.name", "trunk")
	mustGit(t, root, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(root, "go.work"), []byte("go 1.22\n\nuse (\n\t.\n\t../sibling\n)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", "go.work")
	mustGit(t, root, "commit", "-q", "-m", "init")
	return root, parent
}

func registeredCandidates(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	for _, line := range strings.Split(mustGit(t, root, "worktree", "list", "--porcelain"), "\n") {
		if p, ok := strings.CutPrefix(strings.TrimSpace(line), "worktree "); ok && strings.Contains(p, topologyCandidatePrefix) {
			out = append(out, p)
		}
	}
	return out
}

// TestCandidateLeakHelperProcess is the child half of

// TestKilledLandLeaksCandidateUntilSweep: it materializes a candidate, reports its

// path, and blocks inside verify until the parent kills it.

// TestKilledLandLeaksCandidateUntilSweep reproduces the leak: a land killed during

// verification never runs its deferred cleanup, so the candidate checkout and its

// registration survive. The next sweep proves the owner is gone and collects both.

// TestSweepUnlocksRegistrationOfCandidateKilledDuringAdd covers a land killed

// inside `git worktree add`: git's "initializing" lock outlives the directory, and

// `git worktree prune` alone never clears a locked entry.

func TestSweepUnlocksRegistrationOfCandidateKilledDuringAdd(t *testing.T) {
	root, parent := newSiblingTopologyRepo(t)
	dead := filepath.Join(parent, topologyCandidatePrefix+"2147483647-0-7")
	live := filepath.Join(parent, strings.TrimSuffix(topologyCandidatePattern(), "*")+"8")
	for _, wt := range []string{dead, live} {
		mustGit(t, root, "worktree", "add", "--detach", wt, "HEAD")
		mustGit(t, root, "worktree", "lock", "--reason", "initializing", wt)
		if err := os.RemoveAll(wt); err != nil {
			t.Fatal(err)
		}
	}
	mustGit(t, root, "worktree", "prune")
	if got := registeredCandidates(t, root); len(got) != 2 {
		t.Fatalf("locked registrations should survive a plain prune (the leak), got %v", got)
	}

	report := SweepTopologyCandidates(root, parent, nil, CandidateSweepOptions{Apply: true})
	if report.Reaped != 1 || report.WouldReap != 1 || len(report.Failures) != 0 {
		t.Fatalf("sweep did not unlock exactly the dead owner's registration: %+v", report)
	}
	got := registeredCandidates(t, root)
	if len(got) != 1 || !strings.Contains(got[0], filepath.Base(live)) {
		t.Fatalf("want only the live owner's registration kept, got %v", got)
	}
}

func TestSweepTopologyCandidatesKeepsLiveAndYoungLegacy(t *testing.T) {
	parent := t.TempDir()
	now := time.Now()
	start := now.Add(-time.Hour)
	mk := func(name string, mtime time.Time) string {
		p := filepath.Join(parent, name)
		if err := os.MkdirAll(filepath.Join(p, "sub"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, mtime, mtime); err != nil {
			t.Fatal(err)
		}
		return p
	}
	token := strconv.FormatInt(start.UnixMilli(), 36)
	live := mk(topologyCandidatePrefix+"100-"+token+"-1", now)
	exited := mk(topologyCandidatePrefix+"200-"+token+"-2", now)
	reused := mk(topologyCandidatePrefix+"300-"+token+"-3", now)
	unknownStart := mk(topologyCandidatePrefix+"400-"+token+"-4", now)
	legacyOld := mk(topologyCandidatePrefix+"123456", now.Add(-3*time.Hour))
	legacyYoung := mk(topologyCandidatePrefix+"654321", now.Add(-30*time.Minute))
	unrelated := mk(".fak-other-1", now.Add(-48*time.Hour))

	opts := CandidateSweepOptions{
		Now:          now,
		ProcessAlive: func(pid int) bool { return pid != 200 },
		ProcessStart: func(pid int) (time.Time, bool) {
			switch pid {
			case 100:
				return start, true
			case 300:
				return start.Add(time.Minute), true
			}
			return time.Time{}, false
		},
	}
	topologyCandidates := []string{live, exited, reused, unknownStart, legacyOld, legacyYoung}
	dry := SweepTopologyCandidates(parent, parent, noGit, opts)
	if dry.Mode != "dry-run" || dry.WouldReap != 3 || dry.Reaped != 0 || len(dry.Candidates) != len(topologyCandidates) {
		t.Fatalf("dry-run plan wrong: %+v", dry)
	}
	for _, p := range []string{live, exited, reused, unknownStart, legacyOld, legacyYoung, unrelated} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("dry-run removed %s: %v", p, err)
		}
	}

	opts.Apply = true
	applied := SweepTopologyCandidates(parent, parent, noGit, opts)
	if applied.Reaped != 3 || len(applied.Failures) != 0 {
		t.Fatalf("apply result wrong: %+v", applied)
	}
	reasons := map[string]string{}
	for _, item := range applied.Candidates {
		reasons[item.Path] = item.Reason
	}
	for path, want := range map[string]string{
		live:         "owner_process_live",
		exited:       "owner_exited",
		reused:       "owner_pid_reused",
		unknownStart: "owner_process_live",
		legacyYoung:  "legacy_too_young",
	} {
		if reasons[path] != want {
			t.Fatalf("%s reason = %q, want %q", filepath.Base(path), reasons[path], want)
		}
	}
	if !strings.HasPrefix(reasons[legacyOld], "legacy_stale") {
		t.Fatalf("legacy candidate reason = %q, want legacy_stale", reasons[legacyOld])
	}
	for _, p := range []string{exited, reused, legacyOld} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("eligible candidate %s survived apply: %v", filepath.Base(p), err)
		}
	}
	for _, p := range []string{live, unknownStart, legacyYoung, unrelated} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("kept path %s was removed: %v", filepath.Base(p), err)
		}
	}
}

func TestSweepTopologyCandidatesHonorsLimit(t *testing.T) {
	parent := t.TempDir()
	for i := 1; i <= 3; i++ {
		if err := os.Mkdir(filepath.Join(parent, topologyCandidatePrefix+strconv.Itoa(i)+"-0-"+strconv.Itoa(i)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	report := SweepTopologyCandidates(parent, parent, noGit, CandidateSweepOptions{
		Apply: true, Limit: 2, ProcessAlive: func(int) bool { return false },
	})
	if report.WouldReap != 3 || report.Reaped != 2 {
		t.Fatalf("limit not honored: would=%d reaped=%d", report.WouldReap, report.Reaped)
	}
}

func TestParseTopologyCandidateOwner(t *testing.T) {
	pattern := topologyCandidatePattern()
	name := strings.TrimSuffix(pattern, "*") + "987654321"
	owner, ok := parseTopologyCandidateOwner(name)
	if !ok || owner.PID != os.Getpid() {
		t.Fatalf("own pattern %q did not round-trip: owner=%+v ok=%v", pattern, owner, ok)
	}
	for _, bad := range []string{
		topologyCandidatePrefix + "2015920908",
		topologyCandidatePrefix + "x-0-1",
		topologyCandidatePrefix + "0-0-1",
		topologyCandidatePrefix + "12-0-",
		topologyCandidatePrefix + "12-!!-1",
		".fak-other-12-0-1",
	} {
		if _, ok := parseTopologyCandidateOwner(bad); ok {
			t.Fatalf("%q parsed as owner-named", bad)
		}
	}
}

// TestVerifyTopologyCandidateSweepsBeforeCreatingAndCleansUp covers the land-path

// wiring: a dead owner's candidate beside the repository is collected before the

// new candidate is created, and the new one is gone, unregistered, afterwards.

func noGit(string, []string) (int, string) { return 0, "" }
