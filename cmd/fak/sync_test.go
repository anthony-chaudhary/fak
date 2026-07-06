package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncCheckInSyncJSON(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"check", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got safesync.Assessment
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("sync JSON did not decode: %v\n%s", err, out.String())
	}
	if got.State != safesync.StateInSync || !got.OK {
		t.Fatalf("assessment = %+v, want in-sync ok", got)
	}
}

func TestRunSyncCheckInSyncSurfacesDirtyWorktree(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")

	old := syncWorktree
	syncWorktree = func(ctx context.Context, repo string) (safesync.Worktree, bool) {
		if repo != clone {
			t.Fatalf("repo = %q, want %q", repo, clone)
		}
		return safesync.Worktree{
			Dirty:      true,
			TotalDirty: 4,
			Stampable:  1,
			Lanes:      1,
			Junk:       3,
			NextAction: "remove 3 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`",
		}, true
	}
	t.Cleanup(func() { syncWorktree = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"check", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want ok; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"in sync", "worktree dirty: 4 path(s)", "next: remove 3 junk path(s)", "fak sweep --clean-junk"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("sync output missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunSyncCheckInSyncJSONSurfacesDirtyWorktree(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")

	old := syncWorktree
	syncWorktree = func(ctx context.Context, repo string) (safesync.Worktree, bool) {
		if repo != clone {
			t.Fatalf("repo = %q, want %q", repo, clone)
		}
		return safesync.Worktree{
			Dirty:        true,
			TotalDirty:   5,
			Stampable:    2,
			Lanes:        2,
			NoLane:       1,
			Junk:         2,
			OldestPath:   "wave.err",
			OldestAgeSec: 600,
			NextAction:   "remove 2 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`",
		}, true
	}
	t.Cleanup(func() { syncWorktree = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"check", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want ok; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got safesync.Assessment
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("sync JSON did not decode: %v\n%s", err, out.String())
	}
	if got.Worktree == nil || !got.Worktree.Dirty {
		t.Fatalf("worktree = %+v, want dirty metadata", got.Worktree)
	}
	if got.Worktree.TotalDirty != 5 || got.Worktree.NoLane != 1 || got.Worktree.Junk != 2 {
		t.Fatalf("worktree = %+v, want dirty totals", got.Worktree)
	}
	if got.Worktree.OldestPath != "wave.err" || got.Worktree.OldestAgeSec != 600 {
		t.Fatalf("worktree = %+v, want oldest dirty metadata", got.Worktree)
	}
	if !strings.Contains(got.Worktree.NextAction, "remove 2 junk path(s)") || !strings.Contains(got.Worktree.NextAction, "fak sweep --clean-junk") {
		t.Fatalf("next action = %q", got.Worktree.NextAction)
	}
}

func TestRunSyncPushSurfacesDirtyWorktree(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")

	old := syncWorktree
	syncWorktree = func(ctx context.Context, repo string) (safesync.Worktree, bool) {
		if repo != clone {
			t.Fatalf("repo = %q, want %q", repo, clone)
		}
		return safesync.Worktree{
			Dirty:      true,
			TotalDirty: 4,
			Stampable:  1,
			Lanes:      1,
			Junk:       3,
			NextAction: "remove 3 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`",
		}, true
	}
	t.Cleanup(func() { syncWorktree = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want ok; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"pushed work -> origin/work", "worktree dirty: 4 path(s)", "next: remove 3 junk path(s)", "fak sweep --clean-junk"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("sync push output missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunSyncPushJSONSurfacesDirtyWorktree(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")

	old := syncWorktree
	syncWorktree = func(ctx context.Context, repo string) (safesync.Worktree, bool) {
		if repo != clone {
			t.Fatalf("repo = %q, want %q", repo, clone)
		}
		return safesync.Worktree{
			Dirty:      true,
			TotalDirty: 4,
			Stampable:  1,
			Lanes:      1,
			Junk:       3,
			NextAction: "remove 3 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`",
		}, true
	}
	t.Cleanup(func() { syncWorktree = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want ok; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got safesync.PushResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("sync push JSON did not decode: %v\n%s", err, out.String())
	}
	if !got.Pushed || got.Worktree == nil || !got.Worktree.Dirty {
		t.Fatalf("push result = %+v, want pushed with dirty worktree metadata", got)
	}
	if got.Worktree.TotalDirty != 4 || got.Worktree.Junk != 3 {
		t.Fatalf("worktree = %+v, want dirty totals", got.Worktree)
	}
	if !strings.Contains(got.Worktree.NextAction, "remove 3 junk path(s)") || !strings.Contains(got.Worktree.NextAction, "fak sweep --clean-junk") {
		t.Fatalf("next action = %q", got.Worktree.NextAction)
	}
}

func TestRunSyncCheckRefusesDivergentDirtyPath(t *testing.T) {
	clone := syncCLIFixture(t)
	syncWriteFile(t, filepath.Join(clone, "a.txt"), "LOCAL EDIT\n")

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitRefused {
		t.Fatalf("exit = %d, want refused; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "DIVERGES") || !strings.Contains(out.String(), "a.txt") {
		t.Fatalf("human output should name divergent path, got:\n%s", out.String())
	}
	if got := syncRev(t, clone, "HEAD"); got == syncRev(t, clone, "origin/work") {
		t.Fatal("refused check/apply should not move HEAD")
	}
}

func TestRunSyncCheckAheadNamesSafePush(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")
	syncWriteFile(t, filepath.Join(clone, "local.txt"), "local\n")
	syncGit(t, clone, "add", "local.txt")
	syncGit(t, clone, "commit", "-m", "local")

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"check", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitRefused {
		t.Fatalf("exit = %d, want refused; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "fak sync push --remote origin --branch work") {
		t.Fatalf("human output should name safe push path, got:\n%s", out.String())
	}
}

func TestRunSyncCheckAheadSurfacesPushAuditResidual(t *testing.T) {
	clone := syncCLIFixture(t)
	syncGit(t, clone, "merge", "--ff-only", "origin/work")
	syncWriteFile(t, filepath.Join(clone, "local.txt"), "local\n")
	syncGit(t, clone, "add", "local.txt")
	syncGit(t, clone, "commit", "-m", "test(experiments): add data artifacts (fak experiments)")

	old := syncAheadAudit
	syncAheadAudit = func(ctx context.Context, repo, targetRef string) (safesync.PushAudit, bool) {
		if targetRef != "origin/work" {
			t.Fatalf("targetRef = %q, want origin/work", targetRef)
		}
		return safesync.PushAudit{
			OK:    false,
			Range: targetRef + "..HEAD",
			Residuals: []safesync.PushAuditResidual{{
				SHA:       "abc12345",
				Subject:   "test(experiments): add data artifacts (fak experiments)",
				Verdict:   "CLAIM_UNWITNESSED",
				ClaimKind: "test",
				Witness:   "subject-only",
				Reason:    "claims tests but the diff touches no test file",
			}},
		}, true
	}
	t.Cleanup(func() { syncAheadAudit = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"check", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
	if code != syncExitRefused {
		t.Fatalf("exit = %d, want refused; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got safesync.Assessment
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("sync JSON did not decode: %v\n%s", err, out.String())
	}
	if got.PushAudit == nil || got.PushAudit.OK || len(got.PushAudit.Residuals) != 1 {
		t.Fatalf("push audit = %+v, want one blocking residual", got.PushAudit)
	}
	if !strings.Contains(got.Reason, "pre-push audit would block") {
		t.Fatalf("reason should name pre-push audit blocker, got %q", got.Reason)
	}
	if got.PushAudit.Residuals[0].Witness != "subject-only" {
		t.Fatalf("residual = %+v, want subject-only witness", got.PushAudit.Residuals[0])
	}
}

func TestRunSyncApplySafeFastForward(t *testing.T) {
	clone := syncCLIFixture(t)
	syncWriteFile(t, filepath.Join(clone, "a.txt"), "v2\n")     // already target bytes
	syncWriteFile(t, filepath.Join(clone, "mine.txt"), "local") // unrelated dirty work

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"apply", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if got, want := syncRev(t, clone, "HEAD"), syncRev(t, clone, "origin/work"); got != want {
		t.Fatalf("HEAD = %s, want origin/work %s", got, want)
	}
	if got := syncReadFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q", got)
	}
	if got := syncReadFile(t, filepath.Join(clone, "new.txt")); got != "n1\n" {
		t.Fatalf("new.txt = %q", got)
	}
	if got := syncReadFile(t, filepath.Join(clone, "mine.txt")); got != "local" {
		t.Fatalf("unrelated work was not preserved: %q", got)
	}
}

func syncCLIFixture(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	origin := filepath.Join(tmp, "origin")
	if err := os.MkdirAll(origin, 0o755); err != nil {
		t.Fatal(err)
	}
	syncGit(t, origin, "init", "-b", "work")
	syncGit(t, origin, "config", "core.autocrlf", "false")
	syncGit(t, origin, "config", "user.name", "test")
	syncGit(t, origin, "config", "user.email", "test@example.com")
	syncWriteFile(t, filepath.Join(origin, "a.txt"), "v1\n")
	syncWriteFile(t, filepath.Join(origin, "keep.txt"), "keep\n")
	syncGit(t, origin, "add", ".")
	syncGit(t, origin, "commit", "-m", "c1")

	clone := filepath.Join(tmp, "clone")
	syncGit(t, tmp, "clone", origin, clone)
	syncGit(t, clone, "config", "core.autocrlf", "false")
	syncGit(t, clone, "config", "user.name", "test")
	syncGit(t, clone, "config", "user.email", "test@example.com")

	syncWriteFile(t, filepath.Join(origin, "a.txt"), "v2\n")
	syncWriteFile(t, filepath.Join(origin, "new.txt"), "n1\n")
	syncGit(t, origin, "add", ".")
	syncGit(t, origin, "commit", "-m", "c2")
	syncGit(t, clone, "fetch", "origin")
	return clone
}

func syncGit(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func syncRev(t *testing.T, cwd, ref string) string {
	t.Helper()
	cmd := exec.Command("git", "rev-parse", "--verify", ref)
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("rev-parse %s: %v", ref, err)
	}
	return strings.TrimSpace(string(out))
}

func syncWriteFile(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func syncReadFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
