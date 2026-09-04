package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
			JunkPaths:  []string{"wave.err", "route.err", "tick.dryrun.err"},
			NextAction: "remove 3 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`",
		}, true
	}
	t.Cleanup(func() { syncWorktree = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"check", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want ok; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"in sync", "worktree dirty: 4 path(s)", "junk: wave.err, route.err, tick.dryrun.err", "next: remove 3 junk path(s)", "fak sweep --clean-junk"} {
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
			JunkPaths:    []string{"wave.err", "route.err"},
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
	if strings.Join(got.Worktree.JunkPaths, ",") != "wave.err,route.err" {
		t.Fatalf("junk paths = %+v, want wave.err, route.err", got.Worktree.JunkPaths)
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
			JunkPaths:  []string{"wave.err", "route.err", "tick.dryrun.err"},
			NextAction: "remove 3 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`",
		}, true
	}
	t.Cleanup(func() { syncWorktree = old })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitOK {
		t.Fatalf("exit = %d, want ok; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"pushed work -> origin/work", "velocity:", "/ 5s budget", "100/100 A", "worktree dirty: 4 path(s)", "junk: wave.err, route.err, tick.dryrun.err", "next: remove 3 junk path(s)", "fak sweep --clean-junk"} {
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
			JunkPaths:  []string{"wave.err", "route.err", "tick.dryrun.err"},
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
	if !got.Velocity.Qualified || got.Velocity.Score == nil || got.Velocity.BudgetMS != 5000 {
		t.Fatalf("velocity = %+v, want qualified default 5s push score", got.Velocity)
	}
	if got.Worktree.TotalDirty != 4 || got.Worktree.Junk != 3 {
		t.Fatalf("worktree = %+v, want dirty totals", got.Worktree)
	}
	if strings.Join(got.Worktree.JunkPaths, ",") != "wave.err,route.err,tick.dryrun.err" {
		t.Fatalf("junk paths = %+v, want all classified junk paths", got.Worktree.JunkPaths)
	}
	if !strings.Contains(got.Worktree.NextAction, "remove 3 junk path(s)") || !strings.Contains(got.Worktree.NextAction, "fak sweep --clean-junk") {
		t.Fatalf("next action = %q", got.Worktree.NextAction)
	}
}

func TestRunSyncPushRejectsSubMillisecondVelocityBudget(t *testing.T) {
	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--budget", "500us", "--repo", t.TempDir()})
	if code != syncExitUsage || !strings.Contains(errb.String(), "at least 1ms") {
		t.Fatalf("exit=%d stderr=%q, want usage refusal", code, errb.String())
	}
}

func TestRunSyncPushInternalErrorStillEmitsVelocityJSON(t *testing.T) {
	oldCapture := syncCaptureSource
	syncCaptureSource = func(string) (string, error) { return "0f1e2d3c4b5a60718293a4b5c6d7e8f90a1b2c3d", nil }
	t.Cleanup(func() { syncCaptureSource = oldCapture })

	old := syncSafePush
	syncSafePush = func(context.Context, safesync.PushOptions) (safesync.PushResult, error) {
		return safesync.PushResult{
			Reason: safesync.PushReasonInternal,
			Detail: "merge-base unavailable",
			Velocity: safesync.PushVelocity{
				ElapsedMS: 12, BudgetMS: 1000, BudgetRatio: 0.012,
				Grade: "UNSCORED", Notes: []string{"unscored: safe push ended with INTERNAL_ERROR"},
			},
		}, errors.New("merge-base unavailable")
	}
	t.Cleanup(func() { syncSafePush = old })
	oldWorktree := syncWorktree
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) { return safesync.Worktree{}, false }
	t.Cleanup(func() { syncWorktree = oldWorktree })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--budget", "1s", "--repo", ".", "--json"})
	if code != syncExitInternal {
		t.Fatalf("exit=%d stderr=%q, want internal", code, errb.String())
	}
	var got safesync.PushResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, out.String())
	}
	if got.Reason != safesync.PushReasonInternal || got.Velocity.Score != nil || got.Velocity.ElapsedMS != 12 {
		t.Fatalf("result = %+v, want timed unscored internal error", got)
	}
}

// TestRunSyncPushPinsCapturedSource is #4221's push-side witness: `fak sync push` captures
// the source object ONCE (rev-parse HEAD) and hands SafePush that immutable SHA as the pinned
// SourceRef → refs/heads/<branch> refspec — never a mutable branch-tip push. So HEAD/branch
// movement during a retry or backoff cannot sweep a later peer commit into the push: SafePush
// re-pushes that exact refspec on every attempt (see TestSafePush_SourceRefspecClassifiesAgainstSource).
func TestRunSyncPushPinsCapturedSource(t *testing.T) {
	const captured = "0f1e2d3c4b5a60718293a4b5c6d7e8f90a1b2c3d"

	oldCapture := syncCaptureSource
	syncCaptureSource = func(string) (string, error) { return captured, nil }
	t.Cleanup(func() { syncCaptureSource = oldCapture })

	var got safesync.PushOptions
	oldPush := syncSafePush
	syncSafePush = func(_ context.Context, opts safesync.PushOptions) (safesync.PushResult, error) {
		got = opts
		return safesync.PushResult{Pushed: true, Attempts: 1}, nil
	}
	t.Cleanup(func() { syncSafePush = oldPush })

	oldWorktree := syncWorktree
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) { return safesync.Worktree{}, false }
	t.Cleanup(func() { syncWorktree = oldWorktree })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", t.TempDir(), "--branch", "main"})
	if code != syncExitOK {
		t.Fatalf("exit=%d stderr=%q, want ok", code, errb.String())
	}
	if got.SourceRef != captured {
		t.Fatalf("SafePush SourceRef = %q, want the captured HEAD SHA %q — the push is not pinned to the immutable object", got.SourceRef, captured)
	}
	if got.TargetRef != "refs/heads/main" {
		t.Fatalf("SafePush TargetRef = %q, want refs/heads/main (a pinned target, not a mutable branch-tip push)", got.TargetRef)
	}
}

// TestRunSyncPushRefusesOnCaptureFailure pins the other half of #4221's safety: if the source
// object cannot be captured, push REFUSES (INTERNAL) rather than silently falling back to a
// mutable branch-tip push — "do not convert a refusal into success." SafePush is never called.
func TestRunSyncPushRefusesOnCaptureFailure(t *testing.T) {
	oldCapture := syncCaptureSource
	syncCaptureSource = func(string) (string, error) { return "", errors.New("rev-parse HEAD failed") }
	t.Cleanup(func() { syncCaptureSource = oldCapture })

	pushCalls := 0
	oldPush := syncSafePush
	syncSafePush = func(context.Context, safesync.PushOptions) (safesync.PushResult, error) {
		pushCalls++
		return safesync.PushResult{Pushed: true}, nil
	}
	t.Cleanup(func() { syncSafePush = oldPush })

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"push", "--repo", t.TempDir(), "--branch", "main"})
	if code != syncExitInternal {
		t.Fatalf("exit=%d, want internal refusal on capture failure", code)
	}
	if pushCalls != 0 {
		t.Fatalf("SafePush called %d times after a capture failure, want 0 — a failed capture must never fall through to an unpinned push", pushCalls)
	}
}

func TestDefaultSyncWorktreeIncludesJunkPaths(t *testing.T) {
	root := t.TempDir()
	syncGit(t, root, "init")
	syncWriteFile(t, filepath.Join(root, "wave.err"), "stderr\n")
	syncWriteFile(t, filepath.Join(root, "route.err"), "stderr\n")
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	syncWriteFile(t, filepath.Join(root, "docs", "note.md"), "note\n")

	got, ok := defaultSyncWorktree(context.Background(), root)
	if !ok {
		t.Fatal("defaultSyncWorktree returned ok=false")
	}
	if !got.Dirty || got.Junk != 2 {
		t.Fatalf("worktree = %+v, want two junk paths", got)
	}
	if strings.Join(got.JunkPaths, ",") != "route.err,wave.err" {
		t.Fatalf("junk paths = %+v, want route.err,wave.err", got.JunkPaths)
	}
	if !strings.Contains(got.NextAction, "fak sweep --clean-junk") {
		t.Fatalf("next action = %q, want clean-junk guidance", got.NextAction)
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

func TestRunSyncApplyDirtyIdenticalRefusesWithoutCleaning(t *testing.T) {
	clone := syncCLIFixture(t)
	syncWriteFile(t, filepath.Join(clone, "a.txt"), "v2\n")
	headBefore := syncRev(t, clone, "HEAD")

	var out, errb bytes.Buffer
	code := runSync(&out, &errb, []string{"apply", "--repo", clone, "--remote", "origin", "--branch", "work"})
	if code != syncExitRefused {
		t.Fatalf("exit = %d, want refused; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "DIVERGES") || !strings.Contains(out.String(), "a.txt") {
		t.Fatalf("output should name pre-existing dirty path refusal, got:\n%s", out.String())
	}
	if got := syncRev(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on refusal: got %s want %s", got, headBefore)
	}
	if got := syncReadFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("dirty target-identical bytes changed on refusal: %q", got)
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
