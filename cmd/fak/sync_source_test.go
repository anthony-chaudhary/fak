package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// fak-test:runtime fast est=500ms
func TestRunSyncPushExplicitSource(t *testing.T) {
	clone := syncCLIFixture(t)
	approved := syncRev(t, clone, "HEAD")
	// HEAD advances after approval; publishing the approved object must not capture it.
	syncGit(t, clone, "merge", "--ff-only", "origin/work")
	if syncRev(t, clone, "HEAD") == approved {
		t.Fatal("fixture HEAD did not advance")
	}
	oldResolve, oldCapture, oldPush, oldWorktree := syncResolveSource, syncCaptureSource, syncSafePush, syncWorktree
	t.Cleanup(func() {
		syncResolveSource, syncCaptureSource, syncSafePush, syncWorktree = oldResolve, oldCapture, oldPush, oldWorktree
	})
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) { return safesync.Worktree{}, false }

	for _, tc := range []struct {
		name         string
		args         []string
		want         int
		resolveCalls int
		pushCalls    int
	}{
		{"approved", []string{"push", "--source", approved}, syncExitOK, 1, 1},
		{"commit-expression", []string{"push", "--source", "HEAD^"}, syncExitOK, 1, 1},
		{"missing", []string{"push", "--source", "refs/heads/missing-approved-source"}, syncExitInternal, 1, 0},
		{"tree", []string{"push", "--source", "HEAD^{tree}"}, syncExitInternal, 1, 0},
		{"option-like", []string{"push", "--source=--all"}, syncExitInternal, 1, 0},
		{"empty", []string{"push", "--source="}, syncExitUsage, 0, 0},
		{"check", []string{"check", "--source", approved}, syncExitUsage, 0, 0},
		{"dry-run", []string{"push", "--source", approved, "--dry-run"}, syncExitUsage, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolveCalls, captureCalls, pushCalls := 0, 0, 0
			syncResolveSource = func(repo, source string) (string, error) {
				resolveCalls++
				return oldResolve(repo, source)
			}
			syncCaptureSource = func(string) (string, error) {
				captureCalls++
				return "", errors.New("explicit source must not capture HEAD")
			}
			syncSafePush = func(_ context.Context, opts safesync.PushOptions) (safesync.PushResult, error) {
				pushCalls++
				if opts.SourceRef != approved || opts.TargetRef != "refs/heads/work" || opts.Branch != "work" || opts.Remote != "origin" || opts.Repo != clone || opts.MaxRetries != 2 || opts.VelocityBudget != time.Second {
					t.Fatalf("unexpected push options: %+v", opts)
				}
				return safesync.PushResult{Pushed: true}, nil
			}
			args := append(append([]string{}, tc.args...), "--repo", clone, "--branch", "work", "--remote", "origin", "--retries", "2", "--budget", "1s")
			var out, errb bytes.Buffer
			if code := runSync(&out, &errb, args); code != tc.want {
				t.Fatalf("exit=%d want=%d stderr=%s", code, tc.want, errb.String())
			}
			if resolveCalls != tc.resolveCalls || pushCalls != tc.pushCalls || captureCalls != 0 {
				t.Fatalf("resolve=%d push=%d capture=%d; want %d/%d/0", resolveCalls, pushCalls, captureCalls, tc.resolveCalls, tc.pushCalls)
			}
			if tc.want == syncExitInternal && !strings.Contains(errb.String(), "capture push source") {
				t.Fatalf("missing source refusal: %s", errb.String())
			}
		})
	}
}
