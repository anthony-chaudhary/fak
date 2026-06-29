package main

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestReleaseShipExecutesDetachedCutPushTagPublish(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			if !envHas(env, "FAK_RELEASE_LOCK_ROOT=") || !envHas(env, "FAK_RELEASE_OWNER=") {
				t.Fatalf("release_cut env missing shared lock fields: %v", env)
			}
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "push", "origin", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "release-sha\trefs/heads/main\n"
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_tag.py")):
			if !containsArgPair(args, "--trunk", "") {
				t.Fatalf("release_tag must skip local-branch reachability after remote push: %v", args)
			}
			return 0, `{"ok":true,"tag":"v0.35.0","tag_created":true,"tag_pushed":true}`
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_publish.py")):
			return 0, `{"ok":true,"release_created":true,"github_release":{"status":"present","url":"https://example.test/v0.35.0"}}`
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		default:
			t.Fatalf("unexpected command in %s: %s %v", cwd, name, args)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:         true,
		base:            "origin/main",
		remote:          "origin",
		trunk:           "main",
		workflow:        "ci.yml",
		limitCommits:    50,
		ttl:             1800,
		fetch:           true,
		requireCI:       false,
		waitCI:          false,
		skipDryRun:      true,
		ciAppearTimeout: 0,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if result.Worktree != fakeReleaseWorktree {
		t.Fatalf("worktree = %q, want %q", result.Worktree, fakeReleaseWorktree)
	}
	if result.CommitSHA != "release-sha" || result.Version != "0.35.0" || result.Tag != "v0.35.0" {
		t.Fatalf("release outputs missing: %#v", result)
	}
	if result.RemoteBranch["sha"] != "release-sha" {
		t.Fatalf("remote branch = %#v", result.RemoteBranch)
	}
	if result.Cleanup == nil || result.Cleanup["ok"] != true {
		t.Fatalf("cleanup missing/failed: %#v", result.Cleanup)
	}
}

func TestReleaseShipCleansDetachedWorktreeWhenCutRefuses(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 1, `{"ok":false,"aborted":"release_decide held"}`
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		default:
			t.Fatalf("unexpected command in %s: %s %v", cwd, name, args)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:      true,
		base:         "origin/main",
		remote:       "origin",
		trunk:        "main",
		workflow:     "ci.yml",
		limitCommits: 50,
		ttl:          1800,
		fetch:        true,
		skipDryRun:   true,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.Cleanup == nil || result.Cleanup["ok"] != true {
		t.Fatalf("cleanup missing/failed: %#v", result.Cleanup)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "release_cut_refused") {
		t.Fatalf("errors = %#v", result.Errors)
	}
}

const fakeReleaseWorktree = "C:\\tmp\\fak-release-ship-test"

func stubReleaseShipRunner(t *testing.T, runner releaseShipCommandRunner) func() {
	t.Helper()
	oldRunner := releaseShipRunCommand
	oldMkdir := releaseShipMkdirTemp
	releaseShipRunCommand = runner
	releaseShipMkdirTemp = func(parent, pattern string) (string, error) {
		return fakeReleaseWorktree, nil
	}
	return func() {
		releaseShipRunCommand = oldRunner
		releaseShipMkdirTemp = oldMkdir
	}
}

func sameArgs(got []string, want ...string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func containsArgPair(args []string, key, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == key && args[i+1] == value {
			return true
		}
	}
	return false
}

func envHas(env []string, prefix string) bool {
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			return true
		}
	}
	return false
}
