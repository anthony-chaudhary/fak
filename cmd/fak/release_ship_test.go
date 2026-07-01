package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestReleaseShipExecutesDetachedCutPushTagPublish(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			if !envHas(env, "FAK_RELEASE_LOCK_ROOT=") || !envHas(env, "FAK_RELEASE_OWNER=") {
				t.Fatalf("release_lock acquire env missing shared lock fields: %v", env)
			}
			if !containsReleaseShipArg(args, "--ttl") || !containsReleaseShipArg(args, "--note") {
				t.Fatalf("release_lock acquire missing ttl/note: %v", args)
			}
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
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
			if !containsReleaseShipArg(args, "--lock-already-held") {
				t.Fatalf("release_cut must run under the parent release lock: %v", args)
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
			if !containsReleaseShipArg(args, "--lock-already-held") {
				t.Fatalf("release_tag must run under the parent release lock: %v", args)
			}
			return 0, `{"ok":true,"tag":"v0.35.0","tag_created":true,"tag_pushed":true}`
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_publish.py")):
			return 0, `{"ok":true,"release_created":true,"github_release":{"status":"present","url":"https://example.test/v0.35.0"}}`
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
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
	if result.ReleaseLock == nil || result.ReleaseLockRelease == nil {
		t.Fatalf("release lock acquire/release missing: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
	}
	if len(result.ExecutedCommands) == 0 || !strings.HasSuffix(result.ExecutedCommands[0].Args[0], filepath.Join("tools", "release_lock.py")) || result.ExecutedCommands[0].Args[1] != "acquire" {
		t.Fatalf("release lock must be acquired before release work: %#v", result.ExecutedCommands)
	}
}

func TestReleaseShipDryRunReportsSourceAndTargetBranches(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/dev:refs/remotes/origin/dev"):
			return 0, ""
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/dev^{commit}"):
			return 0, "dev-source-sha\n"
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "dev-source-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[{"databaseId":42,"status":"completed","conclusion":"success","headSha":"dev-source-sha","url":"https://example.test/ci"}]`
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.36.0","tag":"v0.36.0","commit_sha":"release-sha"}`
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
		execute:         false,
		base:            "origin/dev",
		sourceBranch:    "dev",
		remote:          "origin",
		trunk:           "main",
		workflow:        "ci.yml",
		limitCommits:    50,
		ttl:             1800,
		fetch:           true,
		requireCI:       true,
		waitCI:          false,
		skipDryRun:      true,
		ciAppearTimeout: 0,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if result.SourceBranch != "dev" || result.SourceSHA != "dev-source-sha" || result.TargetBranch != "main" || result.Base != "origin/dev" {
		t.Fatalf("source/target witness fields wrong: %#v", result)
	}
	if result.SourceCI == nil || result.SourceCI["ok"] != true || result.SourceCI["status"] != "success" {
		t.Fatalf("source CI witness missing or not green: %#v", result.SourceCI)
	}
	if result.CommitSHA != "release-sha" || result.Tag != "v0.36.0" {
		t.Fatalf("release outputs missing: %#v", result)
	}
}

func TestReleaseShipExecuteRefusesMissingSourceCI(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/dev:refs/remotes/origin/dev"):
			return 0, ""
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/dev^{commit}"):
			return 0, "dev-source-sha\n"
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "dev-source-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[]`
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("missing source CI should refuse before release work; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:         true,
		base:            "origin/dev",
		sourceBranch:    "dev",
		remote:          "origin",
		trunk:           "main",
		workflow:        "ci.yml",
		limitCommits:    50,
		ttl:             1800,
		fetch:           true,
		requireCI:       true,
		waitCI:          false,
		skipDryRun:      true,
		ciAppearTimeout: 0,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.SourceCI == nil || result.SourceCI["status"] != "missing" {
		t.Fatalf("source CI witness = %#v, want missing", result.SourceCI)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "source_ci_unconfirmed") {
		t.Fatalf("errors = %#v, want source_ci_unconfirmed", result.Errors)
	}
	if result.Worktree != "" || result.Cut != nil {
		t.Fatalf("release work should not start without source CI: %#v", result)
	}
	if result.ReleaseLock == nil || result.ReleaseLockRelease == nil {
		t.Fatalf("release lock must be released after source CI refusal: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
	}
}

func TestDefaultReleaseShipOptionsUseBranchRoles(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\ndevelopment_branch = \"dev\"\nrelease_branch = \"main\"\nrelease_source = \"dev\"\npublic_front_door = \"main\"\n"), 0o644); err != nil {
		t.Fatalf("write dos.toml: %v", err)
	}
	opts := defaultReleaseShipOptions(root)
	if opts.sourceBranch != "dev" || opts.trunk != "main" || opts.remote != "origin" {
		t.Fatalf("defaults = source %q trunk %q remote %q, want dev/main/origin", opts.sourceBranch, opts.trunk, opts.remote)
	}
}

func TestReleaseShipCleansDetachedWorktreeWhenCutRefuses(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
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
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
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
	if result.ReleaseLock == nil || result.ReleaseLockRelease == nil {
		t.Fatalf("release lock must be released after cut refusal: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
	}
}

func TestReleaseShipRefusesWhenReleaseLockHeldByAnotherOwner(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 3, `{"ok":false,"reason":"held","holder":{"owner":"human-release"}}`
		default:
			t.Fatalf("release ship must refuse before cut work when the release lock is held; got %s %v in %s", name, args, cwd)
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
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "release_lock_refused") {
		t.Fatalf("errors = %#v", result.Errors)
	}
	if result.Worktree != "" || result.Cut != nil || result.ReleaseLockRelease != nil {
		t.Fatalf("release work should not start and lock should not be released when acquire failed: %#v", result)
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

func containsReleaseShipArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
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
