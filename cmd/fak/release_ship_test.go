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
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[{"databaseId":40,"status":"completed","conclusion":"success","headSha":"base-sha","url":"https://example.test/source-ci"}]`
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
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "release-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url"):
			return 0, `[{"databaseId":41,"status":"completed","conclusion":"success","url":"https://example.test/main-ci"}]`
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
		requireCI:       true,
		waitCI:          false,
		skipDryRun:      true,
		ciAppearTimeout: time.Minute,
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
	pushIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "git" && sameArgs(cmd.Args, "push", "origin", "HEAD:refs/heads/main")
	})
	verifyIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "git" && sameArgs(cmd.Args, "ls-remote", "origin", "refs/heads/main")
	})
	ciIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "gh" && sameArgs(cmd.Args, "run", "list", "--workflow", "ci.yml", "--commit", "release-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url")
	})
	tagIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 0 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_tag.py"))
	})
	if !(pushIdx >= 0 && pushIdx < verifyIdx && verifyIdx < ciIdx && ciIdx < tagIdx) {
		t.Fatalf("tag must follow push, remote verify, and promoted CI witness; push=%d verify=%d ci=%d tag=%d commands=%#v", pushIdx, verifyIdx, ciIdx, tagIdx, result.ExecutedCommands)
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
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "main-target-sha\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "main-target-sha", "dev-source-sha"):
			return 0, ""
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			for key, want := range map[string]string{
				"FAK_RELEASE_SOURCE_BRANCH": "dev",
				"FAK_RELEASE_SOURCE_SHA":    "dev-source-sha",
				"FAK_RELEASE_TARGET_BRANCH": "main",
				"FAK_RELEASE_TARGET_SHA":    "main-target-sha",
				"FAK_RELEASE_SOURCE_RANGE":  "main-target-sha..dev-source-sha",
				"FAK_RELEASE_SOURCE_CI":     "success",
			} {
				if got := envValue(env, key); got != want {
					t.Fatalf("%s = %q, want %q in release_cut env %v", key, got, want, env)
				}
			}
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
	if result.TargetSHA != "main-target-sha" || result.TargetAncestry == nil || result.TargetAncestry["ok"] != true || result.TargetAncestry["status"] != "ancestor" {
		t.Fatalf("target ancestry witness wrong: sha=%q ancestry=%#v", result.TargetSHA, result.TargetAncestry)
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

func TestReleaseShipExecuteRefusesNonFastForwardTarget(t *testing.T) {
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
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "main-target-sha\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "main-target-sha", "dev-source-sha"):
			return 1, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("non-fast-forward target should refuse before release work; got %s %v in %s", name, args, cwd)
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
		requireCI:       false,
		waitCI:          false,
		skipDryRun:      true,
		ciAppearTimeout: 0,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.TargetSHA != "main-target-sha" || result.TargetAncestry == nil || result.TargetAncestry["status"] != "non_fast_forward" {
		t.Fatalf("target ancestry = sha %q %#v, want non_fast_forward", result.TargetSHA, result.TargetAncestry)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "target_non_fast_forward") {
		t.Fatalf("errors = %#v, want target_non_fast_forward", result.Errors)
	}
	if result.Worktree != "" || result.Cut != nil {
		t.Fatalf("release work should not start when target cannot fast-forward: %#v", result)
	}
	if result.ReleaseLock == nil || result.ReleaseLockRelease == nil {
		t.Fatalf("release lock must be released after target refusal: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
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

func TestParseReleaseShipOptionsOpenPRRequiresDistinctBranches(t *testing.T) {
	var stderr strings.Builder
	_, err := parseReleaseShipOptions(&stderr, []string{"--open-pr", "--source-branch", "main", "--trunk", "main"})
	if err == nil || !strings.Contains(err.Error(), "--open-pr") {
		t.Fatalf("err = %v, want an --open-pr distinct-branch validation error", err)
	}
}

const openPRFakePromotionLog = "\x1eaaa1111111111111111111111111111111111111\x1ffix(cmd): add release ship open-pr flag (fak cmd)\x1f\x1f\ncmd/fak/release_ship.go\n"
const openPRPromotionBranch = "fak/release/v0.40.0"

func TestReleaseShipOpenPRDryRunPreviewsPlan(t *testing.T) {
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
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "main-target-sha\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "main-target-sha", "dev-source-sha"):
			return 0, ""
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.40.0","tag":"v0.40.0","commit_sha":"cut-sha"}`
		case name == "git" && sameArgs(args, "log", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s%x1f%b%x1f", "main-target-sha..cut-sha"):
			return 0, openPRFakePromotionLog
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
		openPR:          true,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if result.PromotionBranch != openPRPromotionBranch {
		t.Fatalf("promotion branch = %q, want %q", result.PromotionBranch, openPRPromotionBranch)
	}
	if result.PullRequest == nil || result.PullRequest["ok"] != true {
		t.Fatalf("pull request preview missing/not ok: %#v", result.PullRequest)
	}
	if result.PullRequest["head"] != openPRPromotionBranch || result.PullRequest["source_branch"] != "dev" {
		t.Fatalf("preview head/source = %#v, want promotion branch over dev source", result.PullRequest)
	}
	if result.PullRequest["check_ok"] != true {
		t.Fatalf("check_ok = %v, want true", result.PullRequest["check_ok"])
	}
	title, _ := result.PullRequest["title"].(string)
	if !strings.Contains(title, "dev -> main") || !strings.Contains(title, "1 commit") {
		t.Fatalf("title = %q, want a dev->main 1-commit summary", title)
	}
	body, _ := result.PullRequest["body"].(string)
	if !strings.Contains(body, "cmd/fak/release_ship.go") {
		t.Fatalf("body missing file rollup: %q", body)
	}
	if result.PullRequest["url"] != nil {
		t.Fatalf("dry run must not open a real PR: %#v", result.PullRequest)
	}
}

func TestReleaseShipOpenPRExecuteOpensPR(t *testing.T) {
	const wantURL = "https://github.com/anthony-chaudhary/fak/pull/9999"
	promotionProbeCount := 0
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
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "main-target-sha\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "main-target-sha", "dev-source-sha"):
			return 0, ""
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.40.0","tag":"v0.40.0","commit_sha":"cut-sha"}`
		case name == "git" && sameArgs(args, "log", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s%x1f%b%x1f", "main-target-sha..cut-sha"):
			return 0, openPRFakePromotionLog
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/"+openPRPromotionBranch):
			promotionProbeCount++
			if promotionProbeCount == 1 {
				return 0, ""
			}
			return 0, "cut-sha\trefs/heads/" + openPRPromotionBranch + "\n"
		case name == "git" && sameArgs(args, "push", "origin", "HEAD:refs/heads/"+openPRPromotionBranch):
			return 0, ""
		case name == "gh" && sameArgs(args, "pr", "list", "--base", "main", "--head", openPRPromotionBranch, "--state", "open", "--json", "number,url,title,headRefName,baseRefName", "--limit", "1"):
			return 0, `[]`
		case name == "gh" && len(args) == 10 && args[0] == "pr" && args[1] == "create" && args[2] == "--base" && args[3] == "main" && args[4] == "--head" && args[5] == openPRPromotionBranch && args[6] == "--title" && args[8] == "--body-file":
			return 0, wantURL + "\n"
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
		base:            "origin/dev",
		sourceBranch:    "dev",
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
		openPR:          true,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if result.PullRequest == nil || result.PullRequest["url"] != wantURL {
		t.Fatalf("pull request result = %#v, want url %q", result.PullRequest, wantURL)
	}
	if result.PullRequest["head"] != openPRPromotionBranch || result.PullRequest["source_branch"] != "dev" {
		t.Fatalf("pull request head/source = %#v, want promotion branch over dev source", result.PullRequest)
	}
	if result.PullRequest["created"] != true {
		t.Fatalf("pull request created flag = %#v, want true", result.PullRequest["created"])
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("--open-pr must not tag/publish; got tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
}

func TestReleaseShipOpenPRExecuteUpdatesExistingPR(t *testing.T) {
	const wantURL = "https://github.com/anthony-chaudhary/fak/pull/9999"
	promotionProbeCount := 0
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
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "main-target-sha\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "main-target-sha", "dev-source-sha"):
			return 0, ""
		case name == "git" && len(args) >= 5 && sameArgs(args[:3], "worktree", "add", "--detach"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.40.0","tag":"v0.40.0","commit_sha":"cut-sha"}`
		case name == "git" && sameArgs(args, "log", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s%x1f%b%x1f", "main-target-sha..cut-sha"):
			return 0, openPRFakePromotionLog
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/"+openPRPromotionBranch):
			promotionProbeCount++
			if promotionProbeCount == 1 {
				return 0, "old-cut-sha\trefs/heads/" + openPRPromotionBranch + "\n"
			}
			return 0, "cut-sha\trefs/heads/" + openPRPromotionBranch + "\n"
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/"+openPRPromotionBranch+":old-cut-sha", "HEAD:refs/heads/"+openPRPromotionBranch):
			return 0, ""
		case name == "gh" && sameArgs(args, "pr", "list", "--base", "main", "--head", openPRPromotionBranch, "--state", "open", "--json", "number,url,title,headRefName,baseRefName", "--limit", "1"):
			return 0, `[{"number":9999,"url":"` + wantURL + `","title":"old title","headRefName":"` + openPRPromotionBranch + `","baseRefName":"main"}]`
		case name == "gh" && len(args) == 7 && args[0] == "pr" && args[1] == "edit" && args[2] == wantURL && args[3] == "--title" && args[5] == "--body-file":
			return 0, ""
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
		base:            "origin/dev",
		sourceBranch:    "dev",
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
		openPR:          true,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if result.PullRequest == nil || result.PullRequest["url"] != wantURL {
		t.Fatalf("pull request result = %#v, want url %q", result.PullRequest, wantURL)
	}
	if result.PullRequest["existing"] != true || result.PullRequest["updated"] != true {
		t.Fatalf("existing PR should be refreshed, got %#v", result.PullRequest)
	}
	createIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "gh" && len(cmd.Args) > 1 && cmd.Args[0] == "pr" && cmd.Args[1] == "create"
	})
	if createIdx >= 0 {
		t.Fatalf("existing PR rerun must not create a duplicate: %#v", result.ExecutedCommands)
	}
}

const fakeReleaseWorktree = "C:\\tmp\\fak-release-ship-test"

func stubReleaseShipRunner(t *testing.T, runner releaseShipCommandRunner) func() {
	t.Helper()
	t.Setenv("FAK_PYTHON", "python-test")
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

func envValue(env []string, key string) string {
	prefix := key + "="
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			return strings.TrimPrefix(item, prefix)
		}
	}
	return ""
}

func releaseShipCommandIndex(commands []releaseShipCommand, match func(releaseShipCommand) bool) int {
	for i, cmd := range commands {
		if match(cmd) {
			return i
		}
	}
	return -1
}
