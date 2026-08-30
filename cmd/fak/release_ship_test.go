package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

func TestReleaseShipExecutesDetachedCutPushTagPublish(t *testing.T) {
	renewCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			if !envHas(env, "FAK_RELEASE_LOCK_ROOT=") || !envHas(env, "FAK_RELEASE_OWNER=") {
				t.Fatalf("release_lock acquire env missing shared lock fields: %v", env)
			}
			if !containsArgPair(args, "--ttl", "3480") || !containsReleaseShipArg(args, "--note") {
				t.Fatalf("release_lock acquire missing ttl/note: %v", args)
			}
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[{"databaseId":40,"status":"completed","conclusion":"success","headSha":"base-sha","url":"https://example.test/source-ci"}]`
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			if !envHas(env, "FAK_RELEASE_LOCK_ROOT=") || !envHas(env, "FAK_RELEASE_OWNER=") {
				t.Fatalf("release_cut env missing shared lock fields: %v", env)
			}
			if !containsReleaseShipArg(args, "--lock-already-held") {
				t.Fatalf("release_cut must run under the parent release lock: %v", args)
			}
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			if !envHas(env, "FAK_RELEASE_LOCK_ROOT=") || !envHas(env, "FAK_RELEASE_OWNER=") {
				t.Fatalf("release_lock renew env missing shared lock fields: %v", env)
			}
			renewCount++
			switch renewCount {
			case 1:
				if !containsArgPair(args, "--ttl", "3000") {
					t.Fatalf("release_lock before-tag renew ttl = %v, want 3000s", args)
				}
				return 0, `{"ok":true,"renewed":true,"lock":{"ttl":3000}}`
			case 2:
				if !containsArgPair(args, "--ttl", "1800") {
					t.Fatalf("release_lock before-publish renew ttl = %v, want caller ttl 1800s", args)
				}
				return 0, `{"ok":true,"renewed":true,"lock":{"ttl":1800}}`
			default:
				t.Fatalf("unexpected release_lock renew #%d: %v", renewCount, args)
				return 127, "unexpected"
			}
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
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
	if result.RemoteBranchPush == nil || result.RemoteBranchPush["lease_expected_sha"] != "base-sha" || result.RemoteBranchPush["release_commit_sha"] != "release-sha" || result.RemoteBranchPush["used_force_lease"] != true {
		t.Fatalf("remote branch push witness = %#v, want base-sha lease for release-sha", result.RemoteBranchPush)
	}
	if result.RemoteBranchFinalCheck == nil || result.RemoteBranchFinalCheck["sha"] != "release-sha" || result.RemoteBranchFinalCheck["expected_sha"] != "release-sha" {
		t.Fatalf("remote branch final check = %#v, want release-sha", result.RemoteBranchFinalCheck)
	}
	if result.Cleanup == nil || result.Cleanup["ok"] != true {
		t.Fatalf("cleanup missing/failed: %#v", result.Cleanup)
	}
	if result.ReleaseLock == nil || result.ReleaseLockRelease == nil {
		t.Fatalf("release lock acquire/release missing: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
	}
	if len(result.ReleaseLockRenewals) != 2 {
		t.Fatalf("release lock renewals = %#v, want before-tag and before-publish", result.ReleaseLockRenewals)
	}
	if result.ReleaseLockRenewals[0]["label"] != "before_tag" || result.ReleaseLockRenewals[0]["requested_ttl_s"] != 3000 {
		t.Fatalf("before-tag renewal = %#v, want 3000s renewal", result.ReleaseLockRenewals[0])
	}
	if result.ReleaseLockRenewals[1]["label"] != "before_publish" || result.ReleaseLockRenewals[1]["requested_ttl_s"] != 1800 {
		t.Fatalf("before-publish renewal = %#v, want caller ttl renewal", result.ReleaseLockRenewals[1])
	}
	sourceCIIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "gh" && sameArgs(cmd.Args, "run", "list", "--workflow", "ci.yml", "--commit", "base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha")
	})
	lockIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 1 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_lock.py")) && cmd.Args[1] == "acquire"
	})
	worktreeIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "git" && len(cmd.Args) >= 3 && sameArgs(cmd.Args[:3], "worktree", "add", "--detach")
	})
	cutIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 0 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_cut.py"))
	})
	if !(sourceCIIdx >= 0 && sourceCIIdx < lockIdx && lockIdx < worktreeIdx && worktreeIdx < cutIdx) {
		t.Fatalf("release lock must follow read-only source CI and precede release work: sourceCI=%d lock=%d worktree=%d cut=%d commands=%#v", sourceCIIdx, lockIdx, worktreeIdx, cutIdx, result.ExecutedCommands)
	}
	pushIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "git" && sameArgs(cmd.Args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main")
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
	renewBeforeTagIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 1 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_lock.py")) && cmd.Args[1] == "renew" && containsArgPair(cmd.Args, "--ttl", "3000")
	})
	renewBeforePublishIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 1 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_lock.py")) && cmd.Args[1] == "renew" && containsArgPair(cmd.Args, "--ttl", "1800")
	})
	publishIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 0 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_publish.py"))
	})
	if !(pushIdx >= 0 && pushIdx < verifyIdx && verifyIdx < ciIdx && ciIdx < renewBeforeTagIdx && renewBeforeTagIdx < tagIdx && tagIdx < renewBeforePublishIdx && renewBeforePublishIdx < publishIdx) {
		t.Fatalf("tag/publish must follow push, remote verify, promoted CI witness, and lock renewals; push=%d verify=%d ci=%d renewTag=%d tag=%d renewPublish=%d publish=%d commands=%#v", pushIdx, verifyIdx, ciIdx, renewBeforeTagIdx, tagIdx, renewBeforePublishIdx, publishIdx, result.ExecutedCommands)
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
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "dev-source-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			for key, want := range map[string]string{
				"FAK_RELEASE_SOURCE_BRANCH": "dev",
				"FAK_RELEASE_SOURCE_SHA":    "dev-source-sha",
				"FAK_RELEASE_TARGET_BRANCH": "main",
				"FAK_RELEASE_TARGET_SHA":    "main-target-sha",
				"FAK_RELEASE_SOURCE_RANGE":  "main-target-sha..dev-source-sha",
				"FAK_RELEASE_SOURCE_CI":     "green",
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
		execute:              false,
		base:                 "origin/dev",
		sourceBranch:         "dev",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            true,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
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
	if result.SourceCI == nil || result.SourceCI["ok"] != true || result.SourceCI["status"] != "green" {
		t.Fatalf("source CI witness missing or not green: %#v", result.SourceCI)
	}
	if result.CommitSHA != "release-sha" || result.Tag != "v0.36.0" {
		t.Fatalf("release outputs missing: %#v", result)
	}
}

func TestReleaseShipExecuteRefusesStaleTrunkLease(t *testing.T) {
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
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "stale info\n"
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("stale trunk lease should refuse before remote verify, CI, tag, or publish; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:              true,
		base:                 "origin/main",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            false,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.RemoteBranchPush == nil || result.RemoteBranchPush["reason"] != "push_trunk_failed" || result.RemoteBranchPush["lease_expected_sha"] != "base-sha" {
		t.Fatalf("remote branch push refusal = %#v, want leased push_trunk_failed", result.RemoteBranchPush)
	}
	if result.RemoteBranch != nil || result.TagResult != nil || result.Publish != nil {
		t.Fatalf("stale trunk lease must refuse before verify/tag/publish: remote=%#v tag=%#v publish=%#v", result.RemoteBranch, result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "push_trunk_failed") {
		t.Fatalf("errors = %#v, want push_trunk_failed", result.Errors)
	}
}

func TestReleaseShipExecuteRefusesTrunkRewriteBeforeTag(t *testing.T) {
	remoteProbeCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			remoteProbeCount++
			if remoteProbeCount == 1 {
				return 0, "release-sha\trefs/heads/main\n"
			}
			return 0, "rewritten-sha\trefs/heads/main\n"
		// The rewritten tip no longer contains the release commit, so the ancestry
		// probe fails and the release refuses before tagging.
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "release-sha", "origin/main"):
			return 1, ""
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("trunk rewrite before tag should refuse before tag or publish; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:              true,
		base:                 "origin/main",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            false,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.RemoteBranch == nil || result.RemoteBranch["sha"] != "release-sha" {
		t.Fatalf("initial remote verify = %#v, want release-sha", result.RemoteBranch)
	}
	if result.RemoteBranchFinalCheck == nil || result.RemoteBranchFinalCheck["reason"] != "remote_branch_changed_before_tag" || result.RemoteBranchFinalCheck["sha"] != "rewritten-sha" {
		t.Fatalf("final remote check = %#v, want remote_branch_changed_before_tag", result.RemoteBranchFinalCheck)
	}
	if result.RemoteBranchFinalCheck["fast_forwarded"] != nil {
		t.Fatalf("rewrite must not be marked fast_forwarded: %#v", result.RemoteBranchFinalCheck)
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("trunk rewrite before tag must refuse before tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "remote_branch_changed_before_tag") {
		t.Fatalf("errors = %#v, want remote_branch_changed_before_tag", result.Errors)
	}
}

// TestReleaseShipExecuteProceedsOnBenignPeerFastForwardBeforeTag covers the hot
// trunk case: a peer fast-forwards main past the just-landed release commit
// while ship waits for CI to appear. The release commit is still on trunk, so
// the ancestry-aware final check treats it as a benign fast-forward and
// proceeds to tag (content-addressed to the release commit) and publish.
func TestReleaseShipExecuteProceedsOnBenignPeerFastForwardBeforeTag(t *testing.T) {
	remoteProbeCount := 0
	renewCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			remoteProbeCount++
			if remoteProbeCount == 1 {
				return 0, "release-sha\trefs/heads/main\n"
			}
			return 0, "peer-ff-sha\trefs/heads/main\n"
		// The advanced tip still contains the release commit (peer fast-forward),
		// so the ancestry probe confirms it is on trunk and the release proceeds.
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "release-sha", "origin/main"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			renewCount++
			if renewCount == 1 && !containsArgPair(args, "--ttl", "3000") {
				t.Fatalf("before-tag renewal = %v, want 3000s", args)
			}
			if renewCount == 2 && !containsArgPair(args, "--ttl", "1800") {
				t.Fatalf("before-publish renewal = %v, want 1800s", args)
			}
			return 0, `{"ok":true,"renewed":true}`
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_tag.py")):
			if !containsArgPair(args, "--ref", "release-sha") {
				t.Fatalf("release_tag must tag the release commit even after a benign peer fast-forward: %v", args)
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
			t.Fatalf("benign fast-forward must proceed to tag/publish; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:              true,
		base:                 "origin/main",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            false,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
	})

	if !result.OK {
		t.Fatalf("benign peer fast-forward should still ship: %#v", result)
	}
	if result.RemoteBranchFinalCheck == nil || result.RemoteBranchFinalCheck["ok"] != true || result.RemoteBranchFinalCheck["fast_forwarded"] != true || result.RemoteBranchFinalCheck["sha"] != "peer-ff-sha" {
		t.Fatalf("final remote check = %#v, want an ok fast-forward past peer-ff-sha", result.RemoteBranchFinalCheck)
	}
	if result.TagResult == nil || result.Publish == nil {
		t.Fatalf("benign fast-forward must still tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if result.CommitSHA != "release-sha" || result.Tag != "v0.35.0" {
		t.Fatalf("release identity = commit %q tag %q, want the release commit tagged", result.CommitSHA, result.Tag)
	}
}

func TestReleaseShipExecuteRefusesLostLockBeforeTag(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "release-sha\trefs/heads/main\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			if !containsArgPair(args, "--ttl", "3000") {
				t.Fatalf("before-tag renewal ttl = %v, want 3000s", args)
			}
			return 3, `{"ok":false,"reason":"held by another","holder":{"owner":"other-session"}}`
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 3, `{"ok":false,"reason":"not owner","holder":{"owner":"other-session"}}`
		default:
			t.Fatalf("lost lock must refuse before tag or publish; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:              true,
		base:                 "origin/main",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            false,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if len(result.ReleaseLockRenewals) != 1 || result.ReleaseLockRenewals[0]["reason"] != "held by another" {
		t.Fatalf("release lock renewal = %#v, want held-by-another refusal", result.ReleaseLockRenewals)
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("lost lock must refuse before tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "release_lock_renew_failed") {
		t.Fatalf("errors = %#v, want release_lock_renew_failed", result.Errors)
	}
	if len(result.Warnings) == 0 || !strings.Contains(result.Warnings[0], "release lock release failed") {
		t.Fatalf("warnings = %#v, want release failure warning after lost lock", result.Warnings)
	}
}

func TestReleaseShipExecuteReusesMergedReleaseCutForTagPublish(t *testing.T) {
	renewCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "merged-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "merged-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 1, `{"ok":false,"aborted":"git commit failed","detail":"On branch main\nnothing to commit, working tree clean"}`
		case name == "git" && sameArgs(args, "show", "HEAD:VERSION"):
			return 0, "0.40.0\n"
		case name == "git" && sameArgs(args, "cat-file", "-e", "HEAD:docs/releases/v0.40.0.md"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-list", "-n1", "v0.40.0"):
			return 128, "unknown revision\n"
		case name == "git" && sameArgs(args, "tag", "--sort=-v:refname"):
			return 0, "v0.39.0\nv0.38.0\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "merged-sha", "merged-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:merged-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "merged-sha\trefs/heads/main\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			renewCount++
			if renewCount == 1 && !containsArgPair(args, "--ttl", "3000") {
				t.Fatalf("before-tag renewal = %v, want 3000s", args)
			}
			if renewCount == 2 && !containsArgPair(args, "--ttl", "1800") {
				t.Fatalf("before-publish renewal = %v, want 1800s", args)
			}
			return 0, `{"ok":true,"renewed":true}`
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_tag.py")):
			if !containsArgPair(args, "--version", "0.40.0") || !containsArgPair(args, "--ref", "merged-sha") {
				t.Fatalf("release_tag args = %v, want existing merged cut", args)
			}
			return 0, `{"ok":true,"tag":"v0.40.0","tag_created":true,"tag_pushed":true}`
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_publish.py")):
			return 0, `{"ok":true,"release_created":true,"github_release":{"status":"present","url":"https://example.test/v0.40.0"}}`
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
		sourceBranch:    "main",
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
	if result.Cut == nil || result.Cut["existing_cut"] != true || result.Cut["reason"] != "version_ahead_of_latest_tag" {
		t.Fatalf("cut = %#v, want existing merged release cut", result.Cut)
	}
	if result.Version != "0.40.0" || result.Tag != "v0.40.0" || result.CommitSHA != "merged-sha" {
		t.Fatalf("release identity = version %q tag %q commit %q, want merged cut", result.Version, result.Tag, result.CommitSHA)
	}
	if result.TagResult == nil || result.Publish == nil {
		t.Fatalf("existing merged cut must still tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
}

func TestReleaseShipExistingCutRecoveryAllowsFirstReleaseTag(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == "git" && sameArgs(args, "show", "HEAD:VERSION"):
			return 0, "0.1.0\n"
		case name == "git" && sameArgs(args, "cat-file", "-e", "HEAD:docs/releases/v0.1.0.md"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-list", "-n1", "v0.1.0"):
			return 128, "unknown revision\n"
		case name == "git" && sameArgs(args, "tag", "--sort=-v:refname"):
			return 0, ""
		default:
			t.Fatalf("unexpected command in %s: %s %v", cwd, name, args)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := releaseShipResult{BaseSHA: "first-release-sha"}
	cut := recoverReleaseShipExistingCut(&result, fakeReleaseWorktree, releaseShipOptions{
		execute:      true,
		sourceBranch: "main",
		trunk:        "main",
	}, map[string]any{
		"ok":      false,
		"aborted": "git commit failed",
		"detail":  "nothing to commit, working tree clean",
	})

	if cut == nil || cut["existing_cut"] != true || cut["version"] != "0.1.0" || cut["tag"] != "v0.1.0" {
		t.Fatalf("cut = %#v, want first-release existing cut", cut)
	}
	if cut["latest_tag"] != "" || cut["commit_sha"] != "first-release-sha" {
		t.Fatalf("cut witness = %#v, want no latest tag and first-release sha", cut)
	}
}

func TestReleaseShipExecuteRefusesMissingSourceCI(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/dev:refs/remotes/origin/dev"):
			return 0, ""
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/dev^{commit}"):
			return 0, "dev-source-sha\n"
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "dev-source-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[]`
		default:
			t.Fatalf("missing source CI should refuse before the release lock and release work; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:              true,
		base:                 "origin/dev",
		sourceBranch:         "dev",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            true,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
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
	if result.ReleaseLock != nil || result.ReleaseLockRelease != nil {
		t.Fatalf("source CI refusal should happen before release lock acquire/release: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
	}
}

func TestReleaseShipExecuteRefusesNonFastForwardTarget(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
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
		default:
			t.Fatalf("non-fast-forward target should refuse before the release lock and release work; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:              true,
		base:                 "origin/dev",
		sourceBranch:         "dev",
		remote:               "origin",
		trunk:                "main",
		workflow:             "ci.yml",
		limitCommits:         50,
		ttl:                  1800,
		fetch:                true,
		requireCI:            false,
		waitCI:               false,
		skipDryRun:           true,
		ciAppearTimeout:      0,
		allowDirectPromotion: true,
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
	if result.ReleaseLock != nil || result.ReleaseLockRelease != nil {
		t.Fatalf("target refusal should happen before release lock acquire/release: acquire=%#v release=%#v", result.ReleaseLock, result.ReleaseLockRelease)
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
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 3, `{"ok":false,"reason":"held","holder":{"owner":"human-release"}}`
		default:
			t.Fatalf("release ship must refuse before worktree/cut work when the release lock is held; got %s %v in %s", name, args, cwd)
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
	lockIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == releaseShipPython() && len(cmd.Args) > 1 && strings.HasSuffix(cmd.Args[0], filepath.Join("tools", "release_lock.py")) && cmd.Args[1] == "acquire"
	})
	worktreeIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "git" && len(cmd.Args) >= 3 && sameArgs(cmd.Args[:3], "worktree", "add", "--detach")
	})
	if lockIdx < 0 || worktreeIdx >= 0 {
		t.Fatalf("lock should be attempted and worktree should not be added: lock=%d worktree=%d commands=%#v", lockIdx, worktreeIdx, result.ExecutedCommands)
	}
}

func TestParseReleaseShipOptionsOpenPRRequiresDistinctBranches(t *testing.T) {
	var stderr strings.Builder
	_, err := parseReleaseShipOptions(&stderr, []string{"--open-pr", "--source-branch", "main", "--trunk", "main"})
	if err == nil || !strings.Contains(err.Error(), "--open-pr") {
		t.Fatalf("err = %v, want an --open-pr distinct-branch validation error", err)
	}
}

func TestParseReleaseShipOptionsExecuteRequiresOpenPRForDistinctBranches(t *testing.T) {
	var stderr strings.Builder
	_, err := parseReleaseShipOptions(&stderr, []string{"--execute", "--source-branch", "dev", "--trunk", "main", "--base", "origin/dev"})
	if err == nil || !strings.Contains(err.Error(), "must use --open-pr") {
		t.Fatalf("err = %v, want direct promotion guard", err)
	}
}

func TestParseReleaseShipOptionsAllowsDistinctBranchDryRun(t *testing.T) {
	var stderr strings.Builder
	opts, err := parseReleaseShipOptions(&stderr, []string{"--source-branch", "dev", "--trunk", "main", "--base", "origin/dev"})
	if err != nil {
		t.Fatalf("parse err = %v", err)
	}
	if opts.execute || opts.openPR || opts.sourceBranch != "dev" || opts.trunk != "main" {
		t.Fatalf("opts = %+v, want dry-run distinct branch plan", opts)
	}
}

func TestParseReleaseShipOptionsAllowsExplicitDirectPromotion(t *testing.T) {
	var stderr strings.Builder
	opts, err := parseReleaseShipOptions(&stderr, []string{"--execute", "--allow-direct-promotion", "--source-branch", "dev", "--trunk", "main", "--base", "origin/dev"})
	if err != nil {
		t.Fatalf("parse err = %v", err)
	}
	if !opts.execute || !opts.allowDirectPromotion || opts.openPR {
		t.Fatalf("opts = %+v, want explicit direct promotion", opts)
	}
}

func TestParseReleaseShipOptionsRejectsDirectPromotionOverrideWithOpenPR(t *testing.T) {
	var stderr strings.Builder
	_, err := parseReleaseShipOptions(&stderr, []string{"--execute", "--open-pr", "--allow-direct-promotion", "--source-branch", "dev", "--trunk", "main", "--base", "origin/dev"})
	if err == nil || !strings.Contains(err.Error(), "incompatible with --open-pr") {
		t.Fatalf("err = %v, want incompatible override validation", err)
	}
}

func TestReleaseShipExecuteRefusesDirectPromotionBeforeCommands(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		t.Fatalf("direct promotion guard should refuse before commands; got %s %v in %s", name, args, cwd)
		return 127, "unexpected"
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:      true,
		base:         "origin/dev",
		sourceBranch: "dev",
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
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "direct_promotion_requires_open_pr") {
		t.Fatalf("errors = %#v, want direct promotion guard", result.Errors)
	}
	if result.ReleaseLock != nil || result.Worktree != "" || len(result.ExecutedCommands) != 0 {
		t.Fatalf("guard must refuse before lock/worktree/commands: %#v", result)
	}
}

func TestParseReleaseShipOptionsOpenPRRejectsLiveSourcePromotionBranch(t *testing.T) {
	var stderr strings.Builder
	_, err := parseReleaseShipOptions(&stderr, []string{"--open-pr", "--source-branch", "dev", "--trunk", "main", "--promotion-branch", "refs/heads/dev"})
	if err == nil || !strings.Contains(err.Error(), "live --source-branch") {
		t.Fatalf("err = %v, want a live-source promotion-branch validation error", err)
	}
}

func TestParseReleaseShipOptionsOpenPRRejectsInvalidPromotionBranch(t *testing.T) {
	for _, branch := range []string{
		"fak/release/v1..2",
		".hidden/release",
		"fak/release/v1.lock",
		"fak/release/bad branch",
		"fak/release/bad@{branch",
		`fak\release\v1`,
		"fak/release/v1:2",
	} {
		t.Run(branch, func(t *testing.T) {
			var stderr strings.Builder
			_, err := parseReleaseShipOptions(&stderr, []string{"--open-pr", "--source-branch", "dev", "--trunk", "main", "--promotion-branch", branch})
			if err == nil || !strings.Contains(err.Error(), "invalid --promotion-branch") {
				t.Fatalf("err = %v, want invalid promotion-branch validation error", err)
			}
		})
	}
}

func TestReleaseShipRefComponentAvoidsInvalidPromotionBranchPieces(t *testing.T) {
	for _, input := range []string{
		"refs/tags/v1..2.lock",
		"...",
		"release candidate/@{bad}:*.lock",
	} {
		t.Run(input, func(t *testing.T) {
			component := releaseShipRefComponent(input)
			branch := "fak/release/" + component
			if reason := releaseShipInvalidPromotionBranchReason(branch); reason != "" {
				t.Fatalf("component %q made invalid branch %q: %s", component, branch, reason)
			}
		})
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
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "main-target-sha\trefs/heads/main\n"
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/"+openPRPromotionBranch):
			promotionProbeCount++
			if promotionProbeCount == 1 {
				return 0, ""
			}
			return 0, "cut-sha\trefs/heads/" + openPRPromotionBranch + "\n"
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/"+openPRPromotionBranch+":", "HEAD:refs/heads/"+openPRPromotionBranch):
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
	if result.TargetBranchCheck == nil || result.TargetBranchCheck["sha"] != "main-target-sha" || result.TargetBranchCheck["expected_sha"] != "main-target-sha" {
		t.Fatalf("target branch check = %#v, want current main-target-sha", result.TargetBranchCheck)
	}
	if result.TargetBranchFinalCheck == nil || result.TargetBranchFinalCheck["sha"] != "main-target-sha" || result.TargetBranchFinalCheck["expected_sha"] != "main-target-sha" {
		t.Fatalf("final target branch check = %#v, want current main-target-sha", result.TargetBranchFinalCheck)
	}
	targetCheck, _ := result.PullRequest["target_check"].(map[string]any)
	if targetCheck["sha"] != "main-target-sha" || targetCheck["expected_sha"] != "main-target-sha" {
		t.Fatalf("pull request target check = %#v, want current main-target-sha", targetCheck)
	}
	if result.PromotionBranchPush == nil || result.PromotionBranchPush["branch"] != openPRPromotionBranch || result.PromotionBranchPush["sha"] != "cut-sha" {
		t.Fatalf("promotion branch push witness = %#v, want branch %q at cut-sha", result.PromotionBranchPush, openPRPromotionBranch)
	}
	if result.PromotionBranchPush["used_force_lease"] != true || result.PromotionBranchPush["lease_expected_absent"] != true {
		t.Fatalf("new promotion branch must report an absent-ref force-with-lease: %#v", result.PromotionBranchPush)
	}
	push, _ := result.PullRequest["promotion_push"].(map[string]any)
	if push["branch"] != openPRPromotionBranch || push["sha"] != "cut-sha" {
		t.Fatalf("pull request promotion push witness = %#v, want branch %q at cut-sha", push, openPRPromotionBranch)
	}
	if result.PullRequest["created"] != true {
		t.Fatalf("pull request created flag = %#v, want true", result.PullRequest["created"])
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("--open-pr must not tag/publish; got tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
}

func TestReleaseShipOpenPRExecuteRefusesStaleTargetBranch(t *testing.T) {
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
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "new-main-target-sha\trefs/heads/main\n"
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("stale target must refuse before promotion push or gh PR work; got %s %v in %s", name, args, cwd)
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

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.TargetBranchCheck == nil || result.TargetBranchCheck["reason"] != "target_branch_changed_after_preview" {
		t.Fatalf("target branch check = %#v, want target_branch_changed_after_preview", result.TargetBranchCheck)
	}
	if result.PromotionBranchPush != nil {
		t.Fatalf("stale target must refuse before promotion branch push: %#v", result.PromotionBranchPush)
	}
	if result.PullRequest == nil || result.PullRequest["reason"] != "target_branch_changed_after_preview" {
		t.Fatalf("pull request refusal = %#v, want stale target reason", result.PullRequest)
	}
}

func TestReleaseShipOpenPRExecuteRefusesTargetMoveAfterPromotionPush(t *testing.T) {
	targetProbeCount := 0
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
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			targetProbeCount++
			if targetProbeCount == 1 {
				return 0, "main-target-sha\trefs/heads/main\n"
			}
			return 0, "new-main-target-sha\trefs/heads/main\n"
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/"+openPRPromotionBranch):
			promotionProbeCount++
			if promotionProbeCount == 1 {
				return 0, ""
			}
			return 0, "cut-sha\trefs/heads/" + openPRPromotionBranch + "\n"
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/"+openPRPromotionBranch+":", "HEAD:refs/heads/"+openPRPromotionBranch):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("post-push target move must refuse before gh PR work; got %s %v in %s", name, args, cwd)
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

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if result.TargetBranchCheck == nil || result.TargetBranchCheck["sha"] != "main-target-sha" {
		t.Fatalf("first target branch check = %#v, want main-target-sha", result.TargetBranchCheck)
	}
	if result.TargetBranchFinalCheck == nil || result.TargetBranchFinalCheck["reason"] != "target_branch_changed_after_promotion_push" || result.TargetBranchFinalCheck["base_reason"] != "target_branch_changed_after_preview" {
		t.Fatalf("final target branch check = %#v, want post-push target move", result.TargetBranchFinalCheck)
	}
	if result.PromotionBranchPush == nil || result.PromotionBranchPush["sha"] != "cut-sha" {
		t.Fatalf("promotion branch push witness = %#v, want pushed cut-sha before refusal", result.PromotionBranchPush)
	}
	if result.PullRequest == nil || result.PullRequest["reason"] != "target_branch_changed_after_promotion_push" {
		t.Fatalf("pull request refusal = %#v, want post-push target move reason", result.PullRequest)
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
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "main-target-sha\trefs/heads/main\n"
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
	if result.PromotionBranchPush == nil || result.PromotionBranchPush["previous_sha"] != "old-cut-sha" || result.PromotionBranchPush["used_force_lease"] != true {
		t.Fatalf("existing promotion branch update must expose force-with-lease witness: %#v", result.PromotionBranchPush)
	}
	if result.PromotionBranchPush["lease_expected_absent"] != false || result.PromotionBranchPush["lease_expected_sha"] != "old-cut-sha" {
		t.Fatalf("existing promotion branch update must expose the exact lease expectation: %#v", result.PromotionBranchPush)
	}
	createIdx := releaseShipCommandIndex(result.ExecutedCommands, func(cmd releaseShipCommand) bool {
		return cmd.Name == "gh" && len(cmd.Args) > 1 && cmd.Args[0] == "pr" && cmd.Args[1] == "create"
	})
	if createIdx >= 0 {
		t.Fatalf("existing PR rerun must not create a duplicate: %#v", result.ExecutedCommands)
	}
}

func TestRenderReleaseShipIncludesPromotionBranchPush(t *testing.T) {
	var stdout, stderr strings.Builder
	renderReleaseShip(&stdout, &stderr, releaseShipResult{
		OK: true,
		PromotionBranchPush: map[string]any{
			"branch":             openPRPromotionBranch,
			"sha":                "cut-sha",
			"used_force_lease":   true,
			"lease_expected_sha": "old-cut-sha",
		},
	})
	if got := stdout.String(); !strings.Contains(got, "promotion branch: "+openPRPromotionBranch+" cut-sha (force-with-lease: old-cut-sha)") {
		t.Fatalf("stdout = %q, want promotion branch push witness", got)
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRenderReleaseShipIncludesRemoteBranchLease(t *testing.T) {
	var stdout, stderr strings.Builder
	renderReleaseShip(&stdout, &stderr, releaseShipResult{
		OK: true,
		RemoteBranch: map[string]string{
			"remote": "origin",
			"trunk":  "main",
			"sha":    "release-sha",
		},
		RemoteBranchPush: map[string]any{
			"lease_expected_sha": "base-sha",
		},
	})
	if got := stdout.String(); !strings.Contains(got, "pushed: origin/main release-sha (force-with-lease: base-sha)") {
		t.Fatalf("stdout = %q, want remote branch lease witness", got)
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestReleaseShipRetriesTrunkPushAfterPeerAdvance(t *testing.T) {
	revParseCount := 0
	cutCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			revParseCount++
			if revParseCount == 1 {
				return 0, "base-sha\n"
			}
			return 0, "new-base-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			cutCount++
			if cutCount == 1 {
				return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
			}
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha-2"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 0, `{"ok":true,"renewed":true}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "reset", "--hard", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "new-base-sha", "release-sha-2"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:new-base-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "release-sha-2\trefs/heads/main\n"
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_tag.py")):
			if !containsArgPair(args, "--ref", "release-sha-2") || !containsArgPair(args, "--version", "0.35.0") {
				t.Fatalf("release_tag must tag the re-cut commit: %v", args)
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
		sourceBranch:    "main",
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
		pushRetries:     2,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if len(result.PushRetries) != 1 {
		t.Fatalf("push retries = %#v, want exactly one rebase attempt", result.PushRetries)
	}
	retry := result.PushRetries[0]
	if retry["ok"] != true || retry["old_base"] != "base-sha" || retry["new_base"] != "new-base-sha" || retry["commit_sha"] != "release-sha-2" {
		t.Fatalf("retry witness = %#v, want base-sha->new-base-sha re-cut to release-sha-2", retry)
	}
	if result.CommitSHA != "release-sha-2" || result.BaseSHA != "new-base-sha" {
		t.Fatalf("release identity after retry = commit %q base %q, want re-cut on new tip", result.CommitSHA, result.BaseSHA)
	}
	if result.RemoteBranchPush == nil || result.RemoteBranchPush["lease_expected_sha"] != "new-base-sha" {
		t.Fatalf("final push must lease against the advanced tip: %#v", result.RemoteBranchPush)
	}
	if result.RemoteBranch == nil || result.RemoteBranch["sha"] != "release-sha-2" {
		t.Fatalf("verified remote = %#v, want release-sha-2", result.RemoteBranch)
	}
	if result.TagResult == nil || result.Publish == nil {
		t.Fatalf("retry that lands must still tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
}

func TestReleaseShipRetryRefusesDivergedTrunk(t *testing.T) {
	revParseCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			revParseCount++
			if revParseCount == 1 {
				return 0, "base-sha\n"
			}
			return 0, "rewritten-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 0, `{"ok":true,"renewed":true}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "rewritten-sha"):
			return 1, ""
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("diverged trunk must refuse before reset/re-cut/tag; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:         true,
		base:            "origin/main",
		sourceBranch:    "main",
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
		pushRetries:     2,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if len(result.PushRetries) != 1 || result.PushRetries[0]["reason"] != "retry_trunk_diverged" {
		t.Fatalf("push retries = %#v, want a single retry_trunk_diverged refusal", result.PushRetries)
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("diverged trunk must refuse before tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "push_trunk_retry_failed") {
		t.Fatalf("errors = %#v, want push_trunk_retry_failed", result.Errors)
	}
}

func TestReleaseShipRetryBoundedByPushRetries(t *testing.T) {
	revParseCount := 0
	cutCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			revParseCount++
			if revParseCount == 1 {
				return 0, "base-sha\n"
			}
			return 0, "new-base-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			cutCount++
			if cutCount == 1 {
				return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
			}
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha-2"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 0, `{"ok":true,"renewed":true}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "reset", "--hard", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "new-base-sha", "release-sha-2"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:new-base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("bounded retry must stop after one rebase; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:         true,
		base:            "origin/main",
		sourceBranch:    "main",
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
		pushRetries:     1,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if len(result.PushRetries) != 1 {
		t.Fatalf("push retries = %#v, want exactly one rebase before the bound is hit", result.PushRetries)
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("exhausted retries must refuse before tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "push_trunk_failed") {
		t.Fatalf("errors = %#v, want push_trunk_failed after the bound", result.Errors)
	}
}

func TestReleaseShipRetryReWitnessesSourceCIOnRebasedTip(t *testing.T) {
	revParseCount := 0
	cutCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			revParseCount++
			if revParseCount == 1 {
				return 0, "base-sha\n"
			}
			return 0, "new-base-sha\n"
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[{"databaseId":40,"status":"completed","conclusion":"success","headSha":"base-sha","url":"https://example.test/base-ci"}]`
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "new-base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[{"databaseId":41,"status":"completed","conclusion":"success","headSha":"new-base-sha","url":"https://example.test/new-base-ci"}]`
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			cutCount++
			if cutCount == 1 {
				return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
			}
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha-2"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 0, `{"ok":true,"renewed":true}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "reset", "--hard", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "new-base-sha", "release-sha-2"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:new-base-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "release-sha-2\trefs/heads/main\n"
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_tag.py")):
			if !containsArgPair(args, "--ref", "release-sha-2") {
				t.Fatalf("release_tag must tag the re-cut commit: %v", args)
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
		sourceBranch:    "main",
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
		pushRetries:     2,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if len(result.PushRetries) != 1 {
		t.Fatalf("push retries = %#v, want one rebase attempt", result.PushRetries)
	}
	if result.SourceCI == nil || result.SourceCI["source_sha"] != "new-base-sha" || result.SourceCI["ok"] != true {
		t.Fatalf("source CI witness = %#v, want re-witnessed green on new-base-sha", result.SourceCI)
	}
	if result.CommitSHA != "release-sha-2" {
		t.Fatalf("shipped commit = %q, want re-cut release-sha-2", result.CommitSHA)
	}
	if result.TagResult == nil || result.Publish == nil {
		t.Fatalf("re-witnessed retry must still tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
}

func TestReleaseShipRetryRefusesUnconfirmedSourceCIOnRebasedTip(t *testing.T) {
	revParseCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			revParseCount++
			if revParseCount == 1 {
				return 0, "base-sha\n"
			}
			return 0, "new-base-sha\n"
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[{"databaseId":40,"status":"completed","conclusion":"success","headSha":"base-sha","url":"https://example.test/base-ci"}]`
		case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "new-base-sha", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"):
			return 0, `[]`
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 0, `{"ok":true,"renewed":true}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "reset", "--hard", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 0, `{"ok":true,"released":true}`
		default:
			t.Fatalf("unconfirmed re-cut CI must refuse before a second cut/push/tag; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:         true,
		base:            "origin/main",
		sourceBranch:    "main",
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
		pushRetries:     2,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if len(result.PushRetries) != 1 || result.PushRetries[0]["reason"] != "retry_source_ci_unconfirmed" || result.PushRetries[0]["new_base"] != "new-base-sha" {
		t.Fatalf("push retries = %#v, want a single retry_source_ci_unconfirmed refusal on new-base-sha", result.PushRetries)
	}
	if result.SourceCI == nil || result.SourceCI["source_sha"] != "new-base-sha" {
		t.Fatalf("source CI witness = %#v, want refreshed to the unconfirmed new-base-sha", result.SourceCI)
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("unconfirmed re-cut CI must refuse before tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "push_trunk_retry_failed") {
		t.Fatalf("errors = %#v, want push_trunk_retry_failed", result.Errors)
	}
}

func TestReleaseShipRetriesTrunkPushTwiceThenSucceeds(t *testing.T) {
	revParseCount := 0
	cutCount := 0
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			revParseCount++
			switch revParseCount {
			case 1:
				return 0, "base-sha\n"
			case 2:
				return 0, "new-base-sha\n"
			default:
				return 0, "third-base-sha\n"
			}
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			cutCount++
			switch cutCount {
			case 1:
				return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
			case 2:
				return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha-2"}`
			default:
				return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha-3"}`
			}
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 0, `{"ok":true,"renewed":true}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "reset", "--hard", "new-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "new-base-sha", "release-sha-2"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:new-base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "new-base-sha", "third-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "reset", "--hard", "third-base-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "third-base-sha", "release-sha-3"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:third-base-sha", "HEAD:refs/heads/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "ls-remote", "origin", "refs/heads/main"):
			return 0, "release-sha-3\trefs/heads/main\n"
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_tag.py")):
			if !containsArgPair(args, "--ref", "release-sha-3") {
				t.Fatalf("release_tag must tag the twice-recut commit: %v", args)
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
		sourceBranch:    "main",
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
		pushRetries:     2,
	})

	if !result.OK {
		t.Fatalf("result not ok: %#v", result)
	}
	if len(result.PushRetries) != 2 {
		t.Fatalf("push retries = %#v, want two rebase attempts", result.PushRetries)
	}
	if result.PushRetries[0]["new_base"] != "new-base-sha" || result.PushRetries[1]["old_base"] != "new-base-sha" || result.PushRetries[1]["new_base"] != "third-base-sha" {
		t.Fatalf("retry chain = %#v, want base->new-base->new-base-2", result.PushRetries)
	}
	if result.CommitSHA != "release-sha-3" || result.BaseSHA != "third-base-sha" {
		t.Fatalf("release identity = commit %q base %q, want twice-recut on third-base-sha", result.CommitSHA, result.BaseSHA)
	}
	if result.RemoteBranchPush == nil || result.RemoteBranchPush["lease_expected_sha"] != "third-base-sha" {
		t.Fatalf("final push must lease against the twice-advanced tip: %#v", result.RemoteBranchPush)
	}
	if result.RemoteBranch == nil || result.RemoteBranch["sha"] != "release-sha-3" {
		t.Fatalf("verified remote = %#v, want release-sha-3", result.RemoteBranch)
	}
}

func TestReleaseShipRetryAbortsWhenLockLostBeforeRebase(t *testing.T) {
	restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
		switch {
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "acquire":
			return 0, `{"ok":true,"lock":{"owner":"ship-owner"}}`
		case name == "git" && sameArgs(args, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main"):
			return 0, ""
		case name == "git" && sameArgs(args, "rev-parse", "--verify", "origin/main^{commit}"):
			return 0, "base-sha\n"
		case name == "git" && sameArgs(args, "worktree", "add", "--detach", fakeReleaseWorktree, "base-sha"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 0 && strings.HasSuffix(args[0], filepath.Join("tools", "release_cut.py")):
			return 0, `{"ok":true,"version":"0.35.0","tag":"v0.35.0","commit_sha":"release-sha"}`
		case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "base-sha", "release-sha"):
			return 0, ""
		case name == "git" && sameArgs(args, "push", "origin", "--force-with-lease=refs/heads/main:base-sha", "HEAD:refs/heads/main"):
			return 1, "! [rejected] stale info\n"
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "renew":
			return 3, `{"ok":false,"reason":"held by another","holder":{"owner":"other-session"}}`
		case name == "git" && sameArgs(args, "worktree", "remove", "--force", fakeReleaseWorktree):
			return 0, ""
		case name == "git" && sameArgs(args, "worktree", "prune"):
			return 0, ""
		case name == releaseShipPython() && len(args) > 1 && strings.HasSuffix(args[0], filepath.Join("tools", "release_lock.py")) && args[1] == "release":
			return 3, `{"ok":false,"reason":"not owner","holder":{"owner":"other-session"}}`
		default:
			t.Fatalf("lost lock before rebase must refuse before re-cut/tag; got %s %v in %s", name, args, cwd)
			return 127, "unexpected"
		}
	})
	defer restore()

	result := executeReleaseShip(releaseShipOptions{
		execute:         true,
		base:            "origin/main",
		sourceBranch:    "main",
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
		pushRetries:     2,
	})

	if result.OK {
		t.Fatalf("result unexpectedly ok: %#v", result)
	}
	if len(result.PushRetries) != 0 {
		t.Fatalf("push retries = %#v, want none — the rebase must not run after a lost lock", result.PushRetries)
	}
	if len(result.ReleaseLockRenewals) != 1 || result.ReleaseLockRenewals[0]["reason"] != "held by another" {
		t.Fatalf("release lock renewals = %#v, want a single held-by-another refusal", result.ReleaseLockRenewals)
	}
	if result.TagResult != nil || result.Publish != nil {
		t.Fatalf("lost lock must refuse before tag/publish: tag=%#v publish=%#v", result.TagResult, result.Publish)
	}
	if len(result.Errors) == 0 || !strings.Contains(result.Errors[0], "release_lock_renew_failed") {
		t.Fatalf("errors = %#v, want release_lock_renew_failed", result.Errors)
	}
}

func TestReleaseShipPushRetryableGate(t *testing.T) {
	staleLease := map[string]any{"reason": "push_trunk_failed"}
	sameTrunk := releaseShipOptions{pushRetries: 2, sourceBranch: "main", trunk: "main"}
	cases := []struct {
		name string
		opts releaseShipOptions
		push map[string]any
		want bool
	}{
		{"same-trunk stale lease retries", sameTrunk, staleLease, true},
		{"disabled by zero budget", releaseShipOptions{pushRetries: 0, sourceBranch: "main", trunk: "main"}, staleLease, false},
		{"open-pr never rebases", releaseShipOptions{pushRetries: 2, openPR: true, sourceBranch: "main", trunk: "main"}, staleLease, false},
		{"distinct branch never rebases", releaseShipOptions{pushRetries: 2, sourceBranch: "dev", trunk: "main"}, staleLease, false},
		{"empty source never rebases", releaseShipOptions{pushRetries: 2, sourceBranch: "", trunk: "main"}, staleLease, false},
		{"non-lease refusal not retried", sameTrunk, map[string]any{"reason": "release_commit_not_descendant_of_trunk"}, false},
		{"absent reason not retried", sameTrunk, map[string]any{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := releaseShipPushRetryable(tc.opts, tc.push); got != tc.want {
				t.Fatalf("releaseShipPushRetryable = %v, want %v", got, tc.want)
			}
		})
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

func TestReleaseShipExecuteRequiresExplicitReadinessBeforeCommands(t *testing.T) {
	oldRun := releaseShipRunCommand
	t.Cleanup(func() { releaseShipRunCommand = oldRun })
	called := false
	releaseShipRunCommand = func(string, string, []string, []string, time.Duration) (int, string) { called = true; return 0, "" }
	var out, errb bytes.Buffer
	code := runReleaseShip(&out, &errb, []string{"--execute"})
	if code != 2 || called || !strings.Contains(errb.String(), "--readiness") {
		t.Fatalf("code=%d called=%v stderr=%q", code, called, errb.String())
	}
}

func TestReleaseShipExecuteRefusesReadyStateWithoutWitnessedReceipt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "readiness.json")
	data := `{"unit":{"schema":"fak.work-delivery/v1","id":"release-unit","axes":{"authoring":"recorded","compile_admission":"admitted","verification":"passed","integration":"integrated","release":"ready"}},"receipt":{"schema":"fak.work-delivery/v1","unit_id":"release-unit","transition":{"axis":"release","from":"not_ready","to":"ready"},"gate":"release-admission","evidence":[{"kind":"release-readiness","reference":"shipgate://run/9","witnessed":false}],"observed_at":"2026-08-17T18:00:00Z"}}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	oldRun := releaseShipRunCommand
	t.Cleanup(func() { releaseShipRunCommand = oldRun })
	called := false
	releaseShipRunCommand = func(string, string, []string, []string, time.Duration) (int, string) { called = true; return 0, "" }
	var out, errb bytes.Buffer
	code := runReleaseShip(&out, &errb, []string{"--execute", "--readiness", path})
	if code != 1 || called || !strings.Contains(errb.String(), "RELEASE_READINESS_REQUIRED") {
		t.Fatalf("code=%d called=%v stderr=%q", code, called, errb.String())
	}
}

func TestReleaseReadinessConsumesInstalledLaunchQualification(t *testing.T) {
	unit := workdelivery.WorkUnit{Schema: workdelivery.Schema, ID: "release-unit", Axes: workdelivery.Axes{Authoring: workdelivery.AuthoringRecorded, Admission: workdelivery.AdmissionAdmitted, Verification: workdelivery.VerificationPassed, Integration: workdelivery.IntegrationIntegrated, Release: workdelivery.ReleaseReady}}
	receipt := workdelivery.Receipt{Schema: workdelivery.Schema, UnitID: "release-unit", Transition: workdelivery.Transition{Axis: "release", From: "not_ready", To: "ready"}, Gate: "release-admission"}
	write := func(t *testing.T, qualification installedLaunchQualificationReceipt) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), "readiness.json")
		data, err := json.Marshal(map[string]any{"unit": unit, "receipt": receipt, "installed_launch_qualification": qualification})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	ready := installedLaunchQualificationReceipt{Schema: installedLaunchQualificationSchema, Qualified: true}
	if _, _, err := readReleaseReadiness(write(t, ready)); err != nil {
		t.Fatalf("ready qualification rejected: %v", err)
	}
	failed := installedLaunchQualificationReceipt{Schema: installedLaunchQualificationSchema, Qualified: false, Providers: []installedLaunchQualificationProvider{{Provider: "codex", Failure: "UNDERLYING_MISSING"}}}
	if _, _, err := readReleaseReadiness(write(t, failed)); err == nil || !strings.Contains(err.Error(), "not ready") {
		t.Fatalf("failed qualification error = %v, want typed not-ready refusal", err)
	}
}
