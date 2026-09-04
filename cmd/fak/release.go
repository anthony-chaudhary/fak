package main

// fak release is the binary front door for the existing release substrate. The
// release helpers remain the source of truth; this verb makes them discoverable
// from the one command agents already know how to invoke.

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type releaseScriptRunner func(root, script string, args []string, stdout, stderr io.Writer) int

var releaseRunScript releaseScriptRunner = runReleaseScript
var releaseRunShip = runReleaseShip
var releaseRunStatus = runReleaseStatus
var releaseRunReadiness = runReleaseReadiness
var releaseRunNext = runReleaseNext
var releaseLookPath = exec.LookPath

var releaseScripts = map[string]string{
	"plan":           "release_status.py",
	"decide":         "release_decide.py",
	"cut":            "release_cut.py",
	"tag":            "release_tag.py",
	"publish":        "release_publish.py",
	"lock":           "release_lock.py",
	"dry-run":        "release_dry_run.py",
	"dryrun":         "release_dry_run.py",
	"manifest":       "release_manifest.py",
	"stable":         "stable_release_promote.py",
	"stable-context": "stable_release_context.py",
	"next":           "release_next.py",
}

func cmdRelease(argv []string) { os.Exit(runRelease(os.Stdout, os.Stderr, argv)) }

func runRelease(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 {
		switch argv[0] {
		case "-h", "--help", "help":
			releaseUsage(stderr)
			return 0
		}
	}

	mode := "status"
	rest := argv
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		key := strings.ToLower(strings.TrimSpace(argv[0]))
		if key == "staleness" || key == "release-staleness" {
			return runReleaseStaleness(stdout, stderr, argv[1:])
		}
		if key == "status" {
			return releaseRunStatus(stdout, stderr, argv[1:])
		}
		if key == "ship" || key == "auto" {
			return releaseRunShip(stdout, stderr, argv[1:])
		}
		if key == "dispatch" || key == "ondemand" || key == "on-demand" {
			return runReleaseDispatch(stdout, stderr, argv[1:])
		}
		if key == "prplan" {
			return runReleasePRPlan(stdout, stderr, argv[1:])
		}
		if key == "readiness" || key == "release-readiness" || key == "scorecard" || key == "release-scorecard" {
			return releaseRunReadiness(stdout, stderr, argv[1:])
		}
		if key == "next" {
			return releaseRunNext(stdout, stderr, argv[1:])
		}
		if _, ok := releaseScripts[key]; !ok {
			fmt.Fprintf(stderr, "fak release: unknown subcommand %q\n", argv[0])
			releaseUsage(stderr)
			return 2
		}
		mode = key
		rest = argv[1:]
	}
	if mode == "status" {
		return releaseRunStatus(stdout, stderr, rest)
	}

	// Pass the operator's args straight through. The release-substrate dry-run
	// witness (release_dry_run.py, embedded in cut/tag) reads the LATEST EXISTING
	// tag, not the just-bumped VERSION, so it passes on a real version bump; the
	// front door no longer force-injects --skip-dry-run. An operator who wants to
	// skip it for an emergency cut still passes --skip-dry-run explicitly.
	return releaseRunScript(repoRoot(), releaseScripts[mode], rest, stdout, stderr)
}

func runReleaseScript(root, script string, args []string, stdout, stderr io.Writer) int {
	python := releasePython()
	scriptPath := filepath.Join(root, "tools", script)
	if _, err := os.Stat(scriptPath); err != nil {
		fmt.Fprintf(stderr, "fak release: %s is unavailable: %v\n", scriptPath, err)
		return 1
	}
	cmd := exec.Command(python, append([]string{scriptPath}, args...)...)
	cmd.Dir = root
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Env = os.Environ()
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak release: run %s: %v\n", script, err)
		return 127
	}
	return 0
}

func releasePython() string {
	if python := strings.TrimSpace(os.Getenv("FAK_PYTHON")); python != "" {
		return python
	}
	for _, name := range []string{"python", "python3"} {
		if _, err := releaseLookPath(name); err == nil {
			return name
		}
	}
	return "python"
}

func releaseUsage(w io.Writer) {
	fmt.Fprint(w, `fak release - front door for the release helpers

usage:
  fak release [status flags...]
  fak release next [--sync] [--check] [--json]
  fak release ship [--execute] [--json] [ship flags...]
  fak release dispatch [--execute] [--wait] [--timeout 30m] [--json] [--ref main] [--plan-only]
  fak release prplan [--json] [--base <ref>] [--head <ref>] [--check]
  fak release status|staleness|release-staleness|plan|decide|cut|tag|publish|lock|dry-run|manifest|readiness|release-readiness|next [helper flags...]
  fak release stable|stable-context [helper flags...]

examples:
  fak release --json
  fak release next
  fak release next --sync
  fak release next --check
  fak release ship --execute --json
  fak release dispatch --execute --wait --json
  fak release ship --execute --json --open-pr
  fak release prplan --base origin/main --head main
  fak release prplan --check
  fak release staleness --json
  fak release readiness --json
  fak release decide --json --require-ci-green
  fak release cut --json
  fak release cut --execute --json --require-ci-green
  fak release tag --version X.Y.Z --ref HEAD --execute --push --json
  fak release publish --version X.Y.Z --execute --json
  fak release stable-context --codename 2026-06-bedrock --json
  fak release stable --codename 2026-06-bedrock

Canonical local order:
  fak release ship --execute

Canonical on-demand CI order:
  fak release dispatch --execute

Helper order underneath:
  detached worktree at origin/main -> release_decide -> release_lock -> release_cut
  -> push main -> release_tag -> release_publish -> release-artifacts verification

The status, staleness, prplan, and readiness subcommands are native Go folds; legacy helpers are
run in the shipped binary without a Python bootstrap. The deeper release
helpers live in tools/release_*.py / tools/stable_release_*.py and remain the
release contract while their implementation is migrated.
prplan folds the promotion range (release branch .. release source) into PR
units grouped by the (fak <leaf>) ship-stamp — the "PRs managed in advance"
artifact a dev->main promotion opens as its human-legible PR body(ies);
--check gates on unstamped commits in the range.
ship --open-pr reuses that same fold as the body of a real GitHub PR: it pushes
the exact cut commit to --promotion-branch (default fak/release/<tag>) and
opens or refreshes that promotion-branch -> --trunk PR instead of pushing
straight to trunk. The live --source-branch can keep accepting work while a
human operator reviews and merges through the native PR UI; tag/publish then
run in a later ship once the merge lands. Requires distinct --source-branch/
--trunk — a no-op today under the main-only branch-role regime; see
docs/branch-regime-shadow-cutover.md.
In execute mode, distinct --source-branch/--trunk pairs must use --open-pr by
default; --allow-direct-promotion is an explicit operator override.
After an --open-pr promotion is merged, tag/publish the merged release commit
with --source-branch <trunk> --trunk <trunk> --base origin/<trunk>; ship reuses
an existing VERSION/release-note cut instead of creating a no-op release commit.
The ship subcommand is the default hot-tree path: it leaves this checkout's
unrelated modified/untracked files alone by cutting in a transient detached
worktree, while sharing the same single-writer release lock.
ship auto-sizes that lock to the bounded pre-tag work and renews it before
tag/publish; if renewal proves the lock was lost, it refuses before creating
the tag or GitHub release page.
On the same-trunk hot path, if a peer fast-forwards trunk during the cut and the
force-with-lease push refuses, ship re-cuts the VERSION+note commit onto the new
tip and retries the leased push up to --push-retries times (default 2); it never
force-pushes and never tags a lost race — a rewritten/diverged trunk still refuses.
Under --require-ci it re-witnesses source CI on the peer's new tip before the
re-cut, so a retry never ships a commit whose CI was not confirmed green.
Cut/tag executed through this front door run the release-substrate dry-run witness
by default (it reads the latest existing tag, so it passes on a real version bump);
pass --skip-dry-run yourself only for an emergency cut.
`)
}
