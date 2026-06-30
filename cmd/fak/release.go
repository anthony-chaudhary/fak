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
	"readiness":      "release_readiness_scorecard.py",
	"scorecard":      "release_readiness_scorecard.py",
	"stable":         "stable_release_promote.py",
	"stable-context": "stable_release_context.py",
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
		if key == "staleness" {
			return runReleaseStaleness(stdout, stderr, argv[1:])
		}
		if key == "status" {
			return releaseRunStatus(stdout, stderr, argv[1:])
		}
		if key == "ship" || key == "auto" {
			return releaseRunShip(stdout, stderr, argv[1:])
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

	return releaseRunScript(repoRoot(), releaseScripts[mode], releaseArgs(mode, rest), stdout, stderr)
}

func releaseArgs(mode string, args []string) []string {
	if mode != "cut" && mode != "tag" {
		return args
	}
	if !hasReleaseArg(args, "--execute") || hasReleaseArg(args, "--skip-dry-run") {
		return args
	}
	out := make([]string, 0, len(args)+1)
	out = append(out, args...)
	out = append(out, "--skip-dry-run")
	return out
}

func hasReleaseArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}

func runReleaseScript(root, script string, args []string, stdout, stderr io.Writer) int {
	python := strings.TrimSpace(os.Getenv("FAK_PYTHON"))
	if python == "" {
		python = "python"
	}
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

func releaseUsage(w io.Writer) {
	fmt.Fprint(w, `fak release - front door for the release helpers

usage:
  fak release [status flags...]
  fak release ship [--execute] [--json] [ship flags...]
  fak release status|staleness|plan|decide|cut|tag|publish|lock|dry-run|manifest|readiness [helper flags...]
  fak release stable|stable-context [helper flags...]

examples:
  fak release --json
  fak release ship --execute --json
  fak release staleness --json
  fak release decide --json --require-ci-green
  fak release cut --json
  fak release cut --execute --json --require-ci-green
  fak release tag --version X.Y.Z --ref HEAD --execute --push --json
  fak release publish --version X.Y.Z --execute --json
  fak release stable-context --codename 2026-06-bedrock --json
  fak release stable --codename 2026-06-bedrock

Canonical order:
  fak release ship --execute

Helper order underneath:
  detached worktree at origin/main -> release_decide -> release_lock -> release_cut
  -> push main -> release_tag -> release_publish -> release-artifacts verification

The status and staleness subcommands are native Go folds. The deeper release
helpers live in tools/release_*.py / tools/stable_release_*.py and remain the
release contract while their implementation is migrated.
The ship subcommand is the default hot-tree path: it leaves this checkout's
unrelated modified/untracked files alone by cutting in a transient detached
worktree, while sharing the same single-writer release lock.
When cut/tag are executed through this front door, --skip-dry-run is added unless
you supplied it already; the real witness is the green trunk plus the post-tag
release-substrate suite.
`)
}
