package main

import (
	"context"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// commitPreflightFn is the seam the shim calls; it defaults to the real classifier and is
// overridden in tests so runCommitPreflight is exercised with canned evidence — no git, no repo.
var commitPreflightFn = safecommit.PreflightPaths

// runCommitPreflight is the `fak commit preflight` shim (issue #3228). Workers follow the
// shared-trunk discipline of committing by explicit path (`git commit -- <path>`) but
// repeatedly pass a path git cannot match, getting a raw
//
//	error: pathspec '<path>' did not match any file(s) known to git
//
// that gives no structured next step. This preflight classifies each requested pathspec
// against the index + worktree BEFORE the commit and refuses with a NAMED reason plus the fix:
// PATH_UNTRACKED (a present-but-unstaged file → `git add`) or PATH_UNMATCHED (a typo / renamed
// path / stale plan entry). Exit codes mirror `fak commit`: 0 clean, 2 usage, 3 a PRE-commit
// refusal (fix and retry), 1 an infrastructure error.
func runCommitPreflight(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit preflight", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "commit")
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative pathspec to classify (repeatable); paths may also be given after --")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the preflight report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	paths = append(paths, fs.Args()...)
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak commit preflight: at least one --path (or a path after --) is required")
		return 2
	}
	root := resolveRoot(*dir)
	rep, err := commitPreflightFn(context.Background(), root, paths)
	if err != nil {
		fmt.Fprintf(stderr, "fak commit preflight: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak commit preflight: %v\n", err)
			return 1
		}
	} else {
		renderCommitPreflight(stdout, rep)
	}
	return commitPreflightExit(rep)
}

func renderCommitPreflight(w io.Writer, rep safecommit.PathPreflightReport) {
	if rep.OK {
		fmt.Fprintf(w, "commit-preflight OK — all %d pathspec(s) are tracked and committable by path\n", len(rep.Classes))
	} else {
		fmt.Fprintf(w, "commit-preflight: %s\n", rep.Reason)
	}
	for _, c := range rep.Classes {
		if c.State == safecommit.PathTracked {
			fmt.Fprintf(w, "  ok   %s (tracked)\n", c.Path)
			continue
		}
		fmt.Fprintf(w, "  bad  %s (%s)\n", c.Path, c.State)
		if c.Fix != "" {
			fmt.Fprintf(w, "         fix: %s\n", c.Fix)
		}
	}
}

// commitPreflightExit maps a report to the process exit code. An untracked/unmatched pathspec
// (or a not-a-repo) is a refusal on the merits (exit 4, "no — fix the pathspec; re-running the
// identical preflight is refused again"), never the retryable contention exit 3 (#5505 W4); an
// empty pathspec set is a usage error (exit 2).
func commitPreflightExit(rep safecommit.PathPreflightReport) int {
	switch rep.Reason {
	case "":
		return 0
	case safecommit.ReasonNoPath:
		return 2
	default: // PATH_UNTRACKED, PATH_UNMATCHED, NOT_A_REPO
		return safecommit.ExitRefused
	}
}
