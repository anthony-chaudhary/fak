package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

// cmdTreeDoctor — `fak tree-doctor`: diagnose and (with --apply) repair a fak working tree
// that has gone un-tidy under the always-on agent fleet. It reports two reclaimable
// hazards and, on --apply, fixes only the provably-safe ones:
//
//   - a STALE commit lock: .git/fak-commit.lock still owned by a DEAD process, which on
//     Windows can wedge the WHOLE shared-trunk commit lane (the 2026-06-28 56-minute stall);
//   - MERGED, not-live worktrees: left over from prior multi-agent runs, already folded into
//     the trunk and untouched recently.
//
// It NEVER removes a worktree that is unmerged, dirty-with-unmerged-work, or recently
// touched (a live session) — in an always-on tree, a false prune destroys a peer's
// in-flight work. Report-only by default; --apply performs the reclaim.
func cmdTreeDoctor(argv []string) {
	fs := flag.NewFlagSet("tree-doctor", flag.ExitOnError)
	apply := fs.Bool("apply", false, "perform the safe reclaim (reap stale lock, prune merged-not-live worktrees); default is report-only")
	root := fs.String("root", "", "repo root to inspect (default: discover from cwd)")
	trunk := fs.String("trunk", "", "merge target a worktree must be folded into to be prunable (default: origin/main)")
	_ = fs.Parse(argv)

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "tree-doctor: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	opts := treedoctor.Options{RepoRoot: repoRoot, Trunk: *trunk}
	rep, actions := treedoctor.Sweep(context.Background(), gitRunner, opts, *apply)

	// Report.
	if rep.Lock.Present {
		if rep.Lock.Stale {
			fmt.Printf("commit lock: STALE — held by dead PID %d at %s\n", rep.Lock.HolderPID, rep.Lock.Path)
		} else {
			fmt.Printf("commit lock: held by live PID %d (ok)\n", rep.Lock.HolderPID)
		}
	} else {
		fmt.Println("commit lock: none")
	}
	prunable := 0
	for _, w := range rep.Worktrees {
		switch {
		case w.IsMain:
			// don't list the main checkout
		case w.Prunable:
			prunable++
			fmt.Printf("worktree PRUNABLE: %s (merged, not live)\n", w.Path)
		default:
			fmt.Printf("worktree keep:     %s (%s)\n", w.Path, w.Keep)
		}
	}
	if prunable == 0 && !rep.Lock.Stale {
		fmt.Println("tree is clean — nothing to reclaim")
	}

	if len(actions) > 0 {
		verb := "planned"
		if *apply {
			verb = "applied"
		}
		fmt.Printf("\n%s actions (%d):\n", verb, len(actions))
		for _, a := range actions {
			fmt.Println("  - " + a)
		}
		if !*apply {
			fmt.Println("\nrun with --apply to perform these.")
		}
	}
}

// discoverRepoRoot returns the top-level git working directory, or "" if cwd is not a repo.
func discoverRepoRoot() string {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// gitRunner adapts the os/exec git invocation to treedoctor.Runner (combined output, exit
// code; error only when git could not be executed).
func gitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	var ee *exec.ExitError
	if asExit(err, &ee) {
		return string(out), ee.ExitCode(), nil
	}
	return "", -1, err
}

// asExit is errors.As specialized to *exec.ExitError, kept tiny so the file needs no
// extra import beyond os/exec.
func asExit(err error, target **exec.ExitError) bool {
	for err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			*target = ee
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}
