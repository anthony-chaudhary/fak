package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// cmdSelfUpdate — `fak self-update`: converge THIS install on the latest VERIFIED fak.
//
// It is the durable answer to "the running fak/guard is stale": instead of waiting for a
// crash-and-relaunch (which on its own changes nothing, because relaunch re-execs the SAME
// stale file), a watchdog tick — or an operator — runs this. It:
//   1. reads the repo HEAD and compares it to THIS binary's embedded VCS revision
//      (binstamp): nothing to do unless the binary is provably Stale;
//   2. rebuilds fak, then GATES the candidate (vet + a `version` smoke run) and only then
//      atomically swaps it over this binary's path (selfinstall). A tree that is not green
//      is never installed.
//
// --check reports the freshness verdict and exits without building. --force installs even
// when freshness is Unknown/Fresh (e.g. to pick up an uncommitted local build) but STILL
// runs the full gate — force bypasses the staleness check, never the green gate.
func cmdSelfUpdate(argv []string) {
	fs := flag.NewFlagSet("self-update", flag.ExitOnError)
	check := fs.Bool("check", false, "report whether this binary is stale vs HEAD and exit (no build)")
	force := fs.Bool("force", false, "build+gate+install even if not provably stale (still runs the green gate)")
	root := fs.String("root", "", "repo root to build from (default: discover from cwd)")
	target := fs.String("target", "", "binary path to replace (default: this binary's own path). Lets a scheduler update the FLEET binary regardless of which fak it invokes.")
	_ = fs.Parse(argv)

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "self-update: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	// Compare against origin/main, not local HEAD: on a permanently-dirty shared trunk the
	// local tree is always ahead-or-behind with peer WIP, and origin/main is the verified
	// line we actually want guards converged on.
	_, _ = selfinstall.RealRunner(context.Background(), repoRoot, "git", "fetch", "origin", "--quiet")
	headRev := repoRevOf(repoRoot, "origin/main")
	self := binstamp.Self()
	verdict := binstamp.Compare(self, headRev)

	selfRev := self.Revision
	if selfRev == "" {
		selfRev = "(unstamped)"
	} else if len(selfRev) > 12 {
		selfRev = selfRev[:12]
	}
	head := headRev
	if len(head) > 12 {
		head = head[:12]
	}
	fmt.Printf("running: %s%s   HEAD: %s   => %s\n",
		selfRev, dirtyMark(self.Dirty), head, verdict)

	if *check {
		return
	}
	if verdict != binstamp.Stale && !*force {
		if verdict == binstamp.Unknown {
			fmt.Println("self-update: freshness unknown — not rebuilding (pass --force to build+gate+install anyway).")
		} else {
			fmt.Println("self-update: already current — nothing to do.")
		}
		return
	}

	installTarget := strings.TrimSpace(*target)
	if installTarget == "" {
		exe, err := os.Executable()
		if err != nil || strings.TrimSpace(exe) == "" {
			fmt.Fprintln(os.Stderr, "self-update: cannot resolve this binary's own path (pass --target):", err)
			os.Exit(1)
		}
		installTarget = exe
	}

	// Build from a PRISTINE detached origin/main checkout, never the live (peer-dirty)
	// tree: that gives a clean VCS stamp on the installed binary and guarantees we install
	// exactly verified origin/main, not a build contaminated with peers' work-in-progress.
	ctx := context.Background()
	// The build worktree must live OUTSIDE .git (git refuses `worktree add` to a path
	// inside the git dir) and outside the live tree (so it never shows up as peer churn).
	// A per-invocation temp dir under the OS temp root satisfies both.
	buildDir := filepath.Join(os.TempDir(), "fak-selfupdate-build-"+strconv.Itoa(os.Getpid()))
	_, cleanup, perr := selfinstall.PrepareOrigin(ctx, selfinstall.RealRunner, repoRoot, "origin/main", buildDir)
	if perr != nil {
		fmt.Fprintln(os.Stderr, "self-update:", perr)
		os.Exit(1)
	}
	defer cleanup()

	fmt.Printf("self-update: building origin/main + gating, then swapping %s …\n", installTarget)
	res := selfinstall.Install(ctx, selfinstall.RealRunner, selfinstall.OSSwap, selfinstall.Options{
		RepoRoot: buildDir,
		Target:   installTarget,
	})
	fmt.Println(selfinstall.FormatResult(res))
	if !res.Installed {
		os.Exit(1)
	}
}

// repoRevOf returns the full SHA a ref resolves to in the repo at root, or "" on error.
func repoRevOf(root, ref string) string {
	out, ok := selfinstall.RealRunner(context.Background(), root, "git", "rev-parse", ref)
	if !ok {
		return ""
	}
	return strings.TrimSpace(out)
}

func dirtyMark(dirty bool) string {
	if dirty {
		return " +uncommitted"
	}
	return ""
}
