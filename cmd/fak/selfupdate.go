package main

import (
	"context"
	"flag"
	"fmt"
	"os"
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
	_ = fs.Parse(argv)

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "self-update: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	headRev := repoHeadRev(repoRoot)
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

	target, err := os.Executable()
	if err != nil || strings.TrimSpace(target) == "" {
		fmt.Fprintln(os.Stderr, "self-update: cannot resolve this binary's own path:", err)
		os.Exit(1)
	}
	fmt.Printf("self-update: rebuilding + gating, then swapping %s …\n", target)
	res := selfinstall.Install(context.Background(), selfinstall.RealRunner, selfinstall.OSSwap, selfinstall.Options{
		RepoRoot: repoRoot,
		Target:   target,
	})
	fmt.Println(selfinstall.FormatResult(res))
	if !res.Installed {
		os.Exit(1)
	}
}

// repoHeadRev returns the full HEAD SHA of the repo at root, or "" on any error.
func repoHeadRev(root string) string {
	out, ok := selfinstall.RealRunner(context.Background(), root, "git", "rev-parse", "HEAD")
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
