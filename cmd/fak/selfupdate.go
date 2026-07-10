package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// cmdSelfUpdate — `fak self-update`: converge THIS install on the latest VERIFIED fak.
//
// It is the durable answer to "the running fak/guard is stale": instead of waiting for a
// crash-and-relaunch (which on its own changes nothing, because relaunch re-execs the SAME
// stale file), a watchdog tick — or an operator — runs this. It:
//  1. reads the repo HEAD and compares it to THIS binary's embedded VCS revision
//     (binstamp): nothing to do unless the binary is provably Stale;
//  2. rebuilds fak, then GATES the candidate (vet + a `version` smoke run) and only then
//     atomically swaps it over this binary's path (selfinstall). A tree that is not green
//     is never installed.
//
// --check reports the freshness verdict and exits without building. --force installs even
// when freshness is Unknown/Fresh (e.g. to pick up an uncommitted local build) but STILL
// runs the full gate — force bypasses the staleness check, never the green gate.
func cmdSelfUpdate(argv []string) {
	fs := flag.NewFlagSet("self-update", flag.ExitOnError)
	verbFlagUsage(fs, "self-update") // #2232: overview verb -> deep help above the flag dump
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

	// Whose freshness are we judging? When --target names a DIFFERENT binary (the scheduler
	// case: a dev fak invokes self-update to converge the FLEET binary), read the TARGET's
	// embedded stamp — not this invoking binary's. Otherwise the dirty dev binary's "Unknown"
	// would short-circuit the update and the stale fleet binary would never get replaced.
	stamp := binstamp.Self()
	subject := "running"
	fleetTarget := false // --target names a DIFFERENT binary (the scheduler/fleet case)
	if t := strings.TrimSpace(*target); t != "" && !sameBinary(t) {
		subject = "target"
		fleetTarget = true
		if ts, ok := stampOfBinary(t); ok {
			stamp = ts
		}
		// If the target can't self-report, stamp stays as Self() but fleetTarget forces the
		// build below regardless — a fleet binary we cannot prove current gets refreshed.
	}
	verdict := binstamp.Compare(stamp, headRev)

	// Beyond the coarse Fresh/Stale/Unknown, classify the stamp by git ANCESTRY vs origin/main.
	// This is what lets SELF mode tell a binary that is provably BEHIND (Skewed — worth
	// rebuilding) apart from one that is merely AHEAD (a fresh local build not yet pushed):
	// binstamp.Compare collapses BOTH into Stale, so the old `verdict == Stale` rule would
	// rebuild origin/main straight OVER a newer dev binary. AssessStamp reuses the stamp we
	// already resolved (the target's, in fleet mode) and the origin/main we just fetched.
	skew := versionskew.AssessStamp(context.Background(), selfinstall.RealRunner, repoRoot, "origin/main", stamp)

	stampRev := stamp.Revision
	if stampRev == "" {
		stampRev = "(unstamped)"
	} else if len(stampRev) > 12 {
		stampRev = stampRev[:12]
	}
	head := headRev
	if len(head) > 12 {
		head = head[:12]
	}
	fmt.Printf("%s: %s%s   origin/main: %s   => %s (skew: %s)\n",
		subject, stampRev, dirtyMark(stamp.Dirty), head, verdict, skew.Verdict)

	if *check {
		// Observability: also report the swap-aside footprint next to the target binary, so a
		// leak of "<binary>.old.<pid>.<i>" files (the 9 GB class this reaper exists to prevent)
		// is a one-line signal here instead of an invisible pile only `ls` would reveal.
		reportAsideFootprint(installTargetOr(*target))
		return
	}
	// Decide whether to build (see selfUpdateShouldBuild for the SELF/FLEET asymmetry).
	proceed := selfUpdateShouldBuild(*force, fleetTarget, verdict, skew.Verdict)
	if !proceed {
		switch {
		case fleetTarget:
			fmt.Println("self-update: target already current — nothing to do.")
		case skew.Verdict == versionskew.Ahead:
			fmt.Println("self-update: running binary is AHEAD of origin/main (a local build not yet pushed) — not rebuilding (pass --force to build+gate+install origin/main anyway).")
		case skew.Verdict == versionskew.Fresh:
			fmt.Println("self-update: already current — nothing to do.")
		case skew.Verdict == versionskew.Dirty, skew.Verdict == versionskew.Unstamped, skew.Verdict == versionskew.Diverged:
			fmt.Printf("self-update: running binary is %s vs origin/main — not auto-rebuilding a local/off-trunk build (pass --force to build+gate+install origin/main).\n", skew.Verdict)
		default:
			fmt.Println("self-update: freshness unknown — not rebuilding (pass --force to build+gate+install anyway).")
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

	// Single-flight: only one self-update may BUILD at a time on this host. A second
	// invocation (e.g. the scheduled tick firing while a slow build from the last tick is
	// still going on a saturated box) exits immediately rather than stacking another
	// expensive origin checkout + build. The lock is released on return (or process exit).
	release, lerr := selfinstall.TrySingleFlight("")
	if lerr != nil {
		if lerr == selfinstall.ErrBusy {
			fmt.Println("self-update: another self-update is already building — skipping this run.")
			return
		}
		fmt.Fprintln(os.Stderr, "self-update: lock error:", lerr)
		os.Exit(1)
	}
	defer release()

	// Build from a PRISTINE detached origin/main checkout, never the live (peer-dirty)
	// tree: that gives a clean VCS stamp on the installed binary and guarantees we install
	// exactly verified origin/main, not a build contaminated with peers' work-in-progress.
	ctx := context.Background()

	// Self-heal first: reap build worktrees leaked by PRIOR self-update ticks that were
	// killed (saturated box / wall-clock timeout) before their deferred cleanup could run.
	// We hold the single-flight lock, so every fak-selfupdate-build-* worktree on disk
	// belongs to a process that is no longer building; ReapStaleBuilds removes only those
	// whose owning PID is provably dead. Without this, killed ticks accumulate worktrees
	// forever (one prior run leaked dozens) because nothing else ever prunes them.
	if reaped := selfinstall.ReapStaleBuilds(ctx, selfinstall.RealRunner, repoRoot, os.Getpid(), safecommit.ProcessAlive); len(reaped) > 0 {
		fmt.Printf("self-update: reaped %d stale build worktree(s) leaked by killed prior runs\n", len(reaped))
	}

	// Also reap the "<binary>.old.<pid>.<i>" swap-aside files OSSwap leaks on Windows when the
	// old binary was still handle-locked at swap time. Nothing else reclaims them, so one leaks
	// per tick (a real host accumulated 211 of them, ~9 GB). We delete only asides whose owning
	// PID is provably dead — so the old .exe is no longer mapped and the file is safe to remove.
	if reaped := selfinstall.ReapStaleAsides(installTarget, os.Getpid(), safecommit.ProcessAlive); len(reaped) > 0 {
		fmt.Printf("self-update: reaped %d stale swap-aside binary file(s) leaked by prior swaps\n", len(reaped))
	}
	// After reaping, report any REMAINING footprint: asides pinned by still-live owners that we
	// could not reclaim. A large surviving count is the early signal that swaps are outrunning
	// exits (the leak's leading edge) — visible now instead of after it reaches gigabytes.
	if fp := selfinstall.MeasureAsides(installTarget, os.Getpid(), safecommit.ProcessAlive); fp.Count >= 8 {
		fmt.Printf("self-update: NOTE — %d swap-aside file(s) still next to the binary (%s); %d reclaimable once their owners exit\n",
			fp.Count, humanBytes(fp.Bytes), fp.DeadCount)
	}

	// The build worktree must live OUTSIDE .git (git refuses `worktree add` to a path
	// inside the git dir) and outside the live tree (so it never shows up as peer churn).
	// A per-invocation temp dir under the OS temp root satisfies both. BuildDirName encodes
	// our PID so a future run's reaper can tell a live build from a leaked corpse.
	buildDir := filepath.Join(os.TempDir(), selfinstall.BuildDirName(os.Getpid()))
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

// selfUpdateShouldBuild decides whether self-update proceeds to build+gate+install. The two
// modes consult DIFFERENT witnesses on purpose:
//   - FLEET (--target names a different binary): rebuild unless binstamp can PROVE the target
//     is the trunk tip (Fresh). Anything short of proof — including an unresolvable Unknown —
//     rebuilds, because a cheap gated swap is the right default for a binary the fleet runs.
//   - SELF (updating our own binary): rebuild ONLY when versionskew proves the running binary
//     is a strict ANCESTOR of origin/main (Skewed). This is the fix for the case binstamp
//     collapses: a clean local build that is AHEAD of origin/main reads as binstamp.Stale
//     (rev differs) and the old `verdict == Stale` rule would rebuild origin/main straight
//     OVER — downgrading — that newer binary. versionskew keeps Ahead (and Diverged / Dirty /
//     Unstamped) distinct from Skewed, so SELF mode never downgrades a developer's build out
//     from under them; --force is the deliberate escape for those cases.
//
// --force builds in either mode (force bypasses only the staleness check, never the green gate).
func selfUpdateShouldBuild(force, fleetTarget bool, bin binstamp.Freshness, skew versionskew.Verdict) bool {
	if force {
		return true
	}
	if fleetTarget {
		return bin != binstamp.Fresh
	}
	return skew == versionskew.Skewed
}

// installTargetOr resolves the binary path --check should inspect: the explicit --target, or
// this binary's own path. Mirrors the resolution in the main flow so --check reports the same
// target a real self-update would swap (and thus the same install dir the reaper cleans).
func installTargetOr(target string) string {
	if t := strings.TrimSpace(target); t != "" {
		return t
	}
	if exe, err := os.Executable(); err == nil {
		return exe
	}
	return ""
}

// reportAsideFootprint prints a one-line summary of the swap-aside files sitting next to the
// target binary, so `self-update --check` surfaces a leak while it is 5 files instead of 500.
// A clean install dir prints nothing (no noise on the healthy path).
func reportAsideFootprint(target string) {
	if strings.TrimSpace(target) == "" {
		return
	}
	fp := selfinstall.MeasureAsides(target, os.Getpid(), safecommit.ProcessAlive)
	if fp.Count == 0 {
		return
	}
	fmt.Printf("self-update: swap-aside footprint next to %s — %d file(s), %s (%d reclaimable, %s); the next self-update reaps the reclaimable ones\n",
		filepath.Base(target), fp.Count, humanBytes(fp.Bytes), fp.DeadCount, humanBytes(fp.DeadBytes))
}

// sameBinary reports whether path refers to this running executable (so --target pointing
// at ourselves falls back to the in-process binstamp.Self()).
func sameBinary(path string) bool {
	exe, err := os.Executable()
	if err != nil {
		return false
	}
	a, _ := filepath.Abs(filepath.Clean(path))
	b, _ := filepath.Abs(filepath.Clean(exe))
	return strings.EqualFold(a, b)
}

// stampOfBinary reads another fak binary's embedded VCS stamp by running `<bin> version` and
// parsing the "build: <rev>[ +uncommitted]" line cmdVersion prints. A binary too old to have
// that line (or that errors) yields ok=false; the caller treats that as "cannot prove fresh"
// which, in --target mode, correctly lets the update proceed (a fak that cannot self-report
// its build is by definition not the current one).
func stampOfBinary(path string) (binstamp.Stamp, bool) {
	out, ok := selfinstall.RealRunner(context.Background(), "", path, "version")
	if !ok {
		return binstamp.Stamp{}, false
	}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "build:") {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, "build:"))
		dirty := strings.Contains(rest, "+uncommitted")
		// the rev is the first whitespace-delimited token after "build:"
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			return binstamp.Stamp{}, false
		}
		rev := fields[0]
		if rev == "" || rev == "module" || rev == "(no" {
			return binstamp.Stamp{}, false // "module vX" or "(no VCS stamp …)" — no comparable rev
		}
		return binstamp.Stamp{Revision: rev, Dirty: dirty, HasVCS: true}, true
	}
	return binstamp.Stamp{}, false
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
