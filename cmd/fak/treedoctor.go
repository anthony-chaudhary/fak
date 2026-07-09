package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmdTreeDoctor — `fak tree-doctor`: diagnose and (with --apply) repair a fak working tree
// that has gone un-tidy under the always-on agent fleet. It reports three reclaimable
// hazards and, on --apply, fixes only the provably-safe ones:
//
//   - a STALE commit lock: .git/fak-commit.lock still owned by a DEAD process, which on
//     Windows can wedge the WHOLE shared-trunk commit lane (the 2026-06-28 56-minute stall);
//   - ORPHAN lock residue: renamed-aside `.git/*.lock.orphan*` files a lock-recovery step
//     left behind (never an active lock name, so never a live-op race), aged past the live
//     window — cruft that otherwise accumulates in the hot shared .git;
//   - MERGED, not-live worktrees: left over from prior multi-agent runs, already folded into
//     the trunk and untouched recently.
//
// It NEVER removes a worktree that is unmerged, dirty-with-unmerged-work, or recently
// touched (a live session) — in an always-on tree, a false prune destroys a peer's
// in-flight work. Report-only by default; --apply performs the reclaim.
func cmdTreeDoctor(argv []string) {
	fs := flag.NewFlagSet("tree-doctor", flag.ExitOnError)
	apply := fs.Bool("apply", false, "perform the safe reclaim (reap stale lock, prune merged-not-live worktrees); default is report-only")
	asJSON := fs.Bool("json", false, "emit a machine-readable diagnosis/action report")
	root := fs.String("root", "", "repo root to inspect (default: discover from cwd)")
	trunk := fs.String("trunk", "", "merge target a worktree must be folded into to be prunable (default: origin/main)")
	build := fs.Bool("build", false, "probe each untracked file's package with `go build` to flag build poison (opt-in: spawns go, adds host load)")
	abandonAfter := fs.Duration("abandon-after", treedoctor.DefaultAbandonAfter, "an untracked source file older than this and not held by a live owner is surfaced as an abandonment candidate")
	_ = fs.Parse(argv)

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "tree-doctor: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	wopts := treedoctor.WIPOptions{AbandonAfter: *abandonAfter}
	if *build {
		// Opt-in: `go build ./<pkg>/` per untracked file's package flags the shared-trunk
		// build poison that crash-loops peers. Gated because it spawns go (host load) and a
		// whole-package compile is not free — the age/owner inventory works without it.
		wopts.BuildProbe = goBuildProber(repoRoot)
	}
	opts := treedoctor.Options{RepoRoot: repoRoot, Trunk: *trunk, WIP: wopts}
	rep, actions := treedoctor.Sweep(context.Background(), gitRunner, opts, *apply)
	if *asJSON {
		jsonRep := rep
		if *apply && len(actions) > 0 {
			jsonRep = treedoctor.Diagnose(context.Background(), gitRunner, opts)
		}
		if err := renderTreeDoctorJSON(os.Stdout, treeDoctorJSON{
			Schema:      "fak-tree-doctor/1",
			Apply:       *apply,
			RepoRoot:    repoRoot,
			Trunk:       treeDoctorTrunk(*trunk),
			NeedsAction: treeDoctorNeedsAction(jsonRep),
			Report:      jsonRep,
			Actions:     actions,
		}); err != nil {
			fmt.Fprintf(os.Stderr, "tree-doctor: encode json: %v\n", err)
			os.Exit(1)
		}
		return
	}

	renderTreeDoctorText(os.Stdout, rep, actions, *apply)
}

type treeDoctorJSON struct {
	Schema      string            `json:"schema"`
	Apply       bool              `json:"apply"`
	RepoRoot    string            `json:"repo_root"`
	Trunk       string            `json:"trunk"`
	NeedsAction bool              `json:"needs_action"`
	Report      treedoctor.Report `json:"report"`
	Actions     []string          `json:"actions,omitempty"`
}

func renderTreeDoctorJSON(w io.Writer, payload treeDoctorJSON) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(payload)
}

func treeDoctorTrunk(trunk string) string {
	trunk = strings.TrimSpace(trunk)
	if trunk == "" {
		return treedoctor.DefaultTrunk
	}
	return trunk
}

func treeDoctorNeedsAction(rep treedoctor.Report) bool {
	return rep.Lock.Stale || len(rep.SweepableLockResidue()) > 0 || len(rep.PrunableWorktrees()) > 0
}

func renderTreeDoctorText(w io.Writer, rep treedoctor.Report, actions []string, apply bool) {
	if rep.Lock.Present {
		if rep.Lock.Stale {
			fmt.Fprintf(w, "commit lock: STALE — held by dead PID %d at %s\n", rep.Lock.HolderPID, rep.Lock.Path)
		} else {
			fmt.Fprintf(w, "commit lock: held by live PID %d (ok)\n", rep.Lock.HolderPID)
		}
	} else {
		fmt.Fprintln(w, "commit lock: none")
	}
	residue := 0
	for _, f := range rep.LockResidue {
		if f.Sweepable {
			residue++
			fmt.Fprintf(w, "lock residue SWEEPABLE: %s (aged %ds)\n", f.Path, f.AgeSeconds)
		} else {
			fmt.Fprintf(w, "lock residue keep:      %s (fresh — within live window)\n", f.Path)
		}
	}
	prunable := 0
	for _, wt := range rep.Worktrees {
		switch {
		case wt.IsMain:
			// don't list the main checkout
		case wt.Prunable:
			prunable++
			fmt.Fprintf(w, "worktree PRUNABLE: %s (merged, not live)\n", wt.Path)
		default:
			fmt.Fprintf(w, "worktree keep:     %s (%s)\n", wt.Path, wt.Keep)
		}
	}
	if prunable == 0 && residue == 0 && !rep.Lock.Stale {
		fmt.Fprintln(w, "tree is clean — nothing to reclaim")
	}

	renderWIPText(w, rep.WIP)

	if len(actions) > 0 {
		verb := "planned"
		if apply {
			verb = "applied"
		}
		fmt.Fprintf(w, "\n%s actions (%d):\n", verb, len(actions))
		for _, a := range actions {
			fmt.Fprintln(w, "  - "+a)
		}
		if !apply {
			fmt.Fprintln(w, "\nrun with --apply to perform these.")
		}
	}
}

// renderWIPText prints the untracked-source land-or-park inventory: the crash-loop culprits
// (build poison) and aged-unowned residue first, each surfaced for a human to LAND (commit)
// or PARK (move aside) — never removed by tree-doctor itself. A short summary line accounts
// for the live/resident files that are correctly left alone.
func renderWIPText(w io.Writer, wip []treedoctor.WIPFile) {
	if len(wip) == 0 {
		return
	}
	var landOrPark, live, resident int
	for _, f := range wip {
		switch {
		case f.LandOrPark:
			landOrPark++
		case f.Live:
			live++
		default:
			resident++
		}
	}
	fmt.Fprintf(w, "\nuntracked source WIP: %d file(s) — %d land-or-park, %d live, %d resident\n",
		len(wip), landOrPark, live, resident)
	for _, f := range wip {
		if !f.LandOrPark {
			continue
		}
		reason := "aged, no live owner"
		if f.Poison {
			reason = "BUILD POISON — package won't compile"
		}
		owner := f.Owner
		if owner == "" {
			owner = "unknown"
		}
		fmt.Fprintf(w, "  %-9s %s (age %s, owner %s) — %s\n",
			strings.ToUpper(f.Class), f.Path, compactAge(f.AgeSeconds), owner, reason)
	}
	if landOrPark > 0 {
		fmt.Fprintln(w, "  → land (commit by path) or park (move aside); tree-doctor never removes untracked source.")
	}
}

// compactAge renders a second count as a compact h/m/s string for the WIP listing.
func compactAge(sec int64) string {
	switch {
	case sec >= 3600:
		return fmt.Sprintf("%dh%dm", sec/3600, (sec%3600)/60)
	case sec >= 60:
		return fmt.Sprintf("%dm", sec/60)
	default:
		return fmt.Sprintf("%ds", sec)
	}
}

// goBuildProber returns a treedoctor build probe that reports whether a package directory
// (repo-relative, forward slashes) compiles via `go build ./<dir>/` run at repoRoot. A build
// error — the shared-trunk poison signal — returns false. Bounded by a context timeout so a
// wedged compile cannot stall the doctor.
func goBuildProber(repoRoot string) func(string) bool {
	return func(pkgDir string) bool {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		target := "./"
		if pkgDir != "" && pkgDir != "." {
			target = "./" + pkgDir + "/"
		}
		cmd := exec.CommandContext(ctx, "go", "build", target)
		cmd.Dir = repoRoot
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Run() == nil
	}
}

// discoverRepoRoot returns the top-level git working directory, or "" if cwd is not a repo.
func discoverRepoRoot() string {
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// gitRunner adapts the os/exec git invocation to treedoctor.Runner (combined output, exit
// code; error only when git could not be executed). It is shared by treedoctor and sweep.
func gitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	// GIT_OPTIONAL_LOCKS=0: sweep's burst-time reads (`git status --porcelain`,
	// the origin probes) run exactly when peers are committing, and a plain
	// status otherwise opportunistically refreshes the index under
	// .git/index.lock — colliding with the concurrent writers the sweep is
	// grouping for. Optional locks off means reads never contend; mandatory
	// write locks (the doctor's own mutations) are unaffected.
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
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
