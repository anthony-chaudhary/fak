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
