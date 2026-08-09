package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmdGitMaint — `fak git-maint`: the guarded, lock-aware, posture-asserting
// "consolidate-never-prune" object-DB maintenance path for the always-hot shared clone.
// gitgate.go REFUSES the destructive git verbs (gc --prune=now, repack -adb, worktree
// prune); this verb is the safe forward path the guard otherwise only defers.
//
// It runs the always-safe tier (git multi-pack-index write, git commit-graph write
// --reachable — add-only, atomic, safe even mid-commit) unconditionally, and the
// safe-with-grace tier (git maintenance run --task=loose-objects / git prune-packed /
// --task=incremental-repack,
// which unlink ONLY fully-covered redundant copies) only when the lock preflight AND
// the shared no-auto-gc posture both pass, re-checking locks before every mutating step.
// It never prunes unreachable objects, never full-repacks, never edits .git/config, and
// is idempotent. It prints before/after `git count-objects -vH`. The ONE config-editing
// exception is explicitly operator-invoked: --repair-fsmonitor (#5068) clears the #4603
// "core.fsmonitor=true but dead daemon" drift by unsetting the key (default) or starting
// the builtin daemon (--fsmonitor-mode start); the auto-run path never does this.
//
// Default is APPLY (the point of the verb is to consolidate); --dry-run probes locks +
// posture and prints the plan and the before-count without mutating anything. Exit 0 on
// success (grace ran, or was cleanly deferred by a lock); exit 1 when posture drift
// refused the grace tier — an incident an operator must repair in the shared config.
func cmdGitMaint(argv []string) {
	fs := flag.NewFlagSet("git-maint", flag.ExitOnError)
	verbFlagUsage(fs, "git-maint")
	dryRun := fs.Bool("dry-run", false, "probe locks + posture and print the plan and the before-count; mutate nothing")
	asJSON := fs.Bool("json", false, "emit a machine-readable result")
	root := fs.String("root", "", "repo root to consolidate (default: discover from cwd)")
	repairFsmonitor := fs.Bool("repair-fsmonitor", false, "operator-invoked repair of the #4603 fsmonitor drift before the maintenance run (see --fsmonitor-mode)")
	fsmonitorMode := fs.String("fsmonitor-mode", string(gitgate.FsmonitorRepairOff), "fsmonitor repair mode: off (unset core.fsmonitor — the #5068 default for this clone) or start (start the builtin daemon)")
	_ = fs.Parse(argv)

	if *fsmonitorMode != string(gitgate.FsmonitorRepairOff) && *fsmonitorMode != string(gitgate.FsmonitorRepairStart) {
		fmt.Fprintf(os.Stderr, "git-maint: unknown --fsmonitor-mode %q (want off|start)\n", *fsmonitorMode)
		os.Exit(2)
	}

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "git-maint: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}
	commonDir := discoverGitCommonDir(repoRoot)
	if commonDir == "" {
		fmt.Fprintln(os.Stderr, "git-maint: could not resolve the shared .git (--git-common-dir)")
		os.Exit(2)
	}

	// Operator-invoked fsmonitor repair (#5068) runs BEFORE the maintenance run, so
	// the posture line below is the captured after-witness the acceptance gate asks
	// for: on a repaired clone it reads SAFE in the same invocation.
	var repair *gitgate.FsmonitorRepairResult
	if *repairFsmonitor {
		r := gitgate.RepairFsmonitor(context.Background(), gitRunner, gitgate.FsmonitorRepairOptions{
			RepoRoot: repoRoot,
			Mode:     gitgate.FsmonitorRepairMode(*fsmonitorMode),
			Apply:    !*dryRun,
		})
		repair = &r
	}

	res := gitgate.RunMaint(context.Background(), gitRunner, gitgate.MaintOptions{
		RepoRoot:     repoRoot,
		GitCommonDir: commonDir,
		Apply:        !*dryRun,
	})

	if *asJSON {
		if err := renderGitMaintJSON(os.Stdout, res, repair); err != nil {
			fmt.Fprintf(os.Stderr, "git-maint: encode json: %v\n", err)
			os.Exit(1)
		}
	} else {
		if repair != nil {
			renderFsmonitorRepairText(os.Stdout, *repair)
		}
		renderGitMaintText(os.Stdout, res)
	}

	// Posture drift is an incident (an unsupervised auto-gc could prune-race): flag it
	// with a non-zero exit so a loop/CI notices, after the report is printed.
	if res.Incident {
		os.Exit(1)
	}
}

// discoverGitCommonDir resolves the shared object-DB directory via
// `git rev-parse --git-common-dir`, normalized to an absolute path under root (git may
// answer with a bare ".git" relative to the worktree).
func discoverGitCommonDir(root string) string {
	cmd := exec.Command("git", "-C", root, "rev-parse", "--git-common-dir")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	p := strings.TrimSpace(string(out))
	if p == "" {
		return ""
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(root, p)
	}
	return p
}

func renderGitMaintJSON(w io.Writer, res gitgate.MaintResult, repair *gitgate.FsmonitorRepairResult) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(struct {
		Schema string `json:"schema"`
		gitgate.MaintResult
		FsmonitorRepair *gitgate.FsmonitorRepairResult `json:"fsmonitor_repair,omitempty"`
	}{Schema: "fak-git-maint/1", MaintResult: res, FsmonitorRepair: repair})
}

// renderFsmonitorRepairText prints the one-line before/after witness of the
// operator-invoked fsmonitor repair (#5068), ahead of the maintenance report whose
// posture line then shows the post-repair state.
func renderFsmonitorRepairText(w io.Writer, r gitgate.FsmonitorRepairResult) {
	before := "core.fsmonitor=" + displayValue(r.BeforeValue)
	if r.BeforeDaemon != "" {
		before += ", daemon=" + r.BeforeDaemon
	}
	after := "core.fsmonitor=" + displayValue(r.AfterValue)
	if r.AfterDaemon != "" {
		after += ", daemon=" + r.AfterDaemon
	}
	verdict := "NOT CLEARED"
	if r.Cleared {
		verdict = "CLEARED"
	}
	switch {
	case r.Err != "" && !r.Ran:
		fmt.Fprintf(w, "fsmonitor repair (mode=%s): REFUSED — %s\n\n", r.Mode, r.Err)
	case r.Action == gitgate.FsmonitorActionNone:
		fmt.Fprintf(w, "fsmonitor repair (mode=%s): nothing to repair (%s) — %s\n\n", r.Mode, before, verdict)
	case !r.Ran:
		fmt.Fprintf(w, "fsmonitor repair (mode=%s): DRY-RUN — %s; planned: git %s\n\n", r.Mode, before, strings.Join(r.Args, " "))
	default:
		fmt.Fprintf(w, "fsmonitor repair (mode=%s): %s -> git %s (exit %d%s) -> %s — %s\n\n",
			r.Mode, before, strings.Join(r.Args, " "), r.Code, errSuffix(r.Err), after, verdict)
	}
}

// displayValue renders a config value, spelling the unset case as <unset>.
func displayValue(v string) string {
	if strings.TrimSpace(v) == "" {
		return "<unset>"
	}
	return v
}

func renderGitMaintText(w io.Writer, res gitgate.MaintResult) {
	mode := "apply"
	if !res.Apply {
		mode = "dry-run"
	}
	fmt.Fprintf(w, "git-maint (%s)\n", mode)

	if res.Posture.Safe {
		fmt.Fprintf(w, "posture: SAFE (gc.auto=%s, maintenance.auto=%s)\n", res.Posture.GCAuto, res.Posture.MaintenanceAuto)
	} else {
		fmt.Fprintf(w, "posture: DRIFT — %s\n", res.Posture.Drift)
	}
	if len(res.Locks) > 0 {
		fmt.Fprintf(w, "locks:   %d live — %s\n", len(res.Locks), strings.Join(res.Locks, ", "))
	} else {
		fmt.Fprintln(w, "locks:   none")
	}

	fmt.Fprintln(w, "\nsteps:")
	for _, s := range res.Steps {
		verb := "git " + strings.Join(s.Args, " ")
		switch {
		case s.Skipped != "":
			fmt.Fprintf(w, "  - [%-14s] SKIPPED (%s): %s\n", s.Tier, s.Skipped, verb)
		case !s.Ran:
			fmt.Fprintf(w, "  - [%-14s] planned: %s\n", s.Tier, verb)
		case s.Err != "" || s.Code != 0:
			fmt.Fprintf(w, "  - [%-14s] ran (exit %d%s): %s\n", s.Tier, s.Code, errSuffix(s.Err), verb)
		default:
			fmt.Fprintf(w, "  - [%-14s] ran: %s\n", s.Tier, verb)
		}
	}

	if res.GraceRefused != "" {
		fmt.Fprintf(w, "\nsafe-with-grace tier %s: %s\n", refusalWord(res.GraceRefused), res.GraceRefused)
	}

	fmt.Fprintln(w, "\nbefore  git count-objects -vH:")
	writeIndented(w, res.Before.Raw)
	fmt.Fprintln(w, "after   git count-objects -vH:")
	writeIndented(w, res.After.Raw)
	if res.Before.Available && res.After.Available {
		fmt.Fprintf(w, "loose objects folded: %d (%d -> %d); in-pack %d -> %d; packs %d -> %d — nothing pruned\n",
			res.LooseDelta(), res.Before.Count, res.After.Count,
			res.Before.InPack, res.After.InPack, res.Before.Packs, res.After.Packs)
	}
	if res.Before.Available {
		verdict := "ok"
		if res.LooseBacklogHigh {
			verdict = "HIGH"
		}
		fmt.Fprintf(w, "LOOSE_BACKLOG_HIGH: %s (%d >= %d threshold)\n",
			verdict, res.Before.Count, gitgate.LooseBacklogThreshold)
	}
	if res.Incident {
		fmt.Fprintln(w, "\nINCIDENT: posture drift — the shared no-auto-gc config was violated; an auto-gc could prune-race.")
		fmt.Fprintln(w, "  repair the shared config (gc.auto=0, maintenance.auto=false) before rerunning the grace tier.")
	}
}

func errSuffix(err string) string {
	if strings.TrimSpace(err) == "" {
		return ""
	}
	return ", " + err
}

func refusalWord(r gitgate.MaintReason) string {
	if r == gitgate.MaintReasonPostureDrift {
		return "refused"
	}
	return "deferred"
}

func writeIndented(w io.Writer, text string) {
	if strings.TrimSpace(text) == "" {
		fmt.Fprintln(w, "  (unavailable)")
		return
	}
	for _, line := range strings.Split(text, "\n") {
		fmt.Fprintf(w, "  %s\n", strings.TrimRight(line, "\r"))
	}
}
