package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gitdaily"
)

// cmdGitDaily — `fak git-daily`: the once-a-day unattended git-hygiene tick an OS
// scheduler fires so the always-hot shared clone stays fast with no human in the loop.
//
// It is the SCHEDULED composite of two halves that are individually inert here. A git
// process killed mid-transaction leaves its `.lock` behind forever, and gitgate's
// maintenance tiers correctly defer while a lock is live — so `fak git-maint` on a
// timer reports success daily and folds nothing, which is the #4602/#4605 failure
// verbatim (67,885 loose objects, ~2-minute cold git stalls, six lease ghosts up to 4.9
// DAYS old). Reaping locks alone leaves the backlog. This verb reaps the ghosts FIRST,
// then consolidates — see internal/gitdaily for the full safety argument and the exact
// blast radius.
//
// It is safe to point a coarse, catch-up-on-wake trigger at it: mutating ticks are
// serialized by an advisory lock and skipped once the ledger shows an applied run for
// today's local date (--force re-runs). Every applied run appends a `fak-git-daily/1`
// ledger row, so --status answers "has this actually been folding, or deferring as
// LOCKED for a week?" from evidence.
//
// Default is APPLY. Exit 0 on success or a deliberate skip; 1 when the run surfaced
// something an operator must repair (posture drift, failed lock cleanup, or an
// unwritable ledger — which would silently break the once-a-day dedupe); 2 on a usage
// error.
func cmdGitDaily(argv []string) { os.Exit(runGitDaily(os.Stdout, os.Stderr, argv)) }

func runGitDaily(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("git-daily", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "git-daily")
	dryRun := fs.Bool("dry-run", false, "report the dedupe decision, the orphan locks that WOULD be reaped, and the maintenance plan; mutate nothing and write no ledger row")
	force := fs.Bool("force", false, "run even though the ledger already shows an applied run for today (never bypasses the concurrency lock)")
	asJSON := fs.Bool("json", false, "emit a machine-readable result")
	root := fs.String("root", "", "repo root to maintain (default: discover from cwd)")
	ledger := fs.String("ledger", "", "witness ledger path (default: <git-common-dir>/"+gitdaily.LedgerName+")")
	status := fs.Int("status", 0, "instead of running, print the last N ledger rows (0 = do not read back)")
	pruneWorktrees := fs.Bool("prune-worktrees", false, "opt the tick into treedoctor's full sweep (merged/orphan worktree removal); default is locks-only")
	gracePrune := fs.Bool("grace-prune", false, "opt into gitgate's supervised grace-prune tier (quiet window + >=2-week expire floor); default never deletes an object")
	pruneExpire := fs.String("prune-expire", "", "override the grace-prune expire window (must be provably >= 2 weeks; empty = the floor)")
	leaseMaxAge := fs.Duration("lease-lock-max-age", 0, "orphan bound for refs/fak/locks/*.lock (0 = the session-heartbeat TTL)")
	emitUnit := fs.String("emit-unit", "", "instead of running, print an OS scheduler unit that fires this verb: launchd|systemd|taskscheduler")
	interval := fs.Duration("interval", 24*time.Hour, "firing cadence stamped into --emit-unit's unit")
	fakBin := fs.String("fak-bin", "fak", "path to the fak binary the emitted unit invokes")
	label := fs.String("label", "", "unit/task name for --emit-unit (default fak-git-daily)")
	if !parseFlags(fs, argv) {
		return 2
	}

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}

	if strings.TrimSpace(*emitUnit) != "" {
		return emitGitDailyUnit(stdout, stderr, *emitUnit, *label, *fakBin, repoRoot, *interval)
	}

	if repoRoot == "" {
		fmt.Fprintln(stderr, "git-daily: could not resolve a git repo root (pass --root)")
		return 2
	}
	commonDir := discoverGitCommonDir(repoRoot)
	if commonDir == "" {
		fmt.Fprintln(stderr, "git-daily: could not resolve the shared .git (--git-common-dir)")
		return 2
	}

	opts := gitdaily.Options{
		RepoRoot:        repoRoot,
		GitCommonDir:    commonDir,
		Ledger:          strings.TrimSpace(*ledger),
		Apply:           !*dryRun,
		Force:           *force,
		PruneWorktrees:  *pruneWorktrees,
		GracePrune:      *gracePrune,
		PruneExpire:     *pruneExpire,
		LeaseLockMaxAge: *leaseMaxAge,
	}

	// --status is a pure readback: it never runs a tick, so an operator can audit the
	// job's history without perturbing the very ledger they are reading.
	if *status > 0 {
		path := gitdaily.LedgerPath(opts)
		rows := gitdaily.Status(path, *status)
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, gitDailyStatusReport{
				Schema: "fak-git-daily-status/1",
				Ledger: path,
				Rows:   rows,
			}, "fak git-daily")
		}
		writeGitDailyStatus(stdout, path, rows)
		return 0
	}

	res := gitdaily.Run(context.Background(), gitdaily.Runner(gitRunner), opts)

	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, res, "fak git-daily"); code != 0 {
			return code
		}
	} else {
		writeGitDailyText(stdout, res)
	}

	// Every exit here means "an operator must act": posture drift lets an unsupervised
	// auto-gc prune-race, a lock cleanup failure leaves the maintenance wedge in place,
	// and an unwritable ledger silently breaks the once-a-day dedupe (the tick would then
	// redo full work on every trigger).
	if res.Incident || res.LedgerErr != "" {
		return 1
	}
	return 0
}

// emitGitDailyUnit prints an OS scheduler unit whose action is `fak git-daily`, reusing
// the same writers `fak cron emit --command` uses so there is exactly one spelling of
// each unit format in the tree.
//
// The emitted command always carries an ABSOLUTE --root. A scheduler starts its task in
// its own working directory (system32 under Task Scheduler, / under systemd), where
// cwd-based repo discovery finds nothing — so a unit that relied on discovery would exit
// 2 every single day while looking perfectly installed. Refusing to emit without a
// resolvable root turns that silent daily failure into an install-time usage error.
func emitGitDailyUnit(stdout, stderr io.Writer, target, label, fakBin, repoRoot string, interval time.Duration) int {
	if _, ok := cronSources[target]; !ok {
		fmt.Fprintf(stderr, "git-daily: unknown --emit-unit %q (want launchd|systemd|taskscheduler)\n", target)
		return 2
	}
	if interval <= 0 {
		fmt.Fprintln(stderr, "git-daily: --interval must be positive")
		return 2
	}
	if repoRoot == "" {
		fmt.Fprintln(stderr, "git-daily: --emit-unit needs the repo to maintain; run it inside the clone or pass --root <abs path>")
		return 2
	}
	abs, err := filepath.Abs(repoRoot)
	if err != nil {
		fmt.Fprintf(stderr, "git-daily: --root %q is not resolvable: %v\n", repoRoot, err)
		return 2
	}
	if strings.TrimSpace(label) == "" {
		label = "fak-git-daily"
	}
	args := []string{fakBin, "git-daily", "--root", abs}
	descs := cronDescs{
		service: "fak daily git hygiene: reap orphaned git locks, then consolidate the object DB (never prunes)",
		timer:   "Timer for " + label,
		task:    "fak daily git hygiene (reap orphaned git locks, then consolidate the object DB)",
	}
	cronRender(stdout, target, cronSanitizeLabel(label), descs, interval, args)
	return 0
}

// gitDailyStatusReport is the --status --json envelope: the resolved ledger path next to
// the rows, so a reader of the JSON never has to re-derive which file it was read from.
type gitDailyStatusReport struct {
	Schema string         `json:"schema"`
	Ledger string         `json:"ledger"`
	Rows   []gitdaily.Row `json:"rows"`
}

// writeGitDailyText prints the operator report: the skip decision if any, then the lock
// half, then the object-DB half, then the one-line fold witness.
func writeGitDailyText(w io.Writer, res gitdaily.Result) {
	mode := "apply"
	if !res.Apply {
		mode = "dry-run"
	}
	fmt.Fprintf(w, "git-daily (%s) day=%s\n", mode, res.Day)

	if res.Skipped != "" {
		switch res.Skipped {
		case gitdaily.SkipAlreadyRanToday:
			fmt.Fprintf(w, "SKIPPED (%s): an applied run is already recorded for %s — rerun with --force.\n",
				res.Skipped, res.LastRunDay)
		case gitdaily.SkipTickBusy:
			fmt.Fprintf(w, "SKIPPED (%s): another git-daily tick holds the lock and is doing this work now.\n", res.Skipped)
		default:
			fmt.Fprintf(w, "SKIPPED (%s)\n", res.Skipped)
		}
		return
	}
	if res.TickLockErr != "" {
		fmt.Fprintf(w, "TICK LOCK FAILED: %s\n  no cleanup or maintenance ran; restore access to the git common dir, then retry.\n", res.TickLockErr)
		return
	}
	if res.LastRunDay != "" {
		fmt.Fprintf(w, "last applied run: %s\n", res.LastRunDay)
	}

	verb, would := "reaped", ""
	if !res.Apply {
		verb, would = "would reap", "would "
	}
	locks := res.Locks
	fmt.Fprintf(w, "\nlocks:\n")
	fmt.Fprintf(w, "  lease ghosts (refs/fak/locks): %s %d, kept %d fresh\n", verb, len(locks.LeaseReaped), len(locks.LeaseKept))
	for _, p := range locks.LeaseReaped {
		fmt.Fprintf(w, "    - %s\n", p)
	}
	if locks.LeaseErr != "" {
		fmt.Fprintf(w, "    lease sweep error: %s\n", locks.LeaseErr)
	}
	fmt.Fprintf(w, "  index ghosts (.git/index.lock / next-index-*): %s %d\n", verb, len(locks.IndexReaped))
	for _, p := range locks.IndexReaped {
		fmt.Fprintf(w, "    - %s\n", p)
	}
	if locks.IndexErr != "" {
		fmt.Fprintf(w, "    index sweep error: %s\n", locks.IndexErr)
	}
	if locks.StaleCommitLock {
		fmt.Fprintln(w, "  commit lock: STALE (dead holder) — the wedge that stalls the whole commit lane")
	}
	if locks.StaleRefLocks > 0 {
		fmt.Fprintf(w, "  stale ref locks diagnosed: %d\n", locks.StaleRefLocks)
	}
	if len(locks.Actions) == 0 {
		fmt.Fprintf(w, "  tree-doctor: nothing to reclaim\n")
	}
	for _, a := range locks.Actions {
		fmt.Fprintf(w, "  - %s\n", a)
	}

	fmt.Fprintln(w)
	renderGitMaintText(w, res.Maint)

	if res.Apply && res.LedgerErr == "" {
		fmt.Fprintf(w, "\nwitnessed to %s\n", res.LedgerPath)
	}
	if res.LedgerErr != "" {
		fmt.Fprintf(w, "LEDGER WRITE FAILED: %s\n  the once-a-day dedupe cannot hold until this is fixed (%sredo full work on every trigger).\n",
			res.LedgerErr, would)
	}
}

// writeGitDailyStatus prints the ledger readback: one line per run, then the deferred
// streak — the signal that says the ghosts are winning and the backlog is growing.
func writeGitDailyStatus(w io.Writer, path string, rows []gitdaily.Row) {
	fmt.Fprintf(w, "git-daily status — %s\n", path)
	if len(rows) == 0 {
		fmt.Fprintln(w, "no runs recorded yet (the ledger is written by an applied, non-skipped run).")
		return
	}
	for _, r := range rows {
		note := fmt.Sprintf("folded %d loose (%d -> %d), packs %d -> %d, locks cleared %d",
			r.LooseFolded(), r.LooseBefore, r.LooseAfter, r.PacksBefore, r.PacksAfter,
			r.LeaseLocksReaped+r.IndexLocksReaped+r.LockActions)
		if r.GraceRefused != "" {
			note += "; fold tier " + r.GraceRefused
		}
		if r.Incident {
			note += "; INCIDENT"
		}
		fmt.Fprintf(w, "  %s  %s\n", r.Day, note)
	}
	if n, reason := gitdaily.DeferredStreak(rows); n > 1 {
		fmt.Fprintf(w, "\nfold tier has been held back %s for %d consecutive runs — the backlog is growing.\n", reason, n)
	}
}
