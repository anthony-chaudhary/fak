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
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
// LOCKED for a week?" from evidence, and --score GRADES that same evidence (#5587) —
// adoption, outcome health, fold drift — so "is this job still good?" is one command
// with a letter and named defects rather than an operator diffing rows by eye.
//
// Default is APPLY. Exit 0 on success or a deliberate skip; 1 when the run surfaced
// something an operator must repair (posture drift, failed lock cleanup, or an
// unwritable ledger — which would silently break the once-a-day dedupe); 2 on a usage
// error.
func cmdGitDaily(argv []string) { os.Exit(runGitDaily(os.Stdout, os.Stderr, argv)) }

var gitDailyRun = gitdaily.Run

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
	score := fs.Bool("score", false, "instead of running, GRADE the recorded history (adoption, outcome health, fold drift) and exit non-zero on a defect")
	pruneWorktrees := fs.Bool("prune-worktrees", false, "opt the tick into treedoctor's full sweep (merged/orphan worktree removal); default is locks-only")
	gracePrune := fs.Bool("grace-prune", false, "opt into gitgate's supervised grace-prune tier (quiet window + >=2-week expire floor); default never deletes an object")
	pruneExpire := fs.String("prune-expire", "", "override the grace-prune expire window (must be provably >= 2 weeks; empty = the floor)")
	leaseMaxAge := fs.Duration("lease-lock-max-age", 0, "orphan bound for refs/fak/locks/*.lock (0 = the session-heartbeat TTL)")
	goTmpDir := fs.String("gotmp-dir", "", "collect orphaned go-build* WORK dirs under this GOTMPDIR (default: $"+treedoctor.GoTmpDirEnv+"; empty disables the rung)")
	goTmpMinAge := fs.Duration("gotmp-min-age", 0, "quiet period a go-build WORK dir must clear before it is reapable, measured on the newest file ANYWHERE inside it (0 = the default floor)")
	goCache := fs.Bool("go-cache", true, "manage Go build cache lifecycle for this run")
	goCacheHighBytes := fs.Int64("go-cache-high-bytes", 0, "start pruning Go build cache when usage reaches this many bytes (0 = automatic)")
	goCacheLowBytes := fs.Int64("go-cache-low-bytes", 0, "prune Go build cache down to this many bytes once cleanup starts (0 = automatic)")
	goCacheMinAge := fs.Duration("go-cache-min-age", 0, "minimum age a Go build cache entry must reach before it is reapable (0 = automatic)")
	goCacheMinFreeBytes := fs.Int64("go-cache-min-free-bytes", 0, "require at least this many free bytes before skipping Go build cache cleanup (0 = automatic)")
	goCacheMaxEntries := fs.Int("go-cache-max-entries", 0, "limit the Go build cache to at most this many entries (0 = automatic)")
	goCacheDeadline := fs.Duration("go-cache-deadline", 0, "stop Go build cache cleanup after this long (0 = automatic)")
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

	// The GOTMPDIR rung (#6207) defaults to whatever the session actually redirected go's
	// WORK dirs into, so the scheduled tick collects the same tree the builds fill. It is
	// resolved HERE, at the I/O edge — gitdaily itself never guesses a path to delete, so a
	// caller that names nothing gets an inert rung rather than a surprise sweep.
	goTmpRoot := strings.TrimSpace(*goTmpDir)
	if goTmpRoot == "" {
		goTmpRoot = treedoctor.GoTmpRootFromEnv(os.Getenv)
	}
	goCacheRoot := treedoctor.GoCacheRootFromEnv(os.Getenv, os.UserCacheDir)
	goCacheOptions := treedoctor.GoCacheOptions{ActiveBuild: treedoctor.ActiveGoBuild}
	if *goCacheHighBytes < 0 {
		return gitDailyUsagef(stderr, "--go-cache-high-bytes must be >= 0")
	}
	if *goCacheLowBytes < 0 {
		return gitDailyUsagef(stderr, "--go-cache-low-bytes must be >= 0")
	}
	if *goCacheMinAge < 0 {
		return gitDailyUsagef(stderr, "--go-cache-min-age must be >= 0")
	}
	if *goCacheMinFreeBytes < 0 {
		return gitDailyUsagef(stderr, "--go-cache-min-free-bytes must be >= 0")
	}
	if *goCacheMaxEntries < 0 {
		return gitDailyUsagef(stderr, "--go-cache-max-entries must be >= 0")
	}
	if *goCacheDeadline < 0 {
		return gitDailyUsagef(stderr, "--go-cache-deadline must be >= 0")
	}
	if *goCacheHighBytes > 0 {
		goCacheOptions.HighBytes = *goCacheHighBytes
	}
	if *goCacheLowBytes > 0 {
		goCacheOptions.LowBytes = *goCacheLowBytes
	}
	if *goCacheHighBytes > 0 && *goCacheLowBytes > 0 && *goCacheLowBytes > *goCacheHighBytes {
		return gitDailyUsagef(stderr, "--go-cache-low-bytes must be <= --go-cache-high-bytes")
	}
	if *goCacheMinAge > 0 {
		goCacheOptions.MinAge = *goCacheMinAge
	}
	if *goCacheMinFreeBytes > 0 {
		goCacheOptions.MinFreeBytes = *goCacheMinFreeBytes
	}
	if *goCacheMaxEntries > 0 {
		goCacheOptions.MaxWalkEntries = *goCacheMaxEntries
	}
	if *goCacheDeadline > 0 {
		goCacheOptions.Deadline = *goCacheDeadline
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
		GoTmpDir:        goTmpRoot,
		GoTmpMinAge:     *goTmpMinAge,
		GoCacheDir:      "",
		GoCacheOptions:  goCacheOptions,
	}
	if *goCache {
		opts.GoCacheDir = goCacheRoot
	}

	// --status is a pure readback: it never runs a tick, so an operator can audit the
	// job's history without perturbing the very ledger they are reading.
	if *status > 0 {
		path := gitdaily.LedgerPath(opts)
		rows := gitdaily.Status(path, *status)
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, gitDailyStatusReport{
				Schema:   "fak-git-daily-status/1",
				Ledger:   path,
				Outcomes: gitdaily.FoldOutcomes(rows),
				Weekly:   gitdaily.FoldOutcomesByWeek(rows),
				Rows:     rows,
			}, "fak git-daily")
		}
		writeGitDailyStatus(stdout, path, rows)
		return 0
	}

	// --score is the GRADED readback (#5587): the same pure, non-perturbing read as
	// --status, folded into a letter grade with named evidence, so "is this job still
	// good?" is one command rather than an operator diffing rows by eye. It exits 1 on
	// debt, which is what lets a cron/CI caller gate on it without parsing the text.
	if *score {
		path := gitdaily.LedgerPath(opts)
		// Limit 0 = the whole retained history: a grade over the last handful of rows
		// would miss exactly the multi-day drift streak the card exists to catch.
		in := gitDailyHealthInput(gitdaily.Status(path, 0), path)
		payload := metrics.GradeGitDailyHealth(in)
		if *asJSON {
			if code := encodeJSONOrFail(stdout, stderr, payload, "fak git-daily"); code != 0 {
				return code
			}
		} else {
			fmt.Fprintln(stdout, metrics.GitDailyHealthFragment(in, payload))
			fmt.Fprintln(stdout, scorecard.Render(payload, metrics.GitDailyDebtKey))
		}
		if !payload.OK {
			return 1
		}
		return 0
	}

	res := gitDailyRun(context.Background(), gitdaily.Runner(gitRunner), opts)

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

func gitDailyUsagef(stderr io.Writer, format string, args ...any) int {
	fmt.Fprintf(stderr, "git-daily: "+format+"\n", args...)
	return 2
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
//
// Outcomes travels WITH the rows it was folded from (#5586). A consumer that only wants
// "is this job healthy?" reads the counters and stops; one that wants to know why reads
// the rows underneath — and because the summary is a fold of exactly the rows in this
// envelope, the two can never disagree the way a separately-kept counter would.
type gitDailyStatusReport struct {
	Schema   string                 `json:"schema"`
	Ledger   string                 `json:"ledger"`
	Outcomes gitdaily.Outcomes      `json:"outcomes"`
	Weekly   []gitdaily.WeekOutcome `json:"weekly"`
	Rows     []gitdaily.Row         `json:"rows"`
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

	// The GOTMPDIR rung (#6207). The age split prints under the summary because a single
	// total is exactly what made an earlier audit of this tree call in-flight churn a leak.
	fmt.Fprintf(w, "\nbuild scratch:\n  %s\n  %s\n", res.GoTmp.Summary(), res.GoCache.Summary())
	for _, hint := range res.GoCache.CleanupHints {
		fmt.Fprintf(w, "  hint: %s\n", hint)
	}
	for _, band := range res.GoTmp.Bands {
		fmt.Fprintf(w, "    %-9s %3d entries  %d bytes\n", band.Name, band.Entries, band.Bytes)
	}
	for _, e := range res.GoTmp.Entries {
		if e.RemoveErr != "" {
			fmt.Fprintf(w, "    FAILED to remove %s: %s\n", e.Path, e.RemoveErr)
		}
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

// gitDailyHealthInput projects the recorded ledger rows onto the pure fold's witness set
// (#5587). Every field descends from a row the tick already appended — this reads the
// ledger, it never counts anything of its own, so the grade cannot disagree with
// `--status`.
//
// The projection lives HERE, at the I/O edge, and not in internal/metrics: gitdaily sits a
// layer above metrics, so the card cannot import it without redding architest with
// ARCH_LAYER_VIOLATION. Keeping the card a pure fold over plain tallies is also what makes
// it deterministic — no clock, no filesystem, no git.
func gitDailyHealthInput(rows []gitdaily.Row, path string) metrics.GitDailyHealthInput {
	tally := gitdaily.FoldOutcomes(rows)
	outcomes := make([]string, 0, len(rows))
	for _, r := range rows {
		outcomes = append(outcomes, string(r.Outcome()))
	}
	runDays := make([]string, 0, len(rows))
	for _, row := range rows {
		runDays = append(runDays, row.Day)
	}
	now := time.Now()
	return metrics.GitDailyHealthInput{
		Runs:        tally.Runs,
		OK:          tally.OK,
		Refused:     tally.Refused,
		Errors:      tally.Errors,
		Reasons:     tally.Reasons,
		FirstDay:    tally.FirstDay,
		LastDay:     tally.LastDay,
		LooseFolded: tally.LooseFolded,
		// The streak rule lives with the card so "a streak" cannot drift between the
		// grade and the caller that feeds it.
		RefusedStreak: metrics.GitDailyRefusedStreak(outcomes),
		// LOCAL date, matching gitdaily.DayLayout — the ledger's day keys are local, so a
		// UTC "today" would read a full day stale for half the world every evening.
		Today:       now.Format(gitdaily.DayLayout),
		CurrentHour: now.Hour(),
		RunDays:     runDays,
		LedgerPath:  path,
	}
}

// writeGitDailyStatus prints the ledger readback: the outcome counters first, then one
// line per run, then the deferred streak — the signal that says the ghosts are winning
// and the backlog is growing.
func writeGitDailyStatus(w io.Writer, path string, rows []gitdaily.Row) {
	fmt.Fprintf(w, "git-daily status — %s\n", path)
	if len(rows) == 0 {
		fmt.Fprintln(w, "no runs recorded yet (the ledger is written by an applied, non-skipped run).")
		return
	}
	writeGitDailyOutcomes(w, gitdaily.FoldOutcomes(rows))
	writeGitDailyWeekly(w, gitdaily.FoldOutcomesByWeek(rows))
	for _, r := range rows {
		note := fmt.Sprintf("folded %d loose (%d -> %d), packs %d -> %d, locks cleared %d",
			r.LooseFolded(), r.LooseBefore, r.LooseAfter, r.PacksBefore, r.PacksAfter,
			r.LeaseLocksReaped+r.IndexLocksReaped+r.LockActions)
		if r.GraceRefused != "" {
			note += "; fold tier " + r.GraceRefused
		}
		if r.GoTmpReaped > 0 {
			note += fmt.Sprintf("; reaped %d go-build WORK dirs (%d bytes)", r.GoTmpReaped, r.GoTmpReclaimedBytes)
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

// writeGitDailyWeekly prints the adoption fold: one deterministic line per UTC week.
func writeGitDailyWeekly(w io.Writer, weeks []gitdaily.WeekOutcome) {
	for _, week := range weeks {
		fmt.Fprintf(w, "week %s: %d recorded runs, %d ok, %d refused, %d error\n",
			week.Week, week.Total, week.OK, week.Refused, week.Errors)
	}
}

// writeGitDailyOutcomes prints the one-line success/refusal/error tally over the rows
// being shown, plus the refusal breakdown when there is one (#5586).
//
// It says "recorded runs" on purpose. A skipped tick (ALREADY_RAN_TODAY, TICK_BUSY)
// writes no ledger row — that is exactly what makes an hourly catch-up trigger safe — so
// these counters tally the ticks that reached the tiers, NOT the times the scheduler
// fired. Labelling them "runs" flat would invite an operator to read a healthy hourly
// trigger as a job that only ran four times.
func writeGitDailyOutcomes(w io.Writer, o gitdaily.Outcomes) {
	window := o.FirstDay
	if o.LastDay != o.FirstDay {
		window += ".." + o.LastDay
	}
	fmt.Fprintf(w, "%d recorded runs (%s): %d ok, %d refused, %d error; folded %d loose objects\n",
		o.Runs, window, o.OK, o.Refused, o.Errors, o.LooseFolded)
	for _, reason := range o.ReasonsByCount() {
		fmt.Fprintf(w, "  refused %s x%d\n", reason, o.Reasons[reason])
	}
}
