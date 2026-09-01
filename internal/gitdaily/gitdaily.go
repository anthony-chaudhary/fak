// Package gitdaily is the once-a-day unattended git-hygiene tick for an always-hot,
// shared, multi-session clone — the thing an OS scheduler fires at 03:00 so the object
// DB and the lock state stay healthy without a human in the loop.
//
// WHY A COMPOSITE VERB AND NOT JUST A SCHEDULED `git gc`. The two halves of git
// hygiene are causally coupled on this kind of clone, and the ORDER between them is the
// whole point. A git process killed mid-transaction (timeout SIGTERM, crash, power
// loss) leaves its `.lock` behind FOREVER, because git only removes it on the path it
// never got to run — and gitgate's maintenance tiers are, correctly, lock-deferential:
//
//   - The FOLD tier refuses while any transaction lock is live, re-probing before every
//     mutating step. `fak-commit.lock` and `packed-refs.lock` are both in that set and
//     both are exactly what a dead holder leaves behind, so one wedged lock pins the
//     fold tier at LOCKED every run, for as long as the file sits there.
//   - The GRACE-PRUNE tier additionally demands a quiet window, and its probe counts
//     EVERY file under fak's lease namespace, ghosts included — deliberately, since a
//     false "quiet" is the failure mode it exists to prevent. gitgate's own comment on
//     that probe says it: "a stale lease also blocks (reap it first)." Nobody was.
//
// Nothing reaped them, so scheduling `git-maint` alone reproduces the #4602/#4605
// failure exactly: a job that runs daily, reports success, and folds nothing, while the
// loose-object backlog grows (67,885 objects, 86% unreachable, ~2-minute cold git
// stalls) and six `refs/fak/locks/session-*.lock` ghosts sit up to 4.9 DAYS old.
// Reaping locks alone leaves the backlog. Run reaps the ghosts FIRST, then consolidates,
// in one tick — the ordering IS the feature.
//
// The lease ghosts carry a second harm the fold tier can't see: a ghost on a session's
// own ref means that session's next heartbeat CAS cannot take its lock, so a LIVE
// session can be unable to publish liveness. Reaping is what fixes that too.
//
// WHAT IT MAY TOUCH (the whole blast radius, all of it provably safe unattended):
//
//   - Orphaned `<common>/refs/fak/locks/*.lock` older than the very lease they guard
//     (leaseref, namespace-confined to fak's own side refs — never index.lock, never a
//     refs/heads lock).
//   - A frozen `.git/index.lock` and stale `.git/next-index-<pid>.lock` files only when
//     commitlane's existing two-sample/dead-owner decisions prove them abandoned — the
//     same evidence gate used by `fak commit status --reclaim-stale-index-lock`.
//   - A `.git/fak-commit.lock` whose holder PID is DEAD, and renamed-aside lock residue
//     (`*.lock.stale-*` — a name with something after `.lock`, so structurally never a
//     lock git is currently holding), via treedoctor in its LocksOnly mode.
//   - The object DB, through gitgate's tiers only: add-only index builds always;
//     redundant-copy folds when unlocked and posture-safe; and a prune ONLY under the
//     opt-in, quiet-window-gated, >=2-week-expire grace tier.
//   - Orphaned `go-build*` WORK dirs under the configured GOTMPDIR (Options.GoTmpDir),
//     via treedoctor's age-based reap. `go` deletes its own WORK dir on a clean exit, so
//     a survivor was killed; the reap is keyed on the newest file ANYWHERE inside, never
//     the top-level mtime, so a long-running build is never deleted out from under
//     itself. This is the collector half of the build-isolation redirection (#6207):
//     pointing GOTMPDIR into the tree is deliberate, but nothing ever collected it —
//     5.9 GB of orphaned WORK dirs on the reference box.
//
// Worktree pruning is deliberately NOT in the unattended path (Options.PruneWorktrees
// opts in): every reap above is decided by a dead PID or a structurally-inert filename,
// whereas a worktree prune weighs a merged/live judgement whose false positive destroys
// a peer's in-flight work.
//
// IDEMPOTENT AND SELF-DEDUPING. A tick is serialized by an advisory lock (so a manual
// run and a scheduled one cannot interleave) and skipped when the ledger already shows
// an applied run for today's LOCAL date — which is what lets an operator point a
// coarse, catch-up-on-wake OS trigger at it without fearing repeats. Every applied run
// appends one `fak-git-daily/1` ledger row, so "has this actually been folding, or
// deferring as LOCKED for a week?" is a readback (Status) rather than a guess.
//
// Like its collaborators, every git effect goes through an injected Runner, so the whole
// decision tree — dedupe, ordering, refusal plumbing — is unit-testable with no git and
// no real repo.
package gitdaily

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

// Runner runs a git command in dir, returning combined output, the exit code, and an
// error only when git could not be executed at all. It mirrors treedoctor.Runner and
// gitgate.MaintRunner so one adapter feeds all three.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// Schema is the ledger row schema tag.
const Schema = "fak-git-daily/1"

// DayLayout is the ledger's day key: the LOCAL calendar date. Local, not UTC, because
// "daily" is an operator-facing promise about their day — a UTC key would make the tick
// fire twice on one local day (or skip one) for anybody west of Greenwich.
const DayLayout = "2006-01-02"

// LedgerName / TickLockName are the tick's two files, both in the git COMMON dir: it is
// per-clone runtime state that must never be committed, is shared by every linked
// worktree, and is removed with the clone. Neither name can confuse the collaborators
// that read that directory: TickLockName is not in gitgate's transaction-lock set and
// lives outside refs/ and worktrees/ (so it cannot make the fold tier defer on itself),
// and it ends in exactly `.lock` (so treedoctor never classifies it as residue).
const (
	LedgerName   = "fak-git-daily.jsonl"
	TickLockName = "fak-git-daily.lock"
)

// LedgerMaxBytes bounds the active ledger before jsonlledger rotates it. One row is
// ~300 bytes and one run per day writes one row, so this holds decades of history.
const LedgerMaxBytes int64 = 1 << 20

// SkipReason is the closed vocabulary for a tick that deliberately did no work. Both
// values are ordinary, expected outcomes of a coarse OS trigger, never failures.
type SkipReason string

const (
	// SkipAlreadyRanToday: the ledger already carries an applied run for today's local
	// date. This is what makes an hourly/catch-up-on-wake trigger safe.
	SkipAlreadyRanToday SkipReason = "ALREADY_RAN_TODAY"
	// SkipTickBusy: another tick holds the advisory lock right now. Doing nothing is
	// correct — the holder is performing this very run.
	SkipTickBusy SkipReason = "TICK_BUSY"
)

// Options configures one daily tick.
type Options struct {
	// RepoRoot is the main checkout every git verb runs from.
	RepoRoot string
	// GitCommonDir is the shared `.git` (from `git rev-parse --git-common-dir`) holding
	// objects/, refs/, the ghost lease locks, and this tick's ledger + advisory lock.
	GitCommonDir string
	// Ledger overrides the ledger path. Empty => <GitCommonDir>/LedgerName.
	Ledger string
	// Now is the reference time for the day key and every age bound (injectable). Zero
	// => time.Now() at call.
	Now time.Time
	// Apply performs the tick. False is a DRY RUN: it reports the dedupe decision, the
	// orphan locks it WOULD reap, and the maintenance plan, mutating nothing and writing
	// no ledger row.
	Apply bool
	// Force ignores the once-a-day dedupe (an operator re-running after repairing a
	// posture incident, or a first manual proof). It never bypasses the advisory lock:
	// two concurrent mutating ticks are unsafe regardless of intent.
	Force bool
	// PruneWorktrees opts the tick INTO treedoctor's full sweep (merged/orphan worktree
	// removal) instead of the locks-only default. Off by default: see the package header.
	PruneWorktrees bool
	// GracePrune / PruneExpire are forwarded to gitgate's opt-in, default-off grace-prune
	// tier. Leaving GracePrune false means this tick can never delete an object.
	GracePrune  bool
	PruneExpire string
	// LeaseLockMaxAge is the orphan bound for refs/fak/locks/*.lock. Zero =>
	// leaseref.DefaultLockFileMaxAge (the session-heartbeat TTL).
	LeaseLockMaxAge time.Duration
	// IndexLockSweep overrides the evidence-gated .git/index.lock and
	// next-index-<pid>.lock sweep. Nil uses commitlane's production observer and
	// decisions. Tests inject a no-op or fixture-backed sweep so they need no real repo.
	IndexLockSweep IndexLockSweepFunc
	// GoTmpDir is the GOTMPDIR whose orphaned `go-build*` WORK dirs this tick collects
	// (#6207). EMPTY DISABLES THE RUNG ENTIRELY — the caller must name the directory, so a
	// clone with no redirected GOTMPDIR, and every test that does not opt in, keeps
	// today's behavior byte-for-byte and this tick can never guess at a path to delete.
	GoTmpDir string
	// GoTmpMinAge is the quiet period a WORK dir must clear before it is reapable, measured
	// on the newest file anywhere inside it. Zero => treedoctor.DefaultGoTmpMinAge.
	GoTmpMinAge time.Duration
	// GoCacheDir names the resolved Go build cache. Empty disables this lifecycle.
	GoCacheDir     string
	GoCacheOptions treedoctor.GoCacheOptions
}

// LockSweep is the lock half's outcome: the ghost lease locks and the treedoctor reaps.
type LockSweep struct {
	// LeaseReaped are the orphaned refs/fak/locks/*.lock paths removed (in a dry run,
	// the ones that WOULD be removed), relative to that directory.
	LeaseReaped []string `json:"lease_reaped,omitempty"`
	// LeaseKept are the fresh lease locks left alone — a possibly-live CAS is never raced.
	LeaseKept []string `json:"lease_kept,omitempty"`
	LeaseErr  string   `json:"lease_err,omitempty"`
	// IndexReaped are stale .git/index.lock / next-index-<pid>.lock files removed under
	// commitlane's frozen-lock evidence (in a dry run, the files that WOULD be removed).
	IndexReaped []string `json:"index_reaped,omitempty"`
	IndexErr    string   `json:"index_err,omitempty"`
	// Actions is treedoctor's action list (applied, or "would ..." in a dry run).
	Actions []string `json:"actions,omitempty"`
	// StaleCommitLock records that the commit lock was found wedged by a dead holder —
	// the failure mode that once stalled this repo's whole commit lane for 56 minutes.
	StaleCommitLock bool `json:"stale_commit_lock,omitempty"`
	// StaleRefLocks counts the ref locks (packed-refs.lock / AUTO_MERGE.lock) diagnosed
	// stale, whether or not the reap bar let them be swept.
	StaleRefLocks int `json:"stale_ref_locks,omitempty"`
}

// Cleared reports how many lock ghosts the sweep cleared (or, in a dry run, would).
// treedoctor's action stream also contains advisories and explicit FAILED lines; those
// are evidence, not successful cleanup, and must never inflate the ledger witness.
func (s LockSweep) Cleared() int {
	n := len(s.LeaseReaped) + len(s.IndexReaped)
	for _, action := range s.Actions {
		if strings.HasPrefix(action, "reaped ") || strings.HasPrefix(action, "swept ") ||
			strings.HasPrefix(action, "would reap ") || strings.HasPrefix(action, "would sweep ") {
			n++
		}
	}
	return n
}

// Failed reports whether the lock half observed a cleanup failure. The scheduled job
// must exit non-zero for this state: otherwise Task Scheduler records a successful run
// while the lock that prevents maintenance remains in place.
func (s LockSweep) Failed() bool {
	if s.LeaseErr != "" || s.IndexErr != "" {
		return true
	}
	for _, action := range s.Actions {
		if strings.HasPrefix(action, "FAILED ") {
			return true
		}
	}
	return false
}

// Result is the full structured outcome of one tick.
type Result struct {
	Schema string `json:"schema"`
	// Day is the local calendar date this tick was keyed to.
	Day string `json:"day"`
	At  string `json:"at"`
	// Apply is false for a dry run.
	Apply bool `json:"apply"`
	// Skipped is non-empty when the tick deliberately did nothing; Locks and Maint are
	// then zero-valued and NOTHING was mutated.
	Skipped SkipReason `json:"skipped,omitempty"`
	// TickLockErr is set when the serializer itself could not be opened or locked. This
	// is an incident, not TICK_BUSY: proceeding would violate the one-mutator contract,
	// while reporting a healthy contender would hide a persistent permissions failure.
	ConfigErr   string `json:"config_err,omitempty"`
	TickLockErr string `json:"tick_lock_err,omitempty"`
	// LastRunDay is the day key of the newest applied run in the ledger ("" on first run).
	LastRunDay string              `json:"last_run_day,omitempty"`
	Locks      LockSweep           `json:"locks"`
	Maint      gitgate.MaintResult `json:"maint"`
	// GoTmp is the orphaned-WORK-dir reap (#6207). Zero-valued when Options.GoTmpDir is
	// empty, which is the rung's disabled state.
	GoTmp   treedoctor.GoTmpReport   `json:"go_tmp"`
	GoCache treedoctor.GoCacheReport `json:"go_cache"`
	// Incident is true for gitgate posture drift or a lock-cleanup failure. Both need an
	// operator: the tick never edits .git/config, and a lock it could not remove can keep
	// the maintenance wedge in place.
	Incident   bool   `json:"incident"`
	LedgerPath string `json:"ledger_path"`
	LedgerErr  string `json:"ledger_err,omitempty"`
}

// Row is one appended ledger record: the small, stable projection of a Result that makes
// "is this job actually working?" answerable months later without keeping the full report.
type Row struct {
	Schema string `json:"schema"`
	Day    string `json:"day"`
	At     string `json:"at"`
	// LeaseLocksReaped / LockActions are the lock half's counts.
	LeaseLocksReaped int `json:"lease_locks_reaped"`
	IndexLocksReaped int `json:"index_locks_reaped"`
	LockActions      int `json:"lock_actions"`
	// LooseBefore / LooseAfter witness the fold. Packs may stay steady or fall when
	// incremental-repack removes a fully-covered redundant pack; both outcomes preserve
	// every reachable object.
	LooseBefore int `json:"loose_before"`
	LooseAfter  int `json:"loose_after"`
	PacksBefore int `json:"packs_before"`
	PacksAfter  int `json:"packs_after"`
	// GraceRefused / GracePruneRefused carry gitgate's structured refusal reasons, so a
	// readback distinguishes "folded nothing because there was nothing to fold" from
	// "deferred as LOCKED for nine consecutive days".
	GraceRefused      string `json:"grace_refused,omitempty"`
	GracePruneRefused string `json:"grace_prune_refused,omitempty"`
	// GoTmpReaped / GoTmpReclaimedBytes witness the orphaned-WORK-dir reap (#6207) in the
	// same ledger that already witnesses the fold, so "is the collector actually
	// collecting?" is a readback rather than another one-off `du`. Bytes are counted from
	// the dirs the filesystem actually gave up, never from the plan's intent.
	GoTmpReaped         int   `json:"gotmp_reaped,omitempty"`
	GoTmpReclaimedBytes int64 `json:"gotmp_reclaimed_bytes,omitempty"`
	// GoCache records a bounded lifecycle receipt in the same scheduled ledger. Candidate
	// paths stay in the immediate Result rather than inflating the long-lived JSONL row.
	GoCache  GoCacheLedgerReceipt `json:"go_cache,omitempty"`
	Incident bool                 `json:"incident,omitempty"`
}

// GoCacheLedgerReceipt is the bounded projection of one Go build-cache lifecycle run.
type GoCacheLedgerReceipt struct {
	Root             string   `json:"root,omitempty"`
	BytesBefore      int64    `json:"bytes_before,omitempty"`
	BytesAfter       int64    `json:"bytes_after,omitempty"`
	ReclaimedBytes   int64    `json:"reclaimed_bytes,omitempty"`
	Reaped           int      `json:"reaped,omitempty"`
	TriggeredBy      []string `json:"triggered_by,omitempty"`
	ScanComplete     bool     `json:"scan_complete"`
	IncompleteReason string   `json:"incomplete_reason,omitempty"`
	Skipped          string   `json:"skipped,omitempty"`
	Err              string   `json:"err,omitempty"`
}

func goCacheLedgerReceipt(r treedoctor.GoCacheReport) GoCacheLedgerReceipt {
	return GoCacheLedgerReceipt{
		Root:             r.Root,
		BytesBefore:      r.BytesBefore,
		BytesAfter:       r.BytesAfter,
		ReclaimedBytes:   r.ReclaimedBytes,
		Reaped:           len(r.Reaped),
		TriggeredBy:      append([]string(nil), r.TriggeredBy...),
		ScanComplete:     r.ScanComplete,
		IncompleteReason: r.IncompleteReason,
		Skipped:          r.Skipped,
		Err:              r.Err,
	}
}

// LooseFolded reports the loose objects this run folded away.
func (r Row) LooseFolded() int { return r.LooseBefore - r.LooseAfter }

// LedgerPath resolves the ledger path for a set of options.
func LedgerPath(opts Options) string {
	if p := opts.Ledger; p != "" {
		return p
	}
	return filepath.Join(opts.GitCommonDir, LedgerName)
}

// Run performs one daily tick: dedupe, then reap the lock ghosts, then consolidate the
// object DB, then witness the result. The ordering is load-bearing — see the package
// header. It returns a fully-populated Result in every path, including the skips; a
// caller reports Result.Skipped rather than treating it as an error, because a coarse OS
// trigger firing more often than daily is the DESIGN, not a fault.
func Run(ctx context.Context, run Runner, opts Options) Result {
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	ledger := LedgerPath(opts)
	res := Result{
		Schema:     Schema,
		Day:        now.Format(DayLayout),
		At:         now.Format(time.RFC3339),
		Apply:      opts.Apply,
		LedgerPath: ledger,
	}

	if err := opts.Validate(); err != nil {
		res.ConfigErr = err.Error()
		res.Incident = true
		return res
	}

	// Serialize mutating ticks. A dry run takes no lock: it mutates nothing, so it can
	// safely observe a tree a real tick is working on, and an operator previewing the
	// plan should never be told the job is busy.
	if opts.Apply {
		unlock, busy, err := acquireTick(filepath.Join(opts.GitCommonDir, TickLockName))
		if err != nil {
			res.TickLockErr = err.Error()
			res.Incident = true
			return res
		}
		if busy {
			res.Skipped = SkipTickBusy
			return res
		}
		defer unlock()
	}

	// Dedupe against the newest APPLIED run. Read after the lock so two ticks racing at
	// midnight cannot both see an empty ledger and both do the work.
	rows := Status(ledger, 0)
	if n := len(rows); n > 0 {
		res.LastRunDay = rows[n-1].Day
	}
	if !opts.Force && res.LastRunDay == res.Day {
		res.Skipped = SkipAlreadyRanToday
		return res
	}

	res.Locks = sweepLocks(ctx, run, opts, now)

	// Reclaim the orphaned build WORK dirs BEFORE the object-DB fold: repacking wants
	// free space, and on the reference box this rung is the single largest reclaim of the
	// tick (5.9 GB) while the fold moves megabytes. A GoTmp failure is reported but is
	// deliberately NOT an Incident — a WORK dir that resisted deletion costs disk and will
	// be retried tomorrow, whereas Incident means "an operator must act today", and
	// spending that signal here would train the operator to ignore it.
	res.GoTmp = treedoctor.SweepGoTmp(treedoctor.GoTmpOptions{
		Root:   opts.GoTmpDir,
		MinAge: opts.GoTmpMinAge,
		Now:    now,
	}, opts.Apply)

	cacheOpts := opts.GoCacheOptions
	cacheOpts.Root = opts.GoCacheDir
	cacheOpts.Now = now
	res.GoCache = treedoctor.SweepGoCache(cacheOpts, opts.Apply)

	res.Maint = gitgate.RunMaint(ctx, gitgate.MaintRunner(run), gitgate.MaintOptions{
		RepoRoot:     opts.RepoRoot,
		GitCommonDir: opts.GitCommonDir,
		Apply:        opts.Apply,
		GracePrune:   opts.GracePrune,
		PruneExpire:  opts.PruneExpire,
	})
	res.Incident = res.Maint.Incident || res.Locks.Failed()

	if opts.Apply {
		if err := appendRow(ledger, res.row()); err != nil {
			res.LedgerErr = err.Error()
		}
	}
	return res
}

// sweepLocks clears the lock ghosts that would otherwise make the fold tier defer:
// first the refs/fak/locks/*.lock orphans (fak's own namespace, the #4605 ghosts), then
// treedoctor's dead-holder commit lock, renamed-aside residue, and stale ref locks. In a
// dry run both halves report what they WOULD do through the same code paths that would
// do it.
func sweepLocks(ctx context.Context, run Runner, opts Options, now time.Time) LockSweep {
	var s LockSweep

	locksDir := filepath.Join(opts.GitCommonDir, "refs", "fak", "locks")
	sweep := leaseref.ReapLockFilesInDir
	if !opts.Apply {
		sweep = leaseref.ScanLockFilesInDir
	}
	reaped, kept, err := sweep(locksDir, now, opts.LeaseLockMaxAge)
	s.LeaseReaped, s.LeaseKept = reaped, kept
	if err != nil {
		s.LeaseErr = err.Error()
	}

	indexSweep := opts.IndexLockSweep
	if indexSweep == nil {
		indexSweep = SweepIndexLocks
	}
	index := indexSweep(ctx, opts.RepoRoot, now, opts.Apply)
	s.IndexReaped = index.Reaped
	s.IndexErr = index.Err

	rep, actions := treedoctor.Sweep(ctx, treedoctor.Runner(run), treedoctor.Options{
		RepoRoot:  opts.RepoRoot,
		Now:       now,
		LocksOnly: !opts.PruneWorktrees,
	}, opts.Apply)
	s.Actions = actions
	s.StaleCommitLock = rep.Lock.Stale
	s.StaleRefLocks = len(rep.StaleRefLocks())
	return s
}

// row projects a Result into its ledger record.
func (r Result) row() Row {
	return Row{
		Schema:            Schema,
		Day:               r.Day,
		At:                r.At,
		LeaseLocksReaped:  len(r.Locks.LeaseReaped),
		IndexLocksReaped:  len(r.Locks.IndexReaped),
		LockActions:       r.Locks.Cleared() - len(r.Locks.LeaseReaped) - len(r.Locks.IndexReaped),
		LooseBefore:       r.Maint.Before.Count,
		LooseAfter:        r.Maint.After.Count,
		PacksBefore:       r.Maint.Before.Packs,
		PacksAfter:        r.Maint.After.Packs,
		GraceRefused:      string(r.Maint.GraceRefused),
		GracePruneRefused: string(r.Maint.GracePruneRefused),

		GoTmpReaped:         r.GoTmp.ReapCount(),
		GoTmpReclaimedBytes: r.GoTmp.ReapedBytes,
		GoCache:             goCacheLedgerReceipt(r.GoCache),

		Incident: r.Incident,
	}
}

// Status reads the ledger back, oldest first, keeping at most the newest limit rows
// (limit <= 0 means all of them). A missing or unreadable ledger is an empty history,
// not an error — the first-run contract every ledger here shares.
func Status(path string, limit int) []Row {
	rows := jsonlledger.Parse(string(jsonlledger.ReadTail(path, LedgerMaxBytes)), func(r Row) bool {
		return r.Schema == Schema && r.Day != ""
	})
	if limit > 0 && len(rows) > limit {
		rows = rows[len(rows)-limit:]
	}
	return rows
}

// DeferredStreak counts how many of the most recent consecutive runs had their fold tier
// held back, and by which reason. It is the signal the #4602 investigation had to
// reconstruct by hand: a long LOCKED streak means the ghosts are winning and the backlog
// is growing invisibly. A zero streak means the last run folded (or had nothing to fold).
func DeferredStreak(rows []Row) (n int, reason string) {
	for i := len(rows) - 1; i >= 0; i-- {
		if rows[i].GraceRefused == "" {
			break
		}
		if reason == "" {
			reason = rows[i].GraceRefused
		} else if reason != rows[i].GraceRefused {
			break
		}
		n++
	}
	return n, reason
}

// appendRow writes one bounded JSONL record.
func appendRow(path string, row Row) error {
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	return jsonlledger.AppendBounded(path, b, LedgerMaxBytes)
}

// acquireTick takes the advisory tick lock. A contended lock is an ordinary busy result;
// an I/O failure is distinct and fail-closed. Mutating without the serializer would let
// two scheduled/manual ticks both pass the day-ledger check and append duplicate rows.
func acquireTick(path string) (release func(), busy bool, err error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, false, err
	}
	if err := flock.TryLock(f); err != nil {
		_ = f.Close()
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, true, nil
		}
		return nil, false, err
	}
	return func() {
		_ = flock.Unlock(f)
		_ = f.Close()
	}, false, nil
}
