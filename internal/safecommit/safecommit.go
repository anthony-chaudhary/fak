// Package safecommit is the EXECUTOR half of the shared-trunk commit discipline that
// internal/gitgate only declares defensively.
//
// On a multi-session shared development branch, the ordinary sequence
//
//	git add <paths>   # then, separately
//	git commit
//
// is NOT atomic: a peer session can commit in the gap between the two and either sweep
// YOUR staged file under THEIR message, or sweep THEIR staged files/deletions into YOUR
// commit. This has corrupted commits repeatedly. The hard-won manual runbook is:
//
//   - commit by explicit pathspec ON THE COMMIT (`git commit -s -F <msg> -- <paths>`),
//     never a separate `git add`;
//   - use -F <file>, never -m — an em-dash or a multi-line subject misparses as a
//     pathspec on Windows git-bash;
//   - after committing, assert that EXACTLY the requested paths landed; if any extra file
//     appears, a peer raced — surface it, never push, never force-push, never
//     `pull --rebase --autostash`.
//
// gitgate REFUSES the hazardous commands and validates a pure plan
// (gitgate.CheckCollectiveCommit) but reads no repo state and performs no commit. This
// package is the missing positive verb: it lock-guards the commit, commits by pathspec
// with the message in a file, and refuses to report success (or push) unless ONLY the
// requested paths landed. The race becomes structurally hard to hit instead of a
// discipline a human has to remember.
//
// A policy or race outcome is a Result value (Reason set), never a returned error — the
// repo's "deny-as-value, not a crash" discipline (gitgate returns a Verdict, witness an
// Outcome; safecommit a Result). The returned error is reserved for INFRASTRUCTURE
// failure only: git not executable, or the lock file unopenable.
package safecommit

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/safesync"
	"github.com/anthony-chaudhary/fak/internal/witness"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// Runner executes a git subcommand in dir and returns (stdout, exitCode, err). It is the
// SAME contract as witness.Runner: err is non-nil ONLY when git could not be EXECUTED (git
// missing); a non-zero exit with git present is reported via code, not err. Injectable so
// tests drive the whole algorithm with canned evidence and assert the exact argv issued —
// no real git or repo. Unlike witness, the default Runner CAPTURES stderr (folded into
// stdout) so a hook's refusal message can surface in Result.Detail.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// LockFunc acquires an advisory lock and returns the release closure. busy is reported as
// ErrLockBusy (mapped to the LOCK_BUSY reason, a value); any other error is infrastructure
// and propagates as the second return of CommitWith.
type LockFunc func(LockOptions) (unlock func(), err error)

// ErrLockBusy is the sentinel a LockFunc returns when the advisory lock is held by another
// fak writer. CommitWith maps it to Result{Reason: ReasonLockBusy}, never a hard error.
var ErrLockBusy = errors.New("safecommit: commit lock busy")

// LockWaitReceipt is the bounded wait evidence attached to a LOCK_BUSY refusal. It keeps
// the deadline and the last observed holder machine-readable while Detail() provides the
// same facts to the human CLI.
type LockWaitReceipt struct {
	ElapsedNS      int64 `json:"elapsed_ns"`
	DeadlineNS     int64 `json:"deadline_ns"`
	HolderPID      int   `json:"holder_pid,omitempty"`
	HolderAlive    bool  `json:"holder_alive"`
	HolderStale    bool  `json:"holder_stale"`
	HolderForeign  bool  `json:"holder_foreign"`
	LockAgeSeconds int64 `json:"lock_age_seconds,omitempty"`
}

// Detail renders the actionable, bounded refusal receipt without exposing the absolute
// lock path. A missing PID stays explicit instead of being mistaken for holder PID zero.
func (r LockWaitReceipt) Detail() string {
	detail := fmt.Sprintf("elapsed wait: %s; deadline: %s", time.Duration(r.ElapsedNS), time.Duration(r.DeadlineNS))
	if r.HolderPID <= 0 {
		return detail + "; holder pid unavailable"
	}
	holder := fmt.Sprintf("holder pid %d (alive=%t stale=%t foreign=%t)",
		r.HolderPID, r.HolderAlive, r.HolderStale, r.HolderForeign)
	if r.LockAgeSeconds > 0 {
		holder += fmt.Sprintf("; lock age=%s", time.Duration(r.LockAgeSeconds)*time.Second)
	}
	return detail + "; " + holder
}

// LockBusyError preserves errors.Is(err, ErrLockBusy) while carrying the wait receipt
// from the real advisory-lock acquisition.
type LockBusyError struct {
	Receipt LockWaitReceipt
}

func (e *LockBusyError) Error() string { return ErrLockBusy.Error() + ": " + e.Receipt.Detail() }
func (e *LockBusyError) Unwrap() error { return ErrLockBusy }

// LockOptions configures the advisory commit lock.
type LockOptions struct {
	Path    string        // "" => <Dir>/.git/fak-commit.lock
	Timeout time.Duration // 0 => DefaultLockTimeout
	NoWait  bool          // fail LOCK_BUSY immediately instead of waiting
}

// DefaultLockTimeout bounds the wait for the advisory lock before LOCK_BUSY.
const DefaultLockTimeout = 10 * time.Second

// Options is the full request to Commit / CommitWith.
type Options struct {
	Dir      string            // repo dir ("" => git discovery from cwd)
	Paths    []string          // explicit repo-relative pathspec (REQUIRED, >= 1)
	Message  string            // commit message (already assembled from -m / -F / stdin)
	Trunk    string            // expected development branch override ("" => branch_roles.development_branch)
	SignOff  bool              // add the DCO sign-off (-s)
	Push     bool              // push, but ONLY after a verified commit
	Lock     LockOptions       // advisory same-host lock
	Recorder *witness.Recorder // optional decisions-note sink for post-commit assertions
	Window   *Window           // optional adaptive process-local writer window
	Review   *ReviewOptions    // optional pre-commit cross-model review rung
	// CoreLockMaintenanceWitness is an independent witness claim that may clear a
	// hard-self core-lock pathset. Empty means ordinary in-agent hard-self edits are
	// refused before staging with CORE_SELF_MODIFY.
	CoreLockMaintenanceWitness string
	// CoreLockWitnessResolver is injectable for tests; nil uses the real git-backed
	// resolver through the same runner seam as the rest of safecommit.
	CoreLockWitnessResolver abi.WitnessResolver
	// CheckerBaseline pins the bytes of the checker(s) grading this commit. When
	// non-empty, CommitWith re-reads them immediately before any git effect and
	// refuses CHECKER_TAMPERED on drift.
	CheckerBaseline CheckerBaseline
	Now             func() time.Time // optional test clock for lock-hold measurement

	// SessionID is the acting session id. When empty, defaults to
	// FAK_SESSION_ID or CLAUDE_CODE_SESSION_ID env var.
	SessionID string
	// SessionScope specifies the explicit paths/files claimed by this session.
	SessionScope []string
	// PeerWIP maps file paths to owning peer session IDs.
	PeerWIP map[string]string
	// PeerWIPChecker is an optional injectable checker for tests or custom attribution.
	PeerWIPChecker func(path string) (peerSession string, isPeer bool)
	// RestrictToSessionScope, when true, automatically filters expanded directory paths
	// to only those within SessionScope instead of refusing. When false (default),
	// sweeps of peer/unscoped paths under a directory pathspec are refused.
	RestrictToSessionScope bool
}

type ReviewFunc func(context.Context, modelroute.ReviewRequest) (modelroute.ReviewResult, error)

type ReviewOptions struct {
	Model     string
	Objective string
	Reviewer  ReviewFunc
}

// DefaultTrunk is the fallback branch when branch-role config cannot be read.
const DefaultTrunk = "main"

// ExpectedTrunk resolves the branch fak commits should land on. An explicit
// override wins; otherwise the configured development branch is used.
func ExpectedTrunk(dir, override string) string {
	trunk := strings.TrimSpace(override)
	if trunk != "" {
		return trunk
	}
	roles, err := branchrole.Load(dir)
	if err == nil && strings.TrimSpace(roles.DevelopmentBranch) != "" {
		return roles.DevelopmentBranch
	}
	return DefaultTrunk
}

// Reason tokens — the closed, checkable vocabulary the executor stamps into Result.Reason
// and the --json contract a calling loop consumes. Local string constants, the same shape
// session/decide.go's ReasonBudget* family uses; the frozen abi.ReasonCode enum is left
// untouched (a CLI executor's reasons do not belong in the additive-only ABI).
const (
	ReasonNoPath          = "NO_PATHS"          // empty pathspec — the executor dual of gitgate's `add .`/`-a` refusal
	ReasonEmptyMessage    = "EMPTY_MESSAGE"     // blank commit message
	ReasonNotARepo        = "NOT_A_REPO"        // not inside a git work tree
	ReasonOffTrunk        = "OFF_TRUNK"         // HEAD is not the expected trunk (or detached)
	ReasonMergeInProgress = "MERGE_IN_PROGRESS" // a merge is mid-flight; a partial path commit would fail
	ReasonNothingStaged   = "NOTHING_STAGED"    // the pathspec has no change to commit
	ReasonLockBusy        = "LOCK_BUSY"         // another fak writer holds the commit lock (retryable)
	ReasonWindowFull      = "WINDOW_FULL"       // adaptive writer window is full (retryable)
	ReasonHookRefused     = "HOOK_REFUSED"      // git/commit-hook refused the commit (exit != 0)
	ReasonPathspecRace    = "PATHSPEC_RACE"     // a peer swept extra files into the commit — the headline guard
	ReasonMessageRace     = "MESSAGE_RACE"      // landed subject/body differs from the requested message — commit left intact
	ReasonSymlinkEscape   = "SYMLINK_ESCAPE"    // a landed path resolves (through a symlink) to a target outside the lease
	ReasonPushRejected    = "PUSH_REJECTED"     // git push refused (e.g. non-fast-forward)
	ReasonReviewRefuted   = "REVIEW_REFUTED"    // opt-in scout review refuted the diff before commit
	ReasonCoreSelfModify  = "CORE_SELF_MODIFY"  // hard-self core-lock path requires external maintenance witness
	// ReasonPreStagedPathOverlap ("PRESTAGED_PATH_OVERLAP") is part of this vocabulary too;
	// it lives in prestaged.go with the same-file staged-hunk ambiguity guard.
	// ReasonStaleBaseDeletion ("STALE_BASE_DELETION") is part of this closed vocabulary too;
	// it lives in stalebase.go with the content-level merge-base guard that emits it.
	// ReasonSpuriousStagedDeletion ("SPURIOUS_STAGED_DELETION") likewise lives in
	// spuriousdelete.go with the whole-path stale-index guard that emits it.
	// ReasonCachedRemoveWorktreePresent ("CACHED_REMOVE_WORKTREE_PRESENT") lives in
	// cachedremove.go with the git-rm-cached/pathspec guard that emits it.
	// ReasonWriterLeaseHeld ("WRITER_LEASE_HELD") lives in writer_lease.go with the
	// #4240 worktree-writer-lease wiring (#4611) that emits it.
)

// Result is the structured outcome. A non-empty Reason is a refusal/race; a clean commit
// has Committed && Verified && Reason == "". RacedExtra lists the committed files that NO
// requested path covers — the evidence of a raced commit.
type Result struct {
	Committed bool     `json:"committed"`
	SHA       string   `json:"committed_sha,omitempty"`
	Paths     []string `json:"paths"`
	Verified  bool     `json:"verified"`
	Pushed    bool     `json:"pushed"`
	// Evidence is the versioned completion contract. When present, Verified is its derived
	// aggregate; when absent, Verified retains the legacy executor diff-witness contract.
	Evidence         *CommitEvidence  `json:"evidence,omitempty"`
	Value            float64          `json:"value"`
	ValueUnit        string           `json:"value_unit,omitempty"`
	Score            int              `json:"score"`
	LegacyScore      int              `json:"legacy_score,omitempty"`
	LegacyScoreScale int              `json:"legacy_score_scale,omitempty"`
	Grade            string           `json:"grade"`
	ScoreNotes       []string         `json:"score_notes,omitempty"`
	Reason           string           `json:"reason,omitempty"`
	Detail           string           `json:"detail,omitempty"`
	RacedExtra       []string         `json:"raced_extra_paths,omitempty"`
	HeadBefore       string           `json:"head_before,omitempty"`
	LockHoldNS       int64            `json:"lock_hold_ns,omitempty"`
	LockWait         *LockWaitReceipt `json:"lock_wait,omitempty"`
	CoreLockPaths    []string         `json:"core_lock_paths,omitempty"`
	CoreLockWitness  string           `json:"core_lock_witness,omitempty"`
	// CoreLockWitnessCorrelation is whether CoreLockWitness actually named a path
	// this commit changed ("correlated" / "uncorrelated" / "indeterminate", plus
	// why). A CONFIRMED witness is not automatically a RELEVANT one; this is the
	// reading that says which, and it is recorded on the maintenance decision note.
	CoreLockWitnessCorrelation string                   `json:"core_lock_witness_correlation,omitempty"`
	Review                     *modelroute.ReviewResult `json:"review,omitempty"`
	PeerCollisions             []string                 `json:"peer_collisions,omitempty"`
	// BuildCheck is what the COMMITTED_RED prospective-tree compile gate DID (#6006): passed,
	// failed, or skipped — and, when skipped, whether the commit was admitted anyway. The gate
	// runs in cmd/fak before the executor, so CommitWith never sets this; the caller attaches
	// it to the result it emits. Absent means the caller ran no gate at all, which is itself
	// distinguishable from a gate that ran and passed (buildcheck.go).
	BuildCheck *BuildCheckResult `json:"build_check,omitempty"`
	// Delivery records only the authoring/recording transition. Compile admission, verification,
	// integration, and release readiness remain independent receipts.
	Delivery *workdelivery.AdapterObservation `json:"delivery,omitempty"`
	// Velocity is the effect-qualified ship-speed reading (#4241): separate local and
	// push legs, each scored only after the command's authoritative effect fields
	// qualify it (Committed&&Verified for local, additionally Pushed for push). It is
	// distinct from Score (outcome quality) and always populated by CommitWith; a
	// refusal/no-op still carries it with UNSCORED legs and retained timing.
	Velocity *CommitVelocity `json:"velocity,omitempty"`
}

// Commit runs the safe-commit algorithm against the real git binary and a real advisory
// flock (gpulease) on <Dir>/.git/fak-commit.lock. It is the thin production wiring around
// CommitWith.
func Commit(ctx context.Context, opts Options) (Result, error) {
	if opts.Recorder == nil {
		opts.Recorder = witness.NewRecorderWithRunner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			return realRunner(ctx, dir, args...)
		}, opts.Dir)
	}
	if opts.Window == nil {
		opts.Window = DefaultWindow
	}
	return CommitWith(ctx, realRunner, realLock, opts)
}

// buildCommitArgs assembles the `git commit` argv: verbatim cleanup, an optional -s sign-off,
// the message file, and the explicit pathspec. Split out of CommitWith as a pure arg-assembly
// phase — no git, no I/O.
func buildCommitArgs(signOff bool, msgPath string, paths []string) []string {
	commitArgs := []string{"commit", "--cleanup=verbatim"}
	if signOff {
		commitArgs = append(commitArgs, "-s")
	}
	commitArgs = append(commitArgs, "-F", msgPath, "--")
	commitArgs = append(commitArgs, paths...)
	return commitArgs
}

// CommitWith is the testable core: every effect goes through the injected run and lock, so
// a fake Runner + fake LockFunc exercise the whole step-ordered algorithm — including the
// race remedy — with no git and no repo. See the package doc for the discipline it encodes.
func CommitWith(ctx context.Context, run Runner, lock LockFunc, opts Options) (res Result, err error) {
	// Authoritative clock for the effect-qualified velocity legs (#4241). It shares
	// opts.Now with the lock-hold timer so a test's injected clock drives both, and is
	// read ONLY after the lock is acquired — never on a pre-lock refusal path — so a
	// fast no-op cannot manufacture an elapsed reading.
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	var velStart time.Time
	var velStarted bool
	var localElapsed, pushElapsed time.Duration
	var localStamped, pushStamped bool
	defer func() {
		res = ScoreResult(res)
		// Legs that never reached their qualifying boundary still retain timing: an
		// unstamped leg is measured to the terminal instant so a refusal/race reports
		// how long it took, while ScoreCommitVelocity keeps it UNSCORED (nil score).
		localE, pushE := localElapsed, pushElapsed
		if velStarted && (!localStamped || !pushStamped) {
			term := now()
			if !localStamped {
				localE = term.Sub(velStart)
			}
			if !pushStamped {
				pushE = term.Sub(velStart)
			}
		}
		v := ScoreCommitVelocity(res, localE, pushE, DefaultVelocityBudgets)
		res.Velocity = &v
	}()

	trunk := ExpectedTrunk(opts.Dir, opts.Trunk)

	// (0) Normalize + validate — pure, no git. Share gitgate's ONE path rule so the
	// executor and the policy agree on what a repo path is.
	paths, ok := normalizePaths(opts.Paths)
	res = Result{Paths: paths}
	if !ok || len(paths) == 0 {
		res.Reason = ReasonNoPath
		return res, nil
	}
	if strings.TrimSpace(opts.Message) == "" {
		res.Reason = ReasonEmptyMessage
		return res, nil
	}
	if reason, refused := GuardCheckerPin(opts.Dir, opts.CheckerBaseline); refused {
		res.Reason = reason
		res.Detail = "declared checker bytes drifted since task declaration"
		return res, nil
	}
	if release, admitted := opts.Window.TryAcquire(); !admitted {
		res.Reason = ReasonWindowFull
		res.Detail = "adaptive commit window is full; retry after an in-flight writer finishes"
		return res, nil
	} else if release != nil {
		defer func() { release(res) }()
	}

	// (1)-(4c) Pure, lock-free refusal checks before any lock or `git add`: in a work tree,
	// on the expected trunk, no merge mid-flight, the pathspec has a change, and the stale-base
	// / spurious-staged-deletion content guards. A refusal returns the annotated Result as-is
	// (the window-release defer above still fires); the rationale lives on precommitGates.
	if r, refused, gerr := precommitGates(ctx, run, opts, trunk, paths, res); gerr != nil || refused {
		res = r
		return res, gerr
	} else {
		res = r
		paths = res.Paths
	}

	var recordPathspec bool
	var recordVerdict, recordReason, recordAssertion string
	// Registered before the lock's defer so Go's LIFO unwind releases the lock first.
	defer func() {
		if recordPathspec {
			recordPathspecAssertion(ctx, opts, res, recordVerdict, recordReason, recordAssertion)
		}
		recordCoreLockMaintenance(ctx, opts, res)
	}()

	if reviewEnabled(opts.Review) {
		review := runPreCommitReview(ctx, run, opts.Dir, paths, opts.Review)
		res.Review = &review
		if review.Verdict == modelroute.ReviewRefute {
			res.Reason = ReasonReviewRefuted
			res.Detail = review.Reason
			return res, nil
		}
	}

	// (5) Acquire the advisory lock (bounded). Busy is a value, not an error.
	releaseLock, lockStart, busyReason, lockErr := acquireCommitLock(lock, opts, &res)
	if lockErr != nil {
		return res, lockErr
	}
	if busyReason != "" {
		res.Reason = busyReason
		return res, nil
	}
	defer releaseLock()
	// The velocity clock starts at lock acquisition (#4241): only work that held the
	// lock has an effect boundary to time, so velStart is set past the busy return.
	velStart, velStarted = lockStart, true

	// (5b) Honor the cooperative worktree writer lease (#4240 → #4611): sync apply holds
	// it across its whole assess+apply window, and a managed commit must never mutate the
	// tree mid-window. Held is a retryable value (WRITER_LEASE_HELD), not an error; on
	// success the lease is held for the rest of the mutation window so the refusal is
	// symmetric — a concurrent sync apply is refused while this commit writes.
	releaseWriterLease, leaseHeldDetail, wlErr := acquireWorktreeWriterLease(opts)
	if wlErr != nil {
		return res, wlErr
	}
	if leaseHeldDetail != "" {
		res.Reason = ReasonWriterLeaseHeld
		res.Detail = leaseHeldDetail
		return res, nil
	}
	defer releaseWriterLease()

	// (6) Capture HEAD, then commit by pathspec with the message in a file.
	if sha, herr := headSHA(ctx, run, opts.Dir); herr != nil {
		return res, herr
	} else {
		res.HeadBefore = sha
	}

	if augmented, changed, aerr := autoIndexDatedNotes(ctx, run, opts.Dir, paths); aerr != nil {
		return res, fmt.Errorf("safecommit: auto-index notes: %w", aerr)
	} else if changed {
		paths = augmented
		res.Paths = append([]string(nil), paths...)
	}

	// Stage EXACTLY the requested paths, inside the lock, with an explicit pathspec — never
	// an unscoped `git add -A`/`.` (which would sweep a peer's tree). `--all` is deliberately
	// pathspec-scoped here: it stages additions, edits, and deletions for the requested paths,
	// including a path already removed from the index by `git rm`, without touching any other
	// dirty file. The post-commit assertion (step 7) remains the authority — a peer who raced
	// between this add and the commit is caught there.
	addArgs := append([]string{"add", "--all", "--"}, paths...)
	if reason, detail, aerr := runLockRidingMutation(ctx, run, opts.Dir, addArgs); aerr != nil {
		return res, aerr
	} else if reason != "" {
		res.Reason = reason
		res.Detail = detail
		return res, nil
	}

	msgPath, cleanup, err := writeMessageFile(opts.Message)
	if err != nil {
		return res, fmt.Errorf("safecommit: write message file: %w", err)
	}
	defer cleanup()

	commitArgs := buildCommitArgs(opts.SignOff, msgPath, paths)
	if reason, detail, cerr := runLockRidingMutation(ctx, run, opts.Dir, commitArgs); cerr != nil {
		return res, cerr
	} else if reason != "" {
		res.Reason = reason
		res.Detail = detail
		return res, nil
	}

	verification, verr := verifyCommittedEffect(ctx, run, opts, paths, &res)
	if verification.record {
		recordPathspec = true
		recordVerdict, recordReason, recordAssertion = verification.verdict, verification.reason, verification.assertion
	}
	if verr != nil || verification.stop {
		return res, verr
	}
	// Local effect boundary reached: stamp the local velocity leg at the verified-commit
	// instant (#4241), before the lock is released and any push is attempted.
	localElapsed, localStamped = now().Sub(velStart), true
	recordPathspec = true
	recordVerdict, recordReason, recordAssertion = witness.VerdictAssertPass, "", "committed-set==requested-set"

	// The correctness-critical window is done once HEAD was captured and the committed
	// pathset was verified. Release before any remote I/O; a slow push must not stall the
	// shared same-host commit lane.
	releaseLock()

	// (8) Optional push — only after a verified commit, by exact SHA refspec (never --force).
	// Pushing the verified SHA, rather than the mutable branch tip after unlock, prevents a
	// peer's later local commit from being swept into this push. The push itself goes through
	// safesync.SafePush so transient transport/non-ff races get the same retry and
	// integrate-through-`fak sync apply` guidance as `fak sync push`. We never pull --rebase
	// --autostash (it strands .git/rebase-merge).
	res, err = applyVerifiedPush(ctx, run, opts, trunk, res)
	// Push effect boundary reached only on a verified push: stamp the push velocity leg
	// when the commit actually landed on the remote (#4241). A rejected push leaves the
	// leg unstamped, so it retains terminal timing but stays UNSCORED.
	if err == nil && res.Pushed {
		pushElapsed, pushStamped = now().Sub(velStart), true
	}
	return res, err
}

type commitVerification struct {
	record                     bool
	verdict, reason, assertion string
	stop                       bool
}

// verifyCommittedEffect performs the post-commit path, symlink, and message assertions.
func verifyCommittedEffect(ctx context.Context, run Runner, opts Options, paths []string, res *Result) (commitVerification, error) {
	if sha, err := headSHA(ctx, run, opts.Dir); err != nil {
		return commitVerification{}, err
	} else {
		res.SHA = sha
	}
	res.Committed = true

	landed, _, err := run(ctx, opts.Dir, "diff-tree", "--no-commit-id", "--name-only", "--no-renames", "-r", "HEAD")
	if err != nil {
		return commitVerification{}, fmt.Errorf("safecommit: git not executable: %w", err)
	}
	if extra := racedExtra(landed, paths); len(extra) > 0 {
		res.Reason = ReasonPathspecRace
		res.RacedExtra = extra
		res.Detail = "extra files landed in this commit — a peer raced; commit left intact for review, not pushed"
		return commitVerification{record: true, verdict: witness.VerdictAssertFail, reason: ReasonPathspecRace, assertion: "committed-set!=requested-set", stop: true}, nil
	}
	if escaped := landedEscapesLease(opts.Dir, landed, paths); len(escaped) > 0 {
		res.Reason = ReasonSymlinkEscape
		res.RacedExtra = escaped
		res.Detail = "a landed path resolves through a symlink to a target outside the lease; commit left intact for review, not pushed"
		return commitVerification{record: true, verdict: witness.VerdictAssertFail, reason: ReasonSymlinkEscape, assertion: "resolved-targets-within-requested-set=false", stop: true}, nil
	}
	landedMsg, code, err := run(ctx, opts.Dir, "log", "-1", "--format=%B")
	if err != nil {
		return commitVerification{}, fmt.Errorf("safecommit: git not executable: %w", err)
	}
	if code != 0 {
		res.Reason = ReasonMessageRace
		res.Detail = "could not read landed commit message for verification; commit left intact for review, not pushed"
		return commitVerification{record: true, verdict: witness.VerdictAssertFail, reason: ReasonMessageRace, assertion: "committed-message-readable=false", stop: true}, nil
	}
	if !commitMessagesMatch(landedMsg, opts.Message) {
		res.Reason = ReasonMessageRace
		res.Detail = "landed commit message does not match the requested message; commit left intact for review, not pushed"
		return commitVerification{record: true, verdict: witness.VerdictAssertFail, reason: ReasonMessageRace, assertion: "committed-message==requested-message=false", stop: true}, nil
	}
	res.Verified = true
	return commitVerification{record: true, verdict: witness.VerdictAssertPass, assertion: "committed-set==requested-set"}, nil
}

// applyVerifiedPush performs step (8): the optional push of the already-verified commit. It
// pushes only when opts.Push is set, by exact SHA refspec through pushVerifiedCommit, and maps
// a rejected push to ReasonPushRejected (a value, never a force-push). Extracted verbatim from
// CommitWith so the executor core stays under its ceiling; the returned Result/err on every
// branch are exactly what CommitWith previously returned.
func applyVerifiedPush(ctx context.Context, run Runner, opts Options, trunk string, res Result) (Result, error) {
	if opts.Push {
		pushed, err := pushVerifiedCommit(ctx, run, opts.Dir, trunk, res.SHA)
		if err != nil {
			return res, err
		}
		if !pushed.Pushed {
			res.Reason = ReasonPushRejected
			res.Detail = trimDetail(pushed.Detail)
			if res.Detail == "" {
				res.Detail = pushed.Reason
			}
			return res, nil
		}
		res.Pushed = true
	}

	return res, nil
}

// acquireCommitLock performs step (5): acquire the advisory lock (bounded) and build the
// idempotent release closure that stamps the hold duration onto the caller's Result. Busy
// is a value, not an error — a non-empty busyReason (ReasonLockBusy) with a nil err; any
// other lock failure is infrastructure and returns as err. Extracted verbatim from
// CommitWith so the executor core stays under its ceiling; res is a pointer to the
// caller's named result so the release closure records LockHoldNS exactly where the
// original in-function closure did.
func acquireCommitLock(lock LockFunc, opts Options, res *Result) (releaseLock func(), lockStart time.Time, busyReason string, err error) {
	unlock, err := lock(opts.Lock)
	if err != nil {
		if errors.Is(err, ErrLockBusy) {
			var busy *LockBusyError
			if errors.As(err, &busy) {
				receipt := busy.Receipt
				res.LockWait = &receipt
				res.Detail = receipt.Detail()
			} else {
				res.Detail = "elapsed wait unavailable; deadline unavailable; holder evidence unavailable"
			}
			return nil, time.Time{}, ReasonLockBusy, nil
		}
		return nil, time.Time{}, "", fmt.Errorf("safecommit: lock: %w", err)
	}
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	// lockStart is the lock-acquisition instant: it anchors both the lock-hold timer
	// and the velocity legs' start (#4241), so the clock is read exactly once here and
	// never before the lock is held.
	lockStart = now()
	lockReleased := false
	release := func() {
		if lockReleased {
			return
		}
		held := now().Sub(lockStart)
		if held > 0 {
			res.LockHoldNS = held.Nanoseconds()
		}
		unlock()
		lockReleased = true
	}
	return release, lockStart, "", nil
}

// runLockRidingMutation runs one lock-riding git mutation (the pathspec-scoped add, then
// the commit — extracted verbatim from CommitWith, shared by both steps) and maps a
// non-zero exit to the transient-vs-permanent split: a peer's raw git holding index.lock
// (or a ref lock) is TRANSIENT contention — after the in-place retries are spent it
// surfaces as the retryable LOCK_BUSY, never the halt-class HOOK_REFUSED, which is
// reserved for a genuine hook refusal. The returned err is non-nil only when git itself
// could not be executed; on success both reason and err are empty.
func runLockRidingMutation(ctx context.Context, run Runner, dir string, args []string) (reason, detail string, err error) {
	out, code, rerr := runRidingLockContention(ctx, run, dir, args...)
	// Stale-index-lock auto-recovery (#3915): contention that outlives the in-place
	// retries and names the INDEX lock may be a crashed writer's abandoned lock,
	// which never clears on its own and would otherwise force a manual `rm`. Reap it
	// when it is provably stale and retry once; when it is fresh (a live git may
	// hold it), report a precise "another git process is active" message instead of
	// git's generic crash text. Any other failure class is unaffected.
	if rerr == nil && code != 0 && isIndexLockContention(out) {
		if r, d, e, handled := recoverStaleIndexLock(ctx, run, dir, args); handled {
			return r, d, e
		}
	}
	return classifyMutation(out, code, rerr)
}

// classifyMutation maps a finished git mutation (out, exit code, exec err) to the
// safecommit reason/detail split: an exec failure is an infrastructure error; a
// clean exit is success; a lock-contention non-zero is the retryable LOCK_BUSY;
// anything else is the halt-class HOOK_REFUSED.
func classifyMutation(out string, code int, rerr error) (reason, detail string, err error) {
	if rerr != nil {
		return "", "", fmt.Errorf("safecommit: git not executable: %w", rerr)
	}
	if code == 0 {
		return "", "", nil
	}
	if isGitLockContention(out) {
		return ReasonLockBusy, trimDetail(out), nil
	}
	return ReasonHookRefused, trimDetail(out), nil
}

func pushVerifiedCommit(ctx context.Context, run Runner, dir, trunk, sha string) (safesync.PushResult, error) {
	sha = strings.TrimSpace(sha)
	if sha == "" {
		return safesync.PushResult{Reason: safesync.PushReasonError, Detail: "verified commit SHA missing; not pushed"}, nil
	}
	remote := gitConfigValue(ctx, run, dir, "branch."+trunk+".remote")
	if remote == "" {
		remote = "origin"
	}
	mergeRef := gitConfigValue(ctx, run, dir, "branch."+trunk+".merge")
	if mergeRef == "" {
		mergeRef = "refs/heads/" + trunk
	}
	return safesync.SafePush(ctx, safesync.PushOptions{
		Repo:      dir,
		Remote:    remote,
		Branch:    branchFromMergeRef(mergeRef, trunk),
		SourceRef: sha,
		TargetRef: mergeRef,
		Runner:    safeSyncRunner(run),
	})
}

func safeSyncRunner(run Runner) safesync.Runner {
	return func(ctx context.Context, repo string, args ...string) safesync.RunResult {
		out, code, err := run(ctx, repo, args...)
		b := []byte(out)
		return safesync.RunResult{Stdout: b, Stderr: b, Code: code, Err: err}
	}
}

func branchFromMergeRef(mergeRef, fallback string) string {
	const prefix = "refs/heads/"
	mergeRef = strings.TrimSpace(mergeRef)
	if branch, ok := strings.CutPrefix(mergeRef, prefix); ok && strings.TrimSpace(branch) != "" {
		return branch
	}
	return fallback
}

func gitConfigValue(ctx context.Context, run Runner, dir, key string) string {
	out, code, err := run(ctx, dir, "config", "--get", key)
	if err != nil || code != 0 {
		return ""
	}
	return strings.TrimSpace(out)
}

// headSHA runs `git rev-parse HEAD` in dir and returns the trimmed SHA. A non-zero exit
// is not an error — an empty repo has no HEAD yet — it simply yields "". A failure to
// exec git itself is the infrastructure error the caller returns as-is.
func headSHA(ctx context.Context, run Runner, dir string) (string, error) {
	head, code, err := run(ctx, dir, "rev-parse", "HEAD")
	if err != nil {
		return "", fmt.Errorf("safecommit: git not executable: %w", err)
	}
	if code != 0 {
		return "", nil
	}
	return strings.TrimSpace(head), nil
}

// precommitGates runs the pure, lock-free refusal checks before any lock or `git add`
// (steps 1-4c) and returns the (possibly Detail-annotated, for warn modes) Result with
// refused=true when a gate declined — the caller then returns res unchanged. A git
// executable failure surfaces as a non-nil error.
func precommitGates(ctx context.Context, run Runner, opts Options, trunk string, paths []string, res Result) (Result, bool, error) {
	// (1) In a work tree?
	if _, code, err := run(ctx, opts.Dir, "rev-parse", "--git-dir"); err != nil {
		return res, false, fmt.Errorf("safecommit: git not executable: %w", err)
	} else if code != 0 {
		res.Reason = ReasonNotARepo
		return res, true, nil
	}

	// (2) On the expected trunk? symbolic-ref exits non-zero on a detached HEAD rather
	// than printing the literal "HEAD", so this rejects detached state too.
	branch, code, err := run(ctx, opts.Dir, "symbolic-ref", "--short", "HEAD")
	if err != nil {
		return res, false, fmt.Errorf("safecommit: git not executable: %w", err)
	}
	branch = strings.TrimSpace(branch)
	isSanctionedWorker := false
	if code != 0 {
		if workerworktree.IsWorkerWorktree(opts.Dir) {
			isSanctionedWorker = true
		} else if cwd, err := os.Getwd(); err == nil && workerworktree.IsWorkerWorktree(cwd) {
			isSanctionedWorker = true
		} else if isSanctionedWorkerWorktreeDir(opts.Dir) || isSanctionedWorkerWorktreeDir("") {
			isSanctionedWorker = true
		}
	}
	if !isSanctionedWorker && (code != 0 || branch != trunk) {
		res.Reason = ReasonOffTrunk
		// A non-zero symbolic-ref is a detached HEAD; the captured output is git's stderr
		// ("fatal: ref HEAD is not a symbolic ref"), not a branch name — don't echo it.
		if code != 0 || branch == "" {
			branch = "detached HEAD"
		}
		res.Detail = fmt.Sprintf("on %s, expected development branch %s", branch, trunk)
		return res, true, nil
	}

	// (3) A merge mid-flight makes a partial path-scoped commit fail ("cannot do a partial
	// commit during a merge"). Refuse with a clear reason rather than block on the lock —
	// the flock guards fak writers, not a peer's raw merge.
	if out, _, err := run(ctx, opts.Dir, "rev-parse", "-q", "--verify", "MERGE_HEAD"); err != nil {
		return res, false, fmt.Errorf("safecommit: git not executable: %w", err)
	} else if strings.TrimSpace(out) != "" {
		res.Reason = ReasonMergeInProgress
		res.Detail = "a merge is in progress (MERGE_HEAD present); resolve it before committing by path"
		return res, true, nil
	}

	// (4) Does the pathspec actually have a change? Fail fast, lock-free; never
	// --allow-empty. Advisory only — step 7 is the authoritative check.
	statusArgs := append([]string{"status", "--porcelain", "--"}, paths...)
	statusOut := ""
	if out, _, err := run(ctx, opts.Dir, statusArgs...); err != nil {
		return res, false, fmt.Errorf("safecommit: git not executable: %w", err)
	} else if strings.TrimSpace(out) == "" {
		res.Reason = ReasonNothingStaged
		return res, true, nil
	} else {
		statusOut = out
	}

	// (4a) Path attribution & directory sweep guard (#11232).
	// When a pathspec is a directory (e.g. `internal/gateway`), git status matches hierarchically.
	// Cross-reference changed paths with session scope / peer WIP: refuse if a peer's modified
	// or untracked file sits under the directory pathspec, or restrict to session's explicit files.
	if mode := peerWIPGuardMode(); mode != staleBaseOff {
		attrOpts := PathAttributionOptions{
			SessionID:              opts.SessionID,
			SessionScope:           opts.SessionScope,
			PeerWIP:                opts.PeerWIP,
			PeerWIPChecker:         opts.PeerWIPChecker,
			RestrictToSessionScope: opts.RestrictToSessionScope,
		}
		attrRes, aerr := checkPathAttributionFromStatus(ctx, run, opts.Dir, paths, statusOut, attrOpts)
		if aerr != nil {
			return res, false, aerr
		}
		if !attrRes.OK {
			if mode == staleBaseWarn {
				res.Detail = appendDetail(res.Detail, "PEER_WIP_COLLISION (warn): "+attrRes.Detail)
			} else {
				res.Reason = ReasonPeerWIPCollision
				res.Detail = attrRes.Detail
				res.PeerCollisions = attrRes.CollidingPaths
				return res, true, nil
			}
		}
		if len(attrRes.EffectivePaths) > 0 && opts.RestrictToSessionScope {
			paths = attrRes.EffectivePaths
			res.Paths = append([]string(nil), paths...)
			res.PeerCollisions = attrRes.CollidingPaths
		}
	}

	changedForCoreLock := statusChangedPaths(statusOut)
	if len(changedForCoreLock) == 0 {
		changedForCoreLock = paths
	}
	if len(paths) > 0 && opts.RestrictToSessionScope {
		changedForCoreLock = paths
	}
	if f, ok := coreLockHardSelfFinding(changedForCoreLock); ok {
		detail, fired, corr := checkCoreLockHardSelf(ctx, run, opts, changedForCoreLock)
		if fired {
			res.Reason = ReasonCoreSelfModify
			res.Detail = detail
			return res, true, nil
		}
		res.CoreLockPaths = append([]string(nil), f.Paths...)
		res.CoreLockWitness = strings.TrimSpace(opts.CoreLockMaintenanceWitness)
		// The witness cleared the lock; whether it named anything this commit
		// actually changes is a SEPARATE question, answered here and recorded.
		res.CoreLockWitnessCorrelation = corr.String()
	}

	// (4a2) STALE_UNTRACKED — blob-identity, lock-free, before any `git add` (#5408). A path
	// this checkout has never indexed can ALREADY exist on origin/<trunk> when HEAD has fallen
	// behind: `git status` reports it `??`, indistinguishable from new work by local index
	// state alone, and the pathspec commit then lands a working-tree copy that predates the
	// trunk. Only a DIFFERING copy is refused; a byte-identical one supersedes nothing, so it
	// is named in Detail and allowed through (40 of the 69 untracked paths measured in #5408
	// were that no-op case — refusing them would block more honest work than it protects).
	//
	// It runs BEFORE (4b), and whatever it claims is WITHHELD from (4b): (4b)'s line-run
	// reading also trips on this class but describes it wrongly — for an untracked path
	// `git diff origin/<trunk> -- P` shows trunk's blob as wholly deleted, so its "would drop
	// N line(s)" is just trunk's own line count. Reads the already-present-locally
	// remote-tracking ref only (no fetch); every unknown falls back to the prior behavior by
	// leaving the path for (4b). Shares (4b)'s block|warn|off escape.
	staleBasePaths := paths
	if mode := staleBaseGuardMode(); mode != staleBaseOff {
		refusal, advisory, unclaimed := checkStaleUntrackedPath(ctx, run, opts.Dir, trunk, paths)
		staleBasePaths = unclaimed
		if advisory != "" {
			res.Detail = appendDetail(res.Detail, "STALE_UNTRACKED (no-op): "+advisory)
		}
		if refusal != "" {
			if mode == staleBaseWarn {
				res.Detail = appendDetail(res.Detail, "STALE_UNTRACKED (warn): "+refusal)
			} else {
				res.Reason = ReasonStaleUntrackedPath
				res.Detail = appendDetail(res.Detail, refusal)
				return res, true, nil
			}
		}
	}

	// (4b) STALE-BASE-DELETION guard — content-level, lock-free, before any `git add`. The
	// pathspec commit lands the WORKING-TREE blob of each requested path; if that blob predates
	// a block a peer already pushed to origin/<trunk>, the commit SILENTLY deletes the peer's
	// lines (the #1073 incident). PATHSPEC_RACE (step 7) is structurally blind to this — the
	// stale file is one of MY OWN requested paths, inside the set it filters out. This guard
	// reads the already-present-locally origin/<trunk> ref (no network fetch) and refuses if
	// committing P would drop a contiguous peer-added run absent from the working tree. It runs
	// before the lock and before any add, so a refusal stages and commits NOTHING — strictly
	// cleaner than PATHSPEC_RACE, which leaves a commit behind. Gated by FAK_STALE_BASE_GUARD
	// (block|warn|off, default block); off skips entirely, warn records the would-be refusal in
	// Detail and proceeds. It judges only the paths (4a2) left unclaimed.
	if mode := staleBaseGuardMode(); mode != staleBaseOff && len(staleBasePaths) > 0 {
		if detail, fired := checkStaleBaseDeletion(ctx, run, opts.Dir, trunk, staleBasePaths); fired {
			if mode == staleBaseWarn {
				res.Detail = appendDetail(res.Detail, "STALE_BASE_DELETION (warn): "+detail)
			} else {
				res.Reason = ReasonStaleBaseDeletion
				res.Detail = detail
				return res, true, nil
			}
		}
	}

	// (4c) CACHED-REMOVE-WORKTREE-PRESENT guard — index-vs-worktree, lock-free, before
	// any `git add`. `git rm --cached <path>` stages an index deletion while leaving the
	// working file in place; a pathspec commit then reads/re-stages that file and silently
	// clears the deletion instead of recording the intended untrack/delete operation. Gated
	// by FAK_CACHED_REMOVE_GUARD (block|warn|off, default block); off skips, warn records
	// and proceeds.
	if mode := cachedRemoveGuardMode(); mode != staleBaseOff {
		if detail, fired := checkCachedRemoveWorktreePresent(ctx, run, opts.Dir, paths); fired {
			if mode == staleBaseWarn {
				res.Detail = appendDetail(res.Detail, "CACHED_REMOVE_WORKTREE_PRESENT (warn): "+detail)
			} else {
				res.Reason = ReasonCachedRemoveWorktreePresent
				res.Detail = detail
				return res, true, nil
			}
		}
	}

	// (4d) SPURIOUS-STAGED-DELETION guard — whole-path, lock-free, before any `git add`. A
	// requested path can be staged as a DELETION (stale index) while an untracked copy of the
	// same path still sits in the working tree — the shape a peer `git reset`/`git rm` leaves
	// after the file was recreated on a shared clone. Committing it deletes a file HEAD carries,
	// only to resurrect it on the next add (a churn commit whose `git show --stat` reports an
	// unintended deletion). It is the whole-file sibling of the (4b) content guard. Gated by
	// FAK_SPURIOUS_DELETE_GUARD (block|warn|off, default block); off skips, warn records and
	// proceeds.
	if mode := spuriousDeleteGuardMode(); mode != staleBaseOff {
		if detail, fired := checkSpuriousStagedDeletion(ctx, run, opts.Dir, paths); fired {
			if mode == staleBaseWarn {
				res.Detail = "SPURIOUS_STAGED_DELETION (warn): " + detail
			} else {
				res.Reason = ReasonSpuriousStagedDeletion
				res.Detail = detail
				return res, true, nil
			}
		}
	}

	// (4e) PRESTAGED-PATH-OVERLAP guard - same-file ownership, lock-free, before any
	// `git add`. fak commit owns staging for requested paths. If one of those paths already
	// has staged hunks, a shared tree cannot tell whether they are this author's work or a
	// peer's staged same-file work. Refuse by default before folding those hunks into this
	// commit; the remedy is to unstage just the requested path and keep the worktree bytes.
	if mode := preStagedPathGuardMode(); mode != staleBaseOff {
		if detail, fired := checkPreStagedPathOverlap(ctx, run, opts.Dir, paths); fired {
			if mode == staleBaseWarn {
				res.Detail = appendDetail(res.Detail, "PRESTAGED_PATH_OVERLAP (warn): "+detail)
			} else {
				res.Reason = ReasonPreStagedPathOverlap
				res.Detail = detail
				return res, true, nil
			}
		}
	}

	return res, false, nil
}

func appendDetail(existing, next string) string {
	existing = strings.TrimSpace(existing)
	next = strings.TrimSpace(next)
	if existing == "" {
		return next
	}
	if next == "" {
		return existing
	}
	return existing + "; " + next
}

// recordPathspecAssertion appends the post-commit assertion result to the
// dedicated decisions note when a recorder is wired. It is best-effort: the
// assertion result stays in Result and the commit outcome does not depend on a
// side-ref write succeeding.
func recordPathspecAssertion(ctx context.Context, opts Options, res Result, verdict, reason, assertion string) {
	if opts.Recorder == nil || res.SHA == "" {
		return
	}
	d := witness.Decision{
		Op:                "safecommit",
		Verdict:           verdict,
		ReasonClass:       reason,
		Tree:              append([]string(nil), res.Paths...),
		PathspecAssertion: assertion,
	}
	_ = opts.Recorder.AppendDecision(ctx, res.SHA, d)
}

func reviewEnabled(r *ReviewOptions) bool {
	return r != nil && strings.TrimSpace(r.Model) != ""
}

func runPreCommitReview(ctx context.Context, run Runner, dir string, paths []string, opts *ReviewOptions) modelroute.ReviewResult {
	model := strings.TrimSpace(opts.Model)
	diff, err := collectReviewDiff(ctx, run, dir, paths)
	if err != nil {
		return unavailableReview(model, "", fmt.Sprintf("collect diff: %v", err))
	}
	if opts.Reviewer == nil {
		return unavailableReview(model, diff, "no reviewer bound")
	}
	req := modelroute.ReviewRequest{
		Model:     model,
		Objective: opts.Objective,
		Diff:      diff,
	}
	res, err := opts.Reviewer(ctx, req)
	if err != nil {
		return unavailableReview(model, diff, err.Error())
	}
	if strings.TrimSpace(res.Model) == "" {
		res.Model = model
	}
	if strings.TrimSpace(res.DiffSHA256) == "" {
		res.DiffSHA256 = modelroute.DiffSHA256(diff)
	}
	if res.Verdict != modelroute.ReviewPass && res.Verdict != modelroute.ReviewRefute {
		// The reviewer ran and answered, so an unusable verdict is not
		// unavailability. Fail closed instead of letting malformed output read as
		// the same permissive state as an absent reviewer.
		return modelroute.ReviewResult{
			Model:      res.Model,
			Verdict:    modelroute.ReviewRefute,
			Reason:     fmt.Sprintf("reviewer returned unusable verdict %q", res.Verdict),
			DiffSHA256: res.DiffSHA256,
		}
	}
	return res
}

func unavailableReview(model, diff, reason string) modelroute.ReviewResult {
	return modelroute.ReviewResult{
		Model:      strings.TrimSpace(model),
		Verdict:    modelroute.ReviewUnavailable,
		Reason:     strings.TrimSpace(reason),
		DiffSHA256: modelroute.DiffSHA256(diff),
	}
}

func collectReviewDiff(ctx context.Context, run Runner, dir string, paths []string) (string, error) {
	diffArgs := append([]string{"diff", "--no-ext-diff", "--binary", "HEAD", "--"}, paths...)
	out, code, err := run(ctx, dir, diffArgs...)
	if err != nil {
		return "", fmt.Errorf("git not executable: %w", err)
	}
	if code != 0 {
		return "", fmt.Errorf("git diff exited %d: %s", code, trimDetail(out))
	}

	var b strings.Builder
	b.WriteString(out)
	otherArgs := append([]string{"ls-files", "--others", "--exclude-standard", "--"}, paths...)
	others, code, err := run(ctx, dir, otherArgs...)
	if err != nil || code != 0 {
		return b.String(), nil
	}
	for _, p := range strings.Split(others, "\n") {
		p, ok := gitgate.CleanRepoPath(p)
		if !ok {
			continue
		}
		appendUntrackedReviewDiff(&b, dir, p)
	}
	return b.String(), nil
}

func appendUntrackedReviewDiff(b *strings.Builder, dir, p string) {
	if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
		b.WriteByte('\n')
	}
	b.WriteString("diff --git a/")
	b.WriteString(p)
	b.WriteString(" b/")
	b.WriteString(p)
	b.WriteString("\nnew file mode 100644\n--- /dev/null\n+++ b/")
	b.WriteString(p)
	b.WriteByte('\n')
	path := p
	if dir != "" {
		path = filepath.Join(dir, filepath.FromSlash(p))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		b.WriteString("+[untracked file unreadable: ")
		b.WriteString(err.Error())
		b.WriteString("]\n")
		return
	}
	if strings.ContainsRune(string(data), '\x00') {
		b.WriteString("+[binary file omitted]\n")
		return
	}
	for _, line := range strings.SplitAfter(string(data), "\n") {
		if line == "" {
			continue
		}
		b.WriteByte('+')
		b.WriteString(line)
		if !strings.HasSuffix(line, "\n") {
			b.WriteByte('\n')
		}
	}
}

// normalizePaths runs each raw pathspec through gitgate's exported repo-path rule, drops
// anything that cannot be a committed path, and dedups while preserving first-seen order.
func normalizePaths(raw []string) ([]string, bool) {
	seen := map[string]bool{}
	out := make([]string, 0, len(raw))
	for _, r := range raw {
		p, ok := gitgate.CleanRepoPath(r)
		if !ok {
			return nil, false
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out, true
}

// racedExtra returns the committed files (one per line in diff-tree output) that NO
// requested path covers — the empirical signature of a peer-swept commit. A requested
// directory legitimately covers the files under it (gitgate.CoveredByAnyTree), so a coarse
// pathspec does not false-positive. Result is sorted for a stable report.
func racedExtra(diffTreeOut string, requested []string) []string {
	var extra []string
	for _, line := range strings.Split(diffTreeOut, "\n") {
		p, ok := gitgate.CleanRepoPath(line)
		if !ok {
			continue
		}
		if !gitgate.CoveredByAnyTree(p, requested) {
			extra = append(extra, p)
		}
	}
	sort.Strings(extra)
	return extra
}

// writeMessageFile writes the commit message to a temp file OUTSIDE .git (so a `git clean`
// or hook never trips on it) and returns its path plus a cleanup. The whole point of -F is
// that the body never reaches argv to misparse as a pathspec (em-dash / multi-line trap).
func writeMessageFile(msg string) (path string, cleanup func(), err error) {
	f, err := os.CreateTemp("", "fak-commit-msg-*.txt")
	if err != nil {
		return "", func() {}, err
	}
	name := f.Name()
	cleanup = func() { _ = os.Remove(name) }
	if _, err := f.WriteString(msg); err != nil {
		f.Close()
		cleanup()
		return "", func() {}, err
	}
	if err := f.Close(); err != nil {
		cleanup()
		return "", func() {}, err
	}
	return name, cleanup, nil
}

func commitMessagesMatch(landed, requested string) bool {
	return comparableCommitMessage(landed) == comparableCommitMessage(requested)
}

func comparableCommitMessage(msg string) string {
	msg = strings.ReplaceAll(msg, "\r\n", "\n")
	msg = strings.TrimSpace(msg)
	lines := strings.Split(msg, "\n")
	for len(lines) > 0 {
		last := strings.TrimSpace(lines[len(lines)-1])
		if last == "" || strings.HasPrefix(strings.ToLower(last), "signed-off-by:") {
			lines = lines[:len(lines)-1]
			continue
		}
		break
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// trimDetail bounds a captured git/hook stderr+stdout blob so Result.Detail stays a useful
// one-screen message, not an unbounded dump.
func trimDetail(s string) string {
	s = strings.TrimSpace(s)
	const max = 2000
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}

// landedEscapesLease resolves each committed path against the real filesystem under dir
// and returns those whose resolved target escapes every requested tree — the symlink-escape
// (CVE-2025-53109) signature that a path-string comparison (racedExtra) cannot see. The
// containment of the RESOLVED, repo-relative target is decided with the same gitgate rule
// the policy uses. Fail-closed semantics: a path that resolves to a target outside the
// lease is reported; a path that cannot be resolved to a real file (EvalSymlinks errors:
// deleted, or never on disk) is NOT reported here — it carries no on-disk symlink to escape
// through, and the string-level racedExtra guard already covered its tracked path. dir == ""
// disables the check (no tree to resolve against).
func landedEscapesLease(dir string, diffTreeOut string, requested []string) []string {
	if dir == "" {
		return nil
	}
	root, err := filepath.EvalSymlinks(dir)
	if err != nil {
		root = dir // best-effort: compare against the unresolved root
	}
	var escaped []string
	for _, line := range strings.Split(diffTreeOut, "\n") {
		p, ok := gitgate.CleanRepoPath(line)
		if !ok {
			continue
		}
		abs := filepath.Join(root, filepath.FromSlash(p))
		real, rerr := filepath.EvalSymlinks(abs)
		if rerr != nil {
			// Not a resolvable on-disk path: nothing to escape through here.
			continue
		}
		rel, rerr := filepath.Rel(root, real)
		if rerr != nil {
			// Cannot express the target relative to the repo root — it is outside. Refuse.
			escaped = append(escaped, p)
			continue
		}
		rel = filepath.ToSlash(rel)
		clean, ok := gitgate.CleanRepoPath(rel)
		if !ok {
			// rel escapes above the root (".." / absolute) — outside the lease. Refuse.
			escaped = append(escaped, p)
			continue
		}
		if !gitgate.CoveredByAnyTree(clean, requested) {
			escaped = append(escaped, p)
		}
	}
	sort.Strings(escaped)
	return escaped
}

// isSanctionedWorkerWorktreeDir checks whether dir or any of its parent directories
// is a sanctioned worker worktree.
func isSanctionedWorkerWorktreeDir(dir string) bool {
	if dir == "" {
		if cwd, err := os.Getwd(); err == nil {
			dir = cwd
		}
	}
	if dir == "" {
		return false
	}
	curr := filepath.Clean(dir)
	for {
		if workerworktree.IsWorkerWorktree(curr) {
			return true
		}
		parent := filepath.Dir(curr)
		if parent == curr || parent == "." || parent == "" {
			break
		}
		curr = parent
	}
	return false
}
