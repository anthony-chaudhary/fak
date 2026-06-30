// Package safecommit is the EXECUTOR half of the shared-trunk commit discipline that
// internal/gitgate only declares defensively.
//
// On a multi-session shared trunk (the fak `main`), the ordinary sequence
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

	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/witness"
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
	Trunk    string            // expected trunk branch ("" => DefaultTrunk)
	SignOff  bool              // add the DCO sign-off (-s)
	Push     bool              // push, but ONLY after a verified commit
	Lock     LockOptions       // advisory same-host lock
	Recorder *witness.Recorder // optional decisions-note sink for post-commit assertions
	Window   *Window           // optional adaptive process-local writer window
	Review   *ReviewOptions    // optional pre-commit cross-model review rung
}

type ReviewFunc func(context.Context, modelroute.ReviewRequest) (modelroute.ReviewResult, error)

type ReviewOptions struct {
	Model     string
	Objective string
	Reviewer  ReviewFunc
}

// DefaultTrunk is the branch fak commits land on when Options.Trunk is empty.
const DefaultTrunk = "main"

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
	ReasonSymlinkEscape   = "SYMLINK_ESCAPE"    // a landed path resolves (through a symlink) to a target outside the lease
	ReasonPushRejected    = "PUSH_REJECTED"     // git push refused (e.g. non-fast-forward)
	ReasonReviewRefuted   = "REVIEW_REFUTED"    // opt-in scout review refuted the diff before commit
	// ReasonPreStagedPathOverlap ("PRESTAGED_PATH_OVERLAP") is part of this vocabulary too;
	// it lives in prestaged.go with the same-file staged-hunk ambiguity guard.
	// ReasonStaleBaseDeletion ("STALE_BASE_DELETION") is part of this closed vocabulary too;
	// it lives in stalebase.go with the content-level merge-base guard that emits it.
	// ReasonSpuriousStagedDeletion ("SPURIOUS_STAGED_DELETION") likewise lives in
	// spuriousdelete.go with the whole-path stale-index guard that emits it.
)

// Result is the structured outcome. A non-empty Reason is a refusal/race; a clean commit
// has Committed && Verified && Reason == "". RacedExtra lists the committed files that NO
// requested path covers — the evidence of a raced commit.
type Result struct {
	Committed  bool                     `json:"committed"`
	SHA        string                   `json:"committed_sha,omitempty"`
	Paths      []string                 `json:"paths"`
	Verified   bool                     `json:"verified"`
	Pushed     bool                     `json:"pushed"`
	Reason     string                   `json:"reason,omitempty"`
	Detail     string                   `json:"detail,omitempty"`
	RacedExtra []string                 `json:"raced_extra_paths,omitempty"`
	HeadBefore string                   `json:"head_before,omitempty"`
	Review     *modelroute.ReviewResult `json:"review,omitempty"`
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

// CommitWith is the testable core: every effect goes through the injected run and lock, so
// a fake Runner + fake LockFunc exercise the whole step-ordered algorithm — including the
// race remedy — with no git and no repo. See the package doc for the discipline it encodes.
func CommitWith(ctx context.Context, run Runner, lock LockFunc, opts Options) (res Result, err error) {
	trunk := strings.TrimSpace(opts.Trunk)
	if trunk == "" {
		trunk = DefaultTrunk
	}

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
	}

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
	unlock, err := lock(opts.Lock)
	if err != nil {
		if errors.Is(err, ErrLockBusy) {
			res.Reason = ReasonLockBusy
			return res, nil
		}
		return res, fmt.Errorf("safecommit: lock: %w", err)
	}
	defer unlock()

	// (6) Capture HEAD, then commit by pathspec with the message in a file.
	if head, code, herr := run(ctx, opts.Dir, "rev-parse", "HEAD"); herr != nil {
		return res, fmt.Errorf("safecommit: git not executable: %w", herr)
	} else if code == 0 {
		res.HeadBefore = strings.TrimSpace(head)
	}

	// Stage EXACTLY the requested paths, inside the lock, with an explicit pathspec — never
	// an unscoped `git add -A`/`.` (which would sweep a peer's tree). `--all` is deliberately
	// pathspec-scoped here: it stages additions, edits, and deletions for the requested paths,
	// including a path already removed from the index by `git rm`, without touching any other
	// dirty file. The post-commit assertion (step 7) remains the authority — a peer who raced
	// between this add and the commit is caught there.
	addArgs := append([]string{"add", "--all", "--"}, paths...)
	if out, code, aerr := run(ctx, opts.Dir, addArgs...); aerr != nil {
		return res, fmt.Errorf("safecommit: git not executable: %w", aerr)
	} else if code != 0 {
		res.Reason = ReasonHookRefused
		res.Detail = trimDetail(out)
		return res, nil
	}

	msgPath, cleanup, err := writeMessageFile(opts.Message)
	if err != nil {
		return res, fmt.Errorf("safecommit: write message file: %w", err)
	}
	defer cleanup()

	commitArgs := []string{"commit"}
	if opts.SignOff {
		commitArgs = append(commitArgs, "-s")
	}
	commitArgs = append(commitArgs, "-F", msgPath, "--")
	commitArgs = append(commitArgs, paths...)
	if out, code, cerr := run(ctx, opts.Dir, commitArgs...); cerr != nil {
		return res, fmt.Errorf("safecommit: git not executable: %w", cerr)
	} else if code != 0 {
		res.Reason = ReasonHookRefused
		res.Detail = trimDetail(out)
		return res, nil
	}

	// (7) VERIFY — the critical assertion. Use the porcelain name list (diff-tree), NOT
	// --stat: --stat formats names (rename arrows, quoting, truncation) and would make the
	// path-set comparison brittle. diff-tree --name-only gives one repo-relative path per
	// line; a deletion still lists the deleted path (correctly "exactly requested").
	if head, code, herr := run(ctx, opts.Dir, "rev-parse", "HEAD"); herr != nil {
		return res, fmt.Errorf("safecommit: git not executable: %w", herr)
	} else if code == 0 {
		res.SHA = strings.TrimSpace(head)
	}
	res.Committed = true

	landed, _, lerr := run(ctx, opts.Dir, "diff-tree", "--no-commit-id", "--name-only", "--no-renames", "-r", "HEAD")
	if lerr != nil {
		return res, fmt.Errorf("safecommit: git not executable: %w", lerr)
	}
	extra := racedExtra(landed, paths)
	if len(extra) > 0 {
		// A peer raced: extra files landed under our commit. Remedy is honest and
		// NON-DESTRUCTIVE — never reset/revert/force-push (a force-push to "fix" this would
		// clobber the peer). Leave the commit (HeadBefore is recorded for a human) and stop.
		res.Reason = ReasonPathspecRace
		res.RacedExtra = extra
		res.Detail = "extra files landed in this commit — a peer raced; commit left intact for review, not pushed"
		recordPathspecAssertion(ctx, opts, res, witness.VerdictAssertFail, ReasonPathspecRace, "committed-set!=requested-set")
		return res, nil
	}

	// racedExtra compared path STRINGS. A symlink created inside the lease that points
	// outside it would pass that check (the committed path string still starts with a
	// requested prefix) while git tracked a target outside the lease — the CVE-2025-53109
	// symlink-escape class. Resolve each landed path on disk and refuse if its real target
	// escapes the lease. Fail closed on an escaping target; a path that does not resolve to
	// a real file (deleted, or simply not present) carries no symlink to escape through and
	// is left to the string-level guard above.
	if escaped := landedEscapesLease(opts.Dir, landed, paths); len(escaped) > 0 {
		res.Reason = ReasonSymlinkEscape
		res.RacedExtra = escaped
		res.Detail = "a landed path resolves through a symlink to a target outside the lease; commit left intact for review, not pushed"
		recordPathspecAssertion(ctx, opts, res, witness.VerdictAssertFail, ReasonSymlinkEscape, "resolved-targets-within-requested-set=false")
		return res, nil
	}
	res.Verified = true
	recordPathspecAssertion(ctx, opts, res, witness.VerdictAssertPass, "", "committed-set==requested-set")

	// (8) Optional push — only a verified commit, plain push (never --force). A rejection
	// (e.g. non-fast-forward) surfaces honestly; the commit stands for a human to integrate.
	// We never pull --rebase --autostash (it strands .git/rebase-merge).
	if opts.Push {
		if out, code, perr := run(ctx, opts.Dir, "push"); perr != nil {
			return res, fmt.Errorf("safecommit: git not executable: %w", perr)
		} else if code != 0 {
			res.Reason = ReasonPushRejected
			res.Detail = trimDetail(out)
			return res, nil
		}
		res.Pushed = true
	}

	return res, nil
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
	if code != 0 || branch != trunk {
		res.Reason = ReasonOffTrunk
		// A non-zero symbolic-ref is a detached HEAD; the captured output is git's stderr
		// ("fatal: ref HEAD is not a symbolic ref"), not a branch name — don't echo it.
		if code != 0 || branch == "" {
			branch = "detached HEAD"
		}
		res.Detail = fmt.Sprintf("on %s, expected %s", branch, trunk)
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
	if out, _, err := run(ctx, opts.Dir, statusArgs...); err != nil {
		return res, false, fmt.Errorf("safecommit: git not executable: %w", err)
	} else if strings.TrimSpace(out) == "" {
		res.Reason = ReasonNothingStaged
		return res, true, nil
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
	// Detail and proceeds.
	if mode := staleBaseGuardMode(); mode != staleBaseOff {
		if detail, fired := checkStaleBaseDeletion(ctx, run, opts.Dir, trunk, paths); fired {
			if mode == staleBaseWarn {
				res.Detail = "STALE_BASE_DELETION (warn): " + detail
			} else {
				res.Reason = ReasonStaleBaseDeletion
				res.Detail = detail
				return res, true, nil
			}
		}
	}

	// (4c) SPURIOUS-STAGED-DELETION guard — whole-path, lock-free, before any `git add`. A
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

	// (4d) PRESTAGED-PATH-OVERLAP guard - same-file ownership, lock-free, before any
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
		return unavailableReview(model, diff, fmt.Sprintf("reviewer returned verdict %q", res.Verdict))
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
