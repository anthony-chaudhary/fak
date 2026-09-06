package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitintent"
	"github.com/anthony-chaudhary/fak/internal/commitrollup"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/safesync"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// commitFn is the seam the CLI shim calls; it defaults to the real safecommit.Commit and
// is overridden in tests so runCommit is exercised without a real git or repo.
var commitFn = safecommit.Commit

// commitLaneBusyFn is an advisory, lock-free pressure check. The authoritative acquisition
// remains inside safecommit.Commit; this seam only prevents queued contenders from starting
// expensive build checks while a live writer already owns the lane.
var commitLaneBusyFn = commitLaneBusy
var commitLaneWaitFn = waitForCommitLane
var commitLaneNow = time.Now
var commitLaneSleep = time.Sleep
var commitRecordTreeReceipt = recordCommittedTreeReceipt

// runCommitCommand routes `fak commit [<sub>]` to its subcommand handler and returns the
// process exit code. main.go calls it directly (inside the observed-git-operation wrapper)
// so the exit is recorded with the rest of the git lane.
func runCommitCommand(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runCommit(stdout, stderr, argv)
	}
	switch argv[0] {
	case "status":
		return runCommitStatus(stdout, stderr, argv[1:])
	case "patch":
		return runCommitPatch(stdout, stderr, argv[1:])
	case "poison-audit":
		return runCommitPoisonAudit(stdout, stderr, argv[1:])
	case "submit":
		return runCommitSubmit(stdout, stderr, argv[1:])
	case "drain":
		return runCommitDrain(stdout, stderr, argv[1:])
	case "preflight":
		return runCommitPreflight(stdout, stderr, argv[1:])
	}
	return runCommit(stdout, stderr, argv)
}

// pathList is a repeatable --path flag (the loopKVList shape): each --path appends one
// repo-relative pathspec.
type pathList []string

func (p *pathList) String() string { return strings.Join(*p, ",") }
func (p *pathList) Set(v string) error {
	v = strings.TrimSpace(v)
	if v == "" {
		return fmt.Errorf("empty --path")
	}
	*p = append(*p, v)
	return nil
}

// messageList is a repeatable -m flag: each -m appends one paragraph, and the paragraphs are
// joined with a blank line exactly like `git commit -m A -m B` ("their values are concatenated
// as separate paragraphs"). This closes a silent footgun: a plain fs.String -m is last-wins, so
// the muscle-memory call `fak commit -m "<subject> (fak leaf)" -m "<body>"` silently kept only
// the body — dropping the subject AND its ship-stamp, after which deriveCommitMessageStamp
// re-derived a DIFFERENT subject (silent corruption, not even a hard refusal). Joining preserves
// the subject and stamp and matches git's own behavior.
type messageList []string

func (m *messageList) String() string { return m.Joined() }
func (m *messageList) Set(v string) error {
	*m = append(*m, v)
	return nil
}

// Joined concatenates the -m paragraphs with a blank line (git's separator). An empty list
// joins to "" so assembleMessage still treats "no -m" as the -F/stdin/required-message case.
func (m messageList) Joined() string { return strings.Join([]string(m), "\n\n") }

// runCommit is the `fak commit` shim: it assembles a safecommit.Options from flags
// (message from -m / -F / stdin; paths from repeated --path AND/OR positionals after --),
// runs the safe-commit algorithm, and reports the structured Result. Exit codes mirror the
// loop verb's discipline: 0 success; 2 usage error; 3 a PRE-commit refusal (blocked, safe
// to retry/replan); 1 a POST-attempt failure (the commit ran but its result is bad — halt)
// or an infrastructure error.
func runCommit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "commit")
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative path to commit (repeatable); paths may also be given after --")
	var msg messageList
	fs.Var(&msg, "m", "commit message `string` (repeatable; multiple -m join as blank-line-separated paragraphs, exactly like git commit -m A -m B; mutually exclusive with -F)")
	msgFile := fs.String("F", "", "read the commit message from this file ('-' = stdin)")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	trunk := fs.String("trunk", "", "expected development branch override (default: configured development branch)")
	push := fs.Bool("push", false, "push after a VERIFIED commit through the safe sync path (never --force)")
	lockTimeout := fs.Duration("lock-timeout", safecommit.DefaultLockTimeout, "finite deadline for waiting on the advisory commit lock (default 10s); LOCK_BUSY reports elapsed wait and holder evidence")
	noSignoff := fs.Bool("no-signoff", false, "do not add the DCO sign-off (-s is the default)")
	var signoff bool
	fs.BoolVar(&signoff, "s", false, "add the DCO sign-off (default: true; git-compatible flag)")
	fs.BoolVar(&signoff, "signoff", false, "add the DCO sign-off (default: true; git-compatible flag)")
	preview := fs.Bool("preview", false, "LINT-ONLY: check the message+paths and exit WITHOUT touching git (is the subject witness-gradeable, does it carry a bindable `(fak <leaf>)` stamp, does the leaf match the paths' lane?). Exit 0 clean, 1 issues, 2 usage")
	requireIssue := fs.Bool("require-issue", false, "treat a missing bindable issue link (#N in subject / `Closes #N` in body) as BLOCKING, not advisory — the dispatch-worker contract so a close binds in `issue_closure_audit` (#312)")
	noBuildCheck := fs.Bool("no-build-check", false, "skip the COMMITTED_RED prospective-tree compile gate before the commit (default: gate ON — refuses a commit that would red the committed trunk)")
	buildCheckTimeout := fs.Duration("build-check-timeout", defaultValidateTimeout, "maximum duration for prospective validation (default 4m); controls prospective validation, not advisory-lock waiting or earlier build/materialization phases")
	allowBuildCheckTimeout := fs.Bool("allow-build-check-timeout", os.Getenv("FAK_COMMIT_BUILD_CHECK") == "allow-timeout", "land the commit even when the build gate TIMES OUT instead of refusing BUILD_CHECK_TIMEOUT (exit 3): an explicit opt-in to fail open on an unchecked tree, reported as build_check.failed_open in --json and docked in the score (#6006)")
	reviewModel := fs.String("review-model", envOrDefault("FAK_REVIEW_MODEL", ""), "optional scout model id, or comma-separated model ids, that must pass/refute this diff before commit; a multi-model quorum blocks on any refute")
	reviewMinModels := fs.Int("review-min-models", envIntOrDefault("FAK_REVIEW_MIN_MODELS", 0), "minimum usable review verdicts required when --review-model names multiple models (default: 2, or 1 for a single model)")
	reviewObjective := fs.String("review-objective", envOrDefault("FAK_REVIEW_OBJECTIVE", ""), "objective given to --review-model (default: FAK_GOAL_OBJECTIVE, then first commit-message line)")
	reviewEndpoint := fs.String("review-endpoint", envOrDefault("FAK_REVIEW_ENDPOINT", "http://127.0.0.1:8080/v1"), "OpenAI-compatible base URL for --review-model")
	reviewAPIKeyEnv := fs.String("review-api-key-env", envOrDefault("FAK_REVIEW_API_KEY_ENV", "FAK_REVIEW_API_KEY"), "env var holding the bearer token for --review-endpoint (empty value sends no token)")
	coreLockWitness := fs.String("core-lock-maintenance-witness", "", "independent witness claim that clears a hard-self core-lock maintenance commit; the gate runs before any `git add`, so a file this commit ADDS needs changed:<path> (committed:<path> is refuted for it)")
	reclaimLock := fs.Bool("reclaim-stale-index-lock", false, "RECOVERY (no commit): reclaim a stale index lock and next-index residue. Dry-run unless --apply; same path as `fak commit status --reclaim-stale-index-lock`")
	reclaimCommitLock := fs.Bool("reclaim-stale-commit-lock", false, "RECOVERY (no commit): reclaim only <git-dir>/fak-commit.lock when its recorded owner is proven stale or foreign. Dry-run unless --apply")
	reclaimApply := fs.Bool("apply", false, "with a --reclaim-stale-*-lock recovery, actually remove the proven-stale lock file(s) (default: dry-run)")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *reviewMinModels < 0 {
		fmt.Fprintln(stderr, "fak commit: --review-min-models must be non-negative")
		return 2
	}
	if *lockTimeout <= 0 {
		fmt.Fprintln(stderr, "fak commit: --lock-timeout must be greater than zero")
		return 2
	}
	if *buildCheckTimeout <= 0 {
		fmt.Fprintln(stderr, "fak commit: --build-check-timeout must be greater than zero")
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	paths = append(paths, fs.Args()...)
	if *reclaimLock && *reclaimCommitLock {
		fmt.Fprintln(stderr, "fak commit: choose either --reclaim-stale-commit-lock or --reclaim-stale-index-lock, not both")
		return 2
	}

	// The serialized commit lane and git's index lane use different lockfiles and
	// different stale-owner proofs. Keep their recovery modes explicit so clearing a
	// dead fak committer can never sweep unrelated index residue as a side effect.
	if *reclaimCommitLock {
		return runCommitLockReclaimAlias(stdout, stderr, *dir, *reclaimApply)
	}

	// --reclaim-stale-index-lock aliases the `fak commit status` recovery onto `fak commit`
	// itself (#5338). A committer meets the lock wedge HERE, so the way out has to be
	// discoverable from `fak commit --help` — not only from a sibling subcommand they have
	// no reason to run while their commit is refusing. It is a RECOVERY mode, not a commit:
	// it needs neither a message nor paths, and returns before any commit machinery runs.
	if *reclaimLock {
		return runCommitReclaimAlias(stdout, stderr, *dir, *reclaimApply)
	}

	// --preview is a no-op dry run: lint the message + paths so a bad subject/stamp is caught
	// BEFORE the commit lands (on the shared trunk you cannot amend — a sibling may push your
	// local commit first). It needs a message but tolerates zero paths (the lane match is then
	// skipped with a note).
	if *preview {
		message, code, ok := resolveCommitMessage(msg, *msgFile, stderr)
		if !ok {
			return code
		}
		root := resolveRoot(*dir)
		expectedTrunk := safecommit.ExpectedTrunk(root, *trunk)
		renderCommitSyncAdvisory(context.Background(), stderr, root, expectedTrunk)
		return runCommitPreview(stdout, stderr, message, paths, root, expectedTrunk, *asJSON, *requireIssue)
	}

	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak commit: at least one --path (or a path after --) is required")
		return 2
	}

	message, code, ok := resolveCommitMessage(msg, *msgFile, stderr)
	if !ok {
		return code
	}
	root := resolveRoot(*dir)
	if derived, ok := deriveCommitMessageStamp(message, paths, root); ok {
		message = derived
	}
	// COMMIT_MSG advisory: the derivation above already auto-healed the deterministic subject
	// defects; anything still ungradeable lands as an immutable ABSTAIN at the commit-audit
	// witness. Name it HERE — the message is final but nothing has touched git yet, so the
	// warning prints whether or not a later gate refuses. It is advisory ONLY: it returns no
	// value, has no block mode, and never changes the exit code (commit_msg_advisory.go).
	renderCommitMsgAdvisory(stderr, message, paths, root)
	review := commitReviewOptions(*reviewModel, firstNonEmpty(*reviewObjective, os.Getenv("FAK_GOAL_OBJECTIVE"), firstCommitLine(message)), *reviewEndpoint, *reviewAPIKeyEnv, *reviewMinModels)

	// --require-issue pre-lints the message before touching git: a real commit on the shared trunk
	// cannot be amended (a sibling may push it first), so a missing bindable `#N` is caught here as a
	// PRE-commit refusal (exit 4: a verdict on the message, not contention — the same command
	// will be refused again) rather than discovered weeks later as a CLAIMED_CLOSED row (#312).
	if *requireIssue {
		rep := hooks.LintCommitMessageWithOptions(message, paths, root, true)
		if !rep.OK {
			fmt.Fprintln(stderr, "fak commit: --require-issue refused this commit:")
			renderPreview(stderr, rep, "")
			return safecommit.ExitRefused
		}
	}

	if ready, receipt := commitLaneWaitFn(root, *lockTimeout); !ready {
		res := safecommit.ScoreResult(safecommit.Result{
			Paths:    append([]string(nil), paths...),
			Reason:   safecommit.ReasonLockBusy,
			Detail:   receipt.Detail(),
			LockWait: &receipt,
		})
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak commit: %v\n", err)
				return 1
			}
			return safecommit.ExitLockBusy
		}
		fmt.Fprintln(stderr, "LOCK_BUSY: commit lane remained held through its finite deadline; skipped build-check to avoid slowing the active writer")
		fmt.Fprintf(stderr, "  %s\n", receipt.Detail())
		fmt.Fprintln(stderr, "  next: inspect `fak commit status`; retry after it reports ready")
		return safecommit.ExitLockBusy
	}

	// COMMITTED_RED gate (#4152/#9266): validate the PROSPECTIVE committed tree — HEAD's
	// committed bytes + exactly this commit's paths, all other working-tree noise masked — before
	// safecommit can mutate the real index or HEAD. The differential build still admits a red that
	// predates this commit; a green build continues through owned gofmt, importer build/vet, and
	// uncached changed-package tests. The typed outcome rides on Result.BuildCheck into --json,
	// and a gate that could not FINISH refuses unless the caller opted into fail-open (#6006).
	buildCheckOutcome, buildCheckDetail := safecommit.BuildCheckDisabled, ""
	if !*noBuildCheck && os.Getenv("FAK_COMMIT_BUILD_CHECK") != "off" {
		buildCheckOutcome, buildCheckDetail = executeCommitBuildCheck(stderr, root, paths, *buildCheckTimeout)
	} else {
		oldFleet := os.Getenv("FLEET_BUILDCHECK_GUARD")
		oldFak := os.Getenv("FAK_COMMIT_BUILD_CHECK")
		_ = os.Setenv("FLEET_BUILDCHECK_GUARD", "off")
		_ = os.Setenv("FAK_COMMIT_BUILD_CHECK", "off")
		defer func() {
			_ = os.Setenv("FLEET_BUILDCHECK_GUARD", oldFleet)
			_ = os.Setenv("FAK_COMMIT_BUILD_CHECK", oldFak)
		}()
	}
	buildCheck, admitBuild, buildReason := safecommit.DecideBuildCheck(buildCheckOutcome, buildCheckDetail, *allowBuildCheckTimeout)
	if !admitBuild {
		// COMMITTED_RED is a verdict on this pathset (exit 4: an unchanged retry recompiles the
		// same red tree); BUILD_CHECK_TIMEOUT is contention (exit 3: the archive lost a race).
		return refuseCommitBuildCheck(stdout, stderr, paths, buildCheck, buildReason, *asJSON)
	}

	// Assess immediately before the writer enters safecommit. This deliberately uses only the
	// current remote-tracking ref: committing must never hide an implicit fetch or become blocked
	// solely because origin is behind/unavailable. The advisory gives agents who skipped preview
	// the same early integration signal while preserving the local commit needed for recovery.
	renderCommitSyncAdvisory(context.Background(), stderr, root, safecommit.ExpectedTrunk(root, *trunk))
	commitDir := *dir
	if commitDir == "" && root != "" && workerworktree.IsWorkerWorktree(root) {
		commitDir = root
	}
	res, err := commitFn(context.Background(), safecommit.Options{
		Dir:                        commitDir,
		Paths:                      paths,
		Message:                    message,
		Trunk:                      *trunk,
		SignOff:                    !*noSignoff || signoff,
		Push:                       *push,
		Lock:                       safecommit.LockOptions{Timeout: *lockTimeout},
		Review:                     review,
		CoreLockMaintenanceWitness: *coreLockWitness,
	})
	if err != nil {
		// Infrastructure failure (git not executable, lock unopenable): not a refusal.
		fmt.Fprintf(stderr, "fak commit: %v\n", err)
		return 1
	}
	res = finalizeCommitEvidence(stderr, root, res, buildCheck, buildCheckOutcome, *push, *requireIssue)

	if code := emitJSONOrRenderPrefixed(stdout, stderr, "fak commit", *asJSON, res, func(w io.Writer) {
		renderCommitResult(w, res)
	}); code != 0 {
		return code
	}
	return commitExitCode(res)
}

// finalizeCommitEvidence attaches the prospective-tree gate and the delivery/review
// receipts before the result is scored and rendered.
func finalizeCommitEvidence(stderr io.Writer, root string, res safecommit.Result, buildCheck safecommit.BuildCheckResult, buildCheckOutcome safecommit.BuildCheckOutcome, push, requireIssue bool) safecommit.Result {
	// Attach the gate's outcome BEFORE scoring: a commit admitted without its prospective tree
	// ever being compiled must not be graded like one that passed the gate (#6006).
	res.BuildCheck = &buildCheck
	completionClass := safecommit.CompletionVerifiedDelivery
	if buildCheck.Outcome == safecommit.BuildCheckDisabled {
		completionClass = safecommit.CompletionRecordOnly
	}
	res = safecommit.FinalizeEvidence(res, safecommit.EvidenceContract{
		CompletionClass: completionClass,
		RequirePush:     push,
		RequireClosure:  requireIssue,
		ClosureBound:    requireIssue,
	})
	if res.Committed {
		artifacts := make([]workdelivery.Artifact, 0, len(res.Paths))
		for _, path := range res.Paths {
			artifacts = append(artifacts, workdelivery.Artifact{Path: path, Kind: "source"})
		}
		unit := workdelivery.WorkUnit{Schema: workdelivery.Schema, ID: res.SHA, Revision: res.SHA, Artifacts: artifacts, Axes: workdelivery.InitialAxes()}
		if delivery, deliveryErr := workdelivery.RecordingObservation(unit, res.SHA, "fak commit", time.Now()); deliveryErr == nil {
			res.Delivery = &delivery
		} else {
			fmt.Fprintf(stderr, "fak commit: record delivery receipt: %v\n", deliveryErr)
		}
	}
	res = safecommit.ScoreResult(res)
	if res.Committed && buildCheckOutcome == safecommit.BuildCheckPassed {
		commitRecordTreeReceipt(root, time.Now())
	}
	if res.Review != nil {
		if err := recordCommitReviewForLoop(res); err != nil {
			fmt.Fprintf(stderr, "fak commit: record review evidence: %v\n", err)
		}
		if err := appendCommitReviewRefusalToGoal(res); err != nil {
			fmt.Fprintf(stderr, "fak commit: append review refusal: %v\n", err)
		}
	}
	return res
}

// renderCommitSyncAdvisory reports the relationship to origin using safesync's existing
// assessment and vocabulary. It is intentionally output-only: every state, including a missing
// or unreadable upstream, leaves commit admission to the existing safecommit gates.
func renderCommitSyncAdvisory(ctx context.Context, w io.Writer, repo, branch string) {
	info, err := syncAssess(ctx, safesync.Options{
		Repo:   repo,
		Remote: "origin",
		Branch: branch,
		Fetch:  false,
	})
	target := "origin/" + branch
	check := fmt.Sprintf("fak sync check --fetch --remote origin --branch %s", branch)
	if err != nil {
		fmt.Fprintf(w, "commit upstream advisory: unavailable %s using current remote-tracking refs (no fetch): %v\n", target, err)
		fmt.Fprintf(w, "  next: run `%s`; integrate %s in place if needed before accumulating more local commits\n", check, target)
		return
	}

	if info.TargetRef != "" {
		target = info.TargetRef
	}
	if target == "origin/" || target == "" {
		target = "origin upstream"
	}

	switch info.State {
	case safesync.StateInSync:
		fmt.Fprintf(w, "commit upstream advisory: in-sync with %s using current remote-tracking refs (no fetch)\n", target)
	case safesync.StateBehind, safesync.StateDiverged:
		fmt.Fprintf(w, "commit upstream advisory: %s %s using current remote-tracking refs (no fetch)\n", info.State, target)
		fmt.Fprintf(w, "  next: run `%s`; if still behind or diverged, integrate %s in place before accumulating more local commits\n", check, target)
	case safesync.StateNoRemoteRef:
		fmt.Fprintf(w, "commit upstream advisory: unavailable %s using current remote-tracking refs (no fetch): %s\n", target, info.Reason)
		fmt.Fprintf(w, "  next: run `%s` to refresh and reassess upstream before accumulating more local commits\n", check)
	default:
		fmt.Fprintf(w, "commit upstream advisory: %s %s using current remote-tracking refs (no fetch)\n", info.State, target)
	}
}

func recordCommittedTreeReceipt(root string, now time.Time) {
	if tree, err := gitRevParse(root, "HEAD^{tree}"); err == nil {
		recordPrepushSuccessForTree(root, tree, now)
	}
}

func deriveCommitMessageStamp(message string, paths []string, root string) (string, bool) {
	rep := hooks.LintCommitMessageWithOptions(message, paths, root, false)
	if rep.SuggestedSubject == "" {
		return message, false
	}
	return replaceFirstNonEmptyLine(message, rep.SuggestedSubject)
}

func replaceFirstNonEmptyLine(message, line string) (string, bool) {
	parts := strings.Split(message, "\n")
	for i, part := range parts {
		if strings.TrimSpace(part) == "" {
			continue
		}
		parts[i] = line
		return strings.Join(parts, "\n"), true
	}
	return message, false
}

type commitSubmitResult struct {
	Queued    bool                      `json:"queued"`
	QueueDir  string                    `json:"queue_dir"`
	IntentID  string                    `json:"intent_id"`
	Sequence  int64                     `json:"sequence"`
	BaseSHA   string                    `json:"base_sha"`
	Paths     []string                  `json:"paths"`
	Subject   string                    `json:"subject"`
	Stamp     commitintent.Stamp        `json:"stamp"`
	QueueSize int                       `json:"queue_size"`
	Record    commitintent.SubmitRecord `json:"record"`
}

type commitDrainResult struct {
	Drained    bool                           `json:"drained"`
	DryRun     bool                           `json:"dry_run"`
	QueueDir   string                         `json:"queue_dir"`
	BaseSHA    string                         `json:"base_sha"`
	ReadyCount int                            `json:"ready_count"`
	QueueSize  int                            `json:"queue_size,omitempty"`
	MarkedDone []string                       `json:"marked_done,omitempty"`
	Stale      []commitintent.SubmitRecord    `json:"stale,omitempty"`
	Invalid    []commitintent.InvalidRecord   `json:"invalid,omitempty"`
	Plan       commitrollup.Plan              `json:"plan"`
	Commit     *safecommit.Result             `json:"commit,omitempty"`
	Pathset    *commitrollup.PathsetAssertion `json:"pathset,omitempty"`
}

func runCommitSubmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit submit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "commit")
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative path for the future commit (repeatable); paths may also be given after --")
	var msg messageList
	fs.Var(&msg, "m", "commit subject/paragraph `string` for the intent (repeatable; joined as blank-line paragraphs like git commit -m A -m B; mutually exclusive with -F)")
	msgFile := fs.String("F", "", "read the commit subject from this file ('-' = stdin)")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	queueDir := fs.String("queue-dir", "", "commit-intent queue dir (default: <repo>/.fak/commit-intents)")
	id := fs.String("id", "", "stable intent id (default: generated intent-<unix-nanos>)")
	base := fs.String("base", "", "base SHA the intent was authored against (default: git rev-parse HEAD)")
	diffDigest := fs.String("diff-digest", "", "optional sha256:<hex> digest of the authored diff")
	asJSON := fs.Bool("json", false, "emit the submitted record as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	*queueDir = pathutil.ExpandTilde(*queueDir)
	paths = append(paths, fs.Args()...)
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak commit submit: at least one --path (or a path after --) is required")
		return 2
	}
	subject, code, ok := resolveCommitMessage(msg, *msgFile, stderr)
	if !ok {
		return code
	}
	_, baseSHA, code := resolveCommitQueueBase(stderr, "submit", *dir, queueDir, *base)
	if code != 0 {
		return code
	}
	intentID := strings.TrimSpace(*id)
	if intentID == "" {
		intentID = fmt.Sprintf("intent-%d", time.Now().UTC().UnixNano())
	}
	intent := commitintent.Intent{
		ID:         intentID,
		BaseSHA:    baseSHA,
		Paths:      paths,
		DiffDigest: *diffDigest,
		Subject:    subject,
	}
	store := commitintent.Store{Dir: *queueDir}
	queue, rec, err := store.Submit(intent)
	if err != nil {
		fmt.Fprintf(stderr, "fak commit submit: %v\n", err)
		// The intent was rejected on its content (an unstampable subject, a bad base):
		// nothing was queued and resubmitting the identical intent is refused again.
		return safecommit.ExitRefused
	}
	res := commitSubmitResult{
		Queued:    true,
		QueueDir:  *queueDir,
		IntentID:  rec.Intent.ID,
		Sequence:  rec.Sequence,
		BaseSHA:   rec.Intent.BaseSHA,
		Paths:     rec.Intent.Paths,
		Subject:   rec.Intent.Subject,
		Stamp:     rec.Intent.Stamp,
		QueueSize: len(queue.Records),
		Record:    rec,
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak commit submit: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "queued %s as #%d (%d path(s)) in %s\n", res.IntentID, res.Sequence, len(res.Paths), res.QueueDir)
	fmt.Fprintf(stdout, "  base: %s\n", short(res.BaseSHA))
	fmt.Fprintf(stdout, "  stamp: %s %s\n", res.Stamp.Kind, res.Stamp.Text)
	return 0
}

func runCommitDrain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit drain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "commit")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	queueDir := fs.String("queue-dir", "", "commit-intent queue dir (default: <repo>/.fak/commit-intents)")
	base := fs.String("base", "", "current base SHA (default: git rev-parse HEAD)")
	max := fs.Int("max", 0, "maximum ready intents to consider (0 = all pending ready intents)")
	trunk := fs.String("trunk", "", "expected development branch override (default: configured development branch)")
	push := fs.Bool("push", false, "push after a VERIFIED rollup commit through the safe sync path (never --force)")
	noSignoff := fs.Bool("no-signoff", false, "do not add the DCO sign-off (-s is the default)")
	var signoff bool
	fs.BoolVar(&signoff, "s", false, "add the DCO sign-off (default: true; git-compatible flag)")
	fs.BoolVar(&signoff, "signoff", false, "add the DCO sign-off (default: true; git-compatible flag)")
	noBuildCheck := fs.Bool("no-build-check", false, "record the rollup without prospective compile/test verification; recorded work is not eligible to mark intents done")
	allowBuildCheckTimeout := fs.Bool("allow-build-check-timeout", os.Getenv("FAK_COMMIT_BUILD_CHECK") == "allow-timeout", "record the rollup when prospective validation times out; the unchecked receipt cannot mark intents done")
	buildCheckTimeout := fs.Duration("build-check-timeout", defaultValidateTimeout, "maximum duration for prospective validation (default 4m); controls prospective validation, not advisory-lock waiting or earlier build/materialization phases")
	noRollup := fs.Bool("no-rollup", false, "disable batching and drain at most one compatible intent")
	dryRun := fs.Bool("dry-run", false, "plan only; do not commit or update queue state")
	asJSON := fs.Bool("json", false, "emit the drain result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *buildCheckTimeout <= 0 {
		fmt.Fprintln(stderr, "fak commit drain: --build-check-timeout must be greater than zero")
		return 2
	}
	if len(fs.Args()) != 0 {
		fmt.Fprintf(stderr, "fak commit drain: unexpected argument %q\n", fs.Args()[0])
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	*queueDir = pathutil.ExpandTilde(*queueDir)
	root, baseSHA, code := resolveCommitQueueBase(stderr, "drain", *dir, queueDir, *base)
	if code != 0 {
		return code
	}

	store := commitintent.Store{Dir: *queueDir}
	drain, err := store.Drain(baseSHA, *max)
	if err != nil {
		fmt.Fprintf(stderr, "fak commit drain: %v\n", err)
		return 1
	}
	plan := commitrollup.PlanBatch(commitDrainRollupIntents(drain), commitrollup.Config{DisableRollup: *noRollup})
	res := commitDrainResult{
		DryRun:     *dryRun,
		QueueDir:   *queueDir,
		BaseSHA:    strings.TrimSpace(baseSHA),
		ReadyCount: len(drain.Ready),
		Stale:      drain.Stale,
		Invalid:    drain.Invalid,
		Plan:       plan,
	}

	if *dryRun || !plan.OK {
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak commit drain: %v\n", err)
				return 1
			}
		} else {
			renderCommitDrainResult(stdout, res)
		}
		if *dryRun || len(drain.Ready)+len(drain.Stale)+len(drain.Invalid) == 0 {
			return 0
		}
		// The rollup plan is not executable (stale or invalid intents in the queue): a
		// verdict on the queue's contents, not contention, so exit 4 — draining again
		// without repairing the queue produces the same unexecutable plan.
		return safecommit.ExitRefused
	}

	buildCheckOutcome, buildCheckDetail := safecommit.BuildCheckDisabled, ""
	if !*noBuildCheck && os.Getenv("FAK_COMMIT_BUILD_CHECK") != "off" {
		buildCheckOutcome, buildCheckDetail = executeCommitBuildCheck(stderr, root, plan.UnionPaths, *buildCheckTimeout)
	}
	buildCheck, admitBuild, buildReason := safecommit.DecideBuildCheck(buildCheckOutcome, buildCheckDetail, *allowBuildCheckTimeout)
	if !admitBuild {
		commitRes := safecommit.Result{Paths: plan.UnionPaths, Reason: buildReason, Detail: buildCheck.Detail, BuildCheck: &buildCheck}
		commitRes = safecommit.FinalizeEvidence(commitRes, safecommit.EvidenceContract{
			CompletionClass: safecommit.CompletionVerifiedDelivery,
			RequirePush:     *push,
			RequireClosure:  true,
		})
		commitRes = safecommit.ScoreResult(commitRes)
		res.Commit = &commitRes
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak commit drain: %v\n", err)
				return 1
			}
		} else {
			renderCommitDrainResult(stdout, res)
		}
		if code, ok := safecommit.BuildCheckExitCode(buildReason); ok {
			return code
		}
		return safecommit.ExitRefused
	}

	commitRes, err := commitFn(context.Background(), safecommit.Options{
		Dir:     root,
		Paths:   plan.UnionPaths,
		Message: plan.Subject,
		Trunk:   *trunk,
		SignOff: !*noSignoff || signoff,
		Push:    *push,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak commit drain: %v\n", err)
		return 1
	}
	commitRes.BuildCheck = &buildCheck
	pathset := plan.AssertPathset(commitRes.Paths)
	completionClass := safecommit.CompletionVerifiedDelivery
	if buildCheck.Outcome == safecommit.BuildCheckDisabled {
		completionClass = safecommit.CompletionRecordOnly
	}
	commitRes = safecommit.FinalizeEvidence(commitRes, safecommit.EvidenceContract{
		CompletionClass: completionClass,
		RequirePush:     *push,
		RequireClosure:  true,
		ClosureBound:    pathset.OK,
	})
	commitRes = safecommit.ScoreResult(commitRes)
	res.Commit = &commitRes
	res.Pathset = &pathset

	if commitDrainMayMarkDone(commitRes, pathset.OK) {
		states := commitDrainDoneStates(plan.IntentIDs)
		queue, err := store.MarkStates(states)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit drain: mark queue done: %v\n", err)
			return 1
		}
		res.Drained = true
		res.MarkedDone = append([]string(nil), plan.IntentIDs...)
		res.QueueSize = len(queue.Records)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak commit drain: %v\n", err)
			return 1
		}
	} else {
		renderCommitDrainResult(stdout, res)
	}
	if !pathset.OK {
		return 1
	}
	return commitExitCode(commitRes)
}

// commitDrainMayMarkDone is the issue/action boundary: recorded or diff-witnessed work is not
// enough to advance queued intents. Versioned receipts need verified delivery; schema-less
// results retain the legacy gate during the migration window.
func commitDrainMayMarkDone(res safecommit.Result, pathsetOK bool) bool {
	return res.Reason == "" && res.DeliveryVerified() && pathsetOK
}

func commitSubmitHeadSHA(root string) (string, error) {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func commitDrainRollupIntents(plan commitintent.DrainPlan) []commitrollup.Intent {
	out := make([]commitrollup.Intent, 0, len(plan.Ready)+len(plan.Stale)+len(plan.Invalid))
	for _, rec := range plan.Ready {
		out = append(out, commitDrainRollupIntent(rec))
	}
	for _, rec := range plan.Stale {
		in := commitDrainRollupIntent(rec)
		in.Stale = true
		out = append(out, in)
	}
	for _, invalid := range plan.Invalid {
		in := commitDrainRollupIntent(invalid.Record)
		in.Refused = true
		in.RefusedReason = commitrollup.ReasonRefusedInput
		if strings.TrimSpace(invalid.Error) != "" {
			in.Witnesses = append(in.Witnesses, "invalid:"+strings.TrimSpace(invalid.Error))
		}
		out = append(out, in)
	}
	return out
}

func commitDrainRollupIntent(rec commitintent.SubmitRecord) commitrollup.Intent {
	stamp := rec.Intent.Stamp.Leaf
	if stamp == "" {
		stamp = rec.Intent.Stamp.Text
	}
	witnesses := []string{}
	if rec.Intent.BaseSHA != "" {
		witnesses = append(witnesses, "base:"+rec.Intent.BaseSHA)
	}
	if rec.Intent.PathDigest != "" {
		witnesses = append(witnesses, "path_digest:"+rec.Intent.PathDigest)
	}
	if rec.Intent.DiffDigest != "" {
		witnesses = append(witnesses, "diff_digest:"+rec.Intent.DiffDigest)
	}
	if rec.Intent.Metadata.Issue > 0 {
		witnesses = append(witnesses, fmt.Sprintf("issue:#%d", rec.Intent.Metadata.Issue))
	}
	return commitrollup.Intent{
		ID:        rec.Intent.ID,
		Submitter: firstNonEmpty(rec.Intent.Metadata.Requester, rec.Intent.Metadata.Source),
		Paths:     rec.Intent.Paths,
		Stamp:     stamp,
		Witnesses: witnesses,
	}
}

func commitDrainDoneStates(ids []string) map[string]commitintent.State {
	out := make(map[string]commitintent.State, len(ids))
	for _, id := range ids {
		out[id] = commitintent.StateDone
	}
	return out
}

func renderCommitDrainResult(stdout io.Writer, res commitDrainResult) {
	if res.Drained {
		sha := ""
		if res.Commit != nil {
			sha = res.Commit.SHA
		}
		fmt.Fprintf(stdout, "drained %d intent(s) into %s\n", len(res.MarkedDone), short(sha))
		return
	}
	if res.DryRun {
		fmt.Fprintf(stdout, "planned %d intent(s); dry run\n", len(res.Plan.IntentIDs))
	} else if res.Plan.OK {
		fmt.Fprintf(stdout, "planned %d intent(s); commit not drained\n", len(res.Plan.IntentIDs))
	} else {
		fmt.Fprintln(stdout, "no drainable commit intents")
	}
	if res.Plan.Subject != "" {
		fmt.Fprintf(stdout, "  subject: %s\n", res.Plan.Subject)
	}
	if len(res.Plan.UnionPaths) > 0 {
		fmt.Fprintf(stdout, "  paths: %s\n", strings.Join(res.Plan.UnionPaths, ", "))
	}
	for _, refusal := range res.Plan.Refusals {
		fmt.Fprintf(stdout, "  refused %s: %s", refusal.IntentID, refusal.Reason)
		if refusal.Detail != "" {
			fmt.Fprintf(stdout, " (%s)", refusal.Detail)
		}
		fmt.Fprintln(stdout)
		if refusal.Reason == safecommit.ReasonPreStagedPathOverlap || strings.Contains(refusal.Detail, safecommit.ReasonPreStagedPathOverlap) {
			fmt.Fprintln(stdout, "    remedy: unstage pre-existing index changes via `git restore --staged <paths>` (worktree edits stay), then retry `fak commit`")
		}
	}
	if res.Pathset != nil && !res.Pathset.OK {
		fmt.Fprintf(stdout, "  pathset mismatch: missing=%v extra=%v\n", res.Pathset.Missing, res.Pathset.Extra)
	}
}

// commitLaneBusy returns true only when the advisory lock names a process that is
// currently alive and still attributable to the lock. Stale/reused-PID locks flow to the
// authoritative safecommit acquisition, which owns their guarded recovery.
func commitLaneBusy(dir string) (bool, int) {
	lockPath := wipCommitLockPath(context.Background(), dir)
	probe := safecommit.ProbeLock(lockPath)
	return probe.Exists && probe.Alive && !probe.Foreign, probe.HolderPID
}

const commitLaneWaitPoll = 250 * time.Millisecond

// waitForCommitLane keeps expensive prospective-tree validation out of an occupied commit
// lane while giving the holder a documented, finite window to finish. Dead and reused-PID
// residues are not treated as live here; they flow to safecommit's guarded reaper.
func waitForCommitLane(dir string, timeout time.Duration) (bool, safecommit.LockWaitReceipt) {
	started := commitLaneNow()
	deadline := started.Add(timeout)
	for {
		busy, holderPID := commitLaneBusyFn(dir)
		now := commitLaneNow()
		receipt := safecommit.LockWaitReceipt{
			ElapsedNS:   now.Sub(started).Nanoseconds(),
			DeadlineNS:  timeout.Nanoseconds(),
			HolderPID:   holderPID,
			HolderAlive: busy,
		}
		if !busy {
			return true, receipt
		}
		if !now.Before(deadline) {
			return false, receipt
		}
		wait := commitLaneWaitPoll
		if remaining := deadline.Sub(now); remaining < wait {
			wait = remaining
		}
		if wait > 0 {
			commitLaneSleep(wait)
		}
	}
}

// stdin is overridable in tests; defaults to os.Stdin.
var stdin = func() io.Reader { return os.Stdin }

func resolveCommitMessage(msg messageList, file string, stderr io.Writer) (string, int, bool) {
	message, code := assembleMessage(stdin(), msg.Joined(), file, stderr)
	return message, code, code == 0
}

func resolveCommitQueueBase(stderr io.Writer, action, dir string, queueDir *string, base string) (string, string, int) {
	root := resolveRoot(dir)
	if *queueDir == "" {
		*queueDir = commitintent.DefaultQueueDir(root)
	}
	baseSHA := strings.TrimSpace(base)
	if baseSHA != "" {
		return root, baseSHA, 0
	}
	var err error
	baseSHA, err = commitSubmitHeadSHA(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak commit %s: resolve base sha: %v\n", action, err)
		return "", "", 1
	}
	return root, baseSHA, 0
}

// assembleMessage resolves the commit message from exactly one source: -m, -F <file>
// (or -F - for stdin). Returns (message, 0) on success or ("", exitCode) on a usage error.
func assembleMessage(in io.Reader, m, file string, stderr io.Writer) (string, int) {
	m = strings.TrimSpace(m)
	file = strings.TrimSpace(file)
	switch {
	case m != "" && file != "":
		fmt.Fprintln(stderr, "fak commit: -m and -F are mutually exclusive")
		return "", 2
	case m != "":
		return m, 0
	case file == "-":
		b, err := io.ReadAll(in)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit: read message from stdin: %v\n", err)
			return "", 2
		}
		return string(b), 0
	case file != "":
		b, err := os.ReadFile(file)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit: read message file: %v\n", err)
			return "", 2
		}
		return string(b), 0
	default:
		fmt.Fprintln(stderr, "fak commit: a message is required (-m STR, or -F FILE/-)")
		return "", 2
	}
}

// commitExitCode maps a Result to the process exit code. Contention is exit 3 ("the lock
// was busy — nothing landed, retry with backoff"); a refusal on the merits is exit 4
// ("no — retrying the same command cannot change the answer"); a commit that ran but
// produced a bad result (race, push rejection, hook refusal) is exit 1 ("ran, result is
// bad — halt"). See internal/safecommit.RefusalExitCode for the classification (#5505 W4).
func commitExitCode(res safecommit.Result) int {
	switch res.Reason {
	case "":
		return 0
	case safecommit.ReasonNoPath, safecommit.ReasonEmptyMessage:
		return 2
	case safecommit.ReasonNotARepo:
		// A setup/environment error: nothing landed, and re-running here never will —
		// the caller has to move to a work tree first.
		return safecommit.ExitRefused
	}
	// The closed refusal vocabulary is classified by safecommit (the SoT next to
	// RefusalReasons); a test there asserts every reason is covered.
	if code, ok := safecommit.RefusalExitCode(res.Reason); ok {
		return code
	}
	// An unrecognized reason is not safe to auto-retry: keep the halt-class default.
	return safecommit.ExitPostCommitFailure
}

func renderCommitResult(stdout io.Writer, res safecommit.Result) {
	if res.Reason == "" {
		fmt.Fprintf(stdout, "committed %s (%d path(s))%s\n", short(res.SHA), len(res.Paths), pushedSuffix(res))
		renderCommitScore(stdout, res)
		if res.BuildCheck != nil {
			fmt.Fprintf(stdout, "  build check: %s (compiled=%t)\n", res.BuildCheck.Outcome, res.BuildCheck.Compiled)
		}
		renderCommitVelocity(stdout, res)
		renderCommitReview(stdout, res)
		return
	}
	fmt.Fprintf(stdout, "%s", res.Reason)
	if res.Detail != "" {
		fmt.Fprintf(stdout, ": %s", res.Detail)
	}
	fmt.Fprintln(stdout)
	// A LOCK_BUSY refusal is the exact moment a committer needs the reclaim path, and the
	// exact moment they cannot go looking for it. Name it inline (#5338). It stays advisory:
	// the reclaim itself still refuses unless the lane evidence proves the lock orphaned.
	if res.Reason == safecommit.ReasonLockBusy {
		fmt.Fprintln(stdout, "  wedged? `fak commit --reclaim-stale-commit-lock` probes only the serialized commit lock (add --apply to remove a proven stale owner); `fak commit status` shows the live owner")
		fmt.Fprintln(stdout, "  separate git residue: `fak commit --reclaim-stale-index-lock` handles only index.lock and next-index files")
	}
	if res.Reason == safecommit.ReasonPreStagedPathOverlap || strings.Contains(res.Reason, safecommit.ReasonPreStagedPathOverlap) || strings.Contains(res.Detail, safecommit.ReasonPreStagedPathOverlap) {
		fmt.Fprintln(stdout, "  remedy: unstage pre-existing index changes via `git restore --staged <paths>` (worktree edits stay), then retry `fak commit`")
	}
	renderCommitScore(stdout, res)
	renderCommitVelocity(stdout, res)
	renderCommitReview(stdout, res)
	if len(res.RacedExtra) > 0 {
		fmt.Fprintf(stdout, "  raced extra paths: %s\n", strings.Join(res.RacedExtra, ", "))
		if res.SHA != "" {
			fmt.Fprintf(stdout, "  commit %s left intact for review (was %s)\n", short(res.SHA), short(res.HeadBefore))
		}
	}
}

func renderCommitScore(stdout io.Writer, res safecommit.Result) {
	if res.Grade == "" && res.Score == 0 {
		res = safecommit.ScoreResult(res)
	}
	fmt.Fprintf(stdout, "  score: %d/100 (%s)\n", res.Score, res.Grade)
	for _, note := range res.ScoreNotes {
		fmt.Fprintf(stdout, "    score note: %s\n", note)
	}
	if res.LockHoldNS > 0 {
		fmt.Fprintf(stdout, "  lock hold: %s\n", time.Duration(res.LockHoldNS))
	}
}

// renderCommitVelocity prints the effect-qualified ship-speed legs (#4241). It is deliberately
// separate from renderCommitScore: quality answers "how healthy was the outcome", velocity
// answers "how fast did the effect land against its budget". A SCORED leg shows its
// budget-relative score; an UNSCORED leg shows its retained timing and the reason it did not
// qualify (a refusal/race/no-op never earns a score). Nil velocity — a fake result in a test, or
// a pre-scoring path — prints nothing. Because `fak sweep --apply` renders through this same
// path, sweep exposes ship speed without inventing a second score.
func renderCommitVelocity(stdout io.Writer, res safecommit.Result) {
	if res.Velocity == nil {
		return
	}
	renderVelocityLeg(stdout, "local", res.Velocity.Local)
	renderVelocityLeg(stdout, "push", res.Velocity.Push)
}

func renderVelocityLeg(stdout io.Writer, name string, leg safecommit.CommitVelocityLeg) {
	elapsed, budget := time.Duration(leg.ElapsedNS), time.Duration(leg.BudgetNS)
	if leg.Score != nil {
		fmt.Fprintf(stdout, "  velocity %s: %d/100 (%s, %s/%s)\n", name, *leg.Score, leg.Status, elapsed, budget)
		return
	}
	if leg.Note != "" {
		fmt.Fprintf(stdout, "  velocity %s: %s — %s (%s/%s)\n", name, leg.Status, leg.Note, elapsed, budget)
		return
	}
	fmt.Fprintf(stdout, "  velocity %s: %s (%s/%s)\n", name, leg.Status, elapsed, budget)
}

func renderCommitReview(stdout io.Writer, res safecommit.Result) {
	if res.Review == nil {
		return
	}
	fmt.Fprintf(stdout, "  review: %s", res.Review.Verdict)
	if res.Review.Model != "" {
		fmt.Fprintf(stdout, " by %s", res.Review.Model)
	}
	if res.Review.Reason != "" {
		fmt.Fprintf(stdout, " — %s", res.Review.Reason)
	}
	fmt.Fprintln(stdout)
}

func pushedSuffix(res safecommit.Result) string {
	if res.Pushed {
		return " and pushed"
	}
	return ""
}

func short(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}
