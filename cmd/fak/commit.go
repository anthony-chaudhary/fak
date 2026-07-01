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
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// commitFn is the seam the CLI shim calls; it defaults to the real safecommit.Commit and
// is overridden in tests so runCommit is exercised without a real git or repo.
var commitFn = safecommit.Commit

func cmdCommit(argv []string) { os.Exit(runCommitCommand(os.Stdout, os.Stderr, argv)) }

func runCommitCommand(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runCommit(stdout, stderr, argv)
	}
	switch argv[0] {
	case "status":
		return runCommitStatus(stdout, stderr, argv[1:])
	case "submit":
		return runCommitSubmit(stdout, stderr, argv[1:])
	case "drain":
		return runCommitDrain(stdout, stderr, argv[1:])
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

// runCommit is the `fak commit` shim: it assembles a safecommit.Options from flags
// (message from -m / -F / stdin; paths from repeated --path AND/OR positionals after --),
// runs the safe-commit algorithm, and reports the structured Result. Exit codes mirror the
// loop verb's discipline: 0 success; 2 usage error; 3 a PRE-commit refusal (blocked, safe
// to retry/replan); 1 a POST-attempt failure (the commit ran but its result is bad — halt)
// or an infrastructure error.
func runCommit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative path to commit (repeatable); paths may also be given after --")
	msg := fs.String("m", "", "commit message (mutually exclusive with -F)")
	msgFile := fs.String("F", "", "read the commit message from this file ('-' = stdin)")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	trunk := fs.String("trunk", "", "expected development branch override (default: configured development branch)")
	push := fs.Bool("push", false, "push after a VERIFIED commit (plain push, never --force)")
	noSignoff := fs.Bool("no-signoff", false, "do not add the DCO sign-off (-s is the default)")
	preview := fs.Bool("preview", false, "LINT-ONLY: check the message+paths and exit WITHOUT touching git (is the subject witness-gradeable, does it carry a bindable `(fak <leaf>)` stamp, does the leaf match the paths' lane?). Exit 0 clean, 1 issues, 2 usage")
	requireIssue := fs.Bool("require-issue", false, "treat a missing bindable issue link (#N in subject / `Closes #N` in body) as BLOCKING, not advisory — the dispatch-worker contract so a close binds in `issue_closure_audit` (#312)")
	reviewModel := fs.String("review-model", envOrDefault("FAK_REVIEW_MODEL", ""), "optional scout model id that must pass/refute this diff before commit; reviewer errors fail open and are recorded")
	reviewObjective := fs.String("review-objective", envOrDefault("FAK_REVIEW_OBJECTIVE", ""), "objective given to --review-model (default: FAK_GOAL_OBJECTIVE, then first commit-message line)")
	reviewEndpoint := fs.String("review-endpoint", envOrDefault("FAK_REVIEW_ENDPOINT", "http://127.0.0.1:8080/v1"), "OpenAI-compatible base URL for --review-model")
	reviewAPIKeyEnv := fs.String("review-api-key-env", envOrDefault("FAK_REVIEW_API_KEY_ENV", "FAK_REVIEW_API_KEY"), "env var holding the bearer token for --review-endpoint (empty value sends no token)")
	coreLockWitness := fs.String("core-lock-maintenance-witness", "", "independent witness claim that clears a hard-self core-lock maintenance commit")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	paths = append(paths, fs.Args()...)

	// --preview is a no-op dry run: lint the message + paths so a bad subject/stamp is caught
	// BEFORE the commit lands (on the shared trunk you cannot amend — a sibling may push your
	// local commit first). It needs a message but tolerates zero paths (the lane match is then
	// skipped with a note).
	if *preview {
		message, code := assembleMessage(stdin(), *msg, *msgFile, stderr)
		if code != 0 {
			return code
		}
		root := resolveRoot(*dir)
		return runCommitPreview(stdout, stderr, message, paths, root, safecommit.ExpectedTrunk(root, *trunk), *asJSON, *requireIssue)
	}

	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak commit: at least one --path (or a path after --) is required")
		return 2
	}

	message, code := assembleMessage(stdin(), *msg, *msgFile, stderr)
	if code != 0 {
		return code
	}
	root := resolveRoot(*dir)
	if derived, ok := deriveCommitMessageStamp(message, paths, root); ok {
		message = derived
	}
	review := commitReviewOptions(*reviewModel, firstNonEmpty(*reviewObjective, os.Getenv("FAK_GOAL_OBJECTIVE"), firstCommitLine(message)), *reviewEndpoint, *reviewAPIKeyEnv)

	// --require-issue pre-lints the message before touching git: a real commit on the shared trunk
	// cannot be amended (a sibling may push it first), so a missing bindable `#N` is caught here as a
	// PRE-commit refusal (exit 3) rather than discovered weeks later as a CLAIMED_CLOSED row (#312).
	if *requireIssue {
		rep := hooks.LintCommitMessageWithOptions(message, paths, root, true)
		if !rep.OK {
			fmt.Fprintln(stderr, "fak commit: --require-issue refused this commit:")
			renderPreview(stderr, rep, "")
			return 3
		}
	}

	res, err := commitFn(context.Background(), safecommit.Options{
		Dir:                        *dir,
		Paths:                      paths,
		Message:                    message,
		Trunk:                      *trunk,
		SignOff:                    !*noSignoff,
		Push:                       *push,
		Review:                     review,
		CoreLockMaintenanceWitness: *coreLockWitness,
	})
	if err != nil {
		// Infrastructure failure (git not executable, lock unopenable): not a refusal.
		fmt.Fprintf(stderr, "fak commit: %v\n", err)
		return 1
	}
	res = safecommit.ScoreResult(res)
	if res.Review != nil {
		if err := recordCommitReviewForLoop(res); err != nil {
			fmt.Fprintf(stderr, "fak commit: record review evidence: %v\n", err)
		}
		if err := appendCommitReviewRefusalToGoal(res); err != nil {
			fmt.Fprintf(stderr, "fak commit: append review refusal: %v\n", err)
		}
	}

	if *asJSON {
		if encErr := writeIndentedJSON(stdout, res); encErr != nil {
			fmt.Fprintf(stderr, "fak commit: %v\n", encErr)
			return 1
		}
	} else {
		renderCommitResult(stdout, res)
	}
	return commitExitCode(res)
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
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative path for the future commit (repeatable); paths may also be given after --")
	msg := fs.String("m", "", "commit subject for the intent (mutually exclusive with -F)")
	msgFile := fs.String("F", "", "read the commit subject from this file ('-' = stdin)")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	queueDir := fs.String("queue-dir", "", "commit-intent queue dir (default: <repo>/.fak/commit-intents)")
	id := fs.String("id", "", "stable intent id (default: generated intent-<unix-nanos>)")
	base := fs.String("base", "", "base SHA the intent was authored against (default: git rev-parse HEAD)")
	diffDigest := fs.String("diff-digest", "", "optional sha256:<hex> digest of the authored diff")
	asJSON := fs.Bool("json", false, "emit the submitted record as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	*queueDir = pathutil.ExpandTilde(*queueDir)
	paths = append(paths, fs.Args()...)
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak commit submit: at least one --path (or a path after --) is required")
		return 2
	}
	subject, code := assembleMessage(stdin(), *msg, *msgFile, stderr)
	if code != 0 {
		return code
	}
	root := resolveRoot(*dir)
	if *queueDir == "" {
		*queueDir = commitintent.DefaultQueueDir(root)
	}
	baseSHA := strings.TrimSpace(*base)
	if baseSHA == "" {
		var err error
		baseSHA, err = commitSubmitHeadSHA(root)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit submit: resolve base sha: %v\n", err)
			return 1
		}
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
		return 3
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
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	queueDir := fs.String("queue-dir", "", "commit-intent queue dir (default: <repo>/.fak/commit-intents)")
	base := fs.String("base", "", "current base SHA (default: git rev-parse HEAD)")
	max := fs.Int("max", 0, "maximum ready intents to consider (0 = all pending ready intents)")
	trunk := fs.String("trunk", "", "expected development branch override (default: configured development branch)")
	push := fs.Bool("push", false, "push after a VERIFIED rollup commit (plain push, never --force)")
	noSignoff := fs.Bool("no-signoff", false, "do not add the DCO sign-off (-s is the default)")
	noRollup := fs.Bool("no-rollup", false, "disable batching and drain at most one compatible intent")
	dryRun := fs.Bool("dry-run", false, "plan only; do not commit or update queue state")
	asJSON := fs.Bool("json", false, "emit the drain result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if len(fs.Args()) != 0 {
		fmt.Fprintf(stderr, "fak commit drain: unexpected argument %q\n", fs.Args()[0])
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	*queueDir = pathutil.ExpandTilde(*queueDir)
	root := resolveRoot(*dir)
	if *queueDir == "" {
		*queueDir = commitintent.DefaultQueueDir(root)
	}
	baseSHA := strings.TrimSpace(*base)
	if baseSHA == "" {
		var err error
		baseSHA, err = commitSubmitHeadSHA(root)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit drain: resolve base sha: %v\n", err)
			return 1
		}
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
		return 3
	}

	commitRes, err := commitFn(context.Background(), safecommit.Options{
		Dir:     root,
		Paths:   plan.UnionPaths,
		Message: plan.Subject,
		Trunk:   *trunk,
		SignOff: !*noSignoff,
		Push:    *push,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak commit drain: %v\n", err)
		return 1
	}
	commitRes = safecommit.ScoreResult(commitRes)
	pathset := plan.AssertPathset(commitRes.Paths)
	res.Commit = &commitRes
	res.Pathset = &pathset

	if commitRes.Reason == "" && commitRes.Verified && pathset.OK {
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
	}
	if res.Pathset != nil && !res.Pathset.OK {
		fmt.Fprintf(stdout, "  pathset mismatch: missing=%v extra=%v\n", res.Pathset.Missing, res.Pathset.Extra)
	}
}

// stdin is overridable in tests; defaults to os.Stdin.
var stdin = func() io.Reader { return os.Stdin }

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

// commitExitCode maps a Result to the process exit code. PRE-commit refusals are exit 3
// ("blocked — retry or replan"); a commit that ran but produced a bad result (race, push
// rejection, hook refusal) is exit 1 ("ran, result is bad — halt").
func commitExitCode(res safecommit.Result) int {
	switch res.Reason {
	case "":
		return 0
	case safecommit.ReasonNoPath, safecommit.ReasonEmptyMessage:
		return 2
	case safecommit.ReasonNotARepo, safecommit.ReasonOffTrunk,
		safecommit.ReasonMergeInProgress, safecommit.ReasonNothingStaged,
		safecommit.ReasonLockBusy, safecommit.ReasonWindowFull,
		safecommit.ReasonReviewRefuted, safecommit.ReasonStaleBaseDeletion,
		safecommit.ReasonCachedRemoveWorktreePresent,
		safecommit.ReasonSpuriousStagedDeletion, safecommit.ReasonPreStagedPathOverlap,
		safecommit.ReasonCoreSelfModify:
		return 3
	default: // PATHSPEC_RACE, HOOK_REFUSED, PUSH_REJECTED
		return 1
	}
}

func renderCommitResult(stdout io.Writer, res safecommit.Result) {
	if res.Reason == "" {
		fmt.Fprintf(stdout, "committed %s (%d path(s))%s\n", short(res.SHA), len(res.Paths), pushedSuffix(res))
		renderCommitScore(stdout, res)
		renderCommitReview(stdout, res)
		return
	}
	fmt.Fprintf(stdout, "%s", res.Reason)
	if res.Detail != "" {
		fmt.Fprintf(stdout, ": %s", res.Detail)
	}
	fmt.Fprintln(stdout)
	renderCommitScore(stdout, res)
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
