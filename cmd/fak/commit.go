package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitintent"
	"github.com/anthony-chaudhary/fak/internal/commitrollup"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// commitFn is the seam the CLI shim calls; it defaults to the real safecommit.Commit and
// is overridden in tests so runCommit is exercised without a real git or repo.
var commitFn = safecommit.Commit

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
	noSignoff := fs.Bool("no-signoff", false, "do not add the DCO sign-off (-s is the default)")
	preview := fs.Bool("preview", false, "LINT-ONLY: check the message+paths and exit WITHOUT touching git (is the subject witness-gradeable, does it carry a bindable `(fak <leaf>)` stamp, does the leaf match the paths' lane?). Exit 0 clean, 1 issues, 2 usage")
	requireIssue := fs.Bool("require-issue", false, "treat a missing bindable issue link (#N in subject / `Closes #N` in body) as BLOCKING, not advisory — the dispatch-worker contract so a close binds in `issue_closure_audit` (#312)")
	noBuildCheck := fs.Bool("no-build-check", false, "skip the COMMITTED_RED prospective-tree compile gate before the commit (default: gate ON — refuses a commit that would red the committed trunk)")
	reviewModel := fs.String("review-model", envOrDefault("FAK_REVIEW_MODEL", ""), "optional scout model id, or comma-separated model ids, that must pass/refute this diff before commit; a multi-model quorum blocks on any refute")
	reviewMinModels := fs.Int("review-min-models", envIntOrDefault("FAK_REVIEW_MIN_MODELS", 0), "minimum usable review verdicts required when --review-model names multiple models (default: 2, or 1 for a single model)")
	reviewObjective := fs.String("review-objective", envOrDefault("FAK_REVIEW_OBJECTIVE", ""), "objective given to --review-model (default: FAK_GOAL_OBJECTIVE, then first commit-message line)")
	reviewEndpoint := fs.String("review-endpoint", envOrDefault("FAK_REVIEW_ENDPOINT", "http://127.0.0.1:8080/v1"), "OpenAI-compatible base URL for --review-model")
	reviewAPIKeyEnv := fs.String("review-api-key-env", envOrDefault("FAK_REVIEW_API_KEY_ENV", "FAK_REVIEW_API_KEY"), "env var holding the bearer token for --review-endpoint (empty value sends no token)")
	coreLockWitness := fs.String("core-lock-maintenance-witness", "", "independent witness claim that clears a hard-self core-lock maintenance commit")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *reviewMinModels < 0 {
		fmt.Fprintln(stderr, "fak commit: --review-min-models must be non-negative")
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	paths = append(paths, fs.Args()...)

	// --preview is a no-op dry run: lint the message + paths so a bad subject/stamp is caught
	// BEFORE the commit lands (on the shared trunk you cannot amend — a sibling may push your
	// local commit first). It needs a message but tolerates zero paths (the lane match is then
	// skipped with a note).
	if *preview {
		message, code := assembleMessage(stdin(), msg.Joined(), *msgFile, stderr)
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

	message, code := assembleMessage(stdin(), msg.Joined(), *msgFile, stderr)
	if code != 0 {
		return code
	}
	root := resolveRoot(*dir)
	if derived, ok := deriveCommitMessageStamp(message, paths, root); ok {
		message = derived
	}
	review := commitReviewOptions(*reviewModel, firstNonEmpty(*reviewObjective, os.Getenv("FAK_GOAL_OBJECTIVE"), firstCommitLine(message)), *reviewEndpoint, *reviewAPIKeyEnv, *reviewMinModels)

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

	// COMMITTED_RED gate (#4152): compile the PROSPECTIVE committed tree — HEAD's committed
	// bytes + exactly this commit's paths, all other working-tree noise masked — and refuse the
	// commit when it would red the committed trunk under default tags. Promotes the
	// internal/buildwitness CI invariant to the commit boundary. Fails OPEN on infra error and on
	// a pre-existing HEAD red (it never refuses on an inability to check, and never blocks a red
	// this commit did not introduce).
	if !*noBuildCheck && os.Getenv("FAK_COMMIT_BUILD_CHECK") != "off" {
		if okB, reason, detail := commitBuildCheckGate(stderr, root, paths); !okB {
			fmt.Fprintf(stderr, "fak commit: %s\n%s\n", reason, strings.TrimSpace(detail))
			fmt.Fprintln(stderr, "fak commit: the prospective committed tree does not compile under default tags — commit refused so the committed trunk stays green. Commit the missing definition too, or fence not-yet-compiling WIP behind //go:build wip_<feature> (see `fak wip fence`), or pass --no-build-check for an intentional multi-commit landing.")
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
	case safecommit.ReasonNotARepo:
		// A setup/environment error: nothing landed, safe to retry once fixed.
		return safecommit.ExitPreCommitRefusal
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
		renderCommitVelocity(stdout, res)
		renderCommitReview(stdout, res)
		return
	}
	fmt.Fprintf(stdout, "%s", res.Reason)
	if res.Detail != "" {
		fmt.Fprintf(stdout, ": %s", res.Detail)
	}
	fmt.Fprintln(stdout)
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
