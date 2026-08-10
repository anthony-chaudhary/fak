package devcmd

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// issueCreateRunner is the injectable gh-argv executor behind runIssueCreateWith — same
// shape as taskHandoffRunner (taskmgr.go:174) so a nil runner can default straight to the
// existing runTaskHandoffGH (taskmgr.go:343) instead of a second exec.Command wrapper.
type issueCreateRunner func(args []string) (stdout, stderr string, ok bool)

// issueCreateResult is the --json shape: the rendered gh argv is always included (even on
// a dry run) so a caller can see exactly what would run or did run.
type issueCreateResult struct {
	OK     bool     `json:"ok"`
	DryRun bool     `json:"dry_run"`
	Title  string   `json:"title"`
	Repo   string   `json:"repo,omitempty"`
	Labels []string `json:"labels,omitempty"`
	Args   []string `json:"args"`
	URL    string   `json:"url,omitempty"`
	Error  string   `json:"error,omitempty"`
}

func runIssueCreate(stdout, stderr io.Writer, argv []string) int {
	return runIssueCreateWith(stdout, stderr, argv, nil)
}

// runIssueCreateWith is fak's smooth, structurally-ungated path for filing one GitHub
// issue: it shells to `gh issue create` from the trusted fak binary (runner defaults to
// runTaskHandoffGH, taskmgr.go:343-351) rather than the model proposing raw `gh issue
// create` via Bash — which is exactly the invocation internal/adjudicator/reversibility.go
// deliberately escalates (REQUIRE_WITNESS/ESCALATE, by design, not a false positive). A
// compiled verb shelling out internally was never in scope for that classifier, the same
// way `fak commit`/`fak sync push` already sidestep the `git push` family. The runner
// param exists purely so tests can inject a fake and assert on the built gh argv without
// ever invoking real gh.
func runIssueCreateWith(stdout, stderr io.Writer, argv []string, runner issueCreateRunner) int {
	fs := flag.NewFlagSet("issue create", flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "issue title (required)")
	body := fs.String("body", "", "issue body text (exactly one of --body/--body-file)")
	bodyFile := fs.String("body-file", "", "path to a file containing the issue body")
	labels := fs.String("labels", "", "comma-separated labels")
	estimatePoints := fs.Float64("estimate-points", 0, "project-work estimate points (required when body lacks Work estimate)")
	parentBaselinePoints := fs.Float64("parent-baseline-points", 0, "parent production-scope baseline points (required when body lacks Overall completion contribution)")
	contributionPoints := fs.Float64("contribution-points", 0, "parent contribution points (default: estimate-points)")
	completionStandard := fs.String("completion-standard", "production", "production|integrated|staging|development|demo|prototype|experiment|research")
	targetEnvelope := fs.String("target-envelope", "", "production target envelope entries (required unless body declares them)")
	witnessedEnvelope := fs.String("witnessed-envelope", "", "directly witnessed envelope entries (required unless body declares them)")
	rawBody := fs.Bool("raw-body", false, "do not append/review project-work metadata (only for non-dispatchable administrative issues)")
	repo := fs.String("repo", "", "owner/name override (default: gh infers from cwd)")
	dryRun := fs.Bool("dry-run", false, "render the issue + gh argv without calling gh")
	asJSON := fs.Bool("json", false, "emit the machine-readable result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue create: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*title) == "" {
		fmt.Fprintln(stderr, "fak-dev issue create: --title is required")
		return 2
	}
	if strings.TrimSpace(*body) != "" && strings.TrimSpace(*bodyFile) != "" {
		fmt.Fprintln(stderr, "fak-dev issue create: pass exactly one of --body or --body-file")
		return 2
	}
	resolvedBody := *body
	if strings.TrimSpace(*bodyFile) != "" {
		b, err := os.ReadFile(*bodyFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue create: read --body-file: %v\n", err)
			return 2
		}
		resolvedBody = string(b)
	}
	if strings.TrimSpace(resolvedBody) == "" {
		fmt.Fprintln(stderr, "fak-dev issue create: --body or --body-file is required")
		return 2
	}
	if !*rawBody {
		var err error
		resolvedBody, err = issuepolicy.AppendProjectWorkDefaults(resolvedBody, issuepolicy.ProjectWorkAuthoring{
			EstimatePoints: *estimatePoints, ParentBaseline: *parentBaselinePoints,
			ContributionPoints: *contributionPoints, CompletionStandard: *completionStandard,
			TargetEnvelope: *targetEnvelope, WitnessedEnvelope: *witnessedEnvelope,
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue create: %v\n", err)
			return 2
		}
		review := issuepolicy.ReviewIssueDraft(issuepolicy.IssueDraft{Title: strings.TrimSpace(*title), Body: resolvedBody}, issuepolicy.Options{StrictProjectWork: true})
		if review.ProjectWork.Status != issuepolicy.ProjectWorkValid {
			fmt.Fprintf(stderr, "fak-dev issue create: generated project-work contract is %s: %s\n", review.ProjectWork.Status, strings.Join(review.ProjectWork.Invalid, "; "))
			return 2
		}
		if review.OperatingEnvelope.Required && review.OperatingEnvelope.Status != issuepolicy.EnvelopeMet {
			fmt.Fprintf(stderr, "fak-dev issue create: generated production operating envelope is %s; gaps=%d invalid=%s\n", review.OperatingEnvelope.Status, len(review.OperatingEnvelope.Gaps), strings.Join(review.OperatingEnvelope.Invalid, "; "))
			return 2
		}
	}

	labelList := issueFanoutSplit(*labels)
	args := []string{"issue", "create", "--title", *title, "--body", resolvedBody}
	for _, l := range labelList {
		args = append(args, "--label", l)
	}
	if strings.TrimSpace(*repo) != "" {
		args = append(args, "--repo", *repo)
	}

	result := issueCreateResult{DryRun: *dryRun, Title: *title, Repo: *repo, Labels: labelList, Args: args}

	if *dryRun {
		result.OK = true
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue create")
		}
		fmt.Fprintf(stdout, "fak-dev issue create --dry-run: would run `gh %s`\n", strings.Join(args, " "))
		return 0
	}

	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	out, errOut, ok := run(args)
	result.OK = ok
	result.URL = strings.TrimSpace(out)
	if !ok {
		result.Error = strings.TrimSpace(errOut)
		if *asJSON {
			encodeJSONOrFail(stdout, stderr, result, "fak-dev issue create")
		} else {
			fmt.Fprintf(stderr, "fak-dev issue create: gh failed: %s\n", result.Error)
		}
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue create")
	}
	fmt.Fprintln(stdout, result.URL)
	return 0
}
