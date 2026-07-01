package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuesmallness"
)

type dispatchIssueSmallnessSingleReport struct {
	Schema string `json:"schema"`
	Mode   string `json:"mode"`
	Issue  int    `json:"issue,omitempty"`
	issuesmallness.Result
}

var (
	dispatchIssueSmallnessFetchIssueBody  = fetchDispatchIssueSmallnessIssueBody
	dispatchIssueSmallnessFetchOpenIssues = fetchDispatchIssueSmallnessOpenIssues
)

func runDispatchIssueSmallnessLint(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("dispatch issue-smallness-lint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	bodyFile := fs.String("body-file", "", `lint a single issue body file ("-" = stdin)`)
	issue := fs.Int("issue", 0, "fetch and lint issue #N via gh")
	open := fs.Bool("open", false, "fetch and lint open issues via gh (dry-run backlog report)")
	limit := fs.Int("limit", 500, "max open issues to scan with --open")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	asScorecard := fs.Bool("scorecard", false, "with --open: fold the rated backlog into a control-pane payload for `fak scoreboard post --from -`")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *asScorecard && !*open {
		fmt.Fprintln(stderr, "fak dispatch issue-smallness-lint: --scorecard only applies to --open")
		return 2
	}

	modes := 0
	if strings.TrimSpace(*bodyFile) != "" {
		modes++
	}
	if *issue > 0 {
		modes++
	}
	if *open {
		modes++
	}
	if modes != 1 {
		fmt.Fprintln(stderr, "fak dispatch issue-smallness-lint: choose exactly one of --body-file, --issue, or --open")
		return 2
	}

	switch {
	case strings.TrimSpace(*bodyFile) != "":
		body, err := readDispatchIssueSmallnessBody(stdin, *bodyFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch issue-smallness-lint: %v\n", err)
			return 2
		}
		report := dispatchIssueSmallnessSingleReport{
			Schema: issuesmallness.Schema,
			Mode:   "body-file",
			Result: issuesmallness.LintBody(body),
		}
		return writeDispatchIssueSmallnessSingle(stdout, stderr, report, *asJSON)
	case *issue > 0:
		body, err := dispatchIssueSmallnessFetchIssueBody(*issue)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch issue-smallness-lint: %v\n", err)
			return 2
		}
		report := dispatchIssueSmallnessSingleReport{
			Schema: issuesmallness.Schema,
			Mode:   "issue",
			Issue:  *issue,
			Result: issuesmallness.LintBody(body),
		}
		return writeDispatchIssueSmallnessSingle(stdout, stderr, report, *asJSON)
	default:
		issues, err := dispatchIssueSmallnessFetchOpenIssues(*limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch issue-smallness-lint: %v\n", err)
			return 2
		}
		report := issuesmallness.ReportOpen(issues)
		if *asScorecard {
			// The card carries its own OK/ACTION verdict + debt; emit it and return 0 so
			// it pipes cleanly into `fak scoreboard post --from -`. The gate lives on the
			// consumer that reads corpus[issue_smallness_debt], not on this producer.
			if err := writeIndentedJSON(stdout, issueSmallnessScorecard(report)); err != nil {
				fmt.Fprintf(stderr, "fak dispatch issue-smallness-lint: encode json: %v\n", err)
				return 1
			}
			return 0
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, report); err != nil {
				fmt.Fprintf(stderr, "fak dispatch issue-smallness-lint: encode json: %v\n", err)
				return 1
			}
		} else {
			writeDispatchIssueSmallnessOpen(stdout, report)
		}
		if issuesmallness.HasFailReport(report) {
			return 1
		}
		return 0
	}
}

func readDispatchIssueSmallnessBody(stdin io.Reader, path string) (string, error) {
	if strings.TrimSpace(path) == "-" {
		b, err := io.ReadAll(stdin)
		return string(b), err
	}
	b, err := os.ReadFile(path)
	return string(b), err
}

func writeDispatchIssueSmallnessSingle(stdout, stderr io.Writer, report dispatchIssueSmallnessSingleReport, asJSON bool) int {
	if asJSON {
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak dispatch issue-smallness-lint: encode json: %v\n", err)
			return 1
		}
	} else {
		label := report.Mode
		if report.Issue != 0 {
			label = fmt.Sprintf("#%d", report.Issue)
		}
		fmt.Fprintf(stdout, "issue-smallness-lint [%s]: %s - %s\n", label, strings.ToUpper(report.Verdict), report.Reason)
		for _, item := range report.Items {
			fmt.Fprintf(stdout, "  - %s\n", item)
		}
	}
	if report.Verdict == issuesmallness.Fail {
		return 1
	}
	return 0
}

func writeDispatchIssueSmallnessOpen(w io.Writer, report issuesmallness.OpenReport) {
	fmt.Fprintf(w, "issue-smallness-lint - scanned %d open issue(s): %d pass, %d warn, %d fail\n",
		report.Scanned, report.Counts[issuesmallness.Pass], report.Counts[issuesmallness.Warn], report.Counts[issuesmallness.Fail])
	for _, row := range report.Flagged {
		fmt.Fprintf(w, "  [%s] #%d %q -> %s\n", strings.ToUpper(row.Verdict), row.Number, row.Title, row.Reason)
	}
}

func fetchDispatchIssueSmallnessIssueBody(number int) (string, error) {
	var data struct {
		Body string `json:"body"`
	}
	if err := dispatchIssueSmallnessGHJSON(&data, "issue", "view", fmt.Sprint(number), "--json", "body"); err != nil {
		return "", err
	}
	return data.Body, nil
}

func fetchDispatchIssueSmallnessOpenIssues(limit int) ([]issuesmallness.Issue, error) {
	var rows []issuesmallness.Issue
	if err := dispatchIssueSmallnessGHJSON(&rows, "issue", "list", "--state", "open", "--limit", fmt.Sprint(limit), "--json", "number,title,body"); err != nil {
		return nil, err
	}
	return rows, nil
}

func dispatchIssueSmallnessGHJSON(out any, args ...string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	configureDispatchHelperCommand(cmd)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("gh %s: %w (%s)", strings.Join(args, " "), err, strings.TrimSpace(string(b)))
	}
	if err := json.Unmarshal(b, out); err != nil {
		return fmt.Errorf("gh %s produced invalid JSON: %w", strings.Join(args, " "), err)
	}
	return nil
}
