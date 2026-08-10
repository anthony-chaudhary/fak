package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/projectcompletion"
)

type projectCompletionIssue struct {
	Number int                      `json:"number"`
	Title  string                   `json:"title"`
	Body   string                   `json:"body"`
	State  string                   `json:"state"`
	Labels []issuepolicy.IssueLabel `json:"labels,omitempty"`
	URL    string                   `json:"url,omitempty"`
}

func RunProjectCompletion(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak-dev project completion", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromIssues := fs.String("from-issues", "", "GitHub issue JSON (number,title,body,state,labels)")
	asJSON := fs.Bool("json", false, "emit machine-readable weighted completion")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*fromIssues) == "" {
		fmt.Fprintln(stderr, "fak-dev project completion: --from-issues ISSUES.json is required")
		return 2
	}
	var b []byte
	var err error
	if *fromIssues == "-" {
		b, err = io.ReadAll(os.Stdin)
	} else {
		b, err = os.ReadFile(*fromIssues)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev project completion: %v\n", err)
		return 2
	}
	var rows []projectCompletionIssue
	if err = json.Unmarshal(b, &rows); err != nil {
		fmt.Fprintf(stderr, "fak-dev project completion: decode: %v\n", err)
		return 2
	}
	issues := make([]projectcompletion.Issue, 0, len(rows))
	for _, row := range rows {
		review := issuepolicy.ReviewIssueDraft(issuepolicy.IssueDraft{Number: row.Number, Title: row.Title, Body: row.Body, Labels: row.Labels, URL: row.URL}, issuepolicy.Options{})
		issues = append(issues, projectcompletion.Issue{Number: row.Number, Title: row.Title, State: row.State, ProjectWork: review.ProjectWork})
	}
	report := projectcompletion.Summarize(issues)
	if *asJSON {
		return writeJSON(stdout, report)
	}
	// The human render lives in the leaf (projectcompletion.RenderText) so its
	// no-bare-"complete" maturity labeling is testable beside the fold (#4640).
	fmt.Fprint(stdout, projectcompletion.RenderText(report))
	if report.Confidence != "complete" {
		return 1
	}
	return 0
}
