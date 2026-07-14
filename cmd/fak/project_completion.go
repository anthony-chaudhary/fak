package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
	"github.com/anthony-chaudhary/fak/internal/projectcompletion"
)

type projectCompletionIssue struct {
	Number int                        `json:"number"`
	Title  string                     `json:"title"`
	Body   string                     `json:"body"`
	State  string                     `json:"state"`
	Labels []issuecontract.IssueLabel `json:"labels,omitempty"`
	URL    string                     `json:"url,omitempty"`
}

func runProjectCompletion(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak project completion", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromIssues := fs.String("from-issues", "", "GitHub issue JSON (number,title,body,state,labels)")
	asJSON := fs.Bool("json", false, "emit machine-readable weighted completion")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*fromIssues) == "" {
		fmt.Fprintln(stderr, "fak project completion: --from-issues ISSUES.json is required")
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
		fmt.Fprintf(stderr, "fak project completion: %v\n", err)
		return 2
	}
	var rows []projectCompletionIssue
	if err = json.Unmarshal(b, &rows); err != nil {
		fmt.Fprintf(stderr, "fak project completion: decode: %v\n", err)
		return 2
	}
	issues := make([]projectcompletion.Issue, 0, len(rows))
	for _, row := range rows {
		review := issuecontract.ReviewIssueDraft(issuecontract.IssueDraft{Number: row.Number, Title: row.Title, Body: row.Body, Labels: row.Labels, URL: row.URL}, issuecontract.Options{})
		issues = append(issues, projectcompletion.Issue{Number: row.Number, Title: row.Title, State: row.State, ProjectWork: review.ProjectWork})
	}
	report := projectcompletion.Summarize(issues)
	if *asJSON {
		return writeJSON(stdout, report)
	}
	fmt.Fprintf(stdout, "production complete: %.2f/%.2f points (%.1f%%) [%s]\n", report.ProductionCompletePoints, report.BaselinePoints, report.ProductionCompletePct, report.Confidence)
	for _, b := range report.ClosedByStandard {
		fmt.Fprintf(stdout, "closed %-12s %.2f points (%d issues)\n", b.Standard, b.Points, b.Issues)
	}
	fmt.Fprintf(stdout, "open: %.2f points; declared: %.2f points\n", report.OpenPoints, report.DeclaredContribution)
	for _, u := range report.Unknown {
		fmt.Fprintf(stdout, "unknown #%d %s: %s\n", u.Number, u.Title, u.Status)
	}
	for _, d := range report.DenominatorDrift {
		fmt.Fprintf(stdout, "denominator drift: %s\n", d)
	}
	if report.Confidence != "complete" {
		return 1
	}
	return 0
}
