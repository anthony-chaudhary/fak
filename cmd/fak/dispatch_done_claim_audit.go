package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/doneclaimaudit"
)

var doneClaimAuditCommand = func(name string, args ...string) ([]byte, error) {
	return exec.Command(name, args...).CombinedOutput()
}

func runDispatchDoneClaimAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dispatch done-claim-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", ".", "repository working tree")
	limit := fs.Int("limit", 100, "number of recently updated issues to inspect")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak dispatch done-claim-audit [--workspace DIR] [--limit N] [--json]")
		fmt.Fprintln(stderr, "Find GitHub issue comments that claim completion without showing a tracked diff or named untracked path.")
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *limit <= 0 {
		fmt.Fprintln(stderr, "done-claim-audit: --limit must be positive")
		return 2
	}

	issues, err := fetchDoneClaimAuditIssues(*workspace, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "done-claim-audit: %v\n", err)
		return 2
	}
	report := doneclaimaudit.Audit(issues, func(sha string) ([]string, bool) {
		out, err := doneClaimAuditCommand("git", "-C", *workspace, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
		if err != nil {
			return nil, false
		}
		paths := strings.Fields(strings.TrimSpace(string(out)))
		return paths, true
	})

	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "done-claim-audit: encode report: %v\n", err)
			return 2
		}
	} else {
		fmt.Fprintf(stdout, "done-claim-audit %s — issues=%d claims=%d findings=%d\n", report.Verdict, report.IssuesScanned, report.ClaimsScanned, len(report.Findings))
		for _, finding := range report.Findings {
			fmt.Fprintf(stdout, "#%d %s\n  %s\n  %s\n", finding.Number, finding.Title, finding.Reason, finding.CommentURL)
		}
	}
	if report.Verdict == "ACTION" {
		return 1
	}
	return 0
}

func fetchDoneClaimAuditIssues(workspace string, limit int) ([]doneclaimaudit.Issue, error) {
	out, err := doneClaimAuditCommand("gh", "issue", "list", "--state", "all", "--limit", strconv.Itoa(limit),
		"--json", "number,title,state,url,comments")
	if err != nil {
		return nil, fmt.Errorf("gh issue list: %w: %s", err, strings.TrimSpace(string(out)))
	}
	var issues []doneclaimaudit.Issue
	if err := json.Unmarshal(out, &issues); err != nil {
		return nil, fmt.Errorf("decode gh issue list: %w", err)
	}
	return issues, nil
}
