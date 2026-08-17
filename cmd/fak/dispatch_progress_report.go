package main

import (
	"fmt"
	"time"
)

type dispatchProgressSummary struct {
	Available       bool   `json:"available"`
	Verdict         string `json:"verdict,omitempty"`
	Commits         int    `json:"commits,omitempty"`
	IssuesClosed    int    `json:"issues_closed,omitempty"`
	WIPFiles        int    `json:"wip_files,omitempty"`
	OpenIssues      int    `json:"open_issues,omitempty"`
	GitHubAvailable bool   `json:"github_available"`
	Reason          string `json:"reason,omitempty"`
	NextAction      string `json:"next_action,omitempty"`
	Error           string `json:"error,omitempty"`
}

func dispatchProgressFold(root string) *dispatchProgressSummary {
	r, err := collectProgress(root, defaultProgressWindow, defaultProgressBaseline, defaultProgressStallAfter, time.Now())
	if err != nil {
		return &dispatchProgressSummary{Error: err.Error()}
	}
	return &dispatchProgressSummary{
		Available: true, Verdict: r.Verdict, Commits: r.Commits,
		IssuesClosed: r.GitHub.RecentlyClosed, WIPFiles: r.WIP.Files,
		OpenIssues: r.GitHub.OpenTotal, GitHubAvailable: r.GitHub.Available,
		Reason: r.Reason, NextAction: r.NextAction,
	}
}

func dispatchProgressLine(p *dispatchProgressSummary) string {
	if p == nil {
		return ""
	}
	if !p.Available {
		return fmt.Sprintf("progress unavailable: %s", p.Error)
	}
	github := fmt.Sprintf("closed=%d open=%d", p.IssuesClosed, p.OpenIssues)
	if !p.GitHubAvailable {
		github = "GitHub=unavailable"
	}
	return fmt.Sprintf("progress %s: commits=%d wip=%d %s; next=%s", p.Verdict, p.Commits, p.WIPFiles, github, p.NextAction)
}
