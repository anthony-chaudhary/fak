package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardroute"
)

func TestRouteFileIssueFailureDoesNotClaimCreated(t *testing.T) {
	origSync := dogfoodIssuesSync
	origFetch := dogfoodIssuesFetchExisting
	defer func() {
		dogfoodIssuesSync = origSync
		dogfoodIssuesFetchExisting = origFetch
	}()

	dogfoodIssuesFetchExisting = func(repo string, limit int) ([]dogfoodissues.Issue, error) {
		return nil, nil
	}

	// Simulate sync failure
	dogfoodIssuesSync = func(plan []dogfoodissues.PlanRow, repo string, labels []string, runner dogfoodissues.Runner) []dogfoodissues.SyncRow {
		return []dogfoodissues.SyncRow{
			{
				Key:    plan[0].Key,
				Action: "create",
				OK:     false,
				Stderr: "network timeout connecting to api.github.com",
			},
		}
	}

	decision := guardroute.RouteDecision{
		Route:     true,
		FileIssue: true,
		CauseKey:  "guard-journal:unknown_verdict",
	}

	var stderr bytes.Buffer
	res := routeFileIssue(&stderr, decision, []string{"audit.jsonl"}, "test/repo", false)

	if res["ok"] != false {
		t.Fatalf("expected ok=false, got %v", res["ok"])
	}
	if res["reason"] != "issue_sync_failed" {
		t.Fatalf("expected reason=issue_sync_failed, got %v", res["reason"])
	}
	summary, _ := res["summary"].(string)
	if strings.HasPrefix(summary, "created issue") {
		t.Fatalf("summary falsely claimed created issue: %q", summary)
	}
	if !strings.Contains(summary, "failed to create issue") || !strings.Contains(summary, "network timeout") {
		t.Fatalf("summary missing failure context: %q", summary)
	}
}

func TestRouteFileIssueSuccessReportsNumberAndURL(t *testing.T) {
	origSync := dogfoodIssuesSync
	origFetch := dogfoodIssuesFetchExisting
	defer func() {
		dogfoodIssuesSync = origSync
		dogfoodIssuesFetchExisting = origFetch
	}()

	dogfoodIssuesFetchExisting = func(repo string, limit int) ([]dogfoodissues.Issue, error) {
		return nil, nil
	}

	issueNum := 1234
	issueURL := "https://github.com/test/repo/issues/1234"

	dogfoodIssuesSync = func(plan []dogfoodissues.PlanRow, repo string, labels []string, runner dogfoodissues.Runner) []dogfoodissues.SyncRow {
		return []dogfoodissues.SyncRow{
			{
				Key:      plan[0].Key,
				Action:   "create",
				OK:       true,
				Number:   &issueNum,
				URL:      issueURL,
				Verified: true,
			},
		}
	}

	decision := guardroute.RouteDecision{
		Route:     true,
		FileIssue: true,
		CauseKey:  "guard-journal:unknown_verdict",
	}

	var stderr bytes.Buffer
	res := routeFileIssue(&stderr, decision, []string{"audit.jsonl"}, "test/repo", false)

	if res["ok"] != true {
		t.Fatalf("expected ok=true, got %v", res["ok"])
	}
	if res["number"] != issueNum {
		t.Fatalf("expected number=%d, got %v", issueNum, res["number"])
	}
	if res["url"] != issueURL {
		t.Fatalf("expected url=%q, got %v", issueURL, res["url"])
	}
	summary, _ := res["summary"].(string)
	if !strings.Contains(summary, "#1234") || !strings.Contains(summary, issueURL) {
		t.Fatalf("summary missing number or URL: %q", summary)
	}
}
