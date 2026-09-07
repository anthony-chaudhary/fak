package issueorchestrator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLiveIngest_ResolveViewQuery(t *testing.T) {
	// Create a temporary workspace with .github/issue-views.json
	dir := t.TempDir()
	githubDir := filepath.Join(dir, ".github")
	if err := os.MkdirAll(githubDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	viewsCfg := IssueViewsConfig{
		Version: 1,
		Repo:    "anthony-chaudhary/fak",
		Default: "ready-leaves",
		Limit:   300,
		Views: []IssueViewEntry{
			{
				Slug:  "ready-leaves",
				Title: "Ready leaves",
				Query: "is:open -label:epic no:assignee",
			},
			{
				Slug:  "p0-p1",
				Title: "P0/P1 leaves",
				Query: "is:open label:priority/P0,priority/P1",
			},
		},
	}
	b, _ := json.Marshal(viewsCfg)
	if err := os.WriteFile(filepath.Join(githubDir, "issue-views.json"), b, 0644); err != nil {
		t.Fatalf("write views: %v", err)
	}

	// 1. Default view
	q, repo, err := ResolveViewQuery(dir, "")
	if err != nil {
		t.Fatalf("resolve default view: %v", err)
	}
	if q != "is:open -label:epic no:assignee" {
		t.Errorf("unexpected query: %s", q)
	}
	if repo != "anthony-chaudhary/fak" {
		t.Errorf("unexpected repo: %s", repo)
	}

	// 2. Explicit view
	q, _, err = ResolveViewQuery(dir, "p0-p1")
	if err != nil {
		t.Fatalf("resolve p0-p1 view: %v", err)
	}
	if q != "is:open label:priority/P0,priority/P1" {
		t.Errorf("unexpected query: %s", q)
	}

	// 3. Unknown view
	_, _, err = ResolveViewQuery(dir, "nonexistent-view")
	if err == nil {
		t.Fatalf("expected error for nonexistent view, got nil")
	}
}

func makeMockPage(startNum, count int) []byte {
	var issues []Issue
	for i := 0; i < count; i++ {
		num := startNum + i
		issues = append(issues, testIssue(num, fmt.Sprintf("key-%d", num), fmt.Sprintf("Issue %d", num), "lane", []string{"internal/pkg/file.go"}, 2))
	}
	b, _ := json.Marshal(issues)
	return b
}

func TestLiveIngest_DynamicSoftPaging(t *testing.T) {
	pageCalls := 0
	mockFetcher := func(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
		pageCalls++
		if page == 1 {
			return makeMockPage(1, 5), false, nil
		}
		if page == 2 {
			return makeMockPage(6, 5), false, nil
		}
		return []byte("[]"), false, nil
	}

	// Requesting TargetIssues: 4. Page 1 returns 5 issues >= 4, so it should stop at page 1.
	opts := LiveIngestOptions{
		PageSize:     5,
		TargetIssues: 4,
		PageFetcher:  mockFetcher,
	}

	issues, err := FetchLiveIssues(context.Background(), opts)
	if err != nil {
		t.Fatalf("FetchLiveIssues: %v", err)
	}

	if len(issues) != 5 {
		t.Errorf("expected 5 issues from page 1, got %d", len(issues))
	}
	if pageCalls != 1 {
		t.Errorf("expected dynamic soft-paging to stop after page 1, called %d times", pageCalls)
	}
}

func TestLiveIngest_RateLimitSoftDegradation(t *testing.T) {
	var warnings []string
	logWarn := func(msg string) {
		warnings = append(warnings, msg)
	}

	mockFetcher := func(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
		if page == 1 {
			return makeMockPage(1, 3), false, nil
		}
		// Page 2 encounters persistent rate limit
		return nil, true, errors.New("HTTP 429: rate limit exceeded")
	}

	opts := LiveIngestOptions{
		PageSize:         3,
		MaxPages:         5,
		RateLimitTimeout: 100 * time.Millisecond,
		PageFetcher:      mockFetcher,
		LogWarning:       logWarn,
	}

	issues, err := FetchLiveIssues(context.Background(), opts)
	if err != nil {
		t.Fatalf("expected soft degradation without error, got %v", err)
	}

	if len(issues) != 3 {
		t.Errorf("expected 3 issues from page 1, got %d", len(issues))
	}

	foundAdvisory := false
	for _, w := range warnings {
		if strings.Contains(w, "[advisory] live ingestion soft-degraded") {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Errorf("expected soft-degraded advisory warning, got %v", warnings)
	}
}

func TestLiveIngest_MidStreamNetworkFault(t *testing.T) {
	var warnings []string
	logWarn := func(msg string) {
		warnings = append(warnings, msg)
	}

	mockFetcher := func(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
		if page == 1 {
			return makeMockPage(10, 4), false, nil
		}
		// Page 2 simulated network fault
		return nil, false, errors.New("network connection reset by peer")
	}

	opts := LiveIngestOptions{
		PageSize:    4,
		MaxPages:    3,
		PageFetcher: mockFetcher,
		LogWarning:  logWarn,
	}

	issues, err := FetchLiveIssues(context.Background(), opts)
	if err != nil {
		t.Fatalf("expected soft degradation on midstream fault, got %v", err)
	}

	if len(issues) != 4 {
		t.Errorf("expected 4 issues from page 1, got %d", len(issues))
	}

	foundAdvisory := false
	for _, w := range warnings {
		if strings.Contains(w, "[advisory] live ingestion soft-degraded") {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Errorf("expected soft-degraded advisory warning, got %v", warnings)
	}
}

func TestLiveIngest_Page1FailureFails(t *testing.T) {
	mockFetcher := func(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
		return nil, false, errors.New("initial connection refused")
	}

	opts := LiveIngestOptions{
		PageFetcher: mockFetcher,
	}

	_, err := FetchLiveIssues(context.Background(), opts)
	if err == nil {
		t.Fatalf("expected error on initial page failure, got nil")
	}
}
