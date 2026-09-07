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

func TestLiveIngest_ViewResolution(t *testing.T) {
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
				Query: "is:open -label:epic -label:research no:assignee",
			},
			{
				Slug:  "p0-p1",
				Title: "P0/P1 leaves",
				Query: "is:open label:priority/P0,priority/P1 -label:epic sort:created-asc",
			},
			{
				Slug:  "dev-leaves",
				Title: "Dev leaves (product only)",
				Query: "is:open label:class:dev -label:epic no:assignee",
			},
		},
	}
	b, err := json.Marshal(viewsCfg)
	if err != nil {
		t.Fatalf("marshal views: %v", err)
	}
	if err := os.WriteFile(filepath.Join(githubDir, "issue-views.json"), b, 0644); err != nil {
		t.Fatalf("write views: %v", err)
	}

	// 1. Default view resolution
	res, err := ResolveView(dir, "")
	if err != nil {
		t.Fatalf("resolve default view: %v", err)
	}
	if res.Slug != "ready-leaves" {
		t.Errorf("expected slug ready-leaves, got %s", res.Slug)
	}
	if res.State != "open" {
		t.Errorf("expected state open, got %s", res.State)
	}
	if res.Repo != "anthony-chaudhary/fak" {
		t.Errorf("expected repo anthony-chaudhary/fak, got %s", res.Repo)
	}
	if res.Limit != 300 {
		t.Errorf("expected limit 300, got %d", res.Limit)
	}
	if len(res.ExcludedLabels) != 2 || res.ExcludedLabels[0] != "epic" || res.ExcludedLabels[1] != "research" {
		t.Errorf("unexpected excluded labels: %v", res.ExcludedLabels)
	}

	// 2. Explicit view: p0-p1
	resP0P1, err := ResolveView(dir, "p0-p1")
	if err != nil {
		t.Fatalf("resolve p0-p1 view: %v", err)
	}
	if resP0P1.Query != "is:open label:priority/P0,priority/P1 -label:epic sort:created-asc" {
		t.Errorf("unexpected query: %s", resP0P1.Query)
	}
	if len(resP0P1.Labels) != 2 || resP0P1.Labels[0] != "priority/P0" || resP0P1.Labels[1] != "priority/P1" {
		t.Errorf("unexpected labels: %v", resP0P1.Labels)
	}
	if resP0P1.Filters["sort"] != "created-asc" {
		t.Errorf("expected sort filter created-asc, got %s", resP0P1.Filters["sort"])
	}

	// 3. Explicit view: dev-leaves
	resDev, err := ResolveView(dir, "dev-leaves")
	if err != nil {
		t.Fatalf("resolve dev-leaves view: %v", err)
	}
	if len(resDev.Labels) != 1 || resDev.Labels[0] != "class:dev" {
		t.Errorf("unexpected labels: %v", resDev.Labels)
	}

	// 4. ResolveViewQuery compatibility
	q, repo, err := ResolveViewQuery(dir, "ready-leaves")
	if err != nil {
		t.Fatalf("ResolveViewQuery: %v", err)
	}
	if q != "is:open -label:epic -label:research no:assignee" || repo != "anthony-chaudhary/fak" {
		t.Errorf("unexpected ResolveViewQuery result: q=%q repo=%q", q, repo)
	}

	// 5. Nonexistent view
	_, err = ResolveView(dir, "nonexistent-view")
	if err == nil {
		t.Fatalf("expected error for nonexistent view, got nil")
	}
}

func TestLiveIngest_ResolveViewQuery(t *testing.T) {
	TestLiveIngest_ViewResolution(t)
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

func TestLiveIngest_RateLimitBackoff(t *testing.T) {
	attempts := 0
	mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		attempts++
		if attempts == 1 {
			return nil, errors.New("HTTP 429: API rate limit exceeded")
		}
		// Return 2 items (< PageSize 3) so pagination terminates after page 1
		return makeMockPage(1, 2), nil
	}

	opts := LiveIngestOptions{
		PageSize:         3,
		RateLimitTimeout: 200 * time.Millisecond,
		Runner:           mockRunner,
	}

	issues, err := FetchLiveIssues(opts)
	if err != nil {
		t.Fatalf("expected retry to succeed after rate limit, got error: %v", err)
	}
	if len(issues) != 2 {
		t.Errorf("expected 2 issues, got %d", len(issues))
	}
	if attempts != 2 {
		t.Errorf("expected 2 attempts (1 failure + 1 retry), got %d", attempts)
	}
}

func TestLiveIngest_SoftDegradation(t *testing.T) {
	var warnings []string
	logWarn := func(msg string) {
		warnings = append(warnings, msg)
	}

	pageCalls := 0
	mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		pageCalls++
		if pageCalls == 1 {
			return makeMockPage(1, 4), nil
		}
		// Subsequent pages fail with persistent rate limit
		return nil, errors.New("HTTP 429: rate limit exceeded")
	}

	opts := LiveIngestOptions{
		PageSize:         4,
		MaxPages:         5,
		RateLimitTimeout: 50 * time.Millisecond,
		Runner:           mockRunner,
		LogWarning:       logWarn,
	}

	issues, err := FetchLiveIssues(opts)
	if err != nil {
		t.Fatalf("expected soft degradation without error, got %v", err)
	}
	if len(issues) != 4 {
		t.Errorf("expected 4 issues from page 1, got %d", len(issues))
	}

	expectedAdvisory := "[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot (4 issues)"
	foundAdvisory := false
	for _, w := range warnings {
		if strings.Contains(w, expectedAdvisory) {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Errorf("expected advisory %q, got warnings: %v", expectedAdvisory, warnings)
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
