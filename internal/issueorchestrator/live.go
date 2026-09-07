package issueorchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// LivePageFetcher is an injectable function that fetches a single page of issue JSON.
// It returns the raw payload (JSON array or search result object), a boolean indicating
// if a rate limit (HTTP 429/403) was encountered, and any underlying error.
type LivePageFetcher func(ctx context.Context, repo string, query string, perPage int, page int) (payload []byte, isRateLimited bool, err error)

// LiveIngestOptions configures live GitHub backlog ingestion.
type LiveIngestOptions struct {
	Workspace        string
	View             string
	Query            string
	Repo             string
	PageSize         int
	MaxPages         int
	RateLimitTimeout time.Duration
	TargetIssues     int
	PageFetcher      LivePageFetcher
	LogWarning       func(string)
}

// IssueViewsConfig maps the schema of .github/issue-views.json.
type IssueViewsConfig struct {
	Version int              `json:"version"`
	Repo    string           `json:"repo"`
	Default string           `json:"default"`
	Limit   int              `json:"limit"`
	Views   []IssueViewEntry `json:"views"`
}

// IssueViewEntry represents one named view in .github/issue-views.json.
type IssueViewEntry struct {
	Slug  string `json:"slug"`
	Title string `json:"title"`
	Query string `json:"query"`
	Note  string `json:"note"`
}

// ResolveViewQuery reads .github/issue-views.json from workspace and returns the query and repo for viewSlug.
func ResolveViewQuery(workspace string, viewSlug string) (query string, repo string, err error) {
	paths := []string{
		filepath.Join(workspace, ".github", "issue-views.json"),
		filepath.Join(".", ".github", "issue-views.json"),
	}

	var data []byte
	for _, p := range paths {
		if b, readErr := os.ReadFile(p); readErr == nil && len(b) > 0 {
			data = b
			break
		}
	}

	if len(data) == 0 {
		return "", "", fmt.Errorf("could not find .github/issue-views.json in workspace %s", workspace)
	}

	var config IssueViewsConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return "", "", fmt.Errorf("parse .github/issue-views.json: %w", err)
	}

	slug := strings.TrimSpace(viewSlug)
	if slug == "" {
		slug = config.Default
	}

	for _, v := range config.Views {
		if strings.EqualFold(v.Slug, slug) {
			return v.Query, config.Repo, nil
		}
	}

	return "", config.Repo, fmt.Errorf("view slug %q not found in .github/issue-views.json", slug)
}

func defaultGHPageFetcher(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
	ghBin, err := exec.LookPath("gh")
	if err != nil {
		return nil, false, fmt.Errorf("gh CLI executable not found in PATH: %w", err)
	}

	searchQuery := query
	if repo != "" && !strings.Contains(query, "repo:") {
		searchQuery = fmt.Sprintf("repo:%s %s", repo, query)
	}

	cmd := exec.CommandContext(ctx, ghBin, "api", "-X", "GET", "search/issues",
		"-f", "q="+searchQuery,
		"-F", fmt.Sprintf("per_page=%d", perPage),
		"-F", fmt.Sprintf("page=%d", page),
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	errOut := stderr.String()
	outBytes := stdout.Bytes()

	if runErr != nil {
		lowerErr := strings.ToLower(errOut)
		isRateLimit := strings.Contains(lowerErr, "rate limit") ||
			strings.Contains(lowerErr, "429") ||
			strings.Contains(lowerErr, "403") ||
			strings.Contains(lowerErr, "secondary rate")
		return outBytes, isRateLimit, fmt.Errorf("gh api error: %s: %w", strings.TrimSpace(errOut), runErr)
	}

	return outBytes, false, nil
}

func extractIssuesPayload(payload []byte) []byte {
	var searchResp struct {
		Items json.RawMessage `json:"items"`
	}
	if err := json.Unmarshal(payload, &searchResp); err == nil && len(searchResp.Items) > 0 {
		return searchResp.Items
	}
	return payload
}

// FetchLiveIssues ingests issues directly from GitHub with view resolution,
// dynamic soft-paging, exponential rate-limit backoff, and soft degradation.
func FetchLiveIssues(ctx context.Context, opts LiveIngestOptions) ([]Issue, error) {
	workspace := opts.Workspace
	if workspace == "" {
		workspace = "."
	}

	repo := opts.Repo
	query := opts.Query

	if query == "" {
		resolvedQuery, resolvedRepo, err := ResolveViewQuery(workspace, opts.View)
		if err != nil && opts.View != "" {
			return nil, err
		}
		if resolvedQuery != "" {
			query = resolvedQuery
		}
		if repo == "" && resolvedRepo != "" {
			repo = resolvedRepo
		}
	}

	if repo == "" {
		repo = "anthony-chaudhary/fak"
	}
	if query == "" {
		query = "is:open -label:epic"
	}

	pageSize := opts.PageSize
	if pageSize <= 0 {
		pageSize = 50
	}
	if pageSize > 100 {
		pageSize = 100
	}

	rateLimitTimeout := opts.RateLimitTimeout
	if rateLimitTimeout <= 0 {
		rateLimitTimeout = 30 * time.Second
	}

	fetcher := opts.PageFetcher
	if fetcher == nil {
		fetcher = defaultGHPageFetcher
	}

	logWarn := opts.LogWarning
	if logWarn == nil {
		logWarn = func(msg string) {
			fmt.Fprintln(os.Stderr, msg)
		}
	}

	var accumulatedIssues []Issue
	seenKeys := make(map[string]bool)
	seenNumbers := make(map[int]bool)

	maxPages := opts.MaxPages
	if maxPages <= 0 {
		maxPages = 20
	}

	for page := 1; page <= maxPages; page++ {
		var pagePayload []byte
		backoffStart := time.Now()
		retryDelay := 500 * time.Millisecond
		maxRetryDelay := 5 * time.Second

		for {
			if ctx.Err() != nil {
				return accumulatedIssues, ctx.Err()
			}

			payload, isRateLimited, fetchErr := fetcher(ctx, repo, query, pageSize, page)
			if isRateLimited {
				if time.Since(backoffStart) >= rateLimitTimeout {
					if len(accumulatedIssues) > 0 {
						logWarn("[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot")
						return accumulatedIssues, nil
					}
					return nil, fmt.Errorf("rate limit timeout (%v) exceeded on page %d: %w", rateLimitTimeout, page, fetchErr)
				}
				jitter := time.Duration(rand.Intn(400)) * time.Millisecond
				sleepDuration := retryDelay + jitter
				select {
				case <-ctx.Done():
					return accumulatedIssues, ctx.Err()
				case <-time.After(sleepDuration):
				}
				retryDelay *= 2
				if retryDelay > maxRetryDelay {
					retryDelay = maxRetryDelay
				}
				continue
			}

			if fetchErr != nil {
				if page > 1 && len(accumulatedIssues) > 0 {
					logWarn("[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot")
					return accumulatedIssues, nil
				}
				return nil, fmt.Errorf("failed to fetch live issues on page %d: %w", page, fetchErr)
			}

			pagePayload = payload
			break
		}

		itemsBytes := extractIssuesPayload(pagePayload)
		pageIssues, decodeErr := DecodeIssues(itemsBytes)
		if decodeErr != nil {
			if page > 1 && len(accumulatedIssues) > 0 {
				logWarn("[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot")
				return accumulatedIssues, nil
			}
			return nil, fmt.Errorf("decode page %d issues: %w", page, decodeErr)
		}

		if len(pageIssues) == 0 {
			break
		}

		for _, iss := range pageIssues {
			if iss.Number > 0 && seenNumbers[iss.Number] {
				continue
			}
			if iss.Key != "" && seenKeys[iss.Key] {
				continue
			}
			if iss.Number > 0 {
				seenNumbers[iss.Number] = true
			}
			if iss.Key != "" {
				seenKeys[iss.Key] = true
			}
			accumulatedIssues = append(accumulatedIssues, iss)
		}

		// Dynamic soft-paging: stop if target dispatchable issues is met
		if opts.TargetIssues > 0 {
			dispatchableCount := 0
			for _, iss := range accumulatedIssues {
				if !isSubdivideTarget(iss) && !isTriageTarget(iss) {
					dispatchableCount++
				}
			}
			if dispatchableCount >= opts.TargetIssues {
				break
			}
		}

		if len(pageIssues) < pageSize {
			break
		}
	}

	return accumulatedIssues, nil
}
