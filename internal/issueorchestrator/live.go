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

// CommandRunner defines an execution hook for running host commands (such as gh CLI).
type CommandRunner func(ctx context.Context, name string, args ...string) ([]byte, error)

// Run executes the command runner.
func (r CommandRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	if r == nil {
		return nil, nil
	}
	return r(ctx, name, args...)
}

// LiveIngestOptions configures live GitHub backlog ingestion.
type LiveIngestOptions struct {
	Workspace        string
	View             string        // slug matching .github/issue-views.json
	Query            string        // optional direct search query override
	Repo             string        // optional repo override
	PageSize         int           // default: 50, max: 100
	MaxPages         int           // optional upper bound (0 = unlimited)
	TargetIssues     int           // stop soft-paging once this many candidates fetched
	RateLimitTimeout time.Duration // default: 30s
	Runner           CommandRunner // optional for testing / mock execution of gh CLI or HTTP
	PageFetcher      LivePageFetcher
	LogWarning       func(string)
	Context          context.Context
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

// ViewParams represents the parsed query parameters from an issue view.
type ViewParams struct {
	Slug           string            `json:"slug"`
	Title          string            `json:"title"`
	Query          string            `json:"query"`
	Repo           string            `json:"repo"`
	Limit          int               `json:"limit"`
	State          string            `json:"state"`
	Labels         []string          `json:"labels"`
	ExcludedLabels []string          `json:"excluded_labels"`
	Filters        map[string]string `json:"filters"`
}

// tokenizeQuery splits a GitHub search query into tokens preserving double-quoted strings.
func tokenizeQuery(q string) []string {
	var tokens []string
	var cur strings.Builder
	inQuotes := false
	for _, r := range q {
		switch {
		case r == '"':
			inQuotes = !inQuotes
			cur.WriteRune(r)
		case r == ' ' && !inQuotes:
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
		default:
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}

// ParseViewQuery extracts state, labels, excluded labels, and filters from a GitHub query string.
func ParseViewQuery(query string) (state string, labels []string, excludedLabels []string, filters map[string]string) {
	filters = make(map[string]string)
	tokens := tokenizeQuery(query)
	for _, tok := range tokens {
		switch {
		case strings.HasPrefix(tok, "is:"):
			val := strings.TrimPrefix(tok, "is:")
			if val == "open" || val == "closed" {
				state = val
			} else {
				filters["is"] = val
			}
		case strings.HasPrefix(tok, "-label:"):
			val := strings.Trim(strings.TrimPrefix(tok, "-label:"), "\"")
			for _, l := range strings.Split(val, ",") {
				if trimmed := strings.TrimSpace(l); trimmed != "" {
					excludedLabels = append(excludedLabels, trimmed)
				}
			}
		case strings.HasPrefix(tok, "label:"):
			val := strings.Trim(strings.TrimPrefix(tok, "label:"), "\"")
			for _, l := range strings.Split(val, ",") {
				if trimmed := strings.TrimSpace(l); trimmed != "" {
					labels = append(labels, trimmed)
				}
			}
		default:
			parts := strings.SplitN(tok, ":", 2)
			if len(parts) == 2 {
				filters[parts[0]] = strings.Trim(parts[1], "\"")
			}
		}
	}
	return state, labels, excludedLabels, filters
}

// ResolveView reads .github/issue-views.json from workspace and returns the resolved ViewParams for viewSlug.
func ResolveView(workspace string, viewSlug string) (ViewParams, error) {
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
		return ViewParams{}, fmt.Errorf("could not find .github/issue-views.json in workspace %s", workspace)
	}

	var config IssueViewsConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return ViewParams{}, fmt.Errorf("parse .github/issue-views.json: %w", err)
	}

	slug := strings.TrimSpace(viewSlug)
	if slug == "" {
		slug = config.Default
	}

	for _, v := range config.Views {
		if strings.EqualFold(v.Slug, slug) {
			state, labels, exLabels, filters := ParseViewQuery(v.Query)
			return ViewParams{
				Slug:           v.Slug,
				Title:          v.Title,
				Query:          v.Query,
				Repo:           config.Repo,
				Limit:          config.Limit,
				State:          state,
				Labels:         labels,
				ExcludedLabels: exLabels,
				Filters:        filters,
			}, nil
		}
	}

	return ViewParams{Repo: config.Repo}, fmt.Errorf("view slug %q not found in .github/issue-views.json", slug)
}

// ResolveViewQuery reads .github/issue-views.json from workspace and returns the query and repo for viewSlug.
func ResolveViewQuery(workspace string, viewSlug string) (query string, repo string, err error) {
	params, err := ResolveView(workspace, viewSlug)
	if err != nil {
		return "", params.Repo, err
	}
	return params.Query, params.Repo, nil
}

func makeGHPageFetcher(runner CommandRunner) LivePageFetcher {
	return func(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
		searchQuery := query
		if repo != "" && !strings.Contains(query, "repo:") {
			searchQuery = fmt.Sprintf("repo:%s %s", repo, query)
		}

		args := []string{
			"api", "-X", "GET", "search/issues",
			"-f", "q=" + searchQuery,
			"-F", fmt.Sprintf("per_page=%d", perPage),
			"-F", fmt.Sprintf("page=%d", page),
		}

		if runner != nil {
			outBytes, runErr := runner(ctx, "gh", args...)
			if runErr != nil {
				lowerErr := strings.ToLower(runErr.Error() + " " + string(outBytes))
				isRateLimit := strings.Contains(lowerErr, "rate limit") ||
					strings.Contains(lowerErr, "429") ||
					strings.Contains(lowerErr, "403") ||
					strings.Contains(lowerErr, "secondary rate")
				return outBytes, isRateLimit, fmt.Errorf("gh command error: %s: %w", strings.TrimSpace(string(outBytes)), runErr)
			}
			return outBytes, false, nil
		}

		ghBin, err := exec.LookPath("gh")
		if err != nil {
			return nil, false, fmt.Errorf("gh CLI executable not found in PATH: %w", err)
		}

		cmd := exec.CommandContext(ctx, ghBin, args...)
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
}

func defaultGHPageFetcher(ctx context.Context, repo string, query string, perPage int, page int) ([]byte, bool, error) {
	return makeGHPageFetcher(nil)(ctx, repo, query, perPage, page)
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
func FetchLiveIssues(args ...any) ([]Issue, error) {
	ctx := context.Background()
	var opts LiveIngestOptions
	for _, arg := range args {
		switch v := arg.(type) {
		case context.Context:
			ctx = v
		case LiveIngestOptions:
			opts = v
		}
	}
	if opts.Context != nil {
		ctx = opts.Context
	}
	return fetchLiveIssues(ctx, opts)
}

func fetchLiveIssues(ctx context.Context, opts LiveIngestOptions) ([]Issue, error) {
	workspace := opts.Workspace
	if workspace == "" {
		workspace = "."
	}

	repo := opts.Repo
	query := opts.Query

	if query == "" {
		resolved, err := ResolveView(workspace, opts.View)
		if err != nil && opts.View != "" {
			return nil, err
		}
		if resolved.Query != "" {
			query = resolved.Query
		}
		if repo == "" && resolved.Repo != "" {
			repo = resolved.Repo
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
		fetcher = makeGHPageFetcher(opts.Runner)
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
		maxPages = 50
	}

	for page := 1; page <= maxPages; page++ {
		var pagePayload []byte
		backoffStart := time.Now()
		retryDelay := 500 * time.Millisecond
		if rateLimitTimeout < retryDelay {
			retryDelay = 10 * time.Millisecond
		}
		maxRetryDelay := 5 * time.Second
		if rateLimitTimeout < maxRetryDelay {
			maxRetryDelay = rateLimitTimeout
		}

		for {
			if ctx.Err() != nil {
				return accumulatedIssues, ctx.Err()
			}

			payload, isRateLimited, fetchErr := fetcher(ctx, repo, query, pageSize, page)
			if isRateLimited {
				if time.Since(backoffStart) >= rateLimitTimeout {
					if len(accumulatedIssues) > 0 {
						logWarn(fmt.Sprintf("[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot (%d issues)", len(accumulatedIssues)))
						return accumulatedIssues, nil
					}
					return nil, fmt.Errorf("rate limit timeout (%v) exceeded on page %d: %w", rateLimitTimeout, page, fetchErr)
				}
				maxJitterMs := 400
				if retryDelay < 100*time.Millisecond {
					maxJitterMs = 10
				}
				jitter := time.Duration(rand.Intn(maxJitterMs)) * time.Millisecond
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
					logWarn(fmt.Sprintf("[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot (%d issues)", len(accumulatedIssues)))
					return accumulatedIssues, nil
				}
				return nil, fmt.Errorf("failed to fetch live issues on page %d: %w", page, fetchErr)
			}

			pagePayload = payload
			break
		}

		itemsBytes := extractIssuesPayload(pagePayload)
		trimmed := bytes.TrimSpace(itemsBytes)
		if len(trimmed) == 0 || string(trimmed) == "[]" {
			break
		}

		pageIssues, decodeErr := DecodeIssues(itemsBytes)
		if decodeErr != nil {
			if page > 1 && len(accumulatedIssues) > 0 {
				logWarn(fmt.Sprintf("[advisory] live ingestion soft-degraded; proceeding with partial candidate snapshot (%d issues)", len(accumulatedIssues)))
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
			if dispatchableCount >= opts.TargetIssues || len(accumulatedIssues) >= opts.TargetIssues {
				break
			}
		}

		if len(pageIssues) < pageSize {
			break
		}
	}

	return accumulatedIssues, nil
}
