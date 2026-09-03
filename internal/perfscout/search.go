package perfscout

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
)

// DefaultQueries are the targeted queries to unearth fresh, performance-specific OSS repos.
var DefaultQueries = []string{
	"qwen3.8 flash",
	"qwen3.8-flash-next",
	"qwen3.8 27b",
	"qwen3.8 spark",
	"qwen3.8 vllm",
	"qwen3.8 sglang",
	"glm-5.3 flash",
	"glm-5.3",
	"glm53-flash",
	"glm-5.3 spark",
	"glm-5.3 vllm",
	"glm-5.3 exl3",
	"qwen3.8 tok/s",
	"glm-5.3 tok/s",
}

// LiveFetcher executes GitHub search queries via deadlined gh invocations.
type LiveFetcher struct {
	Timeout time.Duration
}

// SearchQuery executes one query using gh search repos.
func (f LiveFetcher) SearchQuery(query string, limit int) ([]GitHubRawRepo, error) {
	timeout := f.Timeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}
	cmd, cancel := ghexec.CommandTimeout(nil, timeout,
		"search", "repos", query,
		"--limit", strconv.Itoa(limit),
		"--sort", "updated",
		"--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language,size",
	)
	defer cancel()
	cmd.WaitDelay = 10 * time.Second

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gh search repos %q: %w: %s", query, err, strings.TrimSpace(stderr.String()))
	}

	var out []GitHubRawRepo
	if strings.TrimSpace(stdout.String()) == "" {
		return nil, nil
	}
	if err := json.Unmarshal([]byte(stdout.String()), &out); err != nil {
		return nil, fmt.Errorf("decode gh search output: %w", err)
	}
	return out, nil
}

// Run executes the search and analysis pipeline.
func Run(opts SearchOptions) (InventoryReport, error) {
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}

	var rawRepos []GitHubRawRepo
	queries := opts.Queries
	if len(queries) == 0 {
		queries = DefaultQueries
	}

	var searchErrors []string
	if opts.FixturePath != "" {
		// Read from offline fixture
		data, err := os.ReadFile(opts.FixturePath)
		if err != nil {
			return InventoryReport{}, fmt.Errorf("read fixture %s: %w", opts.FixturePath, err)
		}
		if err := json.Unmarshal(data, &rawRepos); err != nil {
			return InventoryReport{}, fmt.Errorf("parse fixture JSON: %w", err)
		}
	} else {
		// Live search across queries
		fetcher := LiveFetcher{Timeout: 45 * time.Second}
		limit := opts.LimitPerQuery
		if limit <= 0 {
			limit = 25
		}

		seen := make(map[string]bool)
		for _, q := range queries {
			results, err := fetcher.SearchQuery(q, limit)
			if err != nil {
				searchErrors = append(searchErrors, fmt.Sprintf("%s: %v", q, err))
				continue
			}
			for _, r := range results {
				if !seen[r.FullName] {
					seen[r.FullName] = true
					rawRepos = append(rawRepos, r)
				}
			}
			time.Sleep(200 * time.Millisecond)
		}

		// If live queries failed (e.g. rate limit) and yielded 0 results, fallback to cached corpus if present
		if len(rawRepos) == 0 {
			cachedPaths := []string{
				"testdata/perfscout/raw_corpus.json",
				"../testdata/perfscout/raw_corpus.json",
				"../../testdata/perfscout/raw_corpus.json",
			}
			for _, cp := range cachedPaths {
				if data, err := os.ReadFile(cp); err == nil {
					var cached []GitHubRawRepo
					if err := json.Unmarshal(data, &cached); err == nil && len(cached) > 0 {
						rawRepos = cached
						searchErrors = append(searchErrors, fmt.Sprintf("live search yielded 0 results (likely rate limited); used cached corpus %s (%d repos)", cp, len(cached)))
						break
					}
				}
			}
		}
	}

	maxStars := opts.MaxStars
	if maxStars <= 0 {
		maxStars = 500 // Default unpopular indie threshold
	}

	var scored []ScoredRepo
	qwenCount := 0
	glmCount := 0
	dualCount := 0

	for _, raw := range rawRepos {
		s := AnalyzeRepo(raw, now, maxStars)

		// Must match target family
		if s.ModelFamily == FamilyUnknown {
			continue
		}

		// Optional family filter
		if opts.FamilyFilter != "" && opts.FamilyFilter != FamilyUnknown {
			if s.ModelFamily != opts.FamilyFilter && s.ModelFamily != FamilyDual {
				continue
			}
		}

		// Minimum performance score filter
		if opts.MinScore > 0 && s.PerformanceScore < opts.MinScore {
			continue
		}

		// Maximum stars filter (unpopular focus)
		if maxStars > 0 && s.StargazersCount > maxStars {
			continue
		}

		// Freshness filter
		if opts.MaxAgeDays > 0 && s.FreshnessDays > opts.MaxAgeDays {
			continue
		}

		switch s.ModelFamily {
		case FamilyQwenFlash:
			qwenCount++
		case FamilyGLMFlash:
			glmCount++
		case FamilyDual:
			dualCount++
			qwenCount++
			glmCount++
		}

		scored = append(scored, s)
	}

	// Sort by PerformanceScore descending, then by UpdatedAt descending
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].PerformanceScore != scored[j].PerformanceScore {
			return scored[i].PerformanceScore > scored[j].PerformanceScore
		}
		return scored[i].UpdatedAt.After(scored[j].UpdatedAt)
	})

	cohortCount := opts.CohortCount
	if cohortCount <= 0 {
		cohortCount = 4
	}

	cohorts := PartitionCohorts(scored, cohortCount)

	return InventoryReport{
		GeneratedAt:   now,
		QueriesWalked: queries,
		TotalFetched:  len(rawRepos),
		TotalScored:   len(rawRepos),
		RetainedCount: len(scored),
		QwenCount:     qwenCount,
		GLMCount:      glmCount,
		DualCount:     dualCount,
		SearchErrors:  searchErrors,
		Cohorts:       cohorts,
		Repositories:  scored,
	}, nil
}

// PartitionCohorts distributes repos round-robin into cohortCount cohorts.
func PartitionCohorts(repos []ScoredRepo, cohortCount int) map[int][]ScoredRepo {
	cohorts := make(map[int][]ScoredRepo)
	if cohortCount <= 0 || len(repos) == 0 {
		return cohorts
	}

	for i := range repos {
		cohortID := (i % cohortCount) + 1
		repos[i].CohortID = cohortID
		cohorts[cohortID] = append(cohorts[cohortID], repos[i])
	}

	return cohorts
}
