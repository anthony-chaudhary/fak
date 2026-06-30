package ideascout

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestScoreCandidateTransparentWeights(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	topic := Topic{
		Key:   "prompt-injection-defense",
		Terms: []string{"prompt injection", "agent", "tool"},
	}
	cand := Candidate{
		Title:     "Prompt injection defense for agents",
		Summary:   "Hardens tool routing against untrusted content.",
		Published: "2026-06-20T00:00:00Z",
		Extra: map[string]any{
			"stars":     250,
			"pushed_at": "2026-06-25T00:00:00Z",
		},
	}

	score, reasons := ScoreCandidate(cand, topic, DefaultConfig(), now)
	if score != 69 {
		t.Fatalf("score = %d, want 69 (title hits 20 + body 3 + freshness 34 + stars 2 + push 10), reasons=%v", score, reasons)
	}
	joined := strings.Join(reasons, "; ")
	for _, want := range []string{"prompt injection(title)", "agent(title)", "tool", "recent (10d)", "very fresh", "250 stars (+2)", "pushed <=90d"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("reasons %q missing %q", joined, want)
		}
	}
}

func TestDedupeRungsAndSeenCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	seen := map[string]SeenRecord{
		"arxiv:old": {FiledAt: "2026-06-29", IssueURL: "https://github.com/o/r/issues/1", Score: 42, Topic: "t"},
	}
	if err := SaveSeen(dir, seen); err != nil {
		t.Fatalf("SaveSeen: %v", err)
	}
	loaded, err := LoadSeen(dir)
	if err != nil {
		t.Fatalf("LoadSeen: %v", err)
	}
	if loaded["arxiv:old"].IssueURL == "" {
		t.Fatalf("loaded seen cache missing record: %#v", loaded)
	}

	issues := []ExistingIssue{{
		Title: "idea-scout: Capability policy for tool calls",
		Body:  "Filed earlier\nhttps://example.test/already\n<!-- idea-scout-source: github:owner/repo -->",
	}}
	stamped, titleSets, bodies := ExistingIssueIndex(issues)
	cases := []struct {
		name string
		cand Candidate
		want string
	}{
		{"seen-cache", Candidate{SourceID: "arxiv:old", URL: "https://example.test/old", Title: "Novel thing"}, "seen-cache"},
		{"stamp", Candidate{SourceID: "github:owner/repo", URL: "https://github.com/Owner/Repo", Title: "Other"}, "issue-body"},
		{"url", Candidate{SourceID: "arxiv:url", URL: "https://example.test/already", Title: "Other"}, "issue-body"},
		{"title", Candidate{SourceID: "arxiv:title", URL: "https://example.test/new", Title: "Capability policy for tool calls"}, "title-near"},
		{"fresh", Candidate{SourceID: "arxiv:new", URL: "https://example.test/new", Title: "Different research"}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsDuplicate(tc.cand, loaded, stamped, titleSets, bodies, 0.55); got != tc.want {
				t.Fatalf("duplicate rung = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestRenderIssueAndPlanRanking(t *testing.T) {
	now := time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC)
	topic := Topic{Key: "kv-prefix-cache-reuse", Terms: []string{"kv cache", "prefix cache", "reuse"}, Area: "prompt-caching"}
	candidates := []Candidate{
		{
			Source:    "arxiv",
			SourceID:  "arxiv:1",
			URL:       "https://arxiv.org/abs/1",
			Title:     "KV cache reuse for agent turns",
			Summary:   "A prefix cache reuse paper.",
			Published: "2026-06-25T00:00:00Z",
			Topic:     topic.Key,
			Extra:     map[string]any{"authors": []string{"A", "B"}},
		},
		{
			Source:    "github",
			SourceID:  "github:owner/repo",
			URL:       "https://github.com/Owner/Repo",
			Title:     "owner/repo",
			Summary:   "KV cache prefix cache reuse for agents",
			Published: "2026-01-01T00:00:00Z",
			Topic:     topic.Key,
			Extra:     map[string]any{"stars": 1200, "pushed_at": "2026-06-28T00:00:00Z", "language": "Go"},
		},
		{
			Source:   "arxiv",
			SourceID: "arxiv:1",
			URL:      "https://arxiv.org/abs/1",
			Title:    "duplicate in this run",
			Topic:    topic.Key,
		},
	}

	plans, stats := PlanIssues(candidates, map[string]Topic{topic.Key: topic}, nil, nil, nil, "", Config{RecentDays: 180, MinScore: 1, MaxIssues: 2, DupJaccard: 0.55}, "2026-06-30", now)
	if stats["within-run-dup"] != 1 {
		t.Fatalf("within-run-dup = %d, want 1", stats["within-run-dup"])
	}
	if len(plans) != 2 {
		t.Fatalf("planned = %d, want 2", len(plans))
	}
	if plans[0].Score < plans[1].Score {
		t.Fatalf("plans not sorted by descending score: %#v", plans)
	}

	score, reasons := ScoreCandidate(candidates[0], topic, Config{RecentDays: 180}, now)
	issue := RenderIssue(candidates[0], score, reasons, topic, "2026-06-30")
	if !strings.Contains(issue.Body, "<!-- idea-scout-source: arxiv:1 -->") {
		t.Fatalf("issue body missing source stamp:\n%s", issue.Body)
	}
	if !strings.Contains(issue.Body, "**Authors:** A, B") {
		t.Fatalf("issue body missing arxiv facts:\n%s", issue.Body)
	}
	if !hasLabel(issue.Labels, "prompt-caching") || !hasLabel(issue.Labels, TriageOnlyLabel) {
		t.Fatalf("labels = %v, missing area/triage labels", issue.Labels)
	}
}

func TestParseSourceShapes(t *testing.T) {
	xml := `<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.12345v2</id><title>  A
paper </title><summary>  Summary text </summary><published>2026-06-01T00:00:00Z</published><author><name>Ada</name></author></entry></feed>`
	arxiv := ParseArxivAtom(xml, "topic")
	if len(arxiv) != 1 || arxiv[0].SourceID != "arxiv:2401.12345" || arxiv[0].Title != "A paper" {
		t.Fatalf("ParseArxivAtom = %#v", arxiv)
	}
	repos := ParseGitHubRepos([]GitHubRepo{{FullName: "Owner/Repo", URL: "https://github.com/Owner/Repo", StargazersCount: 99}}, "topic")
	if len(repos) != 1 || repos[0].SourceID != "github:owner/repo" {
		t.Fatalf("ParseGitHubRepos = %#v", repos)
	}
}

func TestRunDryRunDoesNotWriteCache(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{
		"topics": [{"key":"t","github":"fixture","terms":["agent","tool"],"area":"trust-floor"}],
		"thresholds": {"min_score": 1, "max_issues": 1}
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	result, err := Run(RunOptions{
		Workspace:   dir,
		ConfigPath:  cfgPath,
		UseFixtures: true,
		Today:       "2026-06-30",
		Now:         time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC),
		Candidates: []Candidate{{
			Source:    "github",
			SourceID:  "github:o/r",
			URL:       "https://github.com/o/r",
			Title:     "o/r",
			Summary:   "agent tool policy",
			Published: "2026-06-29T00:00:00Z",
			Topic:     "t",
		}},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Mode != "dry-run" || len(result.Planned) != 1 {
		t.Fatalf("result = %#v, want one dry-run plan", result)
	}
	if _, err := os.Stat(CachePath(dir)); !os.IsNotExist(err) {
		t.Fatalf("dry-run should not write seen cache, stat err=%v", err)
	}
}

func hasLabel(labels []string, want string) bool {
	for _, got := range labels {
		if got == want {
			return true
		}
	}
	return false
}
