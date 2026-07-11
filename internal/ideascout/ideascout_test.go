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
	if score != 94 {
		t.Fatalf("score = %d, want 94 (title hits 20 + body 3 + freshness 34 + stars 2 + fresh-push 15 + trending 20), reasons=%v", score, reasons)
	}
	joined := strings.Join(reasons, "; ")
	for _, want := range []string{"prompt injection(title)", "agent(title)", "tool", "recent (10d)", "very fresh", "250 stars (+2)", "pushed <=45d (actively updated)", "trending (25*/day, +20)"} {
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

func TestParseHackerNewsJSON(t *testing.T) {
	body := `{"hits":[
		{"objectID":"111","title":"Show HN: a prompt injection scanner","url":"https://example.test/scanner","points":240,"num_comments":57,"created_at":"2026-07-01T10:00:00Z","story_text":""},
		{"objectID":"222","title":"Ask HN: how do you sandbox tool calls?","url":"","points":85,"num_comments":30,"created_at":"2026-07-01T09:00:00Z","story_text":"<p>We run untrusted <a href=\"x\">tools</a>.</p>"},
		{"objectID":"","title":"dropped: no id","url":"https://example.test/x","points":10,"created_at":"2026-07-01T08:00:00Z"}
	]}`
	cands := ParseHackerNewsJSON(body, "trust-floor")
	if len(cands) != 2 {
		t.Fatalf("parsed %d candidates, want 2 (the id-less hit is dropped): %#v", len(cands), cands)
	}
	first := cands[0]
	if first.Source != "hackernews" || first.SourceID != "hn:111" || first.Topic != "trust-floor" {
		t.Fatalf("first candidate shape = %#v", first)
	}
	if first.URL != "https://example.test/scanner" {
		t.Fatalf("link post should keep its outbound URL, got %q", first.URL)
	}
	if intFromExtra(first.Extra, "points") != 240 || intFromExtra(first.Extra, "num_comments") != 57 {
		t.Fatalf("first candidate extra = %#v", first.Extra)
	}
	if disc := stringFromExtra(first.Extra, "discussion"); disc != "https://news.ycombinator.com/item?id=111" {
		t.Fatalf("discussion permalink = %q", disc)
	}
	second := cands[1]
	if second.URL != "https://news.ycombinator.com/item?id=222" {
		t.Fatalf("self post with no url should fall back to the HN permalink, got %q", second.URL)
	}
	if strings.Contains(second.Summary, "<") || !strings.Contains(second.Summary, "We run untrusted tools") {
		t.Fatalf("summary should be tag-stripped prose, got %q", second.Summary)
	}
}

func TestScoreCandidateHackerNewsPoints(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	topic := Topic{Key: "trust-floor", Terms: []string{"sandbox", "tool"}}
	cand := Candidate{
		Title:     "Sandbox for tool calls",
		Published: "2026-06-30T00:00:00Z",
		Extra:     map[string]any{"points": 240, "num_comments": 57},
	}
	score, reasons := ScoreCandidate(cand, topic, DefaultConfig(), now)
	// title hits sandbox+tool 20 + freshness 34 (recent+very fresh) + points 240/20=12.
	if score != 66 {
		t.Fatalf("score = %d, want 66, reasons=%v", score, reasons)
	}
	if joined := strings.Join(reasons, "; "); !strings.Contains(joined, "240 points (+12)") {
		t.Fatalf("reasons missing points signal: %q", joined)
	}
}

type stubFetcher struct {
	github      []GitHubRepo
	githubFresh []GitHubRepo
	hnJSON      string
	redditJSON  string
}

func (s stubFetcher) FetchArxiv(string, int) (string, error)             { return "", nil }
func (s stubFetcher) FetchGitHub(string, int) ([]GitHubRepo, error)      { return s.github, nil }
func (s stubFetcher) FetchGitHubFresh(string, int) ([]GitHubRepo, error) { return s.githubFresh, nil }
func (s stubFetcher) FetchHackerNews(string, int) (string, error)        { return s.hnJSON, nil }
func (s stubFetcher) FetchReddit(string, int) (string, error)            { return s.redditJSON, nil }
func (s stubFetcher) FetchExistingIssues(int) ([]ExistingIssue, error)   { return nil, nil }
func (s stubFetcher) EnsureLabels() error                                { return nil }
func (s stubFetcher) CreateIssue(IssuePlan, string) (string, error)      { return "", nil }
func (s stubFetcher) AddToProject(string, string, string) error          { return nil }

func TestGatherCandidatesHackerNewsFiltersByPoints(t *testing.T) {
	hn := `{"hits":[
		{"objectID":"1","title":"hot trending agent tool","url":"https://example.test/1","points":300,"created_at":"2026-07-01T00:00:00Z"},
		{"objectID":"2","title":"barely-voted post","url":"https://example.test/2","points":3,"created_at":"2026-07-01T00:00:00Z"}
	]}`
	topics := []Topic{{Key: "t", HN: "agent tool", Terms: []string{"agent", "tool"}}}
	cfg := DefaultConfig() // MinPoints=10
	var errs []string
	cands := GatherCandidates(stubFetcher{hnJSON: hn}, topics, cfg, &errs)
	if len(errs) != 0 {
		t.Fatalf("unexpected gather errors: %v", errs)
	}
	if len(cands) != 1 || cands[0].SourceID != "hn:1" {
		t.Fatalf("want only the 300-point story above MinPoints, got %#v", cands)
	}
}

func TestParseRedditJSON(t *testing.T) {
	body := `{"data":{"children":[
		{"kind":"t3","data":{"id":"abc","title":"Show: a new agent sandbox","url":"https://example.test/repo","permalink":"/r/rust/comments/abc/x/","score":320,"num_comments":44,"created_utc":1704067200,"selftext":"","subreddit":"rust"}},
		{"kind":"t3","data":{"id":"def","title":"How do you sandbox tool calls?","url":"https://www.reddit.com/r/LocalLLaMA/comments/def/y/","permalink":"/r/LocalLLaMA/comments/def/y/","score":90,"num_comments":12,"created_utc":1704067200,"selftext":"<p>We run <a href=\"x\">untrusted</a> tools.</p>","subreddit":"LocalLLaMA"}},
		{"kind":"t3","data":{"id":"","title":"dropped: no id","score":10}}
	]}}`
	cands := ParseRedditJSON(body, "trust-floor")
	if len(cands) != 2 {
		t.Fatalf("parsed %d candidates, want 2 (id-less hit dropped): %#v", len(cands), cands)
	}
	first := cands[0]
	if first.Source != "reddit" || first.SourceID != "reddit:abc" || first.Topic != "trust-floor" {
		t.Fatalf("first candidate shape = %#v", first)
	}
	if first.URL != "https://example.test/repo" {
		t.Fatalf("link post should keep its outbound URL, got %q", first.URL)
	}
	if intFromExtra(first.Extra, "points") != 320 {
		t.Fatalf("score should map to points, extra=%#v", first.Extra)
	}
	if disc := stringFromExtra(first.Extra, "discussion"); disc != "https://www.reddit.com/r/rust/comments/abc/x/" {
		t.Fatalf("discussion permalink = %q", disc)
	}
	if sub := stringFromExtra(first.Extra, "subreddit"); sub != "rust" {
		t.Fatalf("subreddit = %q", sub)
	}
	if !strings.HasPrefix(first.Published, "2024-01-01") {
		t.Fatalf("created_utc epoch should render RFC3339, got %q", first.Published)
	}
	second := cands[1]
	if strings.Contains(second.Summary, "<") || !strings.Contains(second.Summary, "We run untrusted tools") {
		t.Fatalf("selftext should be tag-stripped prose, got %q", second.Summary)
	}
}

func TestGatherCandidatesRedditFiltersByPoints(t *testing.T) {
	rj := `{"data":{"children":[
		{"kind":"t3","data":{"id":"1","title":"hot trending agent tool","url":"https://example.test/1","permalink":"/r/x/comments/1/a/","score":300,"created_utc":1704067200,"subreddit":"x"}},
		{"kind":"t3","data":{"id":"2","title":"barely-voted post","url":"https://example.test/2","permalink":"/r/x/comments/2/b/","score":2,"created_utc":1704067200,"subreddit":"x"}}
	]}}`
	topics := []Topic{{Key: "t", Reddit: "agent tool", Terms: []string{"agent", "tool"}}}
	cfg := DefaultConfig()
	var errs []string
	cands := GatherCandidates(stubFetcher{redditJSON: rj}, topics, cfg, &errs)
	if len(errs) != 0 {
		t.Fatalf("unexpected gather errors: %v", errs)
	}
	if len(cands) != 1 || cands[0].SourceID != "reddit:1" {
		t.Fatalf("want only the 300-point post above MinPoints, got %#v", cands)
	}
}

func TestGatherCandidatesFreshLaneAdmitsYoungRepo(t *testing.T) {
	// A young repo below MinStars(25): the stars lane drops it, the fresh lane
	// (star floor FreshMinStars=3) admits it and tags its provenance.
	young := GitHubRepo{
		FullName:        "newco/fresh-agent",
		URL:             "https://github.com/newco/fresh-agent",
		Description:     "a brand-new agent tool sandbox",
		StargazersCount: 8,
		PushedAt:        "2026-06-28T00:00:00Z",
		CreatedAt:       "2026-06-20T00:00:00Z",
	}
	topics := []Topic{{Key: "t", GitHub: "agent tool", Terms: []string{"agent", "tool"}}}
	cfg := DefaultConfig() // MinStars=25, FreshMinStars=3, FreshPerTopic=6
	var errs []string
	// Offered on BOTH lanes: the stars floor drops it, only the fresh lane admits it.
	cands := GatherCandidates(stubFetcher{github: []GitHubRepo{young}, githubFresh: []GitHubRepo{young}}, topics, cfg, &errs)
	if len(errs) != 0 {
		t.Fatalf("unexpected gather errors: %v", errs)
	}
	if len(cands) != 1 || cands[0].SourceID != "github:newco/fresh-agent" {
		t.Fatalf("want exactly the fresh-lane candidate, got %#v", cands)
	}
	if cands[0].Extra["lane"] != "fresh" {
		t.Fatalf("fresh-lane candidate not tagged lane=fresh, Extra=%#v", cands[0].Extra)
	}
}

func TestGatherCandidatesFreshLaneRespectsFreshMinStars(t *testing.T) {
	toy := GitHubRepo{
		FullName:        "toy/repo",
		URL:             "https://github.com/toy/repo",
		StargazersCount: 1, // below FreshMinStars(3): pure noise, dropped
		PushedAt:        "2026-06-29T00:00:00Z",
		CreatedAt:       "2026-06-25T00:00:00Z",
	}
	topics := []Topic{{Key: "t", GitHub: "agent tool", Terms: []string{"agent"}}}
	cfg := DefaultConfig() // FreshMinStars=3
	var errs []string
	cands := GatherCandidates(stubFetcher{githubFresh: []GitHubRepo{toy}}, topics, cfg, &errs)
	if len(errs) != 0 {
		t.Fatalf("unexpected gather errors: %v", errs)
	}
	if len(cands) != 0 {
		t.Fatalf("1-star toy repo should be dropped by FreshMinStars=3, got %#v", cands)
	}
}

func TestScoreCandidateTrending(t *testing.T) {
	now := time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC)
	topic := Topic{Key: "t", Terms: []string{"agent"}}
	// Same stars, same recent push; only repo age differs. The young repo accrued
	// its stars fast (high stars/day) and earns the trending bonus; the old one didn't.
	young := Candidate{Title: "agent x", Published: "2026-06-10T00:00:00Z", Extra: map[string]any{"stars": 400, "pushed_at": "2026-06-29T00:00:00Z"}}
	old := Candidate{Title: "agent x", Published: "2022-06-10T00:00:00Z", Extra: map[string]any{"stars": 400, "pushed_at": "2026-06-29T00:00:00Z"}}
	youngScore, youngReasons := ScoreCandidate(young, topic, DefaultConfig(), now)
	oldScore, _ := ScoreCandidate(old, topic, DefaultConfig(), now)
	if youngScore <= oldScore {
		t.Fatalf("young high-velocity repo (%d) should outscore old same-star repo (%d)", youngScore, oldScore)
	}
	if joined := strings.Join(youngReasons, "; "); !strings.Contains(joined, "trending") {
		t.Fatalf("young repo reasons missing trending signal: %q", joined)
	}
}

func TestApplyThresholdsFreshKnobs(t *testing.T) {
	cfg := DefaultConfig()
	// JSON numbers decode to float64 — anyInt must coerce them.
	applyThresholds(&cfg, map[string]any{
		"fresh_per_topic":   float64(10),
		"fresh_min_stars":   float64(1),
		"fresh_window_days": float64(30),
	})
	if cfg.FreshPerTopic != 10 || cfg.FreshMinStars != 1 || cfg.FreshWindowDays != 30 {
		t.Fatalf("fresh knobs not applied: %+v", cfg)
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
