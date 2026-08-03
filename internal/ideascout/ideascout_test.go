package ideascout

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
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
		{"stamp", Candidate{SourceID: "github:owner/repo", URL: "https://github.com/Owner/Repo", Title: "Other"}, "filed-stamp"},
		{"stamp-case-folded", Candidate{SourceID: "github:Owner/Repo", URL: "https://example.test/nope", Title: "Other"}, "filed-stamp"},
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

	plans, stats, _ := PlanIssues(candidates, map[string]Topic{topic.Key: topic}, nil, nil, nil, "", Config{RecentDays: 180, MinScore: 1, MaxIssues: 2, DupJaccard: 0.55}, "2026-06-30", now)
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
func (s stubFetcher) FetchScoutIssues(int) ([]ExistingIssue, error)      { return nil, nil }
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

// windowedFetcher models GitHub the way the two dedup corpora actually see it:
// `all` is the tracker newest-first, so FetchExistingIssues truncates it to the
// caller's limit — a RECENCY WINDOW — while FetchScoutIssues answers a query
// TARGETED at the idea-scout label and so returns the scout's whole filing
// history however old, exactly as `gh issue list --label idea-scout` does.
type windowedFetcher struct {
	repos     []GitHubRepo
	all       []ExistingIssue // newest first
	labeled   []ExistingIssue // carries the idea-scout label + stamp
	scoutErr  error
	windowErr error
}

func (f windowedFetcher) FetchArxiv(string, int) (string, error)             { return "", nil }
func (f windowedFetcher) FetchGitHub(string, int) ([]GitHubRepo, error)      { return f.repos, nil }
func (f windowedFetcher) FetchGitHubFresh(string, int) ([]GitHubRepo, error) { return nil, nil }
func (f windowedFetcher) FetchHackerNews(string, int) (string, error)        { return "", nil }
func (f windowedFetcher) FetchReddit(string, int) (string, error)            { return "", nil }
func (f windowedFetcher) EnsureLabels() error                                { return nil }
func (f windowedFetcher) CreateIssue(IssuePlan, string) (string, error)      { return "", nil }
func (f windowedFetcher) AddToProject(string, string, string) error          { return nil }

func (f windowedFetcher) FetchExistingIssues(limit int) ([]ExistingIssue, error) {
	if f.windowErr != nil {
		return nil, f.windowErr
	}
	if limit >= 0 && len(f.all) > limit {
		return f.all[:limit], nil // the tracker's tail falls off the window
	}
	return f.all, nil
}

func (f windowedFetcher) FetchScoutIssues(limit int) ([]ExistingIssue, error) {
	if f.scoutErr != nil {
		return nil, f.scoutErr
	}
	if limit >= 0 && len(f.labeled) > limit {
		return f.labeled[:limit], nil // saturated: gh cannot say whether it truncated
	}
	return f.labeled, nil
}

// agedFiling is the case the 800-issue window lost: an idea-scout issue filed long
// ago (and possibly closed since) that has since fallen out of any recency window.
var agedFiling = ExistingIssue{
	Number: 528,
	Title:  "idea-scout: o/aged",
	Body:   "> Auto-filed by the daily idea-scout.\n\n**Source:** https://github.com/o/aged\n<!-- idea-scout-source: github:o/aged -->",
}

func agedCandidateRepo() GitHubRepo {
	return GitHubRepo{
		FullName:        "o/aged",
		URL:             "https://github.com/o/aged",
		Description:     "an agent tool sandbox",
		StargazersCount: 500,
		PushedAt:        "2026-07-30T00:00:00Z",
		CreatedAt:       "2026-07-01T00:00:00Z",
	}
}

func unrelatedRecentIssues() []ExistingIssue {
	return []ExistingIssue{
		{Number: 5540, Title: "flake in the gateway suite", Body: "unrelated"},
		{Number: 5539, Title: "release ship pins a stale base", Body: "unrelated"},
		{Number: 5538, Title: "docs freshness stamp drifted", Body: "unrelated"},
		{Number: 5537, Title: "lease reaper leaves orphans", Body: "unrelated"},
		{Number: 5536, Title: "push gate rejects a tiered leaf", Body: "unrelated"},
	}
}

func writeScoutConfig(t *testing.T, dir, thresholds string) string {
	t.Helper()
	path := filepath.Join(dir, "config.json")
	body := `{
		"topics": [{"key":"t","github":"agent tool","terms":["agent","tool"],"area":"trust-floor"}],
		"thresholds": {` + thresholds + `}
	}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

// TestFiledStampCatchesASourceFiledBelowTheScanWindow is the regression pin for
// #5544. The defect is SELF-MASKING at today's tip: the duplicates the 800-issue
// window already produced (#5298/#5308/#5309) still sit INSIDE that window and are
// matched, so a green run at the tip proves nothing. The window is shrunk instead
// — issue_scan_limit=2 over a six-issue tracker — so the aged filing is below it,
// which is exactly the state #528 was in when it came back as #5298.
func TestFiledStampCatchesASourceFiledBelowTheScanWindow(t *testing.T) {
	dir := t.TempDir()
	cfgPath := writeScoutConfig(t, dir, `"min_score": 1, "max_issues": 3, "issue_scan_limit": 2, "scout_scan_limit": 50`)
	fetcher := windowedFetcher{
		repos:   []GitHubRepo{agedCandidateRepo()},
		all:     append(unrelatedRecentIssues(), agedFiling), // newest first: the filing is 6th of 6
		labeled: []ExistingIssue{agedFiling},
	}

	// The counterfactual first: through the WINDOW alone the aged filing is
	// invisible, so the pre-fix rungs would have called this source new and filed
	// it a second time. If this ever stops holding, the test below is vacuous.
	window, err := fetcher.FetchExistingIssues(2)
	if err != nil {
		t.Fatalf("window fetch: %v", err)
	}
	winStamped, winTitles, winBodies := ExistingIssueIndex(window)
	aged := Candidate{SourceID: "github:o/aged", URL: "https://github.com/o/aged", Title: "o/aged"}
	if rung := IsDuplicate(aged, nil, winStamped, winTitles, winBodies, 0.55); rung != "" {
		t.Fatalf("fixture does not exercise the defect: the 2-issue window already catches the aged filing via %q", rung)
	}

	// No seen-cache exists in dir: the git-ignored fast path is gone, which is the
	// other half of the failure (#5543). The guarantee has to hold without it.
	result, err := Run(RunOptions{
		Workspace:  dir,
		ConfigPath: cfgPath,
		Fetcher:    fetcher,
		Today:      "2026-08-02",
		Now:        time.Date(2026, 8, 2, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(result.Planned) != 0 {
		t.Fatalf("re-filed a source whose issue is below the scan window: %#v", result.Planned)
	}
	if result.Skipped["filed-stamp"] != 1 {
		t.Fatalf("filed-stamp rung did not fire: skipped=%#v", result.Skipped)
	}
	if result.Skipped["issue-body"] != 0 || result.Skipped["title-near"] != 0 {
		t.Fatalf("the windowed rungs must not be what caught it: skipped=%#v", result.Skipped)
	}
	var attributed bool
	for _, d := range result.Dropped {
		if d.SourceID == "github:o/aged" && d.Rung == "filed-stamp" {
			attributed = true
		}
	}
	if !attributed {
		t.Fatalf("dropped attribution missing github:o/aged on filed-stamp: %#v", result.Dropped)
	}
	if idx := result.DedupIndex; idx.FiledIssuesScanned != 1 || idx.FiledStamps != 1 || !idx.ScoutIndexComplete || idx.WindowIssuesScanned != 2 {
		t.Fatalf("dedup index does not report a complete label-targeted scan alongside the 2-issue window: %#v", idx)
	}
}

// A saturated label scan is ambiguous between complete and truncated, and a
// truncated index is indistinguishable from "this source is new". Growth has to
// surface as a loud stop, not a quiet re-file.
func TestSaturatedScoutIndexRefuses(t *testing.T) {
	dir := t.TempDir()
	cfgPath := writeScoutConfig(t, dir, `"min_score": 1, "scout_scan_limit": 2`)
	fetcher := windowedFetcher{
		repos:   []GitHubRepo{agedCandidateRepo()},
		all:     unrelatedRecentIssues(),
		labeled: []ExistingIssue{agedFiling, {Number: 529, Title: "idea-scout: o/two", Body: "<!-- idea-scout-source: github:o/two -->"}},
	}
	_, err := Run(RunOptions{Workspace: dir, ConfigPath: cfgPath, Fetcher: fetcher, Today: "2026-08-02"})
	if err == nil {
		t.Fatal("a saturated filed-issue index must refuse, not proceed on a possibly-truncated scan")
	}
	if !strings.Contains(err.Error(), "saturated") || !strings.Contains(err.Error(), "scout_scan_limit=2") {
		t.Fatalf("refusal does not name the saturation tripwire: %v", err)
	}
}

// The durable rung is MANDATORY: a populated seen-cache is a node-local fast path
// and cannot stand in for it, because the cache is exactly what was lost in #5543.
func TestScoutIndexFetchFailureRefusesDespiteSeenCache(t *testing.T) {
	dir := t.TempDir()
	if err := SaveSeen(dir, map[string]SeenRecord{"github:o/aged": {FiledAt: "2025-01-01"}}); err != nil {
		t.Fatalf("SaveSeen: %v", err)
	}
	cfgPath := writeScoutConfig(t, dir, `"min_score": 1`)
	fetcher := windowedFetcher{repos: []GitHubRepo{agedCandidateRepo()}, all: unrelatedRecentIssues(), scoutErr: errors.New("gh: not authenticated")}
	_, err := Run(RunOptions{Workspace: dir, ConfigPath: cfgPath, Fetcher: fetcher, Today: "2026-08-02"})
	if err == nil {
		t.Fatal("a failed filed-issue index must refuse even when a seen-cache exists")
	}
	if !strings.Contains(err.Error(), "cannot build the filed-issue index") {
		t.Fatalf("refusal does not name the durable rung: %v", err)
	}
}

// The pre-existing refusal on a failed window scan is not relaxed on the back of
// the new rung: degrading the soft rungs onto a bare local cache is still a worse
// run than no run.
func TestWindowFetchRefusalIsUnchanged(t *testing.T) {
	fetcher := windowedFetcher{repos: []GitHubRepo{agedCandidateRepo()}, labeled: []ExistingIssue{agedFiling}, windowErr: errors.New("gh: rate limited")}

	bare := t.TempDir()
	cfgPath := writeScoutConfig(t, bare, `"min_score": 1`)
	if _, err := Run(RunOptions{Workspace: bare, ConfigPath: cfgPath, Fetcher: fetcher, Today: "2026-08-02"}); err == nil {
		t.Fatal("a failed window scan with no seen-cache must still refuse")
	} else if !strings.Contains(err.Error(), "cannot fetch existing issues and no seen-cache") {
		t.Fatalf("window refusal changed shape: %v", err)
	}

	cached := t.TempDir()
	if err := SaveSeen(cached, map[string]SeenRecord{"github:o/other": {FiledAt: "2025-01-01"}}); err != nil {
		t.Fatalf("SaveSeen: %v", err)
	}
	cachedCfg := writeScoutConfig(t, cached, `"min_score": 1`)
	result, err := Run(RunOptions{Workspace: cached, ConfigPath: cachedCfg, Fetcher: fetcher, Today: "2026-08-02", Now: time.Date(2026, 8, 2, 0, 0, 0, 0, time.UTC)})
	if err != nil {
		t.Fatalf("a failed window scan with a seen-cache should degrade, not refuse: %v", err)
	}
	if result.Skipped["filed-stamp"] != 1 {
		t.Fatalf("the durable rung must still gate when the window is gone: skipped=%#v", result.Skipped)
	}
}

// ============================================================================
// The SHARED corpus (#5547).
//
// testdata/dedup_corpus.json is read by BOTH scouts: these tests and
// tools/idea_scout_test.py. It is the mechanical replacement for the prose
// "Two implementations, one contract" table in docs/idea-scout.md — the tie
// that let the SAME dedup defect be fixed twice, once per implementation
// (cfe66c656 Python/#5543, then 00f270957d2a Go/#5544). A rung that changes
// here and not in tools/idea_scout.py (or the other way round) now reds a test
// instead of aging into a re-filed issue.
// ============================================================================

const corpusPath = "testdata/dedup_corpus.json"

type corpusIssue struct {
	Number int    `json:"number"`
	State  string `json:"state"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

type corpusCandidate struct {
	SourceID string `json:"source_id"`
	URL      string `json:"url"`
	Title    string `json:"title"`
}

type corpusCase struct {
	Name      string          `json:"name"`
	Candidate corpusCandidate `json:"candidate"`
	Want      string          `json:"want"`
	Why       string          `json:"why"`
}

type corpusDedupIndex struct {
	FiledIssuesScanned  int  `json:"filed_issues_scanned"`
	FiledStamps         int  `json:"filed_stamps"`
	ScoutIndexComplete  bool `json:"scout_index_complete"`
	WindowIssuesScanned int  `json:"window_issues_scanned"`
}

type corpusExpect struct {
	Refuse         bool             `json:"refuse"`
	RefuseContains []string         `json:"refuse_contains"`
	Planned        []string         `json:"planned"`
	Skipped        map[string]int   `json:"skipped"`
	Dropped        []DroppedSource  `json:"dropped"`
	DedupIndex     corpusDedupIndex `json:"dedup_index"`
}

type corpusRun struct {
	Name         string                `json:"name"`
	Config       json.RawMessage       `json:"config"`
	Repos        []GitHubRepo          `json:"repos"`
	WindowIssues []corpusIssue         `json:"window_issues"`
	ScoutIssues  []corpusIssue         `json:"scout_issues"`
	WindowError  string                `json:"window_error"`
	ScoutError   string                `json:"scout_error"`
	Seen         map[string]SeenRecord `json:"seen"`
	Expect       corpusExpect          `json:"expect"`
	Why          string                `json:"why"`
}

type dedupCorpus struct {
	Schema          string                `json:"schema"`
	Rungs           []string              `json:"rungs"`
	SkipStatKeys    []string              `json:"skip_stat_keys"`
	DupJaccard      float64               `json:"dup_jaccard"`
	Seen            map[string]SeenRecord `json:"seen"`
	WindowIssues    []corpusIssue         `json:"window_issues"`
	ScoutIssues     []corpusIssue         `json:"scout_issues"`
	DedupCases      []corpusCase          `json:"dedup_cases"`
	WindowOnlyCases []corpusCase          `json:"window_only_cases"`
	Runs            []corpusRun           `json:"runs"`
}

func loadCorpus(t *testing.T) dedupCorpus {
	t.Helper()
	raw, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Fatalf("read shared corpus %s (tools/idea_scout_test.py reads the same file): %v", corpusPath, err)
	}
	var c dedupCorpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("parse shared corpus: %v", err)
	}
	if c.Schema != "fak/idea-scout-dedup-corpus@1" {
		t.Fatalf("corpus schema = %q, want fak/idea-scout-dedup-corpus@1", c.Schema)
	}
	if len(c.DedupCases) == 0 || len(c.WindowOnlyCases) == 0 || len(c.Runs) == 0 {
		t.Fatalf("corpus is empty in at least one section: %d dedup, %d window-only, %d runs",
			len(c.DedupCases), len(c.WindowOnlyCases), len(c.Runs))
	}
	return c
}

func existingIssues(in []corpusIssue) []ExistingIssue {
	out := make([]ExistingIssue, 0, len(in))
	for _, iss := range in {
		out = append(out, ExistingIssue{Number: iss.Number, Title: iss.Title, Body: iss.Body})
	}
	return out
}

func corpusCand(c corpusCandidate) Candidate {
	return Candidate{SourceID: c.SourceID, URL: c.URL, Title: c.Title}
}

// index applies the corpus's index_build_rule: the durable stamps come from the
// label-targeted filing history, unioned with any stamp still visible in the
// recency window; the soft rungs see the window and nothing else.
func (c dedupCorpus) index() (map[string]struct{}, []map[string]struct{}, string) {
	stamped := StampIndex(existingIssues(c.ScoutIssues))
	winStamped, titleSets, bodies := ExistingIssueIndex(existingIssues(c.WindowIssues))
	for sid := range winStamped {
		stamped[sid] = struct{}{}
	}
	return stamped, titleSets, bodies
}

func TestSharedCorpusDedupRungs(t *testing.T) {
	c := loadCorpus(t)
	stamped, titleSets, bodies := c.index()
	for _, tc := range c.DedupCases {
		t.Run(tc.Name, func(t *testing.T) {
			got := IsDuplicate(corpusCand(tc.Candidate), c.Seen, stamped, titleSets, bodies, c.DupJaccard)
			if got != tc.Want {
				t.Fatalf("shared corpus %q: rung = %q, want %q\n  candidate: %+v\n  why: %s\n  (tools/idea_scout_test.py asserts the SAME verdict from the SAME file — a rung that moves in only one implementation must red here)",
					tc.Name, got, tc.Want, tc.Candidate, tc.Why)
			}
		})
	}
}

// The counterfactual half of the corpus: with the durable rung removed (no scout
// index) and the seen-cache gone, every case must come back NEW. That is the
// exact state #5543 was found in, and it is what keeps the filed-stamp cases
// above from passing for some unrelated reason.
func TestSharedCorpusWindowOnlyCounterfactual(t *testing.T) {
	c := loadCorpus(t)
	winStamped, titleSets, bodies := ExistingIssueIndex(existingIssues(c.WindowIssues))
	for _, tc := range c.WindowOnlyCases {
		t.Run(tc.Name, func(t *testing.T) {
			got := IsDuplicate(corpusCand(tc.Candidate), nil, winStamped, titleSets, bodies, c.DupJaccard)
			if got != tc.Want {
				t.Fatalf("shared corpus window-only %q: rung = %q, want %q — the corpus no longer exercises the defect\n  why: %s", tc.Name, got, tc.Want, tc.Why)
			}
		})
	}
}

// The rung VOCABULARY is part of the contract too: renaming, adding or dropping a
// rung on one side only is exactly the drift this corpus exists to catch, and a
// per-case verdict check alone would not see it.
func TestSharedCorpusRungVocabulary(t *testing.T) {
	c := loadCorpus(t)

	_, stats, _ := PlanIssues(nil, nil, nil, nil, nil, "", DefaultConfig(), "2026-08-02", time.Time{})
	var gotKeys []string
	for k := range stats {
		gotKeys = append(gotKeys, k)
	}
	wantKeys := append([]string(nil), c.SkipStatKeys...)
	sort.Strings(gotKeys)
	sort.Strings(wantKeys)
	if !reflect.DeepEqual(gotKeys, wantKeys) {
		t.Fatalf("planner skip-stat keys = %v, corpus skip_stat_keys = %v", gotKeys, wantKeys)
	}

	declared := map[string]bool{}
	for _, r := range c.Rungs {
		declared[r] = true
	}
	exercised := map[string]bool{}
	for _, tc := range c.DedupCases {
		if tc.Want == "" {
			continue
		}
		if !declared[tc.Want] {
			t.Fatalf("case %q expects rung %q, which the corpus does not declare in `rungs`", tc.Name, tc.Want)
		}
		exercised[tc.Want] = true
	}
	for _, r := range c.Rungs {
		if !exercised[r] {
			t.Fatalf("declared rung %q has no case in the shared corpus", r)
		}
		if _, ok := stats[r]; !ok {
			t.Fatalf("declared rung %q is not a planner skip-stat key: %v", r, stats)
		}
	}
}

// corpusFetcher replays one run case. It models GitHub the way the two dedup
// corpora actually see it: FetchExistingIssues TRUNCATES the newest-first tracker
// to the caller's limit (a recency window), while FetchScoutIssues answers a query
// targeted at the idea-scout label and so returns the scout's whole filing history
// however old. tools/idea_scout_test.py stubs the Python fetchers identically.
type corpusFetcher struct {
	repos     []GitHubRepo
	window    []ExistingIssue
	scout     []ExistingIssue
	windowErr error
	scoutErr  error
}

func (f corpusFetcher) FetchArxiv(string, int) (string, error)             { return "", nil }
func (f corpusFetcher) FetchGitHub(string, int) ([]GitHubRepo, error)      { return f.repos, nil }
func (f corpusFetcher) FetchGitHubFresh(string, int) ([]GitHubRepo, error) { return nil, nil }
func (f corpusFetcher) FetchHackerNews(string, int) (string, error)        { return "", nil }
func (f corpusFetcher) FetchReddit(string, int) (string, error)            { return "", nil }
func (f corpusFetcher) EnsureLabels() error                                { return nil }
func (f corpusFetcher) AddToProject(string, string, string) error          { return nil }

func (f corpusFetcher) CreateIssue(IssuePlan, string) (string, error) {
	return "", errors.New("corpus replay is dry-run: nothing may ever be filed")
}

func (f corpusFetcher) FetchExistingIssues(limit int) ([]ExistingIssue, error) {
	if f.windowErr != nil {
		return nil, f.windowErr
	}
	if limit >= 0 && len(f.window) > limit {
		return f.window[:limit], nil
	}
	return f.window, nil
}

func (f corpusFetcher) FetchScoutIssues(limit int) ([]ExistingIssue, error) {
	if f.scoutErr != nil {
		return nil, f.scoutErr
	}
	if limit >= 0 && len(f.scout) > limit {
		return f.scout[:limit], nil
	}
	return f.scout, nil
}

func TestSharedCorpusRuns(t *testing.T) {
	c := loadCorpus(t)
	for _, rc := range c.Runs {
		t.Run(rc.Name, func(t *testing.T) {
			dir := t.TempDir()
			cfgPath := filepath.Join(dir, "config.json")
			if err := os.WriteFile(cfgPath, rc.Config, 0o644); err != nil {
				t.Fatalf("write config: %v", err)
			}
			if len(rc.Seen) > 0 {
				if err := SaveSeen(dir, rc.Seen); err != nil {
					t.Fatalf("SaveSeen: %v", err)
				}
			}
			fetcher := corpusFetcher{
				repos:  rc.Repos,
				window: existingIssues(rc.WindowIssues),
				scout:  existingIssues(rc.ScoutIssues),
			}
			if rc.WindowError != "" {
				fetcher.windowErr = errors.New(rc.WindowError)
			}
			if rc.ScoutError != "" {
				fetcher.scoutErr = errors.New(rc.ScoutError)
			}

			result, err := Run(RunOptions{
				Workspace:  dir,
				ConfigPath: cfgPath,
				Fetcher:    fetcher,
				Today:      "2026-08-02",
				Now:        time.Date(2026, 8, 2, 0, 0, 0, 0, time.UTC),
			})

			if rc.Expect.Refuse {
				if err == nil {
					t.Fatalf("shared corpus run %q must REFUSE, got result %#v\n  why: %s", rc.Name, result, rc.Why)
				}
				for _, want := range rc.Expect.RefuseContains {
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("shared corpus run %q refusal %q missing %q", rc.Name, err.Error(), want)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("shared corpus run %q: %v\n  why: %s", rc.Name, err, rc.Why)
			}

			var planned []string
			for _, p := range result.Planned {
				planned = append(planned, p.SourceID)
			}
			want := rc.Expect.Planned
			if len(planned) == 0 && len(want) == 0 {
				planned, want = nil, nil
			}
			if !reflect.DeepEqual(planned, want) {
				t.Fatalf("shared corpus run %q: planned = %v, want %v\n  why: %s", rc.Name, planned, rc.Expect.Planned, rc.Why)
			}
			for rung, n := range rc.Expect.Skipped {
				if result.Skipped[rung] != n {
					t.Fatalf("shared corpus run %q: skipped[%q] = %d, want %d (all: %#v)", rc.Name, rung, result.Skipped[rung], n, result.Skipped)
				}
			}
			gotDropped := result.Dropped
			if len(gotDropped) == 0 {
				gotDropped = nil
			}
			wantDropped := rc.Expect.Dropped
			if len(wantDropped) == 0 {
				wantDropped = nil
			}
			if !reflect.DeepEqual(gotDropped, wantDropped) {
				t.Fatalf("shared corpus run %q: dropped attribution = %#v, want %#v", rc.Name, gotDropped, wantDropped)
			}
			got := corpusDedupIndex{
				FiledIssuesScanned:  result.DedupIndex.FiledIssuesScanned,
				FiledStamps:         result.DedupIndex.FiledStamps,
				ScoutIndexComplete:  result.DedupIndex.ScoutIndexComplete,
				WindowIssuesScanned: result.DedupIndex.WindowIssuesScanned,
			}
			if got != rc.Expect.DedupIndex {
				t.Fatalf("shared corpus run %q: dedup_index = %+v, want %+v", rc.Name, got, rc.Expect.DedupIndex)
			}
			if _, err := os.Stat(CachePath(dir)); len(rc.Seen) == 0 && !os.IsNotExist(err) {
				t.Fatalf("shared corpus run %q is dry-run and must not write the seen cache (stat err=%v)", rc.Name, err)
			}
		})
	}
}

// ============================================================================
// The SHARED SOURCE corpus (#5549).
//
// testdata/source_corpus.json is read by BOTH scouts: the tests below and
// tools/idea_scout_test.py's SharedSourceCorpusTest. It is the gather-stage
// sibling of dedup_corpus.json — #5547 made the DEDUP contract mechanical after
// the same defect had to be fixed twice; this makes the GATHER contract
// mechanical after `hn` and `reddit` turned out to exist only in Go, so a topic
// naming them on the scheduled Python path gathered nothing and reported success.
// ============================================================================

const sourceCorpusPath = "testdata/source_corpus.json"

type sourceParseCase struct {
	Name    string           `json:"name"`
	Lane    string           `json:"lane"`
	Topic   string           `json:"topic"`
	Payload string           `json:"payload"`
	Want    []map[string]any `json:"want"`
	Why     string           `json:"why"`
}

type sourceConfigCase struct {
	Name           string          `json:"name"`
	Topic          json.RawMessage `json:"topic"`
	Refuse         bool            `json:"refuse"`
	RefuseContains []string        `json:"refuse_contains"`
	Why            string          `json:"why"`
}

type sourceScoreCase struct {
	Name               string    `json:"name"`
	Candidate          Candidate `json:"candidate"`
	Terms              []string  `json:"terms"`
	WantScore          int       `json:"want_score"`
	WantReasonContains string    `json:"want_reason_contains"`
	Why                string    `json:"why"`
}

type sourceGatherCase struct {
	Name              string          `json:"name"`
	Topic             Topic           `json:"topic"`
	MinPoints         int             `json:"min_points"`
	HNPayload         string          `json:"hn_payload"`
	RedditPayload     string          `json:"reddit_payload"`
	HNError           string          `json:"hn_error"`
	RedditError       string          `json:"reddit_error"`
	WantSourceIDs     []string        `json:"want_source_ids"`
	WantErrorsContain []string        `json:"want_errors_contain"`
	Why               string          `json:"why"`
	TopicRaw          json.RawMessage `json:"-"`
}

type sourceCorpus struct {
	Schema        string             `json:"schema"`
	TopicKeys     []string           `json:"topic_keys"`
	MetaKeys      []string           `json:"meta_keys"`
	Lanes         []string           `json:"lanes"`
	DisplayList   string             `json:"display_list"`
	ThresholdKeys []string           `json:"threshold_keys"`
	ConfigCases   []sourceConfigCase `json:"config_cases"`
	ParseCases    []sourceParseCase  `json:"parse_cases"`
	ScoreCases    []sourceScoreCase  `json:"score_cases"`
	GatherCases   []sourceGatherCase `json:"gather_cases"`
}

func loadSourceCorpus(t *testing.T) sourceCorpus {
	t.Helper()
	raw, err := os.ReadFile(sourceCorpusPath)
	if err != nil {
		t.Fatalf("read shared source corpus %s (tools/idea_scout_test.py reads the same file): %v", sourceCorpusPath, err)
	}
	var c sourceCorpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("parse shared source corpus: %v", err)
	}
	if c.Schema != "fak/idea-scout-source-corpus@1" {
		t.Fatalf("corpus schema = %q, want fak/idea-scout-source-corpus@1", c.Schema)
	}
	if len(c.ConfigCases) == 0 || len(c.ParseCases) == 0 || len(c.GatherCases) == 0 || len(c.ScoreCases) == 0 {
		t.Fatalf("source corpus is empty in at least one section: %d config, %d parse, %d score, %d gather",
			len(c.ConfigCases), len(c.ParseCases), len(c.ScoreCases), len(c.GatherCases))
	}
	return c
}

// TestSharedSourceCorpusVocabulary pins the SOURCE VOCABULARY itself: the lanes
// that exist, the topic keys that arm them, and the thresholds a config may set.
// tools/idea_scout_test.py asserts the same three lists against its own
// SOURCE_LANES / TOPIC_META_KEYS / DEFAULTS, so a lane or knob that grows on one
// implementation and not the other reds here instead of silently gathering
// nothing on the path that runs.
func TestSharedSourceCorpusVocabulary(t *testing.T) {
	c := loadSourceCorpus(t)

	var laneLabels []string
	for _, lane := range sourceLanes {
		laneLabels = append(laneLabels, lane.label)
	}
	if !reflect.DeepEqual(laneLabels, c.Lanes) {
		t.Fatalf("gather lane labels = %v, corpus lanes = %v\n  (tools/idea_scout.py's SOURCE_LANES must carry the identical list)", laneLabels, c.Lanes)
	}
	if got := sourceTopicKeys(); !reflect.DeepEqual(got, c.TopicKeys) {
		t.Fatalf("source topic keys = %v, corpus topic_keys = %v", got, c.TopicKeys)
	}
	if !reflect.DeepEqual(topicMetaKeys, c.MetaKeys) {
		t.Fatalf("topic meta keys = %v, corpus meta_keys = %v", topicMetaKeys, c.MetaKeys)
	}
	if got := sourceDisplayList(); got != c.DisplayList {
		t.Fatalf("source display list = %q, corpus display_list = %q", got, c.DisplayList)
	}

	gotThresholds := append([]string(nil), thresholdKeys()...)
	wantThresholds := append([]string(nil), c.ThresholdKeys...)
	sort.Strings(gotThresholds)
	sort.Strings(wantThresholds)
	if !reflect.DeepEqual(gotThresholds, wantThresholds) {
		t.Fatalf("Config threshold keys = %v, corpus threshold_keys = %v\n  (this list caught hn_per_topic/reddit_per_topic/min_points being Go-only knobs)", gotThresholds, wantThresholds)
	}

	// Every declared lane must name a topic key that is itself declared, so the
	// two vocabularies cannot drift from each other inside one implementation.
	declaredKeys := map[string]bool{}
	for _, k := range c.TopicKeys {
		declaredKeys[k] = true
	}
	for _, lane := range sourceLanes {
		if !declaredKeys[lane.topicKey] {
			t.Fatalf("lane %q is armed by topic key %q, which the corpus does not declare", lane.label, lane.topicKey)
		}
	}
}

// TestSharedSourceCorpusConfigCases is the loud-refusal half of #5549: a config
// naming a lane the running implementation cannot serve must FAIL rather than
// gather zero and report success.
func TestSharedSourceCorpusConfigCases(t *testing.T) {
	c := loadSourceCorpus(t)
	for _, tc := range c.ConfigCases {
		t.Run(tc.Name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "config.json")
			doc := "{\"topics\":[" + string(tc.Topic) + "]}"
			if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
				t.Fatalf("write config: %v", err)
			}
			_, _, err := LoadConfig(path)
			if tc.Refuse {
				if err == nil {
					t.Fatalf("config case %q must REFUSE and did not\n  why: %s", tc.Name, tc.Why)
				}
				for _, want := range tc.RefuseContains {
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("config case %q refusal %q missing %q", tc.Name, err.Error(), want)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("config case %q must be ACCEPTED: %v\n  why: %s", tc.Name, err, tc.Why)
			}
		})
	}
}

// TestSharedSourceCorpusParseCases folds the SAME wire bytes tools/idea_scout.py
// folds and asserts the SAME candidates come out, field by field.
func TestSharedSourceCorpusParseCases(t *testing.T) {
	c := loadSourceCorpus(t)
	for _, tc := range c.ParseCases {
		t.Run(tc.Name, func(t *testing.T) {
			var got []Candidate
			switch tc.Lane {
			case "hn":
				got = ParseHackerNewsJSON(tc.Payload, tc.Topic)
			case "reddit":
				got = ParseRedditJSON(tc.Payload, tc.Topic)
			default:
				t.Fatalf("parse case %q names lane %q, which has no parser here", tc.Name, tc.Lane)
			}
			if len(got) != len(tc.Want) {
				t.Fatalf("parse case %q: got %d candidates, want %d\n  got: %+v\n  why: %s", tc.Name, len(got), len(tc.Want), got, tc.Why)
			}
			for i, want := range tc.Want {
				assertCandidateFields(t, tc.Name, i, got[i], want, tc.Why)
			}
		})
	}
}

// assertCandidateFields compares a parsed candidate against the corpus's expected
// object one field at a time. `extra` is compared as a whole map (through a JSON
// round-trip so both sides are float64-normalised), so an extra key present on
// one implementation only is caught rather than ignored.
func assertCandidateFields(t *testing.T, caseName string, i int, got Candidate, want map[string]any, why string) {
	t.Helper()
	gotFields := map[string]any{
		"source":    got.Source,
		"source_id": got.SourceID,
		"url":       got.URL,
		"title":     got.Title,
		"summary":   got.Summary,
		"published": got.Published,
		"topic":     got.Topic,
	}
	for key, wantVal := range want {
		if key == "extra" {
			continue
		}
		gotVal, ok := gotFields[key]
		if !ok {
			t.Fatalf("parse case %q candidate %d: corpus names field %q, which the Go candidate has no counterpart for", caseName, i, key)
		}
		if gotVal != wantVal {
			t.Fatalf("parse case %q candidate %d: %s = %#v, want %#v\n  why: %s\n  (tools/idea_scout_test.py folds the same bytes and asserts the same value)", caseName, i, key, gotVal, wantVal, why)
		}
	}
	wantExtra, ok := want["extra"]
	if !ok {
		return
	}
	gotExtra := normaliseJSON(t, got.Extra)
	if !reflect.DeepEqual(gotExtra, normaliseJSON(t, wantExtra)) {
		t.Fatalf("parse case %q candidate %d: extra = %#v, want %#v\n  why: %s", caseName, i, gotExtra, normaliseJSON(t, wantExtra), why)
	}
}

// normaliseJSON round-trips a value through JSON so numbers on both sides land as
// float64 and the comparison is about content, not Go's int/float distinction.
func normaliseJSON(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal for comparison: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal for comparison: %v", err)
	}
	return out
}

// TestSharedSourceCorpusScoreCases pins the `points` bonus. It was a Go-only
// branch: the same HN story scored 30 here and 10 in Python, so a story that
// cleared min_score on one path could never clear it on the other.
func TestSharedSourceCorpusScoreCases(t *testing.T) {
	c := loadSourceCorpus(t)
	now := time.Date(2026, 8, 2, 0, 0, 0, 0, time.UTC)
	for _, tc := range c.ScoreCases {
		t.Run(tc.Name, func(t *testing.T) {
			score, reasons := ScoreCandidate(tc.Candidate, Topic{Key: "probe", Terms: tc.Terms}, DefaultConfig(), now)
			if score != tc.WantScore {
				t.Fatalf("score case %q: score = %d, want %d (reasons=%v)\n  why: %s", tc.Name, score, tc.WantScore, reasons, tc.Why)
			}
			joined := strings.Join(reasons, "; ")
			if tc.WantReasonContains == "" {
				if strings.Contains(joined, "points") {
					t.Fatalf("score case %q: reasons %q claim a points bonus that was not earned", tc.Name, joined)
				}
				return
			}
			if !strings.Contains(joined, tc.WantReasonContains) {
				t.Fatalf("score case %q: reasons %q missing %q", tc.Name, joined, tc.WantReasonContains)
			}
		})
	}
}

// sourceProbeFetcher serves one gather case: the two points lanes from fixture
// bytes, every other lane empty. tools/idea_scout_test.py stubs the Python
// fetchers identically.
type sourceProbeFetcher struct {
	hn        string
	reddit    string
	hnErr     error
	redditErr error
	corpusFetcher
}

func (f sourceProbeFetcher) FetchHackerNews(string, int) (string, error) {
	return f.hn, f.hnErr
}

func (f sourceProbeFetcher) FetchReddit(string, int) (string, error) {
	return f.reddit, f.redditErr
}

// TestSharedSourceCorpusGatherCases is the BEHAVIOURAL half of the vocabulary
// claim: for each declared topic key, a topic naming only that key must actually
// gather its lane. A key that is admissible at config load and unread at gather
// time is exactly the #5549 defect, and a vocabulary list alone would not see it.
func TestSharedSourceCorpusGatherCases(t *testing.T) {
	c := loadSourceCorpus(t)
	for _, tc := range c.GatherCases {
		t.Run(tc.Name, func(t *testing.T) {
			fetcher := sourceProbeFetcher{hn: tc.HNPayload, reddit: tc.RedditPayload}
			if tc.HNError != "" {
				fetcher.hnErr = errors.New(tc.HNError)
			}
			if tc.RedditError != "" {
				fetcher.redditErr = errors.New(tc.RedditError)
			}
			cfg := DefaultConfig()
			cfg.MinPoints = tc.MinPoints
			var errorsOut []string
			cands := GatherCandidates(fetcher, []Topic{tc.Topic}, cfg, &errorsOut)

			var gotIDs []string
			for _, cand := range cands {
				gotIDs = append(gotIDs, cand.SourceID)
			}
			want := tc.WantSourceIDs
			if len(gotIDs) == 0 && len(want) == 0 {
				gotIDs, want = nil, nil
			}
			if !reflect.DeepEqual(gotIDs, want) {
				t.Fatalf("gather case %q: source_ids = %v, want %v\n  why: %s\n  (tools/idea_scout_test.py runs the same case through gather_candidates)", tc.Name, gotIDs, tc.WantSourceIDs, tc.Why)
			}
			for _, wantErr := range tc.WantErrorsContain {
				if !strings.Contains(strings.Join(errorsOut, "; "), wantErr) {
					t.Fatalf("gather case %q: errors %v missing %q", tc.Name, errorsOut, wantErr)
				}
			}
			if len(tc.WantErrorsContain) == 0 && len(errorsOut) > 0 {
				t.Fatalf("gather case %q recorded unexpected errors: %v", tc.Name, errorsOut)
			}
		})
	}
}

// TestSourceCorpusCoversEveryDeclaredTopicKey keeps the gather cases honest: a
// lane added to the vocabulary with no case proving it actually gathers would let
// the #5549 defect back in under a green suite.
func TestSourceCorpusCoversEveryDeclaredTopicKey(t *testing.T) {
	c := loadSourceCorpus(t)
	covered := map[string]bool{}
	for _, tc := range c.GatherCases {
		for _, key := range sourceTopicKeys() {
			if topicNamesKey(tc.Topic, key) && len(tc.WantSourceIDs) > 0 {
				covered[key] = true
			}
		}
	}
	// arxiv and github are covered by the run corpus and the fresh-lane tests;
	// the points lanes are the ones this corpus exists for.
	for _, key := range []string{"hn", "reddit"} {
		if !covered[key] {
			t.Fatalf("declared topic key %q has no gather case that admits a candidate — the corpus would not notice the lane going missing", key)
		}
	}
}

// TestEveryDeclaredThresholdIsActuallyRead welds the threshold vocabulary to the
// code that CONSUMES it, which is the same weld the gather cases make for source
// lanes. thresholdKeys() is reflected off Config's JSON tags, so a knob becomes
// admissible the moment the field exists — but applyThresholds reads a hand-written
// switch, and a key that is admissible at load and unread at apply is exactly the
// #5549 defect on the threshold half: the setting appears to take and does not.
// Without this, adding a Config field and updating the corpus list is enough to
// ship a silently-ignored knob.
func TestEveryDeclaredThresholdIsActuallyRead(t *testing.T) {
	rt := reflect.TypeOf(Config{})
	for _, key := range thresholdKeys() {
		t.Run(key, func(t *testing.T) {
			idx := -1
			for i := 0; i < rt.NumField(); i++ {
				if name, _, _ := strings.Cut(rt.Field(i).Tag.Get("json"), ","); name == key {
					idx = i
					break
				}
			}
			if idx < 0 {
				t.Fatalf("threshold %q is declared but has no Config field", key)
			}
			cfg := DefaultConfig()
			var probe any
			switch cur := reflect.ValueOf(cfg).Field(idx); cur.Kind() {
			case reflect.Int:
				probe = int(cur.Int()) + 7
			case reflect.Float64:
				probe = cur.Float() + 0.125
			case reflect.String:
				probe = "probe-" + key
			default:
				t.Fatalf("threshold %q has unhandled kind %s — teach this test before adding it", key, cur.Kind())
			}
			applyThresholds(&cfg, map[string]any{key: probe})
			if got := reflect.ValueOf(cfg).Field(idx).Interface(); got != probe {
				t.Fatalf("applyThresholds ignored declared threshold %q: Config.%s = %v, want %v — the key is admissible at load and unread at apply, which is the silent no-op this corpus exists to refuse",
					key, rt.Field(idx).Name, got, probe)
			}
		})
	}
}

func topicNamesKey(t Topic, key string) bool {
	switch key {
	case "arxiv":
		return t.Arxiv != ""
	case "github":
		return t.GitHub != ""
	case "hn":
		return t.HN != ""
	case "reddit":
		return t.Reddit != ""
	}
	return false
}

func hasLabel(labels []string, want string) bool {
	for _, got := range labels {
		if got == want {
			return true
		}
	}
	return false
}
