package ideascout

// The package surface: the tuning constants, the wire/record types every stage
// exchanges, the Fetcher seam the live and fixture paths both satisfy, and the
// RunOptions a caller hands to Run.

import (
	"time"
)

const (
	Schema          = "fleet-idea-scout/1"
	CacheDirname    = ".idea-scout"
	CacheFilename   = "seen.json"
	ScoutLabel      = "idea-scout"
	TriageLabel     = "needs-triage"
	TriageOnlyLabel = "triage-only"
	ArxivAPI        = "http://export.arxiv.org/api/query"
	HNAlgoliaAPI    = "https://hn.algolia.com/api/v1/search_by_date"
	RedditSearchAPI = "https://www.reddit.com/search.json"

	WTitleHit    = 10
	WBodyHit     = 3
	WRecent180   = 12
	WRecent30    = 22
	StarDivisor  = 100
	StarCap      = 30
	HNPointDiv   = 20
	HNPointCap   = 25
	WRecentPush  = 10
	WFreshPush   = 15 // pushed within FreshWindowDays: stronger "actively updated" bonus
	TrendingCap  = 20 // cap on the star-velocity (stars/day) trending bonus
	DefaultToday = "1970-01-01"
)

type Topic struct {
	Key    string   `json:"key"`
	Arxiv  string   `json:"arxiv,omitempty"`
	GitHub string   `json:"github,omitempty"`
	HN     string   `json:"hn,omitempty"`
	Reddit string   `json:"reddit,omitempty"`
	Terms  []string `json:"terms"`
	Area   string   `json:"area,omitempty"`
}

type Config struct {
	RecentDays      int     `json:"recent_days"`
	MinScore        int     `json:"min_score"`
	MaxIssues       int     `json:"max_issues"`
	ArxivPerTopic   int     `json:"arxiv_per_topic"`
	GitHubPerTopic  int     `json:"github_per_topic"`
	HNPerTopic      int     `json:"hn_per_topic"`
	RedditPerTopic  int     `json:"reddit_per_topic"`
	MinStars        int     `json:"min_stars"`
	FreshPerTopic   int     `json:"fresh_per_topic"`   // recency-sorted GitHub repos fetched per topic (0 disables the fresh lane)
	FreshMinStars   int     `json:"fresh_min_stars"`   // fresh-lane star floor: admits young repos the MinStars floor would drop
	FreshWindowDays int     `json:"fresh_window_days"` // pushed within this window earns the strong "actively updated" bonus
	MinPoints       int     `json:"min_points"`
	DupJaccard      float64 `json:"dup_jaccard"`
	// IssueScanLimit feeds the SOFT rungs only (issue-body + title-near, i.e. "did a
	// human already write this up lately"): a recency window over the whole tracker,
	// whose coverage shrinks every time the tracker gets busier. Deliberately NOT the
	// anti-re-file guarantee — that is ScoutScanLimit below.
	IssueScanLimit int `json:"issue_scan_limit"`
	// ScoutScanLimit bounds the DURABLE rung: every issue carrying the idea-scout
	// label, i.e. the scout's own filing history. Bounded by MaxIssues/day (<=3), not
	// by tracker growth, so one number covers years. Saturating it is a REFUSAL, never
	// a silent truncation — see the scout-index gate in Run.
	ScoutScanLimit int    `json:"scout_scan_limit"`
	Milestone      string `json:"milestone,omitempty"`
	Project        string `json:"project,omitempty"`
	ProjectOwner   string `json:"project_owner,omitempty"`
}

type Candidate struct {
	Source    string         `json:"source"`
	SourceID  string         `json:"source_id"`
	URL       string         `json:"url"`
	Title     string         `json:"title"`
	Summary   string         `json:"summary,omitempty"`
	Published string         `json:"published,omitempty"`
	Topic     string         `json:"topic"`
	Extra     map[string]any `json:"extra,omitempty"`
}

type ExistingIssue struct {
	Number int    `json:"number,omitempty"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

type IssuePlan struct {
	Title    string   `json:"title"`
	Body     string   `json:"body,omitempty"`
	Labels   []string `json:"labels"`
	SourceID string   `json:"source_id"`
	URL      string   `json:"url"`
	Score    int      `json:"score"`
	Topic    string   `json:"topic"`
}

type SeenRecord struct {
	FiledAt  string `json:"filed_at,omitempty"`
	IssueURL string `json:"issue_url,omitempty"`
	Score    int    `json:"score,omitempty"`
	Topic    string `json:"topic,omitempty"`
}

type RunResult struct {
	Schema             string          `json:"schema"`
	Date               string          `json:"date"`
	Mode               string          `json:"mode"`
	CandidatesGathered int             `json:"candidates_gathered"`
	DedupIndex         DedupIndex      `json:"dedup_index"`
	Skipped            map[string]int  `json:"skipped"`
	Dropped            []DroppedSource `json:"dropped,omitempty"`
	Planned            []IssuePlan     `json:"planned"`
	Filed              []FiledIssue    `json:"filed"`
	Errors             []string        `json:"errors"`
	SourceDigest       string          `json:"source_digest,omitempty"`
	Topics             int             `json:"topics"`
	Thresholds         map[string]any  `json:"thresholds,omitempty"`
}

// DedupIndex makes the durable rung's coverage auditable in the run record: a
// reader can SEE that the filed-issue index was label-targeted and unsaturated
// rather than having to trust that it was. ScoutIndexComplete is only ever true
// because the incomplete case refuses (Run returns an error) instead of reporting.
type DedupIndex struct {
	FiledIssuesScanned  int  `json:"filed_issues_scanned"`
	FiledStamps         int  `json:"filed_stamps"`
	ScoutScanLimit      int  `json:"scout_scan_limit"`
	ScoutIndexComplete  bool `json:"scout_index_complete"`
	WindowIssuesScanned int  `json:"window_issues_scanned"`
	IssueScanLimit      int  `json:"issue_scan_limit"`
}

// DroppedSource is per-source dedup attribution: which rung stopped which
// source_id. Aggregate counts alone cannot answer "was THIS already-triaged
// source caught, and by which rung", which is the only question that matters
// after a re-file.
type DroppedSource struct {
	SourceID string `json:"source_id"`
	Rung     string `json:"rung"`
}

type FiledIssue struct {
	Title    string `json:"title"`
	IssueURL string `json:"issue_url"`
}

type GitHubRepo struct {
	FullName        string `json:"fullName"`
	Name            string `json:"name"`
	Description     string `json:"description"`
	URL             string `json:"url"`
	StargazersCount int    `json:"stargazersCount"`
	PushedAt        string `json:"pushedAt"`
	UpdatedAt       string `json:"updatedAt"`
	CreatedAt       string `json:"createdAt"`
	Language        string `json:"language"`
}

type Fetcher interface {
	FetchArxiv(query string, maxResults int) (string, error)
	FetchGitHub(query string, limit int) ([]GitHubRepo, error)
	FetchGitHubFresh(query string, limit int) ([]GitHubRepo, error)
	FetchHackerNews(query string, limit int) (string, error)
	FetchReddit(query string, limit int) (string, error)
	FetchExistingIssues(limit int) ([]ExistingIssue, error)
	// FetchScoutIssues returns every issue the scout has ever filed, open or
	// closed, identified by its label rather than by recency. It is a separate
	// method from FetchExistingIssues precisely because the two corpora answer
	// different questions: this one is the never-file-twice guarantee, that one
	// is a best-effort look at what humans opened lately.
	FetchScoutIssues(limit int) ([]ExistingIssue, error)
	EnsureLabels() error
	CreateIssue(issue IssuePlan, milestone string) (string, error)
	AddToProject(issueURL, number, owner string) error
}

type RunOptions struct {
	Workspace    string
	ConfigPath   string
	MaxIssues    *int
	MinScore     *int
	Live         bool
	JSON         bool
	Milestone    *string
	Project      *string
	ProjectOwner *string
	Candidates   []Candidate
	Existing     []ExistingIssue
	// ScoutIssues is the fixture stand-in for the label-targeted filed-issue index
	// (rung 2). When a fixture run leaves it nil, Existing serves as the whole
	// corpus — a replay has no window to be truncated by, so the two collapse.
	ScoutIssues []ExistingIssue
	UseFixtures bool
	Today       string
	Now         time.Time
	Fetcher     Fetcher
	// NOTE: there is deliberately no knob here that waives a dedup refusal.
	// RunOptions used to carry `AllowIssueGap bool`, set by no caller anywhere in
	// the repo and read in exactly one place — to turn OFF the refusal that fires
	// when the window fetch fails with no seen-cache to fall back on. Its only
	// reachable effect was to weaken a guarantee, it had no counterpart in
	// tools/idea_scout.py, and a knob present on one implementation and not the
	// other is precisely the drift that made the same defect need two fixes
	// (#5543 then #5544). Removed for #5547; the refusal is unconditional in both
	// implementations and testdata/dedup_corpus.json pins it that way.
}
