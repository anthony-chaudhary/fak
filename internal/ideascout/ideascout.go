package ideascout

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
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

func DefaultConfig() Config {
	return Config{
		RecentDays:      180,
		MinScore:        25,
		MaxIssues:       3,
		ArxivPerTopic:   8,
		GitHubPerTopic:  6,
		HNPerTopic:      8,
		RedditPerTopic:  8,
		MinStars:        25,
		FreshPerTopic:   6,
		FreshMinStars:   3,
		FreshWindowDays: 45,
		MinPoints:       10,
		DupJaccard:      0.55,
		IssueScanLimit:  800,
		ScoutScanLimit:  5000,
	}
}

func DefaultTopics() []Topic {
	return []Topic{
		{Key: "prompt-injection-defense", Arxiv: `abs:"prompt injection" AND (abs:agent OR abs:LLM OR abs:tool)`, GitHub: "prompt injection defense", HN: "prompt injection", Reddit: "prompt injection agent", Terms: []string{"prompt injection", "indirect", "jailbreak", "guardrail", "defense", "tool", "agent", "untrusted", "quarantine"}, Area: "security"},
		{Key: "tool-call-adjudication", Arxiv: `(abs:"tool use" OR abs:"function calling") AND (abs:safety OR abs:permission OR abs:capability OR abs:policy)`, GitHub: "agent tool security", HN: "agent tool permissions", Reddit: "agent tool sandbox permission", Terms: []string{"tool call", "function calling", "capability", "permission", "policy", "adjudicat", "default-deny", "sandbox", "syscall"}, Area: "trust-floor"},
		{Key: "agent-gateway-serving", Arxiv: `(abs:LLM OR abs:agent) AND (abs:gateway OR abs:proxy OR abs:serving OR abs:router)`, GitHub: "llm gateway proxy", HN: "llm gateway", Reddit: "llm gateway proxy router", Terms: []string{"gateway", "proxy", "serving", "router", "openai", "api", "multi-agent", "shared cache", "audit"}, Area: "agentic-serving"},
		{Key: "kv-prefix-cache-reuse", Arxiv: `(abs:"KV cache" OR abs:"prefix cache" OR abs:"prompt cache") AND (abs:reuse OR abs:sharing OR abs:inference)`, GitHub: "llm kv cache", HN: "prompt caching", Reddit: "kv cache prompt caching inference", Terms: []string{"kv cache", "prefix cache", "prompt cache", "reuse", "radix", "paged", "sharing", "turn", "prefill", "speculative"}, Area: "prompt-caching"},
		{Key: "mcp-security", Arxiv: `abs:"model context protocol" OR (abs:agent AND abs:"tool poisoning")`, GitHub: "MCP security", HN: "model context protocol", Reddit: "model context protocol mcp", Terms: []string{"model context protocol", "mcp", "tool poisoning", "server", "manifest", "untrusted", "supply chain"}, Area: "mcp"},
		{Key: "agent-model-arch", Arxiv: `(abs:agent OR abs:"tool use") AND (abs:"function calling" OR abs:fine-tuning OR abs:training) AND ti:LLM`, GitHub: "function calling agent", HN: "open source llm agent", Reddit: "local llm function calling agent", Terms: []string{"function calling", "tool use", "fine-tun", "training", "checkpoint", "qwen", "llama", "reasoning"}, Area: "model-arch"},
	}
}

var tokenRE = regexp.MustCompile(`[a-z0-9]+`)

func Tokenize(text string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, t := range tokenRE.FindAllString(strings.ToLower(text), -1) {
		if len(t) >= 3 {
			out[t] = struct{}{}
		}
	}
	return out
}

func Jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	inter := 0
	for k := range a {
		if _, ok := b[k]; ok {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

func ScoreCandidate(c Candidate, topic Topic, cfg Config, now time.Time) (int, []string) {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	title := strings.ToLower(c.Title)
	body := strings.ToLower(c.Summary)
	score := 0
	var reasons []string
	var hitTerms []string
	for _, term := range topic.Terms {
		t := strings.ToLower(term)
		switch {
		case strings.Contains(title, t):
			score += WTitleHit
			hitTerms = append(hitTerms, term+"(title)")
		case strings.Contains(body, t):
			score += WBodyHit
			hitTerms = append(hitTerms, term)
		}
	}
	if len(hitTerms) > 0 {
		reasons = append(reasons, "terms: "+strings.Join(hitTerms, ", "))
	}

	if published, ok := parseISO(c.Published); ok {
		age := int(now.Sub(published).Hours() / 24)
		if age >= 0 && age <= cfg.RecentDays {
			score += WRecent180
			reasons = append(reasons, fmt.Sprintf("recent (%dd)", age))
			if age <= 30 {
				score += WRecent30
				reasons = append(reasons, "very fresh (<=30d)")
			}
		}
	}
	stars := intFromExtra(c.Extra, "stars")
	if stars > 0 {
		bonus := stars / StarDivisor
		if bonus > StarCap {
			bonus = StarCap
		}
		if bonus > 0 {
			score += bonus
			reasons = append(reasons, fmt.Sprintf("%d stars (+%d)", stars, bonus))
		}
	}
	points := intFromExtra(c.Extra, "points")
	if points > 0 {
		bonus := points / HNPointDiv
		if bonus > HNPointCap {
			bonus = HNPointCap
		}
		if bonus > 0 {
			score += bonus
			reasons = append(reasons, fmt.Sprintf("%d points (+%d)", points, bonus))
		}
	}
	if pushed, ok := parseISO(stringFromExtra(c.Extra, "pushed_at")); ok {
		days := int(now.Sub(pushed).Hours() / 24)
		window := cfg.FreshWindowDays
		if window <= 0 {
			window = 45
		}
		switch {
		case days >= 0 && days <= window:
			score += WFreshPush
			reasons = append(reasons, fmt.Sprintf("pushed <=%dd (actively updated)", window))
		case days <= 90:
			score += WRecentPush
			reasons = append(reasons, "pushed <=90d")
		}
	}
	// Trending: a young repo already gathering stars (high stars/day) is on the
	// rise; an old repo with the same stars accrued them slowly and scores ~0.
	if stars > 0 {
		if created, ok := parseISO(c.Published); ok {
			ageDays := int(now.Sub(created).Hours() / 24)
			if ageDays < 1 {
				ageDays = 1
			}
			rawVel := stars / ageDays
			if rawVel > 0 {
				bonus := rawVel
				if bonus > TrendingCap {
					bonus = TrendingCap
				}
				score += bonus
				reasons = append(reasons, fmt.Sprintf("trending (%d*/day, +%d)", rawVel, bonus))
			}
		}
	}
	return score, reasons
}

var stampRE = regexp.MustCompile(`idea-scout-source:\s*([^\s>]+)`)

// StampIndex is rung 2's whole payload: every `idea-scout-source:` stamp carried
// by issues, lower-cased. Case is folded on BOTH sides (here and in IsDuplicate)
// because GitHub repo names are case-insensitive while the search API hands back
// whichever casing it feels like — an un-folded compare lets `Acme/Repo` slip
// past a stamp that reads `acme/repo`.
func StampIndex(issues []ExistingIssue) map[string]struct{} {
	out := map[string]struct{}{}
	for _, iss := range issues {
		for _, m := range stampRE.FindAllStringSubmatch(iss.Body, -1) {
			if len(m) > 1 {
				out[strings.ToLower(strings.TrimSpace(m[1]))] = struct{}{}
			}
		}
	}
	return out
}

func ExistingIssueIndex(issues []ExistingIssue) (map[string]struct{}, []map[string]struct{}, string) {
	titleSets := make([]map[string]struct{}, 0, len(issues))
	bodies := make([]string, 0, len(issues))
	for _, iss := range issues {
		bodies = append(bodies, strings.ToLower(iss.Body))
		titleSets = append(titleSets, Tokenize(iss.Title))
	}
	return StampIndex(issues), titleSets, strings.Join(bodies, "\n")
}

// IsDuplicate names the rung that fires ("seen-cache" / "filed-stamp" /
// "issue-body" / "title-near"), or "" if the candidate is genuinely new.
//
// "filed-stamp" and "issue-body" are reported separately on purpose: the first is
// the durable, complete filing history (rung 2) and the second is a best-effort
// URL sighting inside a recency window (rung 3). Collapsing them would make a
// windowed guess indistinguishable from the guarantee in the run report.
func IsDuplicate(c Candidate, seen map[string]SeenRecord, stamped map[string]struct{}, titleSets []map[string]struct{}, bodiesJoined string, dupJaccard float64) string {
	sidLower := strings.ToLower(c.SourceID)
	if _, ok := seen[c.SourceID]; ok {
		return "seen-cache"
	}
	if _, ok := seen[sidLower]; ok {
		return "seen-cache"
	}
	if _, ok := stamped[sidLower]; ok {
		return "filed-stamp"
	}
	if u := strings.ToLower(c.URL); u != "" && strings.Contains(bodiesJoined, u) {
		return "issue-body"
	}
	ctoks := Tokenize(c.Title)
	for _, tset := range titleSets {
		if Jaccard(ctoks, tset) >= dupJaccard {
			return "title-near"
		}
	}
	return ""
}

func RenderIssue(c Candidate, score int, reasons []string, topic Topic, today string) IssuePlan {
	rawTitle := strings.TrimSuffix(strings.TrimSpace(c.Title), ".")
	rawTitle = trimRunes(rawTitle, 100)
	title := "idea-scout: " + rawTitle
	summary := trimRunes(strings.TrimSpace(c.Summary), 700)
	var facts []string
	switch c.Source {
	case "arxiv":
		if authors := stringSliceFromExtra(c.Extra, "authors"); len(authors) > 0 {
			facts = append(facts, "**Authors:** "+strings.Join(authors, ", "))
		}
		if c.Published != "" {
			facts = append(facts, "**Submitted:** "+firstN(c.Published, 10))
		}
	case "hackernews", "reddit":
		if sub := stringFromExtra(c.Extra, "subreddit"); sub != "" {
			facts = append(facts, "**Subreddit:** r/"+sub)
		}
		if points := intFromExtra(c.Extra, "points"); points > 0 {
			facts = append(facts, fmt.Sprintf("**Points:** %d", points))
		}
		if comments := intFromExtra(c.Extra, "num_comments"); comments > 0 {
			facts = append(facts, fmt.Sprintf("**Comments:** %d", comments))
		}
		if disc := stringFromExtra(c.Extra, "discussion"); disc != "" {
			facts = append(facts, "**Discussion:** "+disc)
		}
		if c.Published != "" {
			facts = append(facts, "**Posted:** "+firstN(c.Published, 10))
		}
	default:
		if stars := intFromExtra(c.Extra, "stars"); stars > 0 {
			facts = append(facts, fmt.Sprintf("**Stars:** %d", stars))
		}
		if lang := stringFromExtra(c.Extra, "language"); lang != "" {
			facts = append(facts, "**Language:** "+lang)
		}
		if pushed := stringFromExtra(c.Extra, "pushed_at"); pushed != "" {
			facts = append(facts, "**Last push:** "+firstN(pushed, 10))
		}
	}
	why := strings.Join(reasons, "; ")
	if why == "" {
		why = "matched topic query"
	}
	body := "> Auto-filed by the daily **idea-scout** (`fak idea-scout`, " + today + "). A candidate RELATED idea found on " + sourceLabel(c.Source) + "; **needs human triage** - close as `wontfix`/`duplicate` if it is not worth pursuing.\n\n" +
		"**Source:** " + c.URL + "\n\n"
	if len(facts) > 0 {
		body += strings.Join(facts, "\n") + "\n\n"
	}
	body += fmt.Sprintf("**Why surfaced** (topic `%s`, score %d): %s\n\n", topic.Key, score, why) +
		"### Dispatchability\n" +
		"- dispatchability: `triage_only`\n" +
		"- reason: idea-scout candidates need human scope, lane, witness, and acceptance criteria before they become worker-ready leaves.\n\n"
	if summary != "" {
		body += "**Summary**\n\n" + summary + "\n\n"
	}
	body += "---\n" +
		"_Triage hint: is this a capability fak should adopt, a threat it should defend against, or prior art to cite? If none, close it._\n" +
		"<!-- idea-scout-source: " + c.SourceID + " -->"
	labels := []string{ScoutLabel, TriageLabel, TriageOnlyLabel, "research"}
	if topic.Area != "" {
		labels = append(labels, topic.Area)
	}
	return IssuePlan{Title: title, Body: body, Labels: labels, SourceID: c.SourceID, URL: c.URL, Score: score, Topic: topic.Key}
}

// PlanIssues runs score → dedup → threshold → CAP. It returns the issues to file,
// the per-rung counts, and `dropped`, which names the rung that stopped each
// individual source_id so an auditor can re-run a dry-run and check BY NAME that a
// known-already-triaged source is being caught, and by which rung.
func PlanIssues(candidates []Candidate, topicsByKey map[string]Topic, seen map[string]SeenRecord, stamped map[string]struct{}, titleSets []map[string]struct{}, bodiesJoined string, cfg Config, today string, now time.Time) ([]IssuePlan, map[string]int, []DroppedSource) {
	stats := map[string]int{"seen-cache": 0, "filed-stamp": 0, "issue-body": 0, "title-near": 0, "below-min": 0, "within-run-dup": 0}
	var scored []IssuePlan
	var dropped []DroppedSource
	runSeen := map[string]struct{}{}
	for _, cand := range candidates {
		if _, ok := runSeen[cand.SourceID]; ok {
			stats["within-run-dup"]++
			dropped = append(dropped, DroppedSource{SourceID: cand.SourceID, Rung: "within-run-dup"})
			continue
		}
		runSeen[cand.SourceID] = struct{}{}
		topic, ok := topicsByKey[cand.Topic]
		if !ok {
			topic = Topic{Key: cand.Topic}
		}
		if rung := IsDuplicate(cand, seen, stamped, titleSets, bodiesJoined, cfg.DupJaccard); rung != "" {
			stats[rung]++
			dropped = append(dropped, DroppedSource{SourceID: cand.SourceID, Rung: rung})
			continue
		}
		score, reasons := ScoreCandidate(cand, topic, cfg, now)
		if score < cfg.MinScore {
			stats["below-min"]++
			dropped = append(dropped, DroppedSource{SourceID: cand.SourceID, Rung: "below-min"})
			continue
		}
		scored = append(scored, RenderIssue(cand, score, reasons, topic, today))
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].SourceID < scored[j].SourceID
	})
	sort.Slice(dropped, func(i, j int) bool {
		if dropped[i].Rung != dropped[j].Rung {
			return dropped[i].Rung < dropped[j].Rung
		}
		return dropped[i].SourceID < dropped[j].SourceID
	})
	if cfg.MaxIssues >= 0 && len(scored) > cfg.MaxIssues {
		scored = scored[:cfg.MaxIssues]
	}
	return scored, stats, dropped
}

func ParseArxivAtom(xmlText, topicKey string) []Candidate {
	type author struct {
		Name string `xml:"name"`
	}
	type entry struct {
		ID        string   `xml:"id"`
		Title     string   `xml:"title"`
		Summary   string   `xml:"summary"`
		Published string   `xml:"published"`
		Authors   []author `xml:"author"`
	}
	type feed struct {
		Entries []entry `xml:"entry"`
	}
	var f feed
	if err := xml.Unmarshal([]byte(xmlText), &f); err != nil {
		return nil
	}
	var out []Candidate
	for _, e := range f.Entries {
		rawID := strings.TrimSpace(e.ID)
		if rawID == "" {
			continue
		}
		absID := rawID
		if idx := strings.LastIndex(absID, "/"); idx >= 0 {
			absID = absID[idx+1:]
		}
		absID = regexp.MustCompile(`v\d+$`).ReplaceAllString(absID, "")
		var authors []string
		for _, a := range e.Authors {
			if name := strings.TrimSpace(a.Name); name != "" {
				authors = append(authors, name)
			}
			if len(authors) == 6 {
				break
			}
		}
		out = append(out, Candidate{
			Source:    "arxiv",
			SourceID:  "arxiv:" + absID,
			URL:       "https://arxiv.org/abs/" + absID,
			Title:     squashSpace(e.Title),
			Summary:   squashSpace(e.Summary),
			Published: strings.TrimSpace(e.Published),
			Topic:     topicKey,
			Extra:     map[string]any{"authors": authors},
		})
	}
	return out
}

func ParseGitHubRepos(items []GitHubRepo, topicKey string) []Candidate {
	var out []Candidate
	for _, it := range items {
		full := strmatch.FirstNonBlank(it.FullName, it.Name)
		if full == "" {
			continue
		}
		u := it.URL
		if u == "" {
			u = "https://github.com/" + full
		}
		pushed := strmatch.FirstNonBlank(it.PushedAt, it.UpdatedAt)
		out = append(out, Candidate{
			Source:    "github",
			SourceID:  "github:" + strings.ToLower(full),
			URL:       u,
			Title:     full,
			Summary:   it.Description,
			Published: it.CreatedAt,
			Topic:     topicKey,
			Extra: map[string]any{
				"stars":     it.StargazersCount,
				"pushed_at": pushed,
				"language":  it.Language,
			},
		})
	}
	return out
}

// ParseHackerNewsJSON turns an Algolia HN search response into candidates.
// It is a pure fold over the wire JSON: no network, no clock. Link stories keep
// their outbound URL; text/self posts fall back to the HN item permalink so the
// candidate always resolves to something a triager can open.
func ParseHackerNewsJSON(jsonText, topicKey string) []Candidate {
	var doc struct {
		Hits []struct {
			ObjectID    string `json:"objectID"`
			Title       string `json:"title"`
			StoryTitle  string `json:"story_title"`
			URL         string `json:"url"`
			StoryURL    string `json:"story_url"`
			Author      string `json:"author"`
			Points      int    `json:"points"`
			NumComments int    `json:"num_comments"`
			CreatedAt   string `json:"created_at"`
			StoryText   string `json:"story_text"`
		} `json:"hits"`
	}
	if err := json.Unmarshal([]byte(jsonText), &doc); err != nil {
		return nil
	}
	var out []Candidate
	for _, h := range doc.Hits {
		id := strings.TrimSpace(h.ObjectID)
		if id == "" {
			continue
		}
		title := squashSpace(strmatch.FirstNonBlank(h.Title, h.StoryTitle))
		if title == "" {
			continue
		}
		permalink := "https://news.ycombinator.com/item?id=" + id
		u := strmatch.FirstNonBlank(h.URL, h.StoryURL, permalink)
		out = append(out, Candidate{
			Source:    "hackernews",
			SourceID:  "hn:" + id,
			URL:       u,
			Title:     title,
			Summary:   squashSpace(stripTags(h.StoryText)),
			Published: strings.TrimSpace(h.CreatedAt),
			Topic:     topicKey,
			Extra: map[string]any{
				"points":       h.Points,
				"num_comments": h.NumComments,
				"discussion":   permalink,
				"author":       h.Author,
			},
		})
	}
	return out
}

// ParseRedditJSON turns a Reddit listing/search response into candidates. Like
// the other sources it is a pure fold over the wire JSON. Reddit stamps posts
// with a Unix `created_utc` float rather than an ISO string, so it is converted
// to RFC3339 here (a deterministic transform, no wall clock) to match the shared
// freshness path. Self/text posts carry the permalink in `url`; link posts carry
// the outbound target, and the permalink is always kept as the discussion link.
func ParseRedditJSON(jsonText, topicKey string) []Candidate {
	var doc struct {
		Data struct {
			Children []struct {
				Data struct {
					ID          string  `json:"id"`
					Title       string  `json:"title"`
					URL         string  `json:"url"`
					Permalink   string  `json:"permalink"`
					Score       int     `json:"score"`
					NumComments int     `json:"num_comments"`
					CreatedUTC  float64 `json:"created_utc"`
					Selftext    string  `json:"selftext"`
					Subreddit   string  `json:"subreddit"`
				} `json:"data"`
			} `json:"children"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(jsonText), &doc); err != nil {
		return nil
	}
	var out []Candidate
	for _, ch := range doc.Data.Children {
		h := ch.Data
		id := strings.TrimSpace(h.ID)
		if id == "" {
			continue
		}
		title := squashSpace(h.Title)
		if title == "" {
			continue
		}
		permalink := ""
		if h.Permalink != "" {
			permalink = "https://www.reddit.com" + h.Permalink
		}
		u := strmatch.FirstNonBlank(h.URL, permalink)
		if u == "" {
			u = "https://www.reddit.com/comments/" + id
		}
		published := ""
		if h.CreatedUTC > 0 {
			published = time.Unix(int64(h.CreatedUTC), 0).UTC().Format(time.RFC3339)
		}
		out = append(out, Candidate{
			Source:    "reddit",
			SourceID:  "reddit:" + id,
			URL:       u,
			Title:     title,
			Summary:   squashSpace(stripTags(h.Selftext)),
			Published: published,
			Topic:     topicKey,
			Extra: map[string]any{
				"points":       h.Score,
				"num_comments": h.NumComments,
				"discussion":   strmatch.FirstNonBlank(permalink, u),
				"subreddit":    h.Subreddit,
			},
		})
	}
	return out
}

func LoadConfig(path string) ([]Topic, Config, error) {
	topics := DefaultTopics()
	cfg := DefaultConfig()
	if strings.TrimSpace(path) != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, Config{}, err
		}
		var doc struct {
			Topics     []Topic        `json:"topics"`
			Thresholds map[string]any `json:"thresholds"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			return nil, Config{}, err
		}
		if len(doc.Topics) > 0 {
			topics = doc.Topics
		}
		applyThresholds(&cfg, doc.Thresholds)
	}
	for i, t := range topics {
		if strings.TrimSpace(t.Key) == "" {
			return nil, Config{}, fmt.Errorf("topic[%d] missing non-empty 'key'", i)
		}
		if len(t.Terms) == 0 {
			return nil, Config{}, fmt.Errorf("topic %q missing non-empty 'terms' list", t.Key)
		}
		if strings.TrimSpace(t.Arxiv) == "" && strings.TrimSpace(t.GitHub) == "" && strings.TrimSpace(t.HN) == "" && strings.TrimSpace(t.Reddit) == "" {
			return nil, Config{}, fmt.Errorf("topic %q must set at least one of 'arxiv', 'github', 'hn', 'reddit'", t.Key)
		}
	}
	return topics, cfg, nil
}

func CachePath(workspace string) string {
	return filepath.Join(workspace, CacheDirname, CacheFilename)
}

func LoadSeen(workspace string) (map[string]SeenRecord, error) {
	p := CachePath(workspace)
	raw, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]SeenRecord{}, nil
	}
	if err != nil {
		return map[string]SeenRecord{}, err
	}
	var wrapped struct {
		Schema string                `json:"schema"`
		Seen   map[string]SeenRecord `json:"seen"`
	}
	if err := json.Unmarshal(raw, &wrapped); err == nil && wrapped.Seen != nil {
		return wrapped.Seen, nil
	}
	var flat map[string]SeenRecord
	if err := json.Unmarshal(raw, &flat); err != nil {
		return map[string]SeenRecord{}, nil
	}
	return flat, nil
}

func SaveSeen(workspace string, seen map[string]SeenRecord) error {
	p := CachePath(workspace)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(struct {
		Schema string                `json:"schema"`
		Seen   map[string]SeenRecord `json:"seen"`
	}{Schema: Schema, Seen: seen}, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(p, raw, 0o644)
}

func Run(opts RunOptions) (RunResult, error) {
	workspace := opts.Workspace
	if workspace == "" {
		workspace = "."
	}
	topics, cfg, err := LoadConfig(opts.ConfigPath)
	if err != nil {
		return RunResult{}, fmt.Errorf("config error: %w", err)
	}
	if opts.MaxIssues != nil {
		cfg.MaxIssues = *opts.MaxIssues
	}
	if opts.MinScore != nil {
		cfg.MinScore = *opts.MinScore
	}
	if opts.Milestone != nil {
		cfg.Milestone = *opts.Milestone
	}
	if opts.Project != nil {
		cfg.Project = *opts.Project
	}
	if opts.ProjectOwner != nil {
		cfg.ProjectOwner = *opts.ProjectOwner
	}
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	today := opts.Today
	if today == "" {
		today = now.Format("2006-01-02")
	}
	topicsByKey := map[string]Topic{}
	for _, t := range topics {
		topicsByKey[t.Key] = t
	}

	var errorsOut []string
	candidates := opts.Candidates
	issues := opts.Existing
	if !opts.UseFixtures {
		fetcher := opts.Fetcher
		if fetcher == nil {
			fetcher = LiveFetcher{}
		}
		candidates = GatherCandidates(fetcher, topics, cfg, &errorsOut)
		if len(candidates) == 0 && len(errorsOut) > 0 {
			return RunResult{}, fmt.Errorf("refuse: every source failed: %s", strings.Join(errorsOut, "; "))
		}
		seen, loadErr := LoadSeen(workspace)
		if loadErr != nil {
			errorsOut = append(errorsOut, "seen-cache: "+loadErr.Error())
			seen = map[string]SeenRecord{}
		}

		// ---- Rung 2, the durable one -------------------------------------------
		// The scout's OWN filing history, pulled by label so the query is targeted at
		// exactly the population being deduped. This is what makes "filed once, never
		// filed again" true without trusting the git-ignored local cache. It is
		// MANDATORY: a partial index is indistinguishable from "this source is new"
		// and re-files an already-triaged source, so it cannot be waived by any
		// option or covered by a populated seen-cache.
		scoutLimit := cfg.ScoutScanLimit
		if scoutLimit <= 0 {
			scoutLimit = DefaultConfig().ScoutScanLimit
		}
		scoutIssues, scoutErr := fetcher.FetchScoutIssues(scoutLimit)
		if scoutErr != nil {
			return RunResult{}, fmt.Errorf("refuse: cannot build the filed-issue index (gh issue list --label %s); filing now could re-file an already-triaged source (%w)", ScoutLabel, scoutErr)
		}
		if len(scoutIssues) >= scoutLimit {
			// Saturation is ambiguous — gh returns exactly `limit` both when that is
			// all there is and when it truncated. Refuse loudly rather than let the
			// guarantee rot silently the way the 800-issue window did.
			return RunResult{}, fmt.Errorf("refuse: the filed-issue index came back saturated at scout_scan_limit=%d, so it may be truncated and a previously-filed source could be re-filed; raise thresholds.scout_scan_limit (--config) above the number of issues the scout has ever filed", scoutLimit)
		}

		// ---- Rungs 3/4: the soft, windowed corpus of everything else -------------
		// Human-opened issues referencing the same URL, or carrying a near-identical
		// title. Nice to have, never the guarantee. The pre-existing refusal is kept
		// exactly as it was — nothing here is relaxed on the back of rung 2, because
		// degrading these rungs onto a bare local cache is still a worse run than no
		// run.
		issues, err = fetcher.FetchExistingIssues(cfg.IssueScanLimit)
		if err != nil {
			errorsOut = append(errorsOut, "issues: "+err.Error())
			if len(seen) == 0 {
				return RunResult{}, fmt.Errorf("refuse: cannot fetch existing issues and no seen-cache to fall back on (%w)", err)
			}
			issues = nil
		}
		return finishRun(finishInput{Workspace: workspace, Topics: topicsByKey, Config: cfg, Today: today, Now: now, Candidates: candidates, Issues: issues, ScoutIssues: scoutIssues, ScoutScanLimit: scoutLimit, Errors: errorsOut, Live: opts.Live, Fetcher: fetcher, Seen: seen})
	}
	seen, err := LoadSeen(workspace)
	if err != nil {
		errorsOut = append(errorsOut, "seen-cache: "+err.Error())
		seen = map[string]SeenRecord{}
	}
	// Fixture replay: no network, so nothing can have been truncated. When the
	// caller did not separate the two corpora, the supplied issues stand in for
	// both — the same stamps still gate the durable rung.
	scoutIssues := opts.ScoutIssues
	if scoutIssues == nil {
		scoutIssues = issues
	}
	return finishRun(finishInput{Workspace: workspace, Topics: topicsByKey, Config: cfg, Today: today, Now: now, Candidates: candidates, Issues: issues, ScoutIssues: scoutIssues, ScoutScanLimit: cfg.ScoutScanLimit, Errors: errorsOut, Live: opts.Live, Fetcher: opts.Fetcher, Seen: seen})
}

type finishInput struct {
	Workspace      string
	Topics         map[string]Topic
	Config         Config
	Today          string
	Now            time.Time
	Candidates     []Candidate
	Issues         []ExistingIssue
	ScoutIssues    []ExistingIssue
	ScoutScanLimit int
	Errors         []string
	Live           bool
	Fetcher        Fetcher
	Seen           map[string]SeenRecord
}

func finishRun(in finishInput) (RunResult, error) {
	stamped := StampIndex(in.ScoutIssues)
	winStamped, titleSets, bodiesJoined := ExistingIssueIndex(in.Issues)
	// Union, not replacement: a filed issue whose label a human stripped is still
	// ours, and its stamp is still proof we filed that source.
	for sid := range winStamped {
		stamped[sid] = struct{}{}
	}
	toFile, skipStats, dropped := PlanIssues(in.Candidates, in.Topics, in.Seen, stamped, titleSets, bodiesJoined, in.Config, in.Today, in.Now)
	var filed []FiledIssue
	if in.Live && len(toFile) > 0 {
		if in.Fetcher == nil {
			in.Fetcher = LiveFetcher{}
		}
		if err := in.Fetcher.EnsureLabels(); err != nil {
			in.Errors = append(in.Errors, "labels: "+err.Error())
		}
		for _, issue := range toFile {
			u, err := in.Fetcher.CreateIssue(issue, in.Config.Milestone)
			if err != nil {
				in.Errors = append(in.Errors, "create["+issue.SourceID+"]: "+err.Error())
				continue
			}
			if in.Config.Project != "" {
				if err := in.Fetcher.AddToProject(u, in.Config.Project, in.Config.ProjectOwner); err != nil {
					in.Errors = append(in.Errors, "project["+issue.SourceID+"]: "+err.Error())
				}
			}
			in.Seen[issue.SourceID] = SeenRecord{FiledAt: in.Today, IssueURL: u, Score: issue.Score, Topic: issue.Topic}
			filed = append(filed, FiledIssue{Title: issue.Title, IssueURL: u})
		}
		if len(filed) > 0 {
			if err := SaveSeen(in.Workspace, in.Seen); err != nil {
				return RunResult{}, err
			}
		}
	}
	return RunResult{
		Schema:             Schema,
		Date:               in.Today,
		Mode:               mode(in.Live),
		CandidatesGathered: len(in.Candidates),
		DedupIndex: DedupIndex{
			FiledIssuesScanned:  len(in.ScoutIssues),
			FiledStamps:         len(stamped),
			ScoutScanLimit:      in.ScoutScanLimit,
			ScoutIndexComplete:  true,
			WindowIssuesScanned: len(in.Issues),
			IssueScanLimit:      in.Config.IssueScanLimit,
		},
		Skipped:      skipStats,
		Dropped:      dropped,
		Planned:      publicPlans(toFile),
		Filed:        filed,
		Errors:       in.Errors,
		SourceDigest: candidateDigest(in.Candidates),
		Topics:       len(in.Topics),
		Thresholds: map[string]any{
			"max_issues":  in.Config.MaxIssues,
			"min_score":   in.Config.MinScore,
			"dup_jaccard": in.Config.DupJaccard,
		},
	}, nil
}

func GatherCandidates(fetcher Fetcher, topics []Topic, cfg Config, errorsOut *[]string) []Candidate {
	var cands []Candidate
	for _, topic := range topics {
		if topic.Arxiv != "" {
			xmlText, err := fetcher.FetchArxiv(topic.Arxiv, cfg.ArxivPerTopic)
			if err != nil {
				*errorsOut = append(*errorsOut, "arxiv["+topic.Key+"]: "+err.Error())
			} else {
				cands = append(cands, ParseArxivAtom(xmlText, topic.Key)...)
			}
		}
		if topic.GitHub != "" {
			cands = appendStarLane(cands, errorsOut, "github", topic.Key, cfg.MinStars,
				func() ([]GitHubRepo, error) { return fetcher.FetchGitHub(topic.GitHub, cfg.GitHubPerTopic) },
				nil)
		}
		if topic.GitHub != "" && cfg.FreshPerTopic > 0 {
			// The fresh lane: same topic query, sorted by most-recently-updated,
			// with a low star floor so young/trending repos the MinStars floor
			// would drop enter the pool. Recency itself is rewarded in scoring
			// (which has a clock); here we only admit and tag provenance.
			cands = appendStarLane(cands, errorsOut, "github-fresh", topic.Key, cfg.FreshMinStars,
				func() ([]GitHubRepo, error) { return fetcher.FetchGitHubFresh(topic.GitHub, cfg.FreshPerTopic) },
				func(c *Candidate) {
					if c.Extra == nil {
						c.Extra = map[string]any{}
					}
					c.Extra["lane"] = "fresh"
				})
		}
		if topic.HN != "" {
			cands = appendPointsLane(cands, errorsOut, "hn", topic.Key, cfg.MinPoints,
				func() (string, error) { return fetcher.FetchHackerNews(topic.HN, cfg.HNPerTopic) },
				ParseHackerNewsJSON)
		}
		if topic.Reddit != "" {
			cands = appendPointsLane(cands, errorsOut, "reddit", topic.Key, cfg.MinPoints,
				func() (string, error) { return fetcher.FetchReddit(topic.Reddit, cfg.RedditPerTopic) },
				ParseRedditJSON)
		}
	}
	return cands
}

// appendStarLane runs one GitHub lane end to end: fetch, record a `label[topic]: …`
// error and admit nothing on failure, else keep the repos at or above this lane's own
// star floor and parse them into candidates. `tag` post-processes each admitted
// candidate and may be nil — it is how the fresh lane stamps its provenance while the
// main lane stamps nothing.
func appendStarLane(cands []Candidate, errorsOut *[]string, label, topicKey string, minStars int,
	fetch func() ([]GitHubRepo, error), tag func(*Candidate)) []Candidate {
	items, err := fetch()
	if err != nil {
		*errorsOut = append(*errorsOut, label+"["+topicKey+"]: "+err.Error())
		return cands
	}
	filtered := items[:0]
	for _, item := range items {
		if item.StargazersCount >= minStars {
			filtered = append(filtered, item)
		}
	}
	parsed := ParseGitHubRepos(filtered, topicKey)
	if tag != nil {
		for i := range parsed {
			tag(&parsed[i])
		}
	}
	return append(cands, parsed...)
}

// appendPointsLane runs one points-scored social lane (Hacker News, Reddit): fetch,
// record a `label[topic]: …` error and admit nothing on failure, else admit the parsed
// candidates that clear the shared point floor. The lanes differ only in which fetch
// and which parser they name.
func appendPointsLane(cands []Candidate, errorsOut *[]string, label, topicKey string, minPoints int,
	fetch func() (string, error), parse func(jsonText, topicKey string) []Candidate) []Candidate {
	jsonText, err := fetch()
	if err != nil {
		*errorsOut = append(*errorsOut, label+"["+topicKey+"]: "+err.Error())
		return cands
	}
	for _, c := range parse(jsonText, topicKey) {
		if intFromExtra(c.Extra, "points") >= minPoints {
			cands = append(cands, c)
		}
	}
	return cands
}

type LiveFetcher struct {
	HTTPClient *http.Client
}

func (f LiveFetcher) FetchArxiv(query string, maxResults int) (string, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	q := url.Values{}
	q.Set("search_query", query)
	q.Set("sortBy", "submittedDate")
	q.Set("sortOrder", "descending")
	q.Set("max_results", strconv.Itoa(maxResults))
	req, err := http.NewRequest("GET", ArxivAPI+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "fak-idea-scout/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("arxiv status %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (f LiveFetcher) FetchGitHub(query string, limit int) ([]GitHubRepo, error) {
	var out []GitHubRepo
	err := ghJSON([]string{"search", "repos", query, "--limit", strconv.Itoa(limit), "--sort", "stars", "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language"}, 60*time.Second, &out)
	return out, err
}

// FetchGitHubFresh is the recency-first companion to FetchGitHub: the SAME topic
// query (so the neighborhood stays "relative to ours") sorted by most-recently
// updated instead of all-time stars, so newly-created / trending / freshly-pushed
// repos surface where the stars sort would bury them under incumbents.
func (f LiveFetcher) FetchGitHubFresh(query string, limit int) ([]GitHubRepo, error) {
	var out []GitHubRepo
	err := ghJSON([]string{"search", "repos", query, "--limit", strconv.Itoa(limit), "--sort", "updated", "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language"}, 60*time.Second, &out)
	return out, err
}

func (f LiveFetcher) FetchHackerNews(query string, limit int) (string, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	q := url.Values{}
	q.Set("query", query)
	q.Set("tags", "story")
	q.Set("hitsPerPage", strconv.Itoa(limit))
	req, err := http.NewRequest("GET", HNAlgoliaAPI+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "fak-idea-scout/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("hn status %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (f LiveFetcher) FetchReddit(query string, limit int) (string, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	q := url.Values{}
	q.Set("q", query)
	q.Set("sort", "new")
	q.Set("t", "week")
	q.Set("limit", strconv.Itoa(limit))
	req, err := http.NewRequest("GET", RedditSearchAPI+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	// Reddit rejects requests without a descriptive, non-default User-Agent.
	req.Header.Set("User-Agent", "fak-idea-scout/1.0 (+https://github.com/anthony-chaudhary/fak)")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("reddit status %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// FetchExistingIssues is the rung 3/4 corpus: the `limit` most recent issues,
// whoever opened them. A RECENCY WINDOW — it answers "did a human already write
// this up lately", and that is all it is allowed to answer.
func (f LiveFetcher) FetchExistingIssues(limit int) ([]ExistingIssue, error) {
	var out []ExistingIssue
	err := ghJSON([]string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(limit), "--json", "number,title,body"}, 60*time.Second, &out)
	return out, err
}

// FetchScoutIssues is the rung 2 corpus: every issue the scout has EVER filed,
// open or closed.
//
// TARGETED, not windowed. `--label idea-scout` is a server-side filter, so the
// result set is the scout's own filing history — it does not thin out because
// unrelated issues were opened this week, which is precisely how the recency
// window in FetchExistingIssues lost the guarantee. Every filed issue carries the
// label (RenderIssue always emits ScoutLabel) and the matching
// `<!-- idea-scout-source: … -->` stamp, so label ⊇ stamped-by-us.
//
// `--state all` is load-bearing: a source whose issue was triaged and CLOSED is
// the exact case that must not come back. The longer deadline is deliberate — a
// years-deep index is a bigger page walk than the 60s window fetch, and a timeout
// here refuses the whole run.
func (f LiveFetcher) FetchScoutIssues(limit int) ([]ExistingIssue, error) {
	var out []ExistingIssue
	err := ghJSON([]string{"issue", "list", "--state", "all", "--label", ScoutLabel, "--limit", strconv.Itoa(limit), "--json", "number,title,body"}, 180*time.Second, &out)
	return out, err
}

func (f LiveFetcher) EnsureLabels() error {
	wanted := []struct {
		name  string
		color string
		desc  string
	}{
		{ScoutLabel, "8a63d2", "Auto-filed by the daily idea-scout; needs human triage"},
		{TriageLabel, "d4c5f9", "Needs human scoping before an agent dispatch can take it"},
		{TriageOnlyLabel, "d4c5f9", "Useful issue, but not a worker-ready dispatch leaf"},
	}
	var errs []string
	for _, w := range wanted {
		if _, _, err := runGH([]string{"label", "create", w.name, "--color", w.color, "--description", w.desc, "--force"}, 30*time.Second); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func (f LiveFetcher) CreateIssue(issue IssuePlan, milestone string) (string, error) {
	args := []string{"issue", "create", "--title", issue.Title, "--body", issue.Body}
	for _, lab := range issue.Labels {
		args = append(args, "--label", lab)
	}
	if milestone != "" {
		args = append(args, "--milestone", milestone)
	}
	stdout, _, err := runGH(args, 60*time.Second)
	if err != nil {
		return "", err
	}
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) == 0 {
		return "", nil
	}
	return strings.TrimSpace(lines[len(lines)-1]), nil
}

func (f LiveFetcher) AddToProject(issueURL, number, owner string) error {
	args := []string{"project", "item-add", number, "--url", issueURL}
	if owner != "" {
		args = append(args, "--owner", owner)
	}
	_, _, err := runGH(args, 60*time.Second)
	return err
}

func ghJSON(args []string, timeout time.Duration, out any) error {
	stdout, _, err := runGH(args, timeout)
	if err != nil {
		return err
	}
	if strings.TrimSpace(stdout) == "" {
		return nil
	}
	return json.Unmarshal([]byte(stdout), out)
}

func runGH(args []string, timeout time.Duration) (string, string, error) {
	cmd, cancel := ghexec.CommandTimeout(context.Background(), timeout, args...)
	defer cancel()
	// WaitDelay is the straggler backstop: if the deadline kill leaves a grandchild
	// holding the output pipe open, cmd.Run could still block past the timeout;
	// WaitDelay forces the pipes closed so the deadline is real (issue #3483).
	cmd.WaitDelay = 10 * time.Second
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		return stdout.String(), stderr.String(), fmt.Errorf("gh %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), stderr.String(), nil
}

func ReadCandidates(path string) ([]Candidate, error) {
	var out []Candidate
	if err := readJSONFile(path, &out); err == nil {
		return out, nil
	}
	var wrapped struct {
		Candidates []Candidate `json:"candidates"`
	}
	if err := readJSONFile(path, &wrapped); err != nil {
		return nil, err
	}
	return wrapped.Candidates, nil
}

func ReadExistingIssues(path string) ([]ExistingIssue, error) {
	var out []ExistingIssue
	if err := readJSONFile(path, &out); err == nil {
		return out, nil
	}
	var wrapped struct {
		Issues []ExistingIssue `json:"issues"`
	}
	if err := readJSONFile(path, &wrapped); err != nil {
		return nil, err
	}
	return wrapped.Issues, nil
}

func readJSONFile(path string, out any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}

func RenderHuman(w io.Writer, result RunResult, cfg Config) {
	fmt.Fprintf(w, "idea-scout %s - %s\n", result.Date, result.Mode)
	fmt.Fprintf(w, "  gathered %d candidates from %d topics x (arXiv + GitHub + Hacker News + Reddit)\n", result.CandidatesGathered, result.Topics)
	fmt.Fprintf(w, "  dedup index: %d source stamps from %d filed issue(s) (label-targeted, complete) + %d recent issue(s) for the near-dup rungs\n",
		result.DedupIndex.FiledStamps, result.DedupIndex.FiledIssuesScanned, result.DedupIndex.WindowIssuesScanned)
	var parts []string
	keys := make([]string, 0, len(result.Skipped))
	for k := range result.Skipped {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if result.Skipped[k] != 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", k, result.Skipped[k]))
		}
	}
	if len(parts) == 0 {
		parts = append(parts, "none")
	}
	fmt.Fprintf(w, "  deduped/dropped: %s\n", strings.Join(parts, ", "))
	if len(result.Planned) == 0 {
		fmt.Fprintln(w, "  -> nothing new worth filing today.")
	} else {
		verb := "would file"
		if result.Mode == "live" {
			verb = "FILED"
		}
		fmt.Fprintf(w, "  -> %s %d issue(s) (cap %d, min-score %d):\n", verb, len(result.Planned), cfg.MaxIssues, cfg.MinScore)
		for _, issue := range result.Planned {
			fmt.Fprintf(w, "     [%3d] %s\n", issue.Score, issue.Title)
			fmt.Fprintf(w, "           %s  labels=%s\n", issue.URL, strings.Join(issue.Labels, ","))
		}
	}
	if len(result.Errors) > 0 {
		fmt.Fprintln(w, "  errors:")
		for _, e := range result.Errors {
			fmt.Fprintf(w, "     ! %s\n", e)
		}
	}
	if result.Mode == "dry-run" && len(result.Planned) > 0 {
		fmt.Fprintln(w, "\n  dry-run - file these for real with: fak idea-scout --live")
	}
}

func ResultConfig(path string, maxIssues, minScore *int, milestone, project, projectOwner *string) (Config, error) {
	_, cfg, err := LoadConfig(path)
	if err != nil {
		return Config{}, err
	}
	if maxIssues != nil {
		cfg.MaxIssues = *maxIssues
	}
	if minScore != nil {
		cfg.MinScore = *minScore
	}
	if milestone != nil {
		cfg.Milestone = *milestone
	}
	if project != nil {
		cfg.Project = *project
	}
	if projectOwner != nil {
		cfg.ProjectOwner = *projectOwner
	}
	return cfg, nil
}

func mode(live bool) string {
	if live {
		return "live"
	}
	return "dry-run"
}

func publicPlans(in []IssuePlan) []IssuePlan {
	out := make([]IssuePlan, len(in))
	for i, p := range in {
		p.Body = ""
		out[i] = p
	}
	return out
}

func candidateDigest(cands []Candidate) string {
	if len(cands) == 0 {
		return ""
	}
	raw, _ := json.Marshal(cands)
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func parseISO(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, strings.ReplaceAll(s, "Z", "+00:00"))
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}

func intFromExtra(extra map[string]any, key string) int {
	if extra == nil {
		return 0
	}
	switch v := extra[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case json.Number:
		n, _ := v.Int64()
		return int(n)
	case string:
		n, _ := strconv.Atoi(v)
		return n
	default:
		return 0
	}
}

func stringFromExtra(extra map[string]any, key string) string {
	if extra == nil {
		return ""
	}
	if s, ok := extra[key].(string); ok {
		return s
	}
	return ""
}

func stringSliceFromExtra(extra map[string]any, key string) []string {
	if extra == nil {
		return nil
	}
	switch v := extra[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, x := range v {
			if s, ok := x.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func trimRunes(s string, max int) string {
	rs := []rune(s)
	if len(rs) <= max {
		return s
	}
	if max <= 3 {
		return string(rs[:max])
	}
	return strings.TrimSpace(string(rs[:max-3])) + "..."
}

func firstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func squashSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

var htmlTagRE = regexp.MustCompile(`<[^>]*>`)

// stripTags removes the light HTML (<p>, <a>, <i>) the HN API leaves in
// story_text so the summary is plain prose.
func stripTags(s string) string {
	return htmlTagRE.ReplaceAllString(s, " ")
}

func sourceLabel(source string) string {
	switch source {
	case "arxiv":
		return "arXiv"
	case "github":
		return "GitHub"
	case "hackernews":
		return "Hacker News"
	case "reddit":
		return "Reddit"
	default:
		return source
	}
}

func applyThresholds(cfg *Config, values map[string]any) {
	for k, v := range values {
		switch k {
		case "recent_days":
			cfg.RecentDays = anyInt(v, cfg.RecentDays)
		case "min_score":
			cfg.MinScore = anyInt(v, cfg.MinScore)
		case "max_issues":
			cfg.MaxIssues = anyInt(v, cfg.MaxIssues)
		case "arxiv_per_topic":
			cfg.ArxivPerTopic = anyInt(v, cfg.ArxivPerTopic)
		case "github_per_topic":
			cfg.GitHubPerTopic = anyInt(v, cfg.GitHubPerTopic)
		case "hn_per_topic":
			cfg.HNPerTopic = anyInt(v, cfg.HNPerTopic)
		case "reddit_per_topic":
			cfg.RedditPerTopic = anyInt(v, cfg.RedditPerTopic)
		case "min_stars":
			cfg.MinStars = anyInt(v, cfg.MinStars)
		case "fresh_per_topic":
			cfg.FreshPerTopic = anyInt(v, cfg.FreshPerTopic)
		case "fresh_min_stars":
			cfg.FreshMinStars = anyInt(v, cfg.FreshMinStars)
		case "fresh_window_days":
			cfg.FreshWindowDays = anyInt(v, cfg.FreshWindowDays)
		case "min_points":
			cfg.MinPoints = anyInt(v, cfg.MinPoints)
		case "dup_jaccard":
			cfg.DupJaccard = anyFloat(v, cfg.DupJaccard)
		case "issue_scan_limit":
			cfg.IssueScanLimit = anyInt(v, cfg.IssueScanLimit)
		case "scout_scan_limit":
			cfg.ScoutScanLimit = anyInt(v, cfg.ScoutScanLimit)
		case "milestone":
			cfg.Milestone, _ = v.(string)
		case "project":
			cfg.Project, _ = v.(string)
		case "project_owner":
			cfg.ProjectOwner, _ = v.(string)
		}
	}
}

func anyInt(v any, fallback int) int {
	switch x := v.(type) {
	case int:
		return x
	case float64:
		return int(x)
	case json.Number:
		n, err := x.Int64()
		if err == nil {
			return int(n)
		}
	case string:
		n, err := strconv.Atoi(x)
		if err == nil {
			return n
		}
	}
	return fallback
}

func anyFloat(v any, fallback float64) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case json.Number:
		f, err := x.Float64()
		if err == nil {
			return f
		}
	case string:
		f, err := strconv.ParseFloat(x, 64)
		if err == nil {
			return f
		}
	}
	return fallback
}
