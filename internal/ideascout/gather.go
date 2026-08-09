package ideascout

// Candidate gathering: walk every topic across every enabled source lane,
// applying each lane's own admission floor and recording per-lane fetch errors
// instead of failing the run.

import "strings"

// sourceLane declares one gathering lane: the label it stamps on per-lane fetch
// errors, the topic-config key that arms it, and the human name the run report
// prints. Two lanes may share a topic key — `github` and `github-fresh` run the
// same query on different sorts.
type sourceLane struct {
	label    string
	topicKey string
	display  string
}

// sourceLanes is the scout's SOURCE VOCABULARY, written down once. Three things
// read it, so a lane cannot be half-added: LoadConfig refuses a topic key that is
// not here (#5549 — a key no lane reads used to gather zero and report success),
// RenderHuman names the lanes it walked, and
// testdata/source_corpus.json pins this list against tools/idea_scout.py's
// SOURCE_LANES so a lane that grows on one implementation and not the other reds
// a test instead of silently gathering nothing on the scheduled path.
//
// GatherCandidates below still spells each lane out longhand — the lanes are not
// uniform enough to fold into a table without obscuring them — but a lane that
// appears there and not here is unreachable (its topic key would be refused at
// config load), and a lane here with no body is caught by the corpus's
// per-key gather cases.
var sourceLanes = []sourceLane{
	{label: "arxiv", topicKey: "arxiv", display: "arXiv"},
	{label: "github", topicKey: "github", display: "GitHub"},
	{label: "github-fresh", topicKey: "github", display: "GitHub"},
	{label: "hn", topicKey: "hn", display: "Hacker News"},
	{label: "reddit", topicKey: "reddit", display: "Reddit"},
}

// topicMetaKeys are the topic-config keys that name no source lane: the topic's
// own identity, its relevance terms, and the area label filed issues hang under.
var topicMetaKeys = []string{"key", "terms", "area"}

// sourceTopicKeys is the set of topic-config keys that arm a lane, in declaration
// order and de-duplicated (`github` arms two lanes).
func sourceTopicKeys() []string {
	var out []string
	seen := map[string]bool{}
	for _, lane := range sourceLanes {
		if lane.topicKey == "" || seen[lane.topicKey] {
			continue
		}
		seen[lane.topicKey] = true
		out = append(out, lane.topicKey)
	}
	return out
}

// sourceDisplayList renders the lane vocabulary the way the run report says it:
// "arXiv + GitHub + Hacker News + Reddit". Derived, not spelled out, so the
// report cannot claim a lane the gatherer does not walk.
func sourceDisplayList() string {
	var out []string
	seen := map[string]bool{}
	for _, lane := range sourceLanes {
		if seen[lane.display] {
			continue
		}
		seen[lane.display] = true
		out = append(out, lane.display)
	}
	return strings.Join(out, " + ")
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
			cands = appendStarLane(cands, errorsOut, "github", topic.Key, cfg.MinStars, cfg.MinRepoSizeKB,
				func() ([]GitHubRepo, error) { return fetcher.FetchGitHub(topic.GitHub, cfg.GitHubPerTopic) },
				nil)
		}
		if topic.GitHub != "" && cfg.FreshPerTopic > 0 {
			// The fresh lane: same topic query, sorted by most-recently-updated,
			// with a low star floor so young/trending repos the MinStars floor
			// would drop enter the pool. Recency itself is rewarded in scoring
			// (which has a clock); here we only admit and tag provenance.
			cands = appendStarLane(cands, errorsOut, "github-fresh", topic.Key, cfg.FreshMinStars, cfg.MinRepoSizeKB,
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
func appendStarLane(cands []Candidate, errorsOut *[]string, label, topicKey string, minStars, minRepoSizeKB int,
	fetch func() ([]GitHubRepo, error), tag func(*Candidate)) []Candidate {
	items, err := fetch()
	if err != nil {
		*errorsOut = append(*errorsOut, label+"["+topicKey+"]: "+err.Error())
		return cands
	}
	filtered := items[:0]
	for _, item := range items {
		if item.Size >= minRepoSizeKB && item.StargazersCount >= minStars {
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
