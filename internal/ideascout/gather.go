package ideascout

// Candidate gathering: walk every topic across every enabled source lane,
// applying each lane's own admission floor and recording per-lane fetch errors
// instead of failing the run.

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
