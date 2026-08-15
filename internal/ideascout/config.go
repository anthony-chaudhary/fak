package ideascout

// Configuration: the built-in defaults, the on-disk topic/threshold document, and
// the permissive coercion that lets a threshold arrive as int, float, or string.

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

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
		MinRepoSizeKB:   500,
		FreshPerTopic:   6,
		FreshMinStars:   3,
		FreshWindowDays: 45,
		MinPoints:       10,
		DupJaccard:      0.55,
		IssueScanLimit:  800,
		ScoutScanLimit:  5000,
		UntriagedCap:    DefaultUntriagedCap,
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

func LoadConfig(path string) ([]Topic, Config, error) {
	topics := DefaultTopics()
	cfg := DefaultConfig()
	if strings.TrimSpace(path) != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, Config{}, err
		}
		// Topics are decoded twice on purpose: once as raw objects so an
		// unrecognised key can be REFUSED by name, then as Topic. encoding/json
		// drops unknown fields silently, which is exactly the failure #5549
		// records — a topic naming a lane the implementation does not serve
		// gathered zero candidates and reported a normal success.
		var doc struct {
			Topics     []json.RawMessage `json:"topics"`
			Thresholds map[string]any    `json:"thresholds"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			return nil, Config{}, err
		}
		if err := checkThresholdKeys(doc.Thresholds); err != nil {
			return nil, Config{}, err
		}
		if len(doc.Topics) > 0 {
			decoded := make([]Topic, 0, len(doc.Topics))
			for i, rawTopic := range doc.Topics {
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(rawTopic, &fields); err != nil {
					return nil, Config{}, fmt.Errorf("topic[%d] is not a JSON object: %v", i, err)
				}
				var t Topic
				if err := json.Unmarshal(rawTopic, &t); err != nil {
					return nil, Config{}, err
				}
				if err := checkTopicKeys(i, t.Key, fields); err != nil {
					return nil, Config{}, err
				}
				decoded = append(decoded, t)
			}
			topics = decoded
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
		if !topicNamesASource(t) {
			return nil, Config{}, fmt.Errorf("topic %q must set at least one source: %s",
				t.Key, quotedList(sourceTopicKeys()))
		}
	}
	return topics, cfg, nil
}

func topicNamesASource(t Topic) bool {
	for _, key := range sourceTopicKeys() {
		if strings.TrimSpace(topicQuery(t, key)) != "" {
			return true
		}
	}
	return false
}

// topicQuery is the ONE place a topic-config key is turned into the query string
// it holds. Both readers go through it — the "does this topic arm any lane at all"
// check above and the per-lane attempt count the source-health table needs — so a
// new source key cannot be served by one and silently ignored by the other.
func topicQuery(t Topic, key string) string {
	switch key {
	case "arxiv":
		return t.Arxiv
	case "github":
		return t.GitHub
	case "hn":
		return t.HN
	case "reddit":
		return t.Reddit
	}
	return ""
}

// checkTopicKeys refuses a topic key no lane reads. A key that is merely ignored
// is the #5549 defect: the run gathers nothing from it and still exits 0, so the
// operator learns the lane is missing only by reading both implementations side
// by side. Refusing by name is the loud alternative.
func checkTopicKeys(index int, key string, fields map[string]json.RawMessage) error {
	known := map[string]bool{}
	for _, k := range topicMetaKeys {
		known[k] = true
	}
	for _, k := range sourceTopicKeys() {
		known[k] = true
	}
	var unknown []string
	for k := range fields {
		if !known[k] {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	name := key
	if strings.TrimSpace(name) == "" {
		name = fmt.Sprintf("[%d]", index)
	}
	return fmt.Errorf("topic %q names unknown key(s) %s: no source lane reads them, so they would gather nothing silently. Known source keys: %s; other topic keys: %s",
		name, quotedList(unknown), quotedList(sourceTopicKeys()), quotedList(topicMetaKeys))
}

// checkThresholdKeys refuses a threshold no knob reads, for the same reason
// checkTopicKeys refuses an unknown topic key: `min_points` set against an
// implementation that has no points floor is a setting that appears to take and
// does not.
func checkThresholdKeys(values map[string]any) error {
	known := map[string]bool{}
	for _, k := range thresholdKeys() {
		known[k] = true
	}
	var unknown []string
	for k := range values {
		if !known[k] {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("thresholds name unknown key(s) %s: no knob reads them. Known thresholds: %s",
		quotedList(unknown), quotedList(thresholdKeys()))
}

// thresholdKeys is derived from Config's own JSON tags rather than written out,
// so a knob added to the struct is admissible the moment it exists and the
// vocabulary cannot drift from the type it describes.
func thresholdKeys() []string {
	var out []string
	rt := reflect.TypeOf(Config{})
	for i := 0; i < rt.NumField(); i++ {
		tag := rt.Field(i).Tag.Get("json")
		if tag == "" || tag == "-" {
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if name != "" {
			out = append(out, name)
		}
	}
	return out
}

func quotedList(items []string) string {
	quoted := make([]string, 0, len(items))
	for _, it := range items {
		quoted = append(quoted, strconv.Quote(it))
	}
	return strings.Join(quoted, ", ")
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
		case "min_repo_size_kb":
			cfg.MinRepoSizeKB = anyInt(v, cfg.MinRepoSizeKB)
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
		case "untriaged_cap":
			cfg.UntriagedCap = anyInt(v, cfg.UntriagedCap)
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
