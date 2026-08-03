package ideascout

// Configuration: the built-in defaults, the on-disk topic/threshold document, and
// the permissive coercion that lets a threshold arrive as int, float, or string.

import (
	"encoding/json"
	"fmt"
	"os"
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
