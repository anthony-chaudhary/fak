package ideascout

// Shared leaf helpers: timestamp parsing, typed access into a candidate's Extra
// map, and the small string transforms the parsers and renderer both use.

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"time"
)

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
