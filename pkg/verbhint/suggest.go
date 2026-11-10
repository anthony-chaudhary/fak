package verbhint

import (
	"sort"
	"strings"
)

// Suggest returns up to limit registered verbs within maxDist edits of token,
// ordered by ascending distance then lexical order, so multi-candidate refusal
// text stays deterministic. Matching is case-insensitive; it returns nil when
// nothing qualifies, letting callers omit the hint rather than print noise.
func Suggest(token string, known []string, maxDist, limit int) []string {
	if token == "" || limit <= 0 || maxDist < 0 {
		return nil
	}
	low := strings.ToLower(token)
	type scored struct {
		name string
		dist int
	}
	var hits []scored
	for _, k := range known {
		if k == "" {
			continue
		}
		d := Levenshtein(low, strings.ToLower(k), maxDist)
		if d > maxDist {
			continue
		}
		hits = append(hits, scored{k, d})
	}
	if len(hits) == 0 {
		return nil
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].dist != hits[j].dist {
			return hits[i].dist < hits[j].dist
		}
		return hits[i].name < hits[j].name
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.name
	}
	return out
}
