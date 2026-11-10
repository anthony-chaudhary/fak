// Package verbhint resolves a mistyped or hallucinated agent CLI verb to the
// nearest registered verb, so an unknown-verb refusal costs one call instead of
// a loop. It exists because cheap models invent sub-verbs and flags (see
// docs/tickets/model-class-quirks/NV-06-hallucinated-verbs-and-flags.md): the
// refusal text is the only teacher, so it must name a real verb.
package verbhint

import (
	"sort"
	"strings"
)

// DefaultMaxDistance is the edit distance within which a suggestion is offered.
// A value of 2 catches transpositions and single insert/delete/typo mistakes
// without matching genuinely unrelated verbs.
const DefaultMaxDistance = 2

// Nearest returns the registered verb closest to token within maxDist edits.
// Matching is case-insensitive. Ties are broken by the shorter word, then by
// lexical order, so the result is deterministic. ok is false when no verb is
// within maxDist.
func Nearest(token string, known []string, maxDist int) (string, bool) {
	if token == "" || maxDist < 0 {
		return "", false
	}
	low := strings.ToLower(token)
	best := ""
	bestDist := maxDist + 1
	for _, k := range known {
		if k == "" {
			continue
		}
		d := Levenshtein(low, strings.ToLower(k), bestDist)
		if d > maxDist {
			continue
		}
		switch {
		case d < bestDist:
			best, bestDist = k, d
		case d == bestDist:
			// deterministic tie-break: shorter first, then lexical.
			if len(k) < len(best) || (len(k) == len(best) && k < best) {
				best = k
			}
		}
	}
	if best == "" {
		return "", false
	}
	return best, true
}

// Hint returns the one-line refusal suffix naming the nearest verb and the help
// command, or "" when nothing is close enough.
func Hint(token string, known []string, maxDist int) string {
	n, ok := Nearest(token, known, maxDist)
	if !ok {
		return ""
	}
	return "did you mean \"" + n + "\"? run '" + n + " -h' for its flags"
}

// Levenshtein returns the edit distance between a and b. limit, when >= 0,
// aborts early once the running distance exceeds limit (returning limit+1),
// which keeps a scan over many candidates cheap.
func Levenshtein(a, b string, limit int) int {
	ar, br := []rune(a), []rune(b)
	if len(ar) == 0 {
		return len(br)
	}
	if len(br) == 0 {
		return len(ar)
	}
	prev := make([]int, len(br)+1)
	cur := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ar); i++ {
		cur[0] = i
		rowMin := cur[0]
		for j := 1; j <= len(br); j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			cur[j] = min3(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
			if cur[j] < rowMin {
				rowMin = cur[j]
			}
		}
		if limit >= 0 && rowMin > limit {
			return limit + 1
		}
		prev, cur = cur, prev
	}
	return prev[len(br)]
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}

// Sorted returns a copy of known sorted, used by callers that build their verb
// list from a map so refusal output is stable across runs.
func Sorted(known []string) []string {
	out := append([]string(nil), known...)
	sort.Strings(out)
	return out
}
