package toolseq

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
)

// Call is the minimal observation used to discover workflow concepts.
type Call struct {
	Tool  string
	Error bool
}

// Session is one ordered tool-call episode.
type Session struct {
	ID    string
	Calls []Call
}

// Concept is an explainable recurring workflow family. ID is stable for the
// concept's medoid signature and can be fed back to the reporting command.
type Concept struct {
	ID        string   `json:"id"`
	Label     string   `json:"label"`
	Sessions  int      `json:"sessions"`
	Share     float64  `json:"share"`
	Calls     int      `json:"calls"`
	ErrorRate float64  `json:"error_rate"`
	Signature []string `json:"signature"`
	TopTools  []string `json:"top_tools"`
	Exemplars []string `json:"exemplars"`
	Members   []string `json:"member_sessions,omitempty"`
}

// Discover groups sessions by connected components in an explainable feature
// graph. Features are tool names and adjacent transitions; edges require a
// weighted-Jaccard similarity of at least threshold. This deliberately avoids
// opaque embedding clusters: every grouping can be understood from Signature
// and TopTools, and every concept carries source-session exemplars.
func Discover(sessions []Session, threshold float64) []Concept {
	if threshold <= 0 || threshold > 1 {
		threshold = 0.55
	}
	ss := append([]Session(nil), sessions...)
	sort.Slice(ss, func(i, j int) bool { return ss[i].ID < ss[j].ID })
	if len(ss) == 0 {
		return nil
	}
	features := make([]map[string]float64, len(ss))
	for i := range ss {
		features[i] = featureVector(ss[i].Calls)
	}

	parent := make([]int, len(ss))
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	union := func(a, b int) {
		a, b = find(a), find(b)
		if a != b {
			if a < b {
				parent[b] = a
			} else {
				parent[a] = b
			}
		}
	}
	for i := range ss {
		for j := i + 1; j < len(ss); j++ {
			if similarity(features[i], features[j]) >= threshold {
				union(i, j)
			}
		}
	}
	groups := map[int][]int{}
	for i := range ss {
		r := find(i)
		groups[r] = append(groups[r], i)
	}
	roots := make([]int, 0, len(groups))
	for r := range groups {
		roots = append(roots, r)
	}
	sort.Ints(roots)

	out := make([]Concept, 0, len(roots))
	for _, r := range roots {
		out = append(out, buildConcept(ss, features, groups[r], len(ss)))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Sessions != out[j].Sessions {
			return out[i].Sessions > out[j].Sessions
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func buildConcept(ss []Session, fv []map[string]float64, members []int, total int) Concept {
	medoid := members[0]
	best := -1.0
	for _, i := range members {
		sum := 0.0
		for _, j := range members {
			sum += similarity(fv[i], fv[j])
		}
		if sum > best || (sum == best && ss[i].ID < ss[medoid].ID) {
			medoid, best = i, sum
		}
	}
	sig := collapsedTools(ss[medoid].Calls)
	h := sha256.Sum256([]byte(strings.Join(sig, "→")))
	ids := make([]string, 0, len(members))
	toolCounts := map[string]int{}
	calls, errors := 0, 0
	for _, i := range members {
		ids = append(ids, ss[i].ID)
		for _, c := range ss[i].Calls {
			calls++
			toolCounts[c.Tool]++
			if c.Error {
				errors++
			}
		}
	}
	sort.Strings(ids)
	exemplars := append([]string(nil), ids...)
	if len(exemplars) > 3 {
		exemplars = exemplars[:3]
	}
	top := rankedTools(toolCounts, 4)
	label := strings.Join(top, " + ")
	if label == "" {
		label = "no-tool workflow"
	}
	rate := 0.0
	if calls > 0 {
		rate = float64(errors) / float64(calls)
	}
	return Concept{
		ID: fmt.Sprintf("wf-%x", h[:4]), Label: label, Sessions: len(members),
		Share: float64(len(members)) / float64(total), Calls: calls, ErrorRate: rate,
		Signature: sig, TopTools: top, Exemplars: exemplars, Members: ids,
	}
}

func featureVector(calls []Call) map[string]float64 {
	out := map[string]float64{}
	seq := collapsedTools(calls)
	for _, t := range seq {
		out["tool:"+t]++
	}
	for i := 1; i < len(seq); i++ {
		out["edge:"+seq[i-1]+"→"+seq[i]] += 1.5
	}
	return out
}

func collapsedTools(calls []Call) []string {
	out := make([]string, 0, len(calls))
	for _, c := range calls {
		t := strings.TrimSpace(c.Tool)
		if t == "" || (len(out) > 0 && out[len(out)-1] == t) {
			continue
		}
		out = append(out, t)
	}
	return out
}

func similarity(a, b map[string]float64) float64 {
	keys := map[string]struct{}{}
	for k := range a {
		keys[k] = struct{}{}
	}
	for k := range b {
		keys[k] = struct{}{}
	}
	minSum, maxSum := 0.0, 0.0
	for k := range keys {
		av, bv := a[k], b[k]
		if av < bv {
			minSum += av
			maxSum += bv
		} else {
			minSum += bv
			maxSum += av
		}
	}
	if maxSum == 0 {
		return 1
	}
	return minSum / maxSum
}

func rankedTools(counts map[string]int, limit int) []string {
	names := make([]string, 0, len(counts))
	for n := range counts {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		if counts[names[i]] != counts[names[j]] {
			return counts[names[i]] > counts[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) > limit {
		names = names[:limit]
	}
	return names
}
