package toolseq

import (
	"sort"
	"strings"
)

// Edge is one directed tool-to-tool transition observed in the corpus: a call
// of tool From immediately followed by a call of tool To, within a single
// session. Count is how many times the From->To adjacency occurred across all
// sessions; Prob is the outgoing-normalized transition probability, i.e. Count
// divided by the total number of transitions leaving From. The Prob of every
// edge sharing a From sums to 1 (for any From that has outgoing edges), so an
// edge set reads as a proper per-source distribution.
type Edge struct {
	From  string  `json:"from"`
	To    string  `json:"to"`
	Count int     `json:"count"`
	Prob  float64 `json:"prob"`
}

// SeqCount is one contiguous tool-name n-gram and how often it occurred as a
// window across all sessions. Seq is a fresh copy of length n; Count is its
// total occurrences.
type SeqCount struct {
	Seq   []string `json:"seq"`
	Count int      `json:"count"`
}

// seqSep is a byte that cannot occur inside a tool name, so joining a window
// into a map key is collision-free (two distinct windows can never collapse to
// the same key).
const seqSep = "\x00"

// Transitions folds ordered per-session tool sequences into the directed
// transition graph. Each session contributes every adjacent (session[i],
// session[i+1]) pair; a transition never spans a session boundary, so unrelated
// sessions cannot manufacture an edge. The result is deterministic: sorted by
// Count descending, then From ascending, then To ascending. An empty or nil
// input yields an empty (non-nil) slice.
func Transitions(sessions [][]string) []Edge {
	type key struct{ from, to string }
	counts := map[key]int{}
	fromTotals := map[string]int{}
	for _, s := range sessions {
		for i := 0; i+1 < len(s); i++ {
			counts[key{s[i], s[i+1]}]++
			fromTotals[s[i]]++
		}
	}

	out := make([]Edge, 0, len(counts))
	for k, c := range counts {
		e := Edge{From: k.from, To: k.to, Count: c}
		if t := fromTotals[k.from]; t > 0 {
			e.Prob = float64(c) / float64(t)
		}
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		return out[i].To < out[j].To
	})
	return out
}

// TopSequences counts every contiguous length-n tool-name window across all
// sessions and returns the k most frequent, deterministically ranked by Count
// descending then lexicographically by sequence. A window never spans a session
// boundary. n < 1 is meaningless and returns nil; a session shorter than n
// contributes no window. k <= 0 means "no limit" — return every distinct n-gram
// in ranked order.
func TopSequences(sessions [][]string, n, k int) []SeqCount {
	if n < 1 {
		return nil
	}
	counts := map[string]int{}
	// Keep one representative slice per distinct n-gram so the returned Seq is
	// the real tool names, not a re-split of the joined key.
	rep := map[string][]string{}
	for _, s := range sessions {
		for i := 0; i+n <= len(s); i++ {
			window := s[i : i+n]
			key := strings.Join(window, seqSep)
			counts[key]++
			if _, ok := rep[key]; !ok {
				cp := make([]string, n)
				copy(cp, window)
				rep[key] = cp
			}
		}
	}

	out := make([]SeqCount, 0, len(counts))
	for key, c := range counts {
		out = append(out, SeqCount{Seq: rep[key], Count: c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return lessSeq(out[i].Seq, out[j].Seq)
	})
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// lessSeq is lexicographic order over two sequences, element by element, with
// the shorter sequence ordering first on a shared prefix. It is only the
// deterministic tie-break for equal counts.
func lessSeq(a, b []string) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return len(a) < len(b)
}
