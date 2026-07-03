package clonescan

import (
	"sort"
)

// Match is one tracked site whose normalized token windows overlap the candidate:
// the file it lives in, the source-line span of the overlapping region, and how
// many distinct clone windows the candidate shares with it. More shared windows =
// a longer / more complete duplicate, so Windows is the rank key.
type Match struct {
	File       string `json:"file"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
	Windows    int    `json:"windows"`
	SampleLine string `json:"sample_line,omitempty"`
}

// qualifyingWindows slides the WindowTokens-length window over a token stream and
// returns, for each window that carries enough logic, its (startLine, endLine,
// key). The key is the joined normalized token sequence — the same identity the
// scorecard clones on. A window is skipped unless its effective logic count (its
// logic count, gated to zero when it carries only bare assignments) is >=
// MinLogicTokens, so data/declaration regions never qualify.
func qualifyingWindows(toks []token) (keys []string, spans [][2]int) {
	m := len(toks)
	if m < WindowTokens {
		return nil, nil
	}
	// Prefix-summable logic and non-assignment-logic marks.
	logic := make([]int, m)
	nonassign := make([]int, m)
	for i, t := range toks {
		if t.isLogic {
			logic[i] = 1
			if !assignOps[t.sym] {
				nonassign[i] = 1
			}
		}
	}
	running, runningNA := 0, 0
	for j := 0; j < WindowTokens; j++ {
		running += logic[j]
		runningNA += nonassign[j]
	}
	for start := 0; start <= m-WindowTokens; start++ {
		if start > 0 {
			running += logic[start+WindowTokens-1] - logic[start-1]
			runningNA += nonassign[start+WindowTokens-1] - nonassign[start-1]
		}
		effective := running
		if runningNA < 1 {
			effective = 0 // only bare assignments — a declaration/field-init block
		}
		if effective < MinLogicTokens {
			continue
		}
		key := windowKey(toks, start)
		keys = append(keys, key)
		spans = append(spans, [2]int{toks[start].line, toks[start+WindowTokens-1].line})
	}
	return keys, spans
}

// windowKey joins the WindowTokens symbols starting at `start` into one key. A
// small manual builder avoids allocating an intermediate slice per window.
func windowKey(toks []token, start int) string {
	total := 0
	for j := start; j < start+WindowTokens; j++ {
		total += len(toks[j].sym) + 1
	}
	buf := make([]byte, 0, total)
	for j := start; j < start+WindowTokens; j++ {
		buf = append(buf, toks[j].sym...)
		buf = append(buf, '\x1f') // unit separator — never appears in a token sym
	}
	return string(buf)
}

// candidateKeys returns the set of qualifying window keys for a candidate block.
// Identifiers are kept (normalizeIdents=false) to match the scorecard's precision
// sweet spot: distinct code with distinct names must not false-match.
func candidateKeys(candidate string) map[string]bool {
	keys, _ := qualifyingWindows(goTokens(candidate, false))
	set := make(map[string]bool, len(keys))
	for _, k := range keys {
		set[k] = true
	}
	return set
}

// Query answers the authoring-time question: given a candidate Go block, which
// tracked files hold a token-similar block RIGHT NOW? `tree` maps rel-path ->
// source text (the caller supplies the tracked tree; this package does no I/O so
// it stays pure and testable). Results are ranked most-overlap first, capped at
// maxResults (<=0 means no cap).
//
// A file is skipped when its path equals selfPath — so querying a block that is
// ALREADY committed at some path does not report that path as its own duplicate.
// Pass "" for selfPath when the candidate is unwritten.
func Query(candidate string, tree map[string]string, selfPath string, maxResults int) []Match {
	want := candidateKeys(candidate)
	if len(want) == 0 {
		return nil
	}
	var matches []Match
	// Deterministic file order for stable output.
	files := make([]string, 0, len(tree))
	for rel := range tree {
		if rel == selfPath {
			continue
		}
		files = append(files, rel)
	}
	sort.Strings(files)

	for _, rel := range files {
		keys, spans := qualifyingWindows(goTokens(tree[rel], false))
		var hitSpans [][2]int
		for i, k := range keys {
			if want[k] {
				hitSpans = append(hitSpans, spans[i])
			}
		}
		if len(hitSpans) == 0 {
			continue
		}
		lo, hi := hitSpans[0][0], hitSpans[0][1]
		for _, s := range hitSpans[1:] {
			if s[0] < lo {
				lo = s[0]
			}
			if s[1] > hi {
				hi = s[1]
			}
		}
		matches = append(matches, Match{
			File:      rel,
			StartLine: lo,
			EndLine:   hi,
			Windows:   len(hitSpans),
		})
	}

	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].Windows != matches[j].Windows {
			return matches[i].Windows > matches[j].Windows // most overlap first
		}
		return matches[i].File < matches[j].File
	})
	if maxResults > 0 && len(matches) > maxResults {
		matches = matches[:maxResults]
	}
	return matches
}
