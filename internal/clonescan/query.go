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

// span is one qualifying window's source-line extent: [startLine, endLine].
type span = [2]int

// qualifyingWindows slides the WindowTokens-length window over a token stream and
// returns, for each window that carries enough logic, its (startLine, endLine,
// key). The key is the joined normalized token sequence — the same identity the
// scorecard clones on. A window is skipped unless its effective logic count (its
// logic count, gated to zero when it carries only bare assignments) is >=
// MinLogicTokens, so data/declaration regions never qualify.
func qualifyingWindows(toks []token) (keys []string, spans []span) {
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
		spans = append(spans, span{toks[start].line, toks[start+WindowTokens-1].line})
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

// CandidateKeys returns the set of qualifying window keys for a candidate block.
// Identifiers are kept (normalizeIdents=false) to match the scorecard's precision
// sweet spot: distinct code with distinct names must not false-match. Callers that
// hold a prebuilt *TreeIndex compute this once per candidate and hand it straight
// to TreeIndex.Query.
func CandidateKeys(candidate string) map[string]bool {
	keys, _ := qualifyingWindows(goTokens(candidate, false))
	set := make(map[string]bool, len(keys))
	for _, k := range keys {
		set[k] = true
	}
	return set
}

// TreeIndex is a tree tokenized ONCE: each file's source is run through the same
// deterministic qualifyingWindows(goTokens) exactly once at build time and stored
// as a per-file map of window key -> the source-line spans where that key occurs
// (a key can repeat at several positions in one file, so the value is a slice). A
// caller with many candidates to check against the same tree builds the index once
// and queries each candidate against it, instead of re-tokenizing the whole tree
// per candidate.
//
// selfPath is intentionally NOT baked into the index: the same prebuilt index
// answers queries for different candidates, each excluding its own path at query
// time. `files` is the deterministic sorted rel-path order so a query folds through
// the identical stable ordering the legacy per-call Query used.
type TreeIndex struct {
	files  []string                     // rel paths, sorted, for stable output order
	byFile map[string]map[string][]span // rel path -> (window key -> spans)
}

// BuildTreeIndex tokenizes each file in `tree` exactly once and returns the prebuilt
// index. tree maps rel-path -> source text; this package does no I/O so it stays
// pure and testable.
//
// An optional WindowCache memoizes the pure per-file tokenization keyed on the file's
// exact bytes (#4330): when supplied, an unchanged file is a content-addressed lookup
// instead of a re-lex. The cache is accelerate-never-gate — a nil cache, a miss, or
// any implementation failure computes the windows fresh, so the returned index is
// byte-identical with or without it. Callers pass at most one cache; extras are
// ignored.
func BuildTreeIndex(tree map[string]string, cache ...WindowCache) *TreeIndex {
	var wc WindowCache
	if len(cache) > 0 {
		wc = cache[0]
	}
	idx := &TreeIndex{
		files:  make([]string, 0, len(tree)),
		byFile: make(map[string]map[string][]span, len(tree)),
	}
	for rel := range tree {
		idx.files = append(idx.files, rel)
	}
	sort.Strings(idx.files)
	for _, rel := range idx.files {
		src := tree[rel]
		var keys []string
		var spans []span
		if wc != nil {
			if k, s, ok := wc.Get(src); ok {
				keys, spans = k, s
			} else {
				keys, spans = qualifyingWindows(goTokens(src, false))
				wc.Put(src, keys, spans)
			}
		} else {
			keys, spans = qualifyingWindows(goTokens(src, false))
		}
		if len(keys) == 0 {
			continue
		}
		fi := make(map[string][]span, len(keys))
		for i, k := range keys {
			fi[k] = append(fi[k], spans[i])
		}
		idx.byFile[rel] = fi
	}
	// A persistent implementation may coalesce retention at the consumer-owned end
	// of the batch. The optional hook keeps the WindowCache contract pure for in-memory
	// implementations while avoiding one shared-directory scan per Put.
	if maintainer, ok := wc.(interface{ Maintain() }); ok {
		maintainer.Maintain()
	}
	return idx
}

// Query intersects a candidate's `want` key-set (from CandidateKeys) against the
// prebuilt index and returns the ranked, capped matches. It folds through the
// SAME final sort.SliceStable (Windows desc, File asc) and maxResults cap the
// legacy per-call Query used, so output is byte-identical.
//
// A file is skipped when its path equals selfPath — so querying a block that is
// ALREADY committed at some path does not report that path as its own duplicate.
// Pass "" for selfPath when the candidate is unwritten. maxResults <= 0 means no cap.
func (idx *TreeIndex) Query(want map[string]bool, selfPath string, maxResults int) []Match {
	if len(want) == 0 {
		return nil
	}
	var matches []Match
	for _, rel := range idx.files {
		if rel == selfPath {
			continue
		}
		fi := idx.byFile[rel]
		if len(fi) == 0 {
			continue
		}
		var lo, hi, windows int
		for k := range want {
			for _, s := range fi[k] {
				if windows == 0 {
					lo, hi = s[0], s[1]
				} else {
					if s[0] < lo {
						lo = s[0]
					}
					if s[1] > hi {
						hi = s[1]
					}
				}
				windows++
			}
		}
		if windows == 0 {
			continue
		}
		matches = append(matches, Match{
			File:      rel,
			StartLine: lo,
			EndLine:   hi,
			Windows:   windows,
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

// Query answers the authoring-time question: given a candidate Go block, which
// tracked files hold a token-similar block RIGHT NOW? `tree` maps rel-path ->
// source text (the caller supplies the tracked tree; this package does no I/O so
// it stays pure and testable). Results are ranked most-overlap first, capped at
// maxResults (<=0 means no cap).
//
// A file is skipped when its path equals selfPath — so querying a block that is
// ALREADY committed at some path does not report that path as its own duplicate.
// Pass "" for selfPath when the candidate is unwritten.
//
// Query is a thin wrapper over BuildTreeIndex + TreeIndex.Query kept for callers
// with a single candidate to check; a caller with many candidates against the same
// tree should build the index once and reuse it.
func Query(candidate string, tree map[string]string, selfPath string, maxResults int) []Match {
	return BuildTreeIndex(tree).Query(CandidateKeys(candidate), selfPath, maxResults)
}

// CloneSite is one tracked occurrence site for a clone window.
type CloneSite struct {
	File      string `json:"file"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
}

// CloneWindow represents a candidate clone window and all sites where it occurs.
type CloneWindow struct {
	Key   string      `json:"key,omitempty"`
	Sites []CloneSite `json:"sites"`
}

func cloneWindowsCanCoalesce(w1, w2 CloneWindow) bool {
	if len(w1.Sites) != len(w2.Sites) || len(w1.Sites) == 0 {
		return false
	}
	for i := range w1.Sites {
		s1 := w1.Sites[i]
		s2 := w2.Sites[i]
		if s1.File != s2.File {
			return false
		}
		// Overlap condition:
		// [s1.StartLine, s1.EndLine] and [s2.StartLine, s2.EndLine] overlap
		// iff s1.StartLine <= s2.EndLine && s2.StartLine <= s1.EndLine
		if s1.StartLine > s2.EndLine || s2.StartLine > s1.EndLine {
			return false
		}
	}
	return true
}

func normalizeCloneWindowSites(w *CloneWindow) {
	sort.Slice(w.Sites, func(i, j int) bool {
		if w.Sites[i].File != w.Sites[j].File {
			return w.Sites[i].File < w.Sites[j].File
		}
		if w.Sites[i].StartLine != w.Sites[j].StartLine {
			return w.Sites[i].StartLine < w.Sites[j].StartLine
		}
		return w.Sites[i].EndLine < w.Sites[j].EndLine
	})
}

// CoalesceGroups coalesces candidate clone windows that overlap at every site
// and share matching site cardinality and files into coalesced clone window groups,
// while strictly preserving disjoint clone windows and distinct occurrence sets.
func CoalesceGroups(windows []CloneWindow) []CloneWindow {
	n := len(windows)
	if n == 0 {
		return nil
	}
	normalized := make([]CloneWindow, n)
	for i := range windows {
		normalized[i] = CloneWindow{
			Key:   windows[i].Key,
			Sites: append([]CloneSite(nil), windows[i].Sites...),
		}
		normalizeCloneWindowSites(&normalized[i])
	}

	parent := make([]int, n)
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
	union := func(x, y int) {
		rx, ry := find(x), find(y)
		if rx != ry {
			parent[rx] = ry
		}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if cloneWindowsCanCoalesce(normalized[i], normalized[j]) {
				union(i, j)
			}
		}
	}

	groups := make(map[int][]int)
	for i := 0; i < n; i++ {
		root := find(i)
		groups[root] = append(groups[root], i)
	}

	var rootOrder []int
	for root := range groups {
		rootOrder = append(rootOrder, root)
	}
	sort.Ints(rootOrder)

	var result []CloneWindow
	for _, root := range rootOrder {
		indices := groups[root]
		firstWin := normalized[indices[0]]
		m := len(firstWin.Sites)
		coalescedSites := make([]CloneSite, m)
		for s := 0; s < m; s++ {
			f := firstWin.Sites[s].File
			lo := firstWin.Sites[s].StartLine
			hi := firstWin.Sites[s].EndLine
			for _, idx := range indices[1:] {
				st := normalized[idx].Sites[s]
				if st.StartLine < lo {
					lo = st.StartLine
				}
				if st.EndLine > hi {
					hi = st.EndLine
				}
			}
			coalescedSites[s] = CloneSite{
				File:      f,
				StartLine: lo,
				EndLine:   hi,
			}
		}
		result = append(result, CloneWindow{
			Sites: coalescedSites,
		})
	}
	return result
}

// CoalesceCloneWindows coalesces candidate clone windows that overlap at every site
// and share matching site cardinality and files into representative matches, while
// strictly preserving disjoint clone windows and distinct occurrence sets.
func CoalesceCloneWindows(windows []CloneWindow) []Match {
	n := len(windows)
	if n == 0 {
		return nil
	}
	normalized := make([]CloneWindow, n)
	for i := range windows {
		normalized[i] = CloneWindow{
			Key:   windows[i].Key,
			Sites: append([]CloneSite(nil), windows[i].Sites...),
		}
		normalizeCloneWindowSites(&normalized[i])
	}

	parent := make([]int, n)
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
	union := func(x, y int) {
		rx, ry := find(x), find(y)
		if rx != ry {
			parent[rx] = ry
		}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if cloneWindowsCanCoalesce(normalized[i], normalized[j]) {
				union(i, j)
			}
		}
	}

	groups := make(map[int][]int)
	for i := 0; i < n; i++ {
		root := find(i)
		groups[root] = append(groups[root], i)
	}

	var rootOrder []int
	for root := range groups {
		rootOrder = append(rootOrder, root)
	}
	sort.Ints(rootOrder)

	var matches []Match
	for _, root := range rootOrder {
		indices := groups[root]
		firstWin := normalized[indices[0]]
		m := len(firstWin.Sites)
		distinctKeys := make(map[string]bool)
		for _, idx := range indices {
			if normalized[idx].Key != "" {
				distinctKeys[normalized[idx].Key] = true
			}
		}
		numWindows := len(indices)
		if len(distinctKeys) > 0 {
			numWindows = len(distinctKeys)
		}

		for s := 0; s < m; s++ {
			f := firstWin.Sites[s].File
			lo := firstWin.Sites[s].StartLine
			hi := firstWin.Sites[s].EndLine
			for _, idx := range indices[1:] {
				st := normalized[idx].Sites[s]
				if st.StartLine < lo {
					lo = st.StartLine
				}
				if st.EndLine > hi {
					hi = st.EndLine
				}
			}
			matches = append(matches, Match{
				File:      f,
				StartLine: lo,
				EndLine:   hi,
				Windows:   numWindows,
			})
		}
	}

	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].Windows != matches[j].Windows {
			return matches[i].Windows > matches[j].Windows
		}
		if matches[i].File != matches[j].File {
			return matches[i].File < matches[j].File
		}
		return matches[i].StartLine < matches[j].StartLine
	})
	return matches
}

// CoalesceMatches coalesces overlapping matches within each file into a single
// representative match, while strictly preserving disjoint matches as separately reported.
func CoalesceMatches(matches []Match) []Match {
	if len(matches) == 0 {
		return nil
	}
	byFile := make(map[string][]Match)
	var fileOrder []string
	for _, m := range matches {
		if _, seen := byFile[m.File]; !seen {
			fileOrder = append(fileOrder, m.File)
		}
		byFile[m.File] = append(byFile[m.File], m)
	}
	sort.Strings(fileOrder)

	var result []Match
	for _, file := range fileOrder {
		fileMatches := byFile[file]
		sort.Slice(fileMatches, func(i, j int) bool {
			if fileMatches[i].StartLine != fileMatches[j].StartLine {
				return fileMatches[i].StartLine < fileMatches[j].StartLine
			}
			return fileMatches[i].EndLine < fileMatches[j].EndLine
		})

		var cur *Match
		for _, m := range fileMatches {
			if cur == nil {
				cp := m
				cur = &cp
			} else if m.StartLine <= cur.EndLine && cur.StartLine <= m.EndLine {
				if m.EndLine > cur.EndLine {
					cur.EndLine = m.EndLine
				}
				if m.Windows > cur.Windows {
					cur.Windows = m.Windows
				}
			} else {
				result = append(result, *cur)
				cp := m
				cur = &cp
			}
		}
		if cur != nil {
			result = append(result, *cur)
		}
	}

	sort.SliceStable(result, func(i, j int) bool {
		if result[i].Windows != result[j].Windows {
			return result[i].Windows > result[j].Windows
		}
		if result[i].File != result[j].File {
			return result[i].File < result[j].File
		}
		return result[i].StartLine < result[j].StartLine
	})
	return result
}

// CandidateCloneWindows extracts candidate clone windows for the given wanted keys,
// recording their qualifying occurrence sites across the indexed tree (excluding selfPath).
func (idx *TreeIndex) CandidateCloneWindows(want map[string]bool, selfPath string) []CloneWindow {
	if len(want) == 0 {
		return nil
	}
	keys := make([]string, 0, len(want))
	for k, wanted := range want {
		if wanted {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	var windows []CloneWindow
	for _, k := range keys {
		var sites []CloneSite
		for _, rel := range idx.files {
			if rel == selfPath {
				continue
			}
			spans := idx.byFile[rel][k]
			for _, s := range spans {
				sites = append(sites, CloneSite{
					File:      rel,
					StartLine: s[0],
					EndLine:   s[1],
				})
			}
		}
		if len(sites) > 0 {
			windows = append(windows, CloneWindow{
				Key:   k,
				Sites: sites,
			})
		}
	}
	return windows
}

// QueryCoalesced intersects a candidate's key-set against the prebuilt index and returns
// representative matches where candidate clone windows overlapping at every site and sharing
// matching site cardinality and files are coalesced into a single representative match,
// while disjoint clone windows and distinct occurrence sets remain separately reported.
func (idx *TreeIndex) QueryCoalesced(want map[string]bool, selfPath string, maxResults int) []Match {
	windows := idx.CandidateCloneWindows(want, selfPath)
	if len(windows) == 0 {
		return nil
	}
	matches := CoalesceMatches(CoalesceCloneWindows(windows))
	if maxResults > 0 && len(matches) > maxResults {
		matches = matches[:maxResults]
	}
	return matches
}

// QueryCoalesced is the single-candidate wrapper over BuildTreeIndex + TreeIndex.QueryCoalesced.
func QueryCoalesced(candidate string, tree map[string]string, selfPath string, maxResults int) []Match {
	return BuildTreeIndex(tree).QueryCoalesced(CandidateKeys(candidate), selfPath, maxResults)
}
