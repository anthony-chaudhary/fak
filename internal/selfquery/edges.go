package selfquery

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// edges.go is the edge-extraction half of #3161 (parent epic #1494, the
// query-quality axis): a feature query returns a FLAT ranked list, so the
// cross-references our corpus already carries *in the note bodies themselves*
// are thrown away. Six decision notes about one decision thread come back as six
// unrelated cards, and the question a reader actually has — "why did we land
// here, and what did it supersede?" — is exactly the edge the flat list drops.
//
// This extracts a deterministic note->note graph from text that is already in
// the tree. No embeddings, no LLM on the query path, no index build: same bytes
// in, same edges out. The fences from the issue hold here:
//   - Deterministic only — pure text extraction, sorted and de-duplicated.
//   - Advisory RANKING metadata, never a trust signal: an edge to a stale note
//     does not make it fresh, and edges never re-order or drop a result (the
//     same contract FeatureCard.Freshness carries, see freshness.go).
//   - Read-only — nothing here writes to the corpus.
//
// Producer counts measured over the 476 notes in docs/notes on 2026-08-07, which
// is why the kinds are NOT equally load-bearing:
//
//	#NNNN issue refs         379 notes  <- the load-bearing edge by an order of magnitude
//	[[wikilinks]]              9 notes
//	see_also: / companions:    2 notes
//	supersedes:                1 note, whose value is literally "none" (zero real edges)
//	Resolved by <sha>          0 notes
//
// The last two are wired because the issue names them and they cost one line
// each on top of the same scanner, NOT because the corpus demands them today.
// `Resolved by <sha>` in particular is a *GitHub issue-comment* convention
// (documented as a template in docs/notes/EPIC-CLOSEOUT-METHOD-2026-06-29.md),
// not a note-body one — every "resolved by" in docs/notes is English prose. Do
// not read a zero-edge result for those kinds as a broken extractor.
//
// Unblocks the `supersedes:` front-matter edge left open as a documented
// follow-on by #3163 (freshness.go:40-43), which is gated on exactly this.

// Edge kinds. The vocabulary is closed and matches the issue's typing: five
// surface syntaxes fold into four relationship kinds.
const (
	// EdgeCites is an outbound reference: a [[wikilink]] or a #NNNN issue ref.
	EdgeCites = "cites"
	// EdgeSibling is a hand-declared companion note (see_also: / companions:).
	EdgeSibling = "sibling"
	// EdgeSupersedes is a hand-declared supersession (supersedes:) — the
	// explicit edge the dated-note heuristic in freshness.go cannot see.
	EdgeSupersedes = "supersedes"
	// EdgeResolvedBy points at the commit that resolved the note's subject.
	EdgeResolvedBy = "resolved-by"
)

// maxEdgesPerCard bounds how many edges one card carries, so a note citing a
// hundred issues cannot bloat a query response. The cut is applied AFTER the
// deterministic sort, so the retained set is stable for the same input — but it
// is a real truncation, not a full graph: a card at the cap may have more.
const maxEdgesPerCard = 24

// maxNoteBytes bounds a single note read on the query path. Every note in the
// corpus is far under this; the cap exists so a pathological file cannot stall a
// query.
const maxNoteBytes = 512 << 10

// Edge is one outbound cross-reference extracted from a note body. Target is the
// raw extracted reference (a note slug, a `#NNNN` issue ref, a path, or a commit
// sha) — deliberately NOT resolved to a FeatureCard here, because resolution is
// a ranking concern and extraction must stay a pure function of the bytes.
type Edge struct {
	Kind   string `json:"kind"`
	Target string `json:"target"`
}

var (
	// wikilinkRE matches a [[target]] reference. The target may carry a `|alias`
	// or `#anchor` suffix, both trimmed by cleanWikilink.
	wikilinkRE = regexp.MustCompile(`\[\[([^\[\]\n]+)\]\]`)
	// issueRefRE matches a bare #NNNN issue reference. One to five digits: fak
	// issue numbers are four digits today, and the upper bound is what keeps a
	// six-digit hex colour (#1a2b3c has letters, but #123456 does not) out.
	// Boundary checks are done by hand in extractIssueRefs — RE2 has no
	// lookbehind.
	issueRefRE = regexp.MustCompile(`#\d{1,5}`)
	// siblingKeyRE matches the head of a see_also: / companions: line in either
	// style the corpus actually uses: YAML-ish front matter (`see_also: [...]`)
	// and a bold or plain prose line (`**Companions:** ...`). List and emphasis
	// markers are allowed before the key.
	siblingKeyRE = regexp.MustCompile(`(?i)^[\s>*\-]*\**\s*(see[ _]also|companions)\s*\**\s*:`)
	// supersedesKeyRE matches a `supersedes:` declaration under the same
	// leading-marker allowance.
	supersedesKeyRE = regexp.MustCompile(`(?i)^[\s>*\-]*\**\s*supersedes\s*\**\s*:`)
	// resolvedByRE matches the commit-anchored closure convention, `Resolved by
	// <sha>`, with the sha optionally backticked or escaped. Seven hex chars is
	// git's short-sha floor, so a shorter token is not admitted.
	resolvedByRE = regexp.MustCompile("(?i)resolved by\\s+[`\\\\]*([0-9a-f]{7,40})\\b")
	// mdLinkDestRE pulls the destination out of a markdown link, so a companion
	// written as [`POLICY.md`](../../POLICY.md) yields the path, not the label.
	mdLinkDestRE = regexp.MustCompile(`\]\(([^)\s]+)`)
)

// ExtractEdges returns the outbound edges of one note body, de-duplicated and
// deterministically ordered (by kind, then target). It is a pure function: the
// same bytes always yield the same edges, which is what lets a query response
// carry a subgraph without an index build or a cache.
//
// This is the engine a `fak index edges <slug>` verb wraps — the issue's first
// checkable step — kept here so the extraction is testable and reusable without
// the CLI.
func ExtractEdges(body string) []Edge {
	seen := map[Edge]bool{}
	var out []Edge
	add := func(kind, target string) {
		target = strings.TrimSpace(target)
		if target == "" {
			return
		}
		e := Edge{Kind: kind, Target: target}
		if seen[e] {
			return
		}
		seen[e] = true
		out = append(out, e)
	}

	for _, m := range wikilinkRE.FindAllStringSubmatch(body, -1) {
		add(EdgeCites, cleanWikilink(m[1]))
	}
	for _, ref := range extractIssueRefs(body) {
		add(EdgeCites, ref)
	}
	for _, m := range resolvedByRE.FindAllStringSubmatch(body, -1) {
		add(EdgeResolvedBy, strings.ToLower(m[1]))
	}
	for _, line := range strings.Split(body, "\n") {
		if loc := siblingKeyRE.FindStringIndex(line); loc != nil {
			for _, t := range splitDeclaredTargets(line[loc[1]:]) {
				add(EdgeSibling, t)
			}
			continue
		}
		if loc := supersedesKeyRE.FindStringIndex(line); loc != nil {
			for _, t := range splitDeclaredTargets(line[loc[1]:]) {
				add(EdgeSupersedes, t)
			}
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Target < out[j].Target
	})
	return out
}

// cleanWikilink reduces a raw [[...]] payload to its target: the part before an
// `|alias` or `#anchor`, with surrounding whitespace and backticks removed.
// Returns "" for the two shapes the corpus proves are NOT links — a POSIX
// character class (`[[:space:]]`, which leads with a colon) and an empty body.
func cleanWikilink(raw string) string {
	s := raw
	if i := strings.IndexAny(s, "|#"); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(strings.Trim(strings.TrimSpace(s), "`"))
	if s == "" || strings.HasPrefix(s, ":") {
		return ""
	}
	return s
}

// extractIssueRefs finds bare #NNNN issue references. A match is admitted only
// when the `#` is not preceded by a word character or another `#` (so a markdown
// heading, an anchor like `file.md#2`, and `##3` are all rejected) and the digits
// are not followed by a word character (so `#12abc` is rejected).
func extractIssueRefs(body string) []string {
	var out []string
	for _, loc := range issueRefRE.FindAllStringIndex(body, -1) {
		if loc[0] > 0 && isRefWordByte(body[loc[0]-1]) {
			continue
		}
		if loc[1] < len(body) && isRefWordByte(body[loc[1]]) {
			continue
		}
		out = append(out, body[loc[0]:loc[1]])
	}
	return out
}

// isRefWordByte reports whether b would make an adjacent #NNNN part of a larger
// token rather than a standalone issue reference.
func isRefWordByte(b byte) bool {
	switch {
	case b >= '0' && b <= '9', b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z':
		return true
	case b == '_', b == '#':
		return true
	}
	return false
}

// splitDeclaredTargets parses the value side of a see_also: / companions: /
// supersedes: declaration into individual targets. It handles both shapes the
// corpus uses — a bracketed comma list and a prose comma list of markdown links
// or backticked filenames — by splitting on commas and, per item, preferring a
// markdown link DESTINATION over its label, then stripping list brackets,
// backticks, and any trailing parenthetical gloss.
//
// The literal "none" is dropped: it is a declared ABSENCE of a target, and the
// one `supersedes:` producer in the corpus writes exactly that.
func splitDeclaredTargets(value string) []string {
	value = strings.TrimSpace(value)
	value = strings.TrimSuffix(strings.TrimPrefix(value, "["), "]")
	var out []string
	for _, part := range strings.Split(value, ",") {
		item := strings.TrimSpace(part)
		if item == "" {
			continue
		}
		if m := mdLinkDestRE.FindStringSubmatch(item); m != nil {
			item = m[1]
		} else {
			if i := strings.Index(item, " ("); i >= 0 {
				item = item[:i]
			}
			item = strings.Trim(strings.TrimSpace(item), "[]`*.\"'")
		}
		item = strings.TrimSpace(item)
		if item == "" || strings.EqualFold(item, "none") {
			continue
		}
		out = append(out, item)
	}
	return out
}

// NoteEdges reads one note by its repo-relative reference and returns its
// outbound edges. root is the checkout the ref resolves against. It is the
// one-note read path — the direct engine for the issue's first checkable step
// ("print the outbound edges of ONE note") — and returns nil, not an error, for
// a ref that is not a readable in-repo markdown file, so a caller can offer it
// over any card without pre-classifying the ref.
func NoteEdges(root, ref string) []Edge {
	body, ok := readNoteBody(root, ref)
	if !ok {
		return nil
	}
	return ExtractEdges(body)
}

// readNoteBody reads the markdown file a DetailRef cites, under the same
// precision-first admission the staleness check uses (citedRepoPath): no URLs,
// no cap refs, no parent-escaping paths. Additionally restricted to `.md` — the
// edge syntaxes are a markdown-corpus convention, and scanning a Go file for
// `#NNNN` would mine comments, not a note graph.
func readNoteBody(root, ref string) (string, bool) {
	if strings.TrimSpace(root) == "" {
		return "", false
	}
	clean, ok := citedRepoPath(ref)
	if !ok || !strings.HasSuffix(strings.ToLower(clean), ".md") {
		return "", false
	}
	f, err := os.Open(filepath.Join(root, filepath.FromSlash(clean)))
	if err != nil {
		return "", false
	}
	defer f.Close()
	buf := make([]byte, maxNoteBytes)
	n, err := f.Read(buf)
	if n <= 0 && err != nil {
		return "", false
	}
	return string(buf[:n]), true
}

// applyEdges stamps the 1-hop outbound edges onto each card whose DetailRef
// names a readable in-repo note — the issue's query mode (a), "expand the top-K
// along 1-hop edges so neighbours surface with the hit".
//
// It runs over the RANKED, POST-LIMIT result rather than the full candidate set
// (which is what freshness does), because this reads file bodies rather than
// stat'ing paths: bounding the reads to what is actually returned keeps the hot
// path proportional to --limit. Identical refs are read once via a small cache.
//
// defaultRoot is the checkout to resolve against for cards that carry no Root of
// their own; a cross-repo card (#3435) resolves against the root it was loaded
// from, so a merged result never reads the wrong checkout.
//
// It only writes the advisory Related field — it never re-orders or drops a card.
func applyEdges(cards []FeatureCard, defaultRoot string) {
	cache := map[string][]Edge{}
	for i := range cards {
		if edges := edgesForCard(cards[i], defaultRoot, cache); len(edges) > 0 {
			cards[i].Related = edges
		}
	}
}

// edgesForCard resolves one card's edges against the checkout it was loaded from
// (its own Root, else defaultRoot), memoising by root+ref so a repeated
// DetailRef is read once per call. cache may be nil for a one-shot lookup.
func edgesForCard(c FeatureCard, defaultRoot string, cache map[string][]Edge) []Edge {
	root := c.Root
	if root == "" {
		root = defaultRoot
	}
	if root == "" {
		return nil
	}
	key := root + "\x00" + c.DetailRef
	if edges, done := cache[key]; done {
		return edges
	}
	edges := NoteEdges(root, c.DetailRef)
	if len(edges) > maxEdgesPerCard {
		edges = edges[:maxEdgesPerCard]
	}
	if cache != nil {
		cache[key] = edges
	}
	return edges
}
