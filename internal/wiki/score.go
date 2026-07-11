package wiki

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// This is L7 (#4284): wiki quality as a MEASURABLE number, gate-able in CI — not a
// vibe. The angle is CodeWiki's (arXiv:2510.24428, CodeWikiBench): repository-doc
// quality must be scored, not assumed. fak's on-brand metrics fall straight out of
// the two halves it already owns:
//
//   - citation-resolve rate — the fraction of code cites that resolve (L3, #4280).
//   - leaf coverage         — the fraction of declared leaves that have a page (L1).
//   - freshness             — the fraction of pages that pin a generated_at_sha (L4).
//
// ComputeScore is a pure view over a page set + the catalog. The only IO is the cite
// resolver stat-ing cited files (via VerifyCitations); no LLM, no clock, no network.

// PageInput is one generated page handed to the scorer: its id relative to the wiki
// pages root with the ".md" stripped (e.g. "core-features/gateway", matching the
// Structure page ID scheme), and its raw markdown.
type PageInput struct {
	RelID    string
	Markdown []byte
}

// Score is the wiki quality report. Each rate is in [0,1]; a vacuous rate (no
// denominator) is 1.0 for the cite/fresh ratios — a page set that cites nothing has
// nothing unresolved — while LeafCoverage of an empty wiki is 0.0, so an empty or
// missing wiki is caught by the coverage floor rather than passing vacuously.
type Score struct {
	Repo                string    `json:"repo"`
	Pages               int       `json:"pages"`
	Leaves              int       `json:"leaves"`
	LeavesCovered       int       `json:"leaves_covered"`
	LeafCoverage        float64   `json:"leaf_coverage"`
	Citations           int       `json:"citations"`
	CitationsResolved   int       `json:"citations_resolved"`
	CitationResolveRate float64   `json:"citation_resolve_rate"`
	FreshPages          int       `json:"fresh_pages"`
	FreshRate           float64   `json:"fresh_rate"`
	Danglers            []Dangler `json:"danglers,omitempty"`
}

// ComputeScore folds a page set into the quality report. root is the tree the code
// citations resolve against; cat supplies the declared-leaf denominator for
// coverage; pages is every generated page found under the wiki pages root.
//
// Coverage counts a leaf as covered when a page with the id "core-features/<leaf>"
// is present — the exact id Structure mints for that leaf, so L7 measures against
// L1's own scheme. Resolve rate divides resolved cites by total cites across all
// pages. Fresh rate divides pages carrying a generated_at_sha by total pages.
func ComputeScore(root string, cat *devindex.Catalog, pages []PageInput) Score {
	s := Score{Repo: repoName(root), Pages: len(pages), Leaves: len(cat.Leaves)}

	present := make(map[string]bool, len(pages))
	for _, p := range pages {
		present[normalizeRel(p.RelID)] = true

		s.Citations += countCitations(p.Markdown)
		dangs := VerifyCitations(root, p.Markdown)
		s.Danglers = append(s.Danglers, dangs...)

		if ParseFrontmatter(p.Markdown).GeneratedAtSHA != "" {
			s.FreshPages++
		}
	}
	s.CitationsResolved = s.Citations - len(s.Danglers)

	for _, lf := range cat.Leaves {
		if present["core-features/"+strings.ToLower(lf.Name)] || present["core-features/"+lf.Name] {
			s.LeavesCovered++
		}
	}

	s.LeafCoverage = ratio(s.LeavesCovered, s.Leaves, 0) // empty wiki => 0, not vacuously 1
	s.CitationResolveRate = ratio(s.CitationsResolved, s.Citations, 1)
	s.FreshRate = ratio(s.FreshPages, s.Pages, 1)
	return s
}

// Passes reports whether the score clears the two floors: every code cite resolves
// at >= minResolve, and leaf coverage is at >= minCoverage. It is the predicate
// behind `fak wiki score --check`.
func (s Score) Passes(minResolve, minCoverage float64) bool {
	return s.CitationResolveRate >= minResolve && s.LeafCoverage >= minCoverage
}

// countCitations counts the well-formed code citations in the markdown (the same
// `[path:line]` shape VerifyCitations resolves, filtered by the same looksLikePath
// prose guard). It is the denominator of the resolve rate.
func countCitations(md []byte) int {
	n := 0
	for _, m := range citationRE.FindAllStringSubmatch(string(md), -1) {
		if looksLikePath(strings.TrimSpace(m[1])) {
			n++
		}
	}
	return n
}

// ratio is num/den as a float, returning vacuous when den == 0. Keeping the
// empty-denominator convention in one place makes each rate's zero-case explicit at
// the call site.
func ratio(num, den int, vacuous float64) float64 {
	if den == 0 {
		return vacuous
	}
	return float64(num) / float64(den)
}
