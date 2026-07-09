// Package agentsindex is a stdlib-only, tier-1 view over AGENTS.md (issue #3535,
// epic #3229). AGENTS.md is ~41.7 KB (~10.4k EST. tokens); a dispatched worker that
// Reads it whole on turn 1 pays that entire tax before doing any work. This package
// parses the file into a deterministic section model so a worker can hold a compact
// resident TOC (<= agentsindex.TOCBudgetTokens) and fault in only the one section a
// task needs — the same Read-avoidance idea the cold-tool-deferral levers (#3231/#3232)
// apply to tool schemas, here applied to a big always-read doc.
//
// It changes NO AGENTS.md content — only its load discipline. The parse is byte-exact:
// a section's Raw is sliced verbatim from the source bytes (CRLF preserved), so a
// `--section <slug>` fetch and the `--full` escape hatch are byte-identical to what a
// whole Read would have returned. Header detection is fence-aware (a line-start '#'
// inside a ``` / ~~~ block is never a heading) so a commented script line can never
// fabricate a section. ATX headers only — AGENTS.md has no setext headers (documented
// limitation).
//
// Tier-1 contract (internal/architest): stdlib-only, off the hot path, no fak imports.
package agentsindex

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// FileName is the orientation doc this package indexes, resolved under a repo root.
const FileName = "AGENTS.md"

// Section is one ATX-headed span of AGENTS.md. A level-2 ('##') section's Raw includes
// its nested level-3 ('###') spans; a level-3 section's Raw is just its own span. The
// two therefore deliberately overlap in the source bytes — SectionBySlug("parent")
// returns the whole subtree, SectionBySlug("child") returns only the child.
type Section struct {
	Slug      string // short kebab slug derived from the heading head
	Title     string // full heading text, leading/trailing '#' stripped
	Level     int    // ATX level: 2 for '##', 3 for '###', …
	Line      int    // 1-based source line of the heading
	Raw       string // byte-exact body span, heading line through (exclusive) the next same-or-shallower heading
	EstTokens int    // (len(Raw)+3)/4 — ESTIMATED, the house per-section price
}

// Doc is a parsed AGENTS.md: the whole file bytes plus its level>=2 sections.
type Doc struct {
	Raw      []byte
	Sections []Section
}

// EstTokens is the ESTIMATED whole-file price — the cost a full turn-1 Read pays.
func (d *Doc) EstTokens() int { return EstTokensOf(string(d.Raw)) }

// EstTokensOf is the house 4-bytes-per-token estimate, matching the per-section price
// and the mcpfootprint floor estimator's provenance ("ESTIMATED", never a live counter).
func EstTokensOf(s string) int { return (len(s) + 3) / 4 }

// Load reads root/AGENTS.md and parses it. A caller with an explicit file path uses
// ParseFile instead.
func Load(root string) (*Doc, error) { return ParseFile(filepath.Join(root, FileName)) }

// ParseFile reads and parses an explicit AGENTS.md path.
func ParseFile(path string) (*Doc, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b), nil
}

// Parse builds the section model from raw AGENTS.md bytes. Deterministic and total:
// any bytes parse (a file with no headers yields zero sections, which the callers'
// budget gates treat as fail-closed).
func Parse(raw []byte) *Doc {
	offsets, lines := splitLines(raw)

	type hdr struct {
		level   int
		title   string
		lineIdx int
	}
	var hdrs []hdr
	inFence := false
	for i, ln := range lines {
		if isFence(ln) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		if lvl, title, ok := atxHeader(ln); ok {
			hdrs = append(hdrs, hdr{lvl, title, i})
		}
	}

	doc := &Doc{Raw: raw}
	seen := map[string]int{}
	for hi, h := range hdrs {
		if h.level < 2 { // the level-1 file title / preamble is reachable only via --full
			continue
		}
		start := offsets[h.lineIdx]
		end := len(raw)
		for hj := hi + 1; hj < len(hdrs); hj++ {
			if hdrs[hj].level <= h.level { // a same-or-shallower heading closes this span
				end = offsets[hdrs[hj].lineIdx]
				break
			}
		}
		body := raw[start:end]

		slug := slugify(slugHead(h.title))
		if slug == "" {
			slug = "section"
		}
		if n, dup := seen[slug]; dup {
			seen[slug] = n + 1
			slug = fmt.Sprintf("%s-%d", slug, n+1)
		} else {
			seen[slug] = 1
		}

		doc.Sections = append(doc.Sections, Section{
			Slug:      slug,
			Title:     h.title,
			Level:     h.level,
			Line:      h.lineIdx + 1,
			Raw:       string(body),
			EstTokens: EstTokensOf(string(body)),
		})
	}
	return doc
}

// SectionBySlug returns the section with an exact slug match; ok=false on a miss (never
// a fuzzy guess — the CLI surfaces near-miss suggestions from Search separately).
func (d *Doc) SectionBySlug(slug string) (Section, bool) {
	for _, s := range d.Sections {
		if s.Slug == slug {
			return s, true
		}
	}
	return Section{}, false
}

// Search ranks sections against a free-text query with a devindex-style lexical score:
// a term in the heading weighs more than a term in the body. Deterministic — ties break
// by source order — and returns only positive-scoring sections.
func (d *Doc) Search(query string) []Section {
	terms := tokenize(query)
	if len(terms) == 0 {
		return nil
	}
	type scored struct {
		s     Section
		score int
	}
	var ranked []scored
	for _, s := range d.Sections {
		title := strings.ToLower(s.Title)
		body := strings.ToLower(s.Raw)
		sc := 0
		for _, t := range terms {
			if strings.Contains(title, t) {
				sc += 3
			}
			if strings.Contains(body, t) {
				sc++
			}
		}
		if sc > 0 {
			ranked = append(ranked, scored{s, sc})
		}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].s.Line < ranked[j].s.Line
	})
	out := make([]Section, len(ranked))
	for i, r := range ranked {
		out[i] = r.s
	}
	return out
}

// FindRoot walks up from start looking for a dos.toml, returning the repo root and ok.
// Mirrors the devindex freshness-test root walk so the live-tree gates can locate the
// real AGENTS.md without importing devindex (tier-1 stays fak-free).
func FindRoot(start string) (string, bool) {
	dir := start
	for i := 0; i < 12; i++ {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", false
}

// --- lexers -------------------------------------------------------------------

// splitLines returns each line's start byte offset and its text with the trailing
// CR/LF stripped. Offsets index into raw so a section span can be sliced byte-exact
// (Raw keeps the original CRLF); the text is only used for header/fence detection.
func splitLines(raw []byte) (offsets []int, lines []string) {
	off := 0
	for off < len(raw) {
		nl := bytes.IndexByte(raw[off:], '\n')
		end := len(raw)
		if nl >= 0 {
			end = off + nl
		}
		line := bytes.TrimSuffix(raw[off:end], []byte("\r"))
		offsets = append(offsets, off)
		lines = append(lines, string(line))
		if nl < 0 {
			break
		}
		off = end + 1
	}
	return offsets, lines
}

// isFence reports whether a line opens/closes a fenced code block (``` or ~~~), allowing
// up to three leading spaces per CommonMark.
func isFence(line string) bool {
	t := strings.TrimLeft(line, " ")
	return strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~")
}

// atxHeader detects an ATX heading and returns its level and cleaned title. Requires a
// space after the '#' run (so "#foo" and "###" alone are not headings) and strips a
// trailing '#' closing sequence.
func atxHeader(line string) (level int, title string, ok bool) {
	i := 0
	for i < len(line) && i < 4 && line[i] == ' ' {
		i++
	}
	h := 0
	for i < len(line) && line[i] == '#' {
		h++
		i++
	}
	if h < 1 || h > 6 {
		return 0, "", false
	}
	if i >= len(line) || line[i] != ' ' {
		return 0, "", false
	}
	title = strings.TrimSpace(line[i:])
	title = strings.TrimSpace(strings.TrimRight(title, "#"))
	if title == "" {
		return 0, "", false
	}
	return h, title, true
}

// slugHead returns the heading text up to the first '(', ':', or dash-delimiter, so a
// long parenthetical/qualified title yields a short stable slug.
func slugHead(title string) string {
	cut := len(title)
	for _, delim := range []string{"(", ":", "—", "–"} { // '(' ':' em-dash en-dash
		if idx := strings.Index(title, delim); idx >= 0 && idx < cut {
			cut = idx
		}
	}
	return strings.TrimSpace(title[:cut])
}

// slugify lowercases and collapses every non-alphanumeric run to a single '-', trimming
// leading/trailing dashes. Unicode-safe (an em-dash or accented char becomes a dash).
func slugify(s string) string {
	var b strings.Builder
	prevDash := false
	for _, r := range strings.ToLower(s) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			prevDash = false
		default:
			if !prevDash && b.Len() > 0 {
				b.WriteByte('-')
				prevDash = true
			}
		}
	}
	return strings.TrimRight(b.String(), "-")
}

// tokenize splits a query into lowercase alphanumeric terms.
func tokenize(s string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}
