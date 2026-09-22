// Package docsearch loads and searches the repository's curated documentation map.
// It is deliberately independent of devindex so runtime discovery can use the
// documentation surface without importing repository-development tooling.
//
// Invariant: doc search ranking is fail-closed and deterministic across all catalog queries.
// Precondition: empty or whitespace queries return nil without modifying catalog state.
// Guard: missing documentation sources degrade safely to an empty catalog.
package docsearch

import (
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/trigram"
)

// Doc is one entry of the curated documentation map.
type Doc struct {
	Title   string   `json:"title"`
	Path    string   `json:"path"`
	Blurb   string   `json:"blurb,omitempty"`
	Sources []string `json:"sources,omitempty"`
	Approx  bool     `json:"approx,omitempty"`
	// Discovered marks a doc found by walking the tree rather than parsed from a
	// curated source file, so a caller can tell a curated row from a discovered one.
	Discovered bool `json:"discovered,omitempty"`
}

// Catalog is the narrow documentation authority shared by runtime discovery and
// the development index.
type Catalog struct {
	Root string `json:"root"`
	Docs []Doc  `json:"docs"`
}

// Load preserves the runtime discovery contract: root must identify a fak
// repository (by its dos.toml), while missing documentation sources degrade to
// an empty catalog.
func Load(root string) (*Catalog, error) {
	if _, err := os.ReadFile(filepath.Join(root, "dos.toml")); err != nil {
		return nil, err
	}
	return LoadDocs(root), nil
}

// LoadDocs reads only the documentation sources. Development indexing already
// validates dos.toml as its taxonomy authority before calling this helper.
// Tree discovery runs AFTER the curated sources so curated rows keep precedence;
// a missing docs/ tree is a no-op, keeping bare temp roots valid.
func LoadDocs(root string) *Catalog {
	c := &Catalog{Root: root}
	for _, source := range []string{"INDEX.md", "llms.txt", "README.md", "AGENTS.md"} {
		if data, err := os.ReadFile(filepath.Join(root, source)); err == nil {
			c.parse(source, string(data))
		}
	}
	for _, discovered := range DiscoverDocs(root) {
		if c.hasPath(discovered.Path) {
			continue
		}
		c.Docs = append(c.Docs, discovered)
	}
	return c
}

// hasPath reports whether the catalog already holds a doc at the normalized path.
func (c *Catalog) hasPath(path string) bool {
	want := normPath(path)
	for i := range c.Docs {
		if normPath(c.Docs[i].Path) == want {
			return true
		}
	}
	return false
}

// h1RE matches an ATX H1 heading and captures the heading text.
var h1RE = regexp.MustCompile(`^#\s+(.+)$`)

// discoverHeadBytes bounds the per-file read used to find an H1. Large notes are
// common under docs/, so discovery never reads a whole file.
const discoverHeadBytes = 64 << 10

// DiscoverDocs walks ONLY the top level of root/docs (not subdirectories; docs/
// holds thousands of dated notes and deeper trees are out of scope) and returns
// every regular .md/.txt file directly inside it as a Doc. It is deterministic
// (sorted by Path) and degrades quietly: an absent or unreadable docs/ tree
// yields no docs, never an error.
//
// A discovered row's Title is the file's humanized NAME (pathTitle, e.g.
// "born bottlenecks" for born-bottlenecks.md) and its Blurb is the file's first
// ATX H1 within the first 64 KiB, so a query matches on either the filename or
// the heading. This mirrors how the curated linker titles (pathTitle) and
// link-line blurbs are already surfaced, so discovery and curation score alike —
// only the provenance tier in SearchDocs orders a curated row first.
func DiscoverDocs(root string) []Doc {
	entries, err := os.ReadDir(filepath.Join(root, "docs"))
	if err != nil {
		return nil
	}
	var docs []Doc
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".md" && ext != ".txt" {
			continue
		}
		path := "docs/" + filepath.ToSlash(entry.Name())
		h1 := discoverH1(filepath.Join(root, "docs", entry.Name()))
		if h1 == "" {
			h1 = pathTitle(path)
		}
		docs = append(docs, Doc{
			Title:      pathTitle(path),
			Path:       path,
			Blurb:      h1,
			Sources:    []string{"tree"},
			Discovered: true,
		})
	}
	sort.Slice(docs, func(i, j int) bool { return docs[i].Path < docs[j].Path })
	return docs
}

// discoverH1 returns the file's first ATX H1 heading text, or "" when the head
// holds no H1 or the file cannot be read. It reads at most discoverHeadBytes so a
// large note never costs a full read.
func discoverH1(file string) string {
	f, err := os.Open(file)
	if err != nil {
		return ""
	}
	defer f.Close()
	head, err := io.ReadAll(io.LimitReader(f, discoverHeadBytes))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(head), "\n") {
		if m := h1RE.FindStringSubmatch(strings.TrimRight(line, "\r")); m != nil {
			if title := strings.TrimSpace(strings.TrimLeft(m[1], "#")); title != "" {
				return title
			}
		}
	}
	return ""
}

var docLineRE = regexp.MustCompile(`^\s*[-*]\s*\[(.+?)\]\(([^)]+)\)\s*(?:[—–-]\s*(.*))?$`)
var inlineCodePathRE = regexp.MustCompile("`((?:docs/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\\.(?:md|txt))`")
var anyMarkdownLinkRE = regexp.MustCompile(`\[[^]]+\]\(([^)]+\.(?:md|txt))(?:#[^)]+)?\)`)

func (c *Catalog) parse(source, text string) {
	seen := map[string]bool{}
	for _, raw := range strings.Split(text, "\n") {
		title, path, blurb, ok := ParseBullet(raw)
		if !ok {
			for _, linked := range markdownLinkedDocs(raw, source) {
				c.merge(linked)
			}
			continue
		}
		if path == "" || seen[title+"\x00"+path] {
			continue
		}
		seen[title+"\x00"+path] = true
		merged := false
		for i := range c.Docs {
			if normPath(c.Docs[i].Path) != normPath(path) {
				continue
			}
			if !contains(c.Docs[i].Sources, source) {
				c.Docs[i].Sources = append(c.Docs[i].Sources, source)
			}
			if c.Docs[i].Blurb == "" && blurb != "" {
				c.Docs[i].Blurb = blurb
			}
			merged = true
			break
		}
		if !merged {
			c.Docs = append(c.Docs, Doc{Title: title, Path: path, Blurb: blurb, Sources: []string{source}})
		}
		for _, extraPath := range InlinePaths(raw) {
			c.merge(Doc{Title: pathTitle(extraPath), Path: extraPath, Blurb: blurb, Sources: []string{source}})
		}
	}
}

// ParseBullet parses the curated Markdown doc-map grammar used by both loading
// and the development index's committed-tree freshness check.
func ParseBullet(line string) (title, path, blurb string, ok bool) {
	m := docLineRE.FindStringSubmatch(line)
	if m == nil {
		return "", "", "", false
	}
	return strings.TrimSpace(strings.ReplaceAll(m[1], "`", "")), strings.TrimSpace(m[2]), strings.TrimSpace(m[3]), true
}

func markdownLinkedDocs(line, source string) []Doc {
	var docs []Doc
	for _, match := range anyMarkdownLinkRE.FindAllStringSubmatch(line, -1) {
		path := match[1]
		if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
			continue
		}
		docs = append(docs, Doc{Title: pathTitle(path), Path: path, Blurb: strings.TrimSpace(strings.ReplaceAll(line, "`", "")), Sources: []string{source}})
	}
	return docs
}

// InlinePaths returns Markdown-linked and inline-code documentation paths.
func InlinePaths(line string) []string {
	var paths []string
	for _, match := range inlineCodePathRE.FindAllStringSubmatch(line, -1) {
		paths = append(paths, match[1])
	}
	for _, match := range anyMarkdownLinkRE.FindAllStringSubmatch(line, -1) {
		paths = append(paths, match[1])
	}
	return paths
}

func pathTitle(path string) string {
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	return strings.ReplaceAll(strings.ReplaceAll(base, "-", " "), "_", " ")
}

func (c *Catalog) merge(candidate Doc) {
	for i := range c.Docs {
		if normPath(c.Docs[i].Path) != normPath(candidate.Path) {
			continue
		}
		for _, source := range candidate.Sources {
			if !contains(c.Docs[i].Sources, source) {
				c.Docs[i].Sources = append(c.Docs[i].Sources, source)
			}
		}
		return
	}
	c.Docs = append(c.Docs, candidate)
}

// SearchDocs returns lexically ranked documentation matches, falling back to
// trigram near-matches only when exact scoring yields no result.
func (c *Catalog) SearchDocs(query string) []Doc {
	toks := tokens(query)
	if len(toks) == 0 {
		return nil
	}
	type scored struct {
		d        Doc
		s        int
		coverage int
		tier     int
	}
	var hits []scored
	for _, d := range c.Docs {
		title, path, blurb := strings.ToLower(d.Title), strings.ToLower(d.Path), strings.ToLower(d.Blurb)
		score, coverage := 0, 0
		for _, tk := range toks {
			matched := false
			if strings.Contains(title, tk) {
				score += 3
				matched = true
			}
			if strings.Contains(path, tk) {
				score += 2
				matched = true
			}
			if strings.Contains(blurb, tk) {
				score++
				matched = true
			}
			if matched {
				coverage++
			}
		}
		if score == 0 {
			continue
		}
		if len(toks) > 1 {
			score += canonicalBonus(d)
		}
		// Discovered rows rank BELOW curated ones. A constant subtraction cannot
		// express that: a discovered row matching more fields than a curated row
		// would out-grow any fixed penalty. So provenance is a categorical, primary
		// sort key (curated tier 0, discovered tier 1) and the score ranks within a
		// tier. The per-token penalty orders discovered rows among themselves.
		if d.Discovered {
			score -= discoveredPenalty * coverage
		}
		// Admit on score>=0, not score>0: the per-token penalty can land a genuine
		// discovered match on exactly 0 when its only signal is the H1 blurb (the
		// discovered shape is Title=filename, Blurb=H1, so an H1-only query scores
		// raw 1 - 1 = 0). A curated row can never score <0 here (its raw score is
		// positive and it takes no penalty), so this relaxation admits only
		// discovered rows a stricter test would have silently dropped — the precise
		// "unlisted doc is invisible" failure #1656 exists to remove.
		if score >= 0 {
			hits = append(hits, scored{d: d, s: score, coverage: coverage, tier: provenanceTier(d)})
		}
	}
	if len(hits) == 0 {
		return c.fuzzy(toks)
	}
	sort.SliceStable(hits, func(i, j int) bool {
		// Provenance first: every curated row ranks above every discovered row,
		// regardless of score. Curated rows are all tier 0, so this clause is inert
		// for the pre-existing curated-only ranking (their relative order is decided
		// below, unchanged); it only demotes discovered rows wholesale.
		if hits[i].tier != hits[j].tier {
			return hits[i].tier < hits[j].tier
		}
		if len(toks) > 1 {
			if hits[i].coverage != hits[j].coverage {
				return hits[i].coverage > hits[j].coverage
			}
			iNote := strings.HasPrefix(strings.ToLower(normPath(hits[i].d.Path)), "docs/notes/")
			jNote := strings.HasPrefix(strings.ToLower(normPath(hits[j].d.Path)), "docs/notes/")
			if iNote != jNote {
				return !iNote
			}
		}
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		if len(toks) > 1 {
			iTitle, jTitle := strings.ToLower(hits[i].d.Title), strings.ToLower(hits[j].d.Title)
			if iTitle != jTitle {
				return iTitle < jTitle
			}
		}
		if hits[i].d.Title != hits[j].d.Title {
			return hits[i].d.Title < hits[j].d.Title
		}
		// Path is the final, total tiebreak so the ordering is deterministic even
		// for a hand-built catalog holding two rows with an identical title.
		return normPath(hits[i].d.Path) < normPath(hits[j].d.Path)
	})
	out := make([]Doc, len(hits))
	for i, hit := range hits {
		out[i] = hit.d
	}
	return out
}

// provenanceTier is the primary ranking key SearchDocs sorts on: 0 for a row
// parsed from a curated source file, 1 for a row discovered by walking the tree.
// Curated rows therefore always outrank discovered rows for the same query, as a
// categorical fact rather than a subtraction any score can out-grow.
func provenanceTier(d Doc) int {
	if d.Discovered {
		return 1
	}
	return 0
}

// discoveredPenalty is the per-matched-token subtraction applied to a discovered
// row's score so its own matches still rank among themselves. It is deliberately
// small — 1, the weight of a blurb hit — so it never drops a genuine match: the
// worst case (an H1-only match, raw 1 - 1 = 0) is still admitted by the `>= 0`
// test, and a filename match scores 3 - 1 = 2. Cross-provenance precedence is
// enforced by provenanceTier, NOT by this constant.
const discoveredPenalty = 1

func canonicalBonus(d Doc) int {
	bonus := 2 * len(d.Sources)
	p := strings.ToLower(normPath(d.Path))
	switch {
	case strings.HasPrefix(p, "docs/notes/"), strings.HasPrefix(p, "docs/_witnesses/"), strings.HasPrefix(p, "docs/generated/"):
		bonus -= 3
	case !strings.Contains(p, "/"):
		bonus += 3
	case strings.Count(p, "/") == 1:
		bonus += 2
	}
	return bonus
}

const fuzzyThreshold = 0.34

func (c *Catalog) fuzzy(toks []string) []Doc {
	type scored struct {
		d Doc
		s float64
	}
	var hits []scored
	for _, d := range c.Docs {
		s := fuzzyScore(toks, weightedField{strings.ToLower(d.Title), 3}, weightedField{strings.ToLower(d.Path), 2}, weightedField{strings.ToLower(d.Blurb), 1})
		if s > 0 {
			d.Approx = true
			hits = append(hits, scored{d, s})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		// Provenance first, exactly as the exact-match path orders: a curated row
		// outranks a discovered row even in the near-miss fallback, so a typo query
		// cannot let a tree-discovered doc jump a curated one.
		if it, jt := provenanceTier(hits[i].d), provenanceTier(hits[j].d); it != jt {
			return it < jt
		}
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		if hits[i].d.Title != hits[j].d.Title {
			return hits[i].d.Title < hits[j].d.Title
		}
		return normPath(hits[i].d.Path) < normPath(hits[j].d.Path)
	})
	out := make([]Doc, len(hits))
	for i, hit := range hits {
		out[i] = hit.d
	}
	return out
}

type weightedField struct {
	text   string
	weight int
}

func fuzzyScore(toks []string, fields ...weightedField) float64 {
	best := 0.0
	for _, field := range fields {
		for _, word := range strings.FieldsFunc(field.text, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
			for _, tk := range toks {
				sim := trigram.Similarity(tk, word)
				if sim >= fuzzyThreshold && sim*float64(field.weight) > best {
					best = sim * float64(field.weight)
				}
			}
		}
	}
	return best
}

func tokens(query string) []string { return strings.Fields(strings.ToLower(query)) }
func normPath(path string) string {
	return strings.TrimPrefix(strings.ReplaceAll(path, "\\", "/"), "./")
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
