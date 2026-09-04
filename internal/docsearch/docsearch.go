// Package docsearch loads and searches the repository's curated documentation map.
// It is deliberately independent of devindex so runtime discovery can use the
// documentation surface without importing repository-development tooling.
//
// Invariant: doc search ranking is fail-closed and deterministic across all catalog queries.
// Precondition: empty or whitespace queries return nil without modifying catalog state.
// Guard: missing documentation sources degrade safely to an empty catalog.
package docsearch

import (
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
func LoadDocs(root string) *Catalog {
	c := &Catalog{Root: root}
	for _, source := range []string{"INDEX.md", "llms.txt", "README.md", "AGENTS.md"} {
		if data, err := os.ReadFile(filepath.Join(root, source)); err == nil {
			c.parse(source, string(data))
		}
	}
	return c
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
		if score > 0 {
			if len(toks) > 1 {
				score += canonicalBonus(d)
			}
			hits = append(hits, scored{d: d, s: score, coverage: coverage})
		}
	}
	if len(hits) == 0 {
		return c.fuzzy(toks)
	}
	sort.SliceStable(hits, func(i, j int) bool {
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
		return hits[i].d.Title < hits[j].d.Title
	})
	out := make([]Doc, len(hits))
	for i, hit := range hits {
		out[i] = hit.d
	}
	return out
}

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
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].d.Title < hits[j].d.Title
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
