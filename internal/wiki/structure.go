package wiki

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"strconv"
)

// Tree is the whole wiki structure: an ordered list of sections, each a list of
// pages. It is the deterministic projection DeepWiki spends an LLM+FAISS pass to
// re-infer (`page.tsx:746-795`); fak reads it straight from the self-index. The
// JSON shape is `wiki.json` — stable across runs for a fixed tree, so a diff of
// two wiki.json files is a real change in the repo's structure, never LLM noise.
type Tree struct {
	Repo     string    `json:"repo"`
	Sections []Section `json:"sections"`
}

// Section is one top-level grouping (Overview / System Architecture / Core
// Features …), adopting DeepWiki's fixed section taxonomy as the sectioning.
type Section struct {
	Title string `json:"title"`
	Pages []Page `json:"pages"`
}

// Page is one navigable page: a slug id, a human title, the section it sits
// under, a one-line summary, and RelevantFiles — the source globs the page is
// grounded in. For a leaf page RelevantFiles is the lane's DECLARED trees from
// dos.toml (ground truth), not files an embedding pass guessed were relevant.
type Page struct {
	ID            string   `json:"id"`
	Title         string   `json:"title"`
	Section       string   `json:"section"`
	Summary       string   `json:"summary,omitempty"`
	RelevantFiles []string `json:"relevant_files,omitempty"`
}

// topLevelDocs are the repo-root onboarding docs an Overview page is grounded in.
// Only the ones that actually resolve under the catalog root are cited, so the
// Overview page never carries a dangling RelevantFiles entry (the L3 invariant
// applied to the structure step itself).
var topLevelDocs = []string{"README.md", "AGENTS.md", "CLAUDE.md", "llms.txt", "CONTRIBUTING.md"}

// Structure projects the wiki section→page tree from the loaded self-index. It is
// a pure function of the catalog: same catalog in, byte-identical Tree out (pages
// are sorted, so ordering is not dos.toml-file-order dependent). No LLM.
//
// The taxonomy:
//
//   - Overview          — one Introduction page grounded in the root onboarding docs.
//   - System Architecture — one Lane & Tier Map page grounded in dos.toml + the tier map.
//   - Core Features     — one page PER LEAF, RelevantFiles = the leaf's declared trees.
//
// L2 (content) fills each page's prose; L4 pins a generated_at_sha; L5 adds the
// Mermaid graph; L6 the claims banner. L1 emits only the witnessed skeleton.
func Structure(cat *devindex.Catalog) Tree {
	t := Tree{Repo: repoName(cat.Root)}

	// Overview: the root onboarding docs that exist.
	overview := Page{
		ID:      "overview/introduction",
		Title:   "Introduction",
		Section: "Overview",
		Summary: "Repo entry point: what fak is and where to start.",
	}
	for _, d := range topLevelDocs {
		if fileExists(cat.Root, d) {
			overview.RelevantFiles = append(overview.RelevantFiles, d)
		}
	}
	t.Sections = append(t.Sections, Section{Title: "Overview", Pages: []Page{overview}})

	// System Architecture: the lane/tier map itself.
	arch := Page{
		ID:            "architecture/lane-tier-map",
		Title:         "Lane & Tier Map",
		Section:       "System Architecture",
		Summary:       laneSummary(len(cat.Leaves)),
		RelevantFiles: archFiles(cat.Root),
	}
	t.Sections = append(t.Sections, Section{Title: "System Architecture", Pages: []Page{arch}})

	// Core Features: one page per declared leaf, grounded in the leaf's trees.
	pages := make([]Page, 0, len(cat.Leaves))
	for _, lf := range cat.Leaves {
		pages = append(pages, Page{
			ID:            "core-features/" + lf.Name,
			Title:         lf.Name,
			Section:       "Core Features",
			Summary:       lf.Desc,
			RelevantFiles: splitTrees(lf.Tree),
		})
	}
	sort.Slice(pages, func(i, j int) bool { return pages[i].Title < pages[j].Title })
	t.Sections = append(t.Sections, Section{Title: "Core Features", Pages: pages})

	return t
}

// PageCount is the total number of pages across all sections — the coverage
// numerator L7 (`fak wiki score`) divides by the declared-leaf count.
func (t Tree) PageCount() int {
	n := 0
	for _, s := range t.Sections {
		n += len(s.Pages)
	}
	return n
}

// splitTrees turns a leaf's comma-joined Tree glob string ("internal/x/**,
// cmd/fak/y*.go") into the individual globs used as RelevantFiles.
func splitTrees(tree string) []string {
	var out []string
	for _, g := range strings.Split(tree, ",") {
		if g = strings.TrimSpace(g); g != "" {
			out = append(out, g)
		}
	}
	return out
}

// archFiles are the architecture-defining files that exist under root: the lane
// taxonomy (dos.toml) and the tier map (architest_test.go).
func archFiles(root string) []string {
	var out []string
	for _, f := range []string{"dos.toml", "internal/architest/architest_test.go"} {
		if fileExists(root, f) {
			out = append(out, f)
		}
	}
	return out
}

func laneSummary(nLeaves int) string {
	return "The declared lane taxonomy and import-tier map: " + strconv.Itoa(nLeaves) + " leaves."
}
