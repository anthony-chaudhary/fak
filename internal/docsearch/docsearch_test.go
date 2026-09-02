// Tests for docsearch's curated doc-map grammar and discovery ranking: the
// ParseBullet link-line grammar, InlinePaths extraction, Load's dos.toml
// authority plus cross-source merge/dedupe, and SearchDocs' exact scoring,
// notes-deprecation ordering, and trigram fuzzy fallback.
package docsearch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseBullet(t *testing.T) {
	cases := []struct {
		name  string
		line  string
		title string
		path  string
		blurb string
		ok    bool
	}{
		{"em-dash blurb", "- [Gateway](docs/gateway.md) — the performance gate", "Gateway", "docs/gateway.md", "the performance gate", true},
		{"hyphen blurb", "* [Guide](README.md) - how to start", "Guide", "README.md", "how to start", true},
		{"no blurb", "- [Bare](docs/bare.md)", "Bare", "docs/bare.md", "", true},
		{"backticked title", "- [`Backticked`](docs/b.md)", "Backticked", "docs/b.md", "", true},
		{"plain prose is not a bullet", "Gateway lives in docs/gateway.md.", "", "", "", false},
		{"missing link target", "- [Title] no parens", "", "", "", false},
		{"numbered list is not a bullet", "1. [Item](docs/i.md)", "", "", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			title, path, blurb, ok := ParseBullet(tc.line)
			if ok != tc.ok || title != tc.title || path != tc.path || blurb != tc.blurb {
				t.Fatalf("ParseBullet(%q) = (%q, %q, %q, %v), want (%q, %q, %q, %v)",
					tc.line, title, path, blurb, ok, tc.title, tc.path, tc.blurb, tc.ok)
			}
		})
	}
}

func TestInlinePaths(t *testing.T) {
	cases := []struct {
		name string
		line string
		want []string
	}{
		{"inline code path", "configure it in `docs/inner/guide.md` first", []string{"docs/inner/guide.md"}},
		{"markdown link anchor stripped", "read [Other](docs/other.md#section) next", []string{"docs/other.md"}},
		{"no doc paths", "see the README for usage", nil},
		{"txt extension counts", "notes live in `docs/plan.txt`", []string{"docs/plan.txt"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := InlinePaths(tc.line)
			if strings.Join(got, "\x00") != strings.Join(tc.want, "\x00") {
				t.Fatalf("InlinePaths(%q) = %v, want %v", tc.line, got, tc.want)
			}
		})
	}
}

func TestLoadRequiresDosToml(t *testing.T) {
	root := t.TempDir()
	if _, err := Load(root); err == nil {
		t.Fatal("Load accepted a root with no dos.toml")
	}
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load rejected a root with dos.toml: %v", err)
	}
	if c.Root != root || len(c.Docs) != 0 {
		t.Fatalf("Load produced Root=%q with %d docs; want %q with 0 docs", c.Root, len(c.Docs), root)
	}
}

func TestLoadDocsMergesAcrossSources(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"INDEX.md":  "- [Gateway](docs/gateway.md) — the performance gate\n",
		"llms.txt":  "- [Gateway Guide](docs/gateway.md)\n- [Only In Llms](docs/only-llms.md) — blurb only here\n",
		"README.md": "Deeper context: [Linked Page](docs/linked.md) explains the seam.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c := LoadDocs(root)
	if c.Root != root {
		t.Fatalf("LoadDocs Root = %q, want %q", c.Root, root)
	}
	byPath := map[string]Doc{}
	for _, d := range c.Docs {
		if _, dup := byPath[d.Path]; dup {
			t.Fatalf("path %s appears in more than one doc entry: %+v", d.Path, c.Docs)
		}
		byPath[d.Path] = d
	}
	gateway, ok := byPath["docs/gateway.md"]
	if !ok {
		t.Fatalf("docs/gateway.md missing from catalog: %+v", c.Docs)
	}
	if gateway.Title != "Gateway" {
		t.Errorf("gateway title = %q, want first-seen %q", gateway.Title, "Gateway")
	}
	if gateway.Blurb != "the performance gate" {
		t.Errorf("gateway blurb = %q, want first non-empty %q", gateway.Blurb, "the performance gate")
	}
	if strings.Join(gateway.Sources, ",") != "INDEX.md,llms.txt" {
		t.Errorf("gateway sources = %v, want [INDEX.md llms.txt]", gateway.Sources)
	}
	if d, ok := byPath["docs/only-llms.md"]; !ok || strings.Join(d.Sources, ",") != "llms.txt" {
		t.Errorf("docs/only-llms.md entry = %+v, want sourced only from llms.txt", byPath["docs/only-llms.md"])
	}
	linked, ok := byPath["docs/linked.md"]
	if !ok || linked.Title != "linked" {
		t.Errorf("prose-linked docs/linked.md entry = %+v, want titled %q", linked, "linked")
	}
}

func searchCatalog() *Catalog {
	return &Catalog{
		Root: ".",
		Docs: []Doc{
			{Title: "Gateway", Path: "docs/gateway.md", Blurb: "the performance gate", Sources: []string{"INDEX.md"}},
			{Title: "Overview", Path: "docs/overview.md", Blurb: "the gateway map", Sources: []string{"INDEX.md"}},
			{Title: "Cache", Path: "docs/cache.md", Blurb: "cache eviction", Sources: []string{"INDEX.md"}},
			{Title: "Cache", Path: "docs/notes/cache.md", Blurb: "cache eviction", Sources: []string{"llms.txt"}},
			{Title: "Alpha", Path: "docs/alpha.md", Sources: []string{"INDEX.md"}},
		},
	}
}

func TestSearchDocsExactRanking(t *testing.T) {
	cases := []struct {
		name  string
		query string
		want  []string
	}{
		{
			name:  "multi-token coverage dominates",
			query: "gateway overview",
			want:  []string{"docs/overview.md", "docs/gateway.md"},
		},
		{
			name:  "single token score ordering",
			query: "alpha",
			want:  []string{"docs/alpha.md"},
		},
		{
			name:  "notes paths rank below canonical docs on ties",
			query: "cache eviction",
			want:  []string{"docs/cache.md", "docs/notes/cache.md"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := searchCatalog().SearchDocs(tc.query)
			paths := make([]string, len(got))
			for i, d := range got {
				paths[i] = d.Path
			}
			if strings.Join(paths, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("SearchDocs(%q) = %v, want %v", tc.query, paths, tc.want)
			}
			for _, d := range got {
				if d.Approx {
					t.Fatalf("exact match for %q was marked approximate: %+v", tc.query, d)
				}
			}
		})
	}
}

func TestSearchDocsFuzzyFallback(t *testing.T) {
	// "gatewayy" shares 5 of 6 trigrams with "gateway", so the trigram fallback
	// finds it while exact substring matching scores zero for every doc.
	got := searchCatalog().SearchDocs("gatewayy")
	if len(got) == 0 {
		t.Fatal("typo query returned no fuzzy fallback matches")
	}
	if !got[0].Approx {
		t.Fatalf("fuzzy fallback hit %q was not marked approximate", got[0].Path)
	}
	if got[0].Path != "docs/gateway.md" {
		t.Fatalf("fuzzy fallback top hit = %q, want docs/gateway.md", got[0].Path)
	}
}

func TestSearchDocsEmptyQuery(t *testing.T) {
	for _, query := range []string{"", "   ", "\t"} {
		if got := searchCatalog().SearchDocs(query); got != nil {
			t.Fatalf("SearchDocs(%q) = %v, want nil", query, got)
		}
	}
}
