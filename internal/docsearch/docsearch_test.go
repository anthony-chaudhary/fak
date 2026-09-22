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

func TestDiscoverDocsFindsUnlistedDoc(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	body := "# Born Bottlenecks\n\nA doc that is never linked from a curated source.\n"
	if err := os.WriteFile(filepath.Join(root, "docs", "only-on-disk.md"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got := DiscoverDocs(root)
	if len(got) != 1 {
		t.Fatalf("DiscoverDocs returned %d docs, want 1: %+v", len(got), got)
	}
	d := got[0]
	if d.Path != "docs/only-on-disk.md" {
		t.Errorf("Path = %q, want docs/only-on-disk.md", d.Path)
	}
	if d.Title != "only on disk" {
		t.Errorf("Title = %q, want humanized filename %q", d.Title, "only on disk")
	}
	if d.Blurb != "Born Bottlenecks" {
		t.Errorf("Blurb = %q, want H1 %q", d.Blurb, "Born Bottlenecks")
	}
	if !d.Discovered {
		t.Errorf("Discovered = false, want true")
	}
	if strings.Join(d.Sources, ",") != "tree" {
		t.Errorf("Sources = %v, want [tree]", d.Sources)
	}
}

func TestDiscoverDocsIsShallowAndBounded(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs", "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "top.md"), []byte("# Top\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "note.txt"), []byte("# Note\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "skip.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "nested", "deep.md"), []byte("# Deep\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := DiscoverDocs(root)
	if len(got) != 2 {
		t.Fatalf("DiscoverDocs = %+v, want only the two top-level .md/.txt files", got)
	}
	if got[0].Path != "docs/note.txt" || got[1].Path != "docs/top.md" {
		t.Fatalf("DiscoverDocs paths = %q,%q, want sorted docs/note.txt,docs/top.md", got[0].Path, got[1].Path)
	}
}

func TestDiscoverDocsMissingTreeIsNil(t *testing.T) {
	if got := DiscoverDocs(t.TempDir()); got != nil {
		t.Fatalf("DiscoverDocs on a root with no docs/ = %+v, want nil", got)
	}
}

func TestLoadDocsIncludesUnlistedDoc(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"INDEX.md":  "- [Gateway](docs/gateway.md) — the performance gate\n",
		"llms.txt":  "- [Only In Llms](docs/only-llms.md) — blurb only here\n",
		"README.md": "Deeper context: [Linked Page](docs/linked.md) explains the seam.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "unlisted-observability.md"), []byte("# Observability Ledger\n\nbody\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c := LoadDocs(root)
	byPath := map[string]Doc{}
	for _, d := range c.Docs {
		if _, dup := byPath[d.Path]; dup {
			t.Fatalf("path %s appears twice: %+v", d.Path, c.Docs)
		}
		byPath[d.Path] = d
	}
	unlisted, ok := byPath["docs/unlisted-observability.md"]
	if !ok {
		t.Fatalf("docs/unlisted-observability.md missing from catalog: %+v", c.Docs)
	}
	if !unlisted.Discovered || unlisted.Title != "unlisted observability" || unlisted.Blurb != "Observability Ledger" {
		t.Errorf("unlisted doc = %+v, want discovered filename-title %q with H1 blurb %q", unlisted, "unlisted observability", "Observability Ledger")
	}
	gateway, ok := byPath["docs/gateway.md"]
	if !ok {
		t.Fatalf("curated docs/gateway.md missing from catalog: %+v", c.Docs)
	}
	if gateway.Discovered {
		t.Errorf("curated gateway doc marked Discovered: %+v", gateway)
	}
}

func TestLoadDocsDeduplicatesCuratedPath(t *testing.T) {
	root := t.TempDir()
	body := "# Gateway Contract\n"
	if err := os.WriteFile(filepath.Join(root, "INDEX.md"), []byte("- [Gateway](docs/gateway.md) — the real blurb\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "gateway.md"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	c := LoadDocs(root)
	n := 0
	for _, d := range c.Docs {
		if d.Path == "docs/gateway.md" {
			n++
			if d.Discovered {
				t.Errorf("curated path was re-added as discovered: %+v", d)
			}
			if d.Blurb != "the real blurb" {
				t.Errorf("curated blurb was overwritten by discovery: %q", d.Blurb)
			}
		}
	}
	if n != 1 {
		t.Fatalf("docs/gateway.md appears %d times, want 1 (curated dedup, discovery must not duplicate)", n)
	}
}

func TestSearchDocsReturnsUnlistedDoc(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "INDEX.md"), []byte("- [Gateway](docs/gateway.md) — the performance gate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"unlisted-observability.md": "# Observability Ledger\n\nbody\n",
		"born-bottlenecks.md":       "# Born Bottlenecks\n\nbody\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, "docs", name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c := LoadDocs(root)

	got := c.SearchDocs("observability")
	if len(got) == 0 || got[0].Path != "docs/unlisted-observability.md" {
		t.Fatalf("SearchDocs(observability) = %+v, want docs/unlisted-observability.md first", got)
	}
	if got[0].Approx {
		t.Errorf("filename match on an unlisted doc fell through to the fuzzy fallback: %+v", got[0])
	}

	multi := c.SearchDocs("born bottlenecks")
	found := false
	for _, d := range multi {
		if d.Path == "docs/born-bottlenecks.md" {
			found = true
		}
	}
	if !found {
		t.Fatalf("SearchDocs(born bottlenecks) did not return docs/born-bottlenecks.md: %+v", multi)
	}
}

func TestSearchDocsCuratedPrecedenceOverDiscovered(t *testing.T) {
	c := &Catalog{
		Root: ".",
		Docs: []Doc{
			{Title: "Gateway", Path: "docs/notes/gateway.md", Sources: []string{"INDEX.md"}},
			{Title: "discovered gateway", Path: "docs/discovered-gateway.md", Sources: []string{"tree"}, Discovered: true},
		},
	}
	got := c.SearchDocs("gateway")
	if len(got) < 2 {
		t.Fatalf("SearchDocs(gateway) = %+v, want both curated and discovered rows", got)
	}
	if got[0].Path != "docs/notes/gateway.md" {
		t.Fatalf("SearchDocs(gateway) top hit = %q, want curated docs/notes/gateway.md", got[0].Path)
	}
	if got[1].Path != "docs/discovered-gateway.md" {
		t.Fatalf("SearchDocs(gateway) second hit = %q, want discovered docs/discovered-gateway.md", got[1].Path)
	}
}

// TestSearchDocsPrecedenceNotOutgrownByExtraFields is the adversarial regression
// for the falsified constant-penalty design: a discovered row matching MORE fields
// (title+path+blurb) must not outrank a curated row at equal token coverage whose
// score is lower. Provenance is the primary key, so curated still wins.
func TestSearchDocsPrecedenceNotOutgrownByExtraFields(t *testing.T) {
	c := &Catalog{
		Root: ".",
		Docs: []Doc{
			{Title: "memory skill guide", Path: "docs/notes/memory-skill-guide.md", Blurb: "memory skill", Sources: []string{"INDEX.md"}},
			{Title: "skill memory", Path: "docs/skill-memory.md", Blurb: "skill memory", Sources: []string{"tree"}, Discovered: true},
		},
	}
	got := c.SearchDocs("skill memory")
	if len(got) < 2 {
		t.Fatalf("SearchDocs(skill memory) = %+v, want both rows", got)
	}
	if got[0].Path != "docs/notes/memory-skill-guide.md" {
		t.Fatalf("top hit = %q, want curated docs/notes/memory-skill-guide.md ahead of the higher-scoring discovered row", got[0].Path)
	}
}

func TestSearchDocsPrecedenceSingleTokenBlurbOnly(t *testing.T) {
	c := &Catalog{
		Root: ".",
		Docs: []Doc{
			{Title: "zeta", Path: "docs/notes/zeta.md", Blurb: "gateway related", Sources: []string{"INDEX.md"}},
			{Title: "gateway", Path: "docs/gateway.md", Blurb: "gateway", Sources: []string{"tree"}, Discovered: true},
		},
	}
	got := c.SearchDocs("gateway")
	if len(got) < 2 {
		t.Fatalf("SearchDocs(gateway) = %+v, want both rows", got)
	}
	if got[0].Path != "docs/notes/zeta.md" {
		t.Fatalf("top hit = %q, want curated blurb-only docs/notes/zeta.md first", got[0].Path)
	}
}

// TestSearchDocsPrecedenceHoldsInFuzzyFallback pins the second adversarial finding:
// the typo fallback bypassed the tier clause, so a discovered row could jump a
// curated one on a near-miss query.
func TestSearchDocsPrecedenceHoldsInFuzzyFallback(t *testing.T) {
	c := &Catalog{
		Root: ".",
		Docs: []Doc{
			{Title: "zeta guide", Path: "docs/notes/gateway-guide.md", Blurb: "gateway usage", Sources: []string{"INDEX.md"}},
			{Title: "gateway", Path: "docs/gateway.md", Blurb: "gateway", Sources: []string{"tree"}, Discovered: true},
		},
	}
	got := c.SearchDocs("gatway")
	if len(got) < 2 {
		t.Fatalf("SearchDocs(gatway) = %+v, want both near-miss rows", got)
	}
	if got[0].Discovered {
		t.Fatalf("fuzzy fallback top hit = %q (discovered), want the curated row first", got[0].Path)
	}
	if !got[0].Approx {
		t.Errorf("fuzzy hit %q not marked approximate", got[0].Path)
	}
}

// TestSearchDocsUnlistedOnlyMatchNotDropped pins that an unlisted on-disk doc is
// returned for a query matching only its H1 (blurb) and for a filename query — the
// discovery payoff (#1656). The per-token penalty must not silently drop these.
func TestSearchDocsUnlistedOnlyMatchNotDropped(t *testing.T) {
	c := &Catalog{
		Root: ".",
		Docs: []Doc{
			{Title: "xyz", Path: "docs/xyz.md", Blurb: "Quantum Foam", Sources: []string{"tree"}, Discovered: true},
			{Title: "born bottlenecks", Path: "docs/born-bottlenecks.md", Blurb: "Born Bottlenecks", Sources: []string{"tree"}, Discovered: true},
		},
	}
	got := c.SearchDocs("quantum")
	if len(got) != 1 || got[0].Path != "docs/xyz.md" {
		t.Fatalf("SearchDocs(quantum) = %+v, want the H1-only discovered doc docs/xyz.md", got)
	}
	if got[0].Approx {
		t.Errorf("H1-only discovered match dropped from the exact path and rescued as Approx: %+v", got[0])
	}
	if got := c.SearchDocs("bottlenecks"); len(got) != 1 || got[0].Path != "docs/born-bottlenecks.md" {
		t.Fatalf("SearchDocs(bottlenecks) = %+v, want the filename-matched discovered doc", got)
	}
}

func TestLoadDocsMissingDocsTreeDegradesQuietly(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"INDEX.md": "- [Gateway](docs/gateway.md) — the performance gate\n",
		"llms.txt": "- [Only In Llms](docs/only-llms.md) — blurb only here\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c := LoadDocs(root)
	if len(c.Docs) != 2 {
		t.Fatalf("LoadDocs with no docs/ tree returned %d docs, want 2 curated rows: %+v", len(c.Docs), c.Docs)
	}
	for _, d := range c.Docs {
		if d.Discovered {
			t.Errorf("curated doc marked Discovered with no docs/ tree: %+v", d)
		}
	}
}

// TestSearchDocsDeterministicTiebreak pins the third adversarial finding: two rows
// with an equal title must still order deterministically (by path), so a
// hand-built catalog cannot produce input-order-dependent results.
func TestSearchDocsDeterministicTiebreak(t *testing.T) {
	a := Doc{Title: "same", Path: "docs/a.md", Sources: []string{"INDEX.md"}}
	b := Doc{Title: "same", Path: "docs/b.md", Sources: []string{"INDEX.md"}}
	first := (&Catalog{Root: ".", Docs: []Doc{a, b}}).SearchDocs("same")
	second := (&Catalog{Root: ".", Docs: []Doc{b, a}}).SearchDocs("same")
	if len(first) != 2 || len(second) != 2 {
		t.Fatalf("SearchDocs(same) = %v / %v, want two rows each", first, second)
	}
	if first[0].Path != second[0].Path || first[1].Path != second[1].Path {
		t.Fatalf("tiebreak is input-order dependent: %q,%q vs %q,%q",
			first[0].Path, first[1].Path, second[0].Path, second[1].Path)
	}
	if first[0].Path != "docs/a.md" {
		t.Fatalf("tiebreak top = %q, want docs/a.md (path order)", first[0].Path)
	}
}
