package conceptcatalog

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func good() Catalog {
	return Catalog{
		Meta: Metadata{Families: []Family{{ID: "cache", Name: "Cache", Roots: []string{"cache"}}}},
		Rows: []Row{
			{ID: "cache-a", Canonical: "Cache A", Family: "cache", Kind: "symbol", Definition: "a definition long enough", Distinction: "not cache b", DistinctFrom: []string{"cache-b"}, Grounding: "CacheA", GroundingKind: "symbol", GlossaryAnchor: "docs/glossary.md", Verdict: "crystal", Source: "rows.json"},
			{ID: "cache-b", Canonical: "Cache B", Family: "cache", Kind: "symbol", Definition: "another definition", Distinction: "not cache a", DistinctFrom: []string{"cache-a"}, Grounding: "CacheB", GroundingKind: "symbol", GlossaryAnchor: "docs/glossary.md", Verdict: "crystal", Source: "rows.json"},
		},
	}
}

func TestValidateMalformedCatalogMutations(t *testing.T) {
	tests := []struct {
		name, code string
		mut        func(*Catalog)
	}{
		{"unresolved distinct_from", "unresolved_reference", func(c *Catalog) { c.Rows[0].DistinctFrom = []string{"missing"} }},
		{"canonical distinct_from", "canonical_reference", func(c *Catalog) { c.Rows[0].DistinctFrom = []string{"Cache B"} }},
		{"duplicate id", "duplicate_id", func(c *Catalog) { c.Rows[1].ID = c.Rows[0].ID }},
		{"duplicate canonical", "duplicate_canonical", func(c *Catalog) { c.Rows[1].Canonical = c.Rows[0].Canonical }},
		{"duplicate grounding", "duplicate_grounding", func(c *Catalog) { c.Rows[1].Grounding = c.Rows[0].Grounding }},
		{"wrong grounding kind", "wrong_grounding_kind", func(c *Catalog) { c.Rows[0].GroundingKind = "test" }},
		{"stale alias", "stale_alias", func(c *Catalog) { c.Rows[0].Aliases = []string{"Cache A"} }},
		{"classification masks positioned concept", "classification_conflict", func(c *Catalog) { c.Meta.Families[0].Ignore = []string{"CacheA"} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := good()
			tt.mut(&c)
			ds := ValidateStrict(c)
			found := false
			for _, d := range ds {
				if d.Code == tt.code {
					found = true
					if d.File == "" || d.RowID == "" || d.Field == "" || d.Value == "" || d.Repair == "" {
						t.Fatalf("diagnostic lacks repair context: %+v", d)
					}
				}
			}
			if !found {
				t.Fatalf("mutation not detected as %s: %+v", tt.code, ds)
			}
		})
	}
}

func TestValidatePositiveKindsAliasesAndClassifications(t *testing.T) {
	c := good()
	c.Rows = append(c.Rows,
		Row{ID: "cache-metric", Canonical: "Cache Metric", Family: "cache", Kind: "metric", Definition: "metric definition", Distinction: "metric rather than symbol", DistinctFrom: []string{"cache-a"}, Aliases: []string{"cache_hits_total"}, Grounding: "cache_hits_total", GroundingKind: "metric", GlossaryAnchor: "docs/glossary.md", Verdict: "crystal", Source: "rows.json"},
		Row{ID: "cache-doc", Canonical: "Cache Guide", Family: "cache", Kind: "doc", Definition: "documentation definition", Distinction: "guide rather than metric", DistinctFrom: []string{"cache-metric"}, Grounding: "cache-guide", GroundingKind: "doc", GlossaryAnchor: "docs/glossary.md", Verdict: "crystal", Source: "rows.json"},
	)
	// Exact false-positive substring exclusions, test-only classifications, and
	// build-tag-only classifications are metadata, not positioned concepts.
	c.Meta.Families[0].Exclude = []string{"cachedtesthelper"}
	c.Meta.Families[0].Ignore = []string{"cache_fixture_test", "cache_wip_build"}
	if ds := ValidateStrict(c); len(ds) != 0 {
		t.Fatalf("valid symbols/docs/metrics/aliases and classifications rejected: %+v", ds)
	}
}

func TestValidateTreeRejectsExcludedCorpusOnlyGrounding(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "cache_test.go"), []byte("package cache\nconst TestOnlyGround = 1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "wip.go"), []byte("//go:build wip_cache\n\npackage cache\nconst BuildOnlyGround = 1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, grounding := range []string{"TestOnlyGround", "BuildOnlyGround"} {
		t.Run(grounding, func(t *testing.T) {
			c := good()
			c.Rows[0].Grounding = grounding
			ds := ValidateTree(c, root)
			found := false
			for _, d := range ds {
				if d.Code == "excluded_corpus_grounding" && d.File != "" && d.RowID != "" && d.Field == "grounding" && d.Value == grounding && d.Repair != "" {
					found = true
				}
			}
			if !found {
				t.Fatalf("excluded-only grounding %q not diagnosed with repair context: %+v", grounding, ds)
			}
		})
	}
}
func TestLoadDirReportsMalformedFile(t *testing.T) {
	d := t.TempDir()
	_ = os.WriteFile(filepath.Join(d, "_meta.json"), []byte(`{"families":[]}`), 0600)
	_ = os.WriteFile(filepath.Join(d, "rows-bad.json"), []byte(`{"rows":[`), 0600)
	_, err := LoadDir(d)
	if err == nil || !strings.Contains(err.Error(), "rows-bad.json") {
		t.Fatalf("want file-qualified parse error, got %v", err)
	}
}
func TestCurrentCatalogSemanticContract(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	c, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if ds := ValidateTree(c, root); len(ds) > 0 {
		b, _ := json.MarshalIndent(ds, "", "  ")
		t.Fatalf("current 100%%-coverage catalog invalid:\n%s", b)
	}
}
