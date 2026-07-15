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

func TestValidatePositiveKindsAliasesAndExclusion(t *testing.T) {
	c := good()
	c.Rows = append(c.Rows, Row{ID: "cache-metric", Canonical: "Cache Metric", Family: "cache", Kind: "metric", Definition: "metric definition", Distinction: "metric rather than symbol", DistinctFrom: []string{"cache-a"}, Aliases: []string{"cache_hits_total"}, Grounding: "cache_hits_total", GroundingKind: "metric", GlossaryAnchor: "docs/glossary.md", Verdict: "crystal", Source: "rows.json"})
	c.Meta.Families[0].Exclude = []string{"cachedtesthelper"}
	if ds := Validate(c); len(ds) != 0 {
		t.Fatalf("valid symbols/metrics/aliases and unpositioned exclusion rejected: %+v", ds)
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
	if ds := Validate(c); len(ds) > 0 {
		b, _ := json.MarshalIndent(ds, "", "  ")
		t.Fatalf("current catalog invalid:\n%s", b)
	}
}
