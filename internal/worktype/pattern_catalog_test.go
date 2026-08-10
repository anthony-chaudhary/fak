package worktype

import (
	"bytes"
	"strings"
	"testing"
)

func TestSeedPatternCatalogValidDeterministicRoundTrip(t *testing.T) {
	c := SeedPatternCatalog()
	if len(c.Patterns) < 8 || len(c.Subpatterns) < 12 {
		t.Fatalf("seed too small: %d/%d", len(c.Patterns), len(c.Subpatterns))
	}
	a, err := c.DeterministicJSON()
	if err != nil {
		t.Fatal(err)
	}
	b, err := c.DeterministicJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatal("nondeterministic JSON")
	}
	got, err := ParsePatternCatalog(a)
	if err != nil {
		t.Fatal(err)
	}
	again, err := got.DeterministicJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, again) {
		t.Fatal("round trip changed bytes")
	}
}
func TestPatternCatalogRejectsMalformedAndDangling(t *testing.T) {
	tests := []struct {
		name string
		mut  func(*PatternCatalog)
		want string
	}{
		{"duplicate", func(c *PatternCatalog) { c.Subpatterns[1].ID = c.Subpatterns[0].ID }, "duplicate id"},
		{"alias ambiguity", func(c *PatternCatalog) { c.Subpatterns[0].Aliases = []string{c.Subpatterns[1].Name} }, "ambiguous"},
		{"dangling", func(c *PatternCatalog) { c.Patterns[0].Subpatterns = []string{"sp.missing"} }, "dangling"},
		{"unknown enum", func(c *PatternCatalog) { c.Patterns[0].Provenance = "invented" }, "unknown provenance"},
		{"missing boundary", func(c *PatternCatalog) { c.Patterns[0].ExcludeWhen = "" }, "required"},
		{"cycle", func(c *PatternCatalog) {
			c.Subpatterns[0].Composes = []string{c.Subpatterns[1].ID}
			c.Subpatterns[1].Composes = []string{c.Subpatterns[0].ID}
		}, "cycle"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := SeedPatternCatalog()
			tt.mut(&c)
			if err := c.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v want %q", err, tt.want)
			}
		})
	}
}
func TestSubpatternComposesAcrossPatterns(t *testing.T) {
	c := SeedPatternCatalog()
	n := 0
	for _, p := range c.Patterns {
		for _, s := range p.Subpatterns {
			if s == "sp.independent-adjudication" {
				n++
			}
		}
	}
	if n < 2 {
		t.Fatalf("composition reuse=%d", n)
	}
}

func TestPatternCatalogRejectsUnknownJSONField(t *testing.T) {
	b, err := SeedPatternCatalog().DeterministicJSON()
	if err != nil {
		t.Fatal(err)
	}
	bad := bytes.Replace(b, []byte(`"schema":`), []byte(`"unexpected":true,"schema":`), 1)
	if _, err := ParsePatternCatalog(bad); err == nil {
		t.Fatal("unknown field accepted")
	}
}
