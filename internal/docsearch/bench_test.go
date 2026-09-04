package docsearch

import (
	"testing"
)

func BenchmarkDocSearch(b *testing.B) {
	c := searchCatalog()
	queries := []string{
		"gateway",
		"gateway overview",
		"cache eviction",
		"alpha",
		"gatewayy",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		results := c.SearchDocs(query)
		if len(results) == 0 {
			b.Fatalf("expected results for query %q, got none", query)
		}
	}
}

func BenchmarkDocSearchExactVsFuzzy(b *testing.B) {
	c := searchCatalog()
	b.Run("ExactMatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = c.SearchDocs("gateway overview")
		}
	})
	b.Run("FuzzyFallback", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = c.SearchDocs("gatewayy")
		}
	})
}
func TestBenchmarkSanity(t *testing.T) {
	c := searchCatalog()
	res := c.SearchDocs("gateway")
	if len(res) == 0 {
		t.Fatal("expected results")
	}
}
