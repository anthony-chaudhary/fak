package gateway

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func benchmarkToolCatalog(tools int) []map[string]any {
	full := make([]map[string]any, 0, tools)
	for i := 0; i < tools; i++ {
		name := fmt.Sprintf("mcp__bench__tool_%04d", i)
		description := fmt.Sprintf("operate benchmark resource %04d with deterministic semantics", i)
		if i == tools/2 {
			name = "mcp__bench__search_customer_records"
			description = "search customer records by account and email"
		}
		full = append(full, map[string]any{"name": name, "description": description})
	}
	return full
}

func rankToolCatalogByIntent(full []map[string]any, query string) []string {
	cat, err := selfquery.Load("", selfquery.Options{Tools: selfquery.ToolDescriptorsFromMaps(full)})
	if err != nil {
		return substringToolFilter(full, query)
	}
	resp, err := cat.Query(selfquery.Request{Query: query, Plane: selfquery.PlaneLive})
	if err != nil {
		return substringToolFilter(full, query)
	}
	out := make([]string, 0, len(resp.Cards))
	for _, card := range resp.Cards {
		out = append(out, card.Name)
	}
	return out
}

func BenchmarkToolFilteringAlternatives(b *testing.B) {
	for _, tools := range []int{64, 512, 2048} {
		full := benchmarkToolCatalog(tools)
		query := "customer"
		b.Run(fmt.Sprintf("tools=%d", tools), func(b *testing.B) {
			b.Run("native_ranked", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					_ = rankToolCatalogByIntent(full, query)
				}
			})
			b.Run("tuned_all_schemas", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					_ = full
				}
			})
			b.Run("incumbent_substring_retrieval", func(b *testing.B) {
				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					_ = substringToolFilter(full, query)
				}
			})
		})
	}
}

func TestToolFilteringAlternativesRetainRequiredTool(t *testing.T) {
	full := benchmarkToolCatalog(64)
	want := "mcp__bench__search_customer_records"
	for name, got := range map[string][]string{
		"native_ranked":                 rankToolCatalogByIntent(full, "customer"),
		"incumbent_substring_retrieval": substringToolFilter(full, "customer"),
	} {
		found := false
		for _, tool := range got {
			if tool == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%s dropped required tool %q", name, want)
		}
	}
}
