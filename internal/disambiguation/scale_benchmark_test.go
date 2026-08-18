package disambiguation

import (
	"fmt"
	"runtime"
	"testing"
)

const projectedScaleEntries = 4096

func projectedIndex(b *testing.B) []Entry {
	b.Helper()
	entries := make([]Entry, 0, projectedScaleEntries)
	seed := publicEntries[0]
	for i := 0; i < projectedScaleEntries; i++ {
		entry := cloneEntry(seed)
		entry.Identity.CanonicalTerm = fmt.Sprintf("projected term %04d", i)
		entry.Identity.Aliases = []string{fmt.Sprintf("projected alias %04d", i)}
		entry.Scope = Scope{Kind: "projected", Value: fmt.Sprintf("%04d", i)}
		entry.Contrasts = []Contrast{{CanonicalTerm: fmt.Sprintf("projected term %04d", (i+1)%projectedScaleEntries), Explanation: "scale fixture contrast", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}}
		entries = append(entries, entry)
	}
	return entries
}

func benchmarkScope(b *testing.B, operation, baseline string) {
	b.ReportMetric(float64(projectedScaleEntries), "entries")
	b.ReportMetric(1, "OBSERVED_provenance")
	b.ReportMetric(1, "tuned_indexed_baseline")
	b.Logf("scope operation=%s dataset_entries=%d baseline=%s hardware=%s/%s cpu=%s provenance=OBSERVED", operation, projectedScaleEntries, baseline, runtime.GOOS, runtime.GOARCH, runtime.Version())
}

func BenchmarkGenerateProjectedIndex(b *testing.B) {
	entries := projectedIndex(b)
	benchmarkScope(b, "generate", "canonical sort + JSON encode")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GenerateIndex(entries); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkExactQueryProjectedIndex(b *testing.B) {
	entries := projectedIndex(b)
	index, err := NewIndex(entries)
	if err != nil {
		b.Fatal(err)
	}
	benchmarkScope(b, "exact-query", "prebuilt map lookup")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, _ := index.queryCanonical("projected term 2048"); !ok {
			b.Fatal("miss")
		}
	}
}
func BenchmarkAliasQueryProjectedIndex(b *testing.B) {
	entries := projectedIndex(b)
	index, err := NewIndex(entries)
	if err != nil {
		b.Fatal(err)
	}
	benchmarkScope(b, "alias-query", "prebuilt map lookup")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, ok := index.resolve("projected alias 2048"); !ok {
			b.Fatal("miss")
		}
	}
}
func BenchmarkReverseQueryProjectedIndex(b *testing.B) {
	entries := projectedIndex(b)
	index, err := NewIndex(entries)
	if err != nil {
		b.Fatal(err)
	}
	benchmarkScope(b, "reverse-query", "indexed canonical entries with exact locator scan")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		response, err := index.ReverseLookup(ReverseSourcePath, entries[0].Sources[0].Locator)
		if err != nil || len(response.Matches) == 0 {
			b.Fatal("miss", err)
		}
	}
}
