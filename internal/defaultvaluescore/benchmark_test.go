package defaultvaluescore

import (
	"testing"
	"time"
)

// BenchmarkDefaultValueScore measures end-to-end evaluation and folding of the default-value scorecard.
func BenchmarkDefaultValueScore(b *testing.B) {
	asOf := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := BuildAsOf("../..", asOf)
		if p.Schema != Schema {
			b.Fatalf("unexpected schema: got %q want %q", p.Schema, Schema)
		}
	}
}

// BenchmarkParseFlags measures parsing and classification throughput for flag declarations.
func BenchmarkParseFlags(b *testing.B) {
	const sampleSource = `package main
func registerFlags(fs *flag.FlagSet) {
	fs.Bool("compact-anchor-head", true, "retaining stable head protects provider-cache reuse")
	fs.Int("compact-history-budget", 4096, "bounded history compaction budget")
	fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine")
	fs.Bool("vdso", true, "in-process fast path")
	fs.Bool("elide-result-bytes", true, "result elision removes repeated payload bytes")
	fs.String("session-id", "", "trace and session identifier")
	fs.Int("context-budget-tokens", 0, "hard context token limit")
}
`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		flags := ParseFlags(sampleSource, "cmd/fak/guard.go")
		if len(flags) == 0 {
			b.Fatal("unexpected empty parsed flags")
		}
	}
}

// BenchmarkKPIEvaluation measures in-memory evaluation throughput across all four scorecard KPIs.
func BenchmarkKPIEvaluation(b *testing.B) {
	asOf := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	flags := []valueFlag{
		{name: "compact-anchor-head", kind: "Bool", defLit: "true", defaultOn: true, source: "cmd/fak/guard.go"},
		{name: "compact-anchor-head", kind: "Bool", defLit: "true", defaultOn: true, source: "cmd/fak/serve.go"},
		{name: "compact-history-budget", kind: "Int", defLit: "4096", defaultOn: true, source: "cmd/fak/guard.go"},
		{name: "compact-history-budget", kind: "Int", defLit: "4096", defaultOn: true, source: "cmd/fak/serve.go"},
		{name: "vdso", kind: "Bool", defLit: "true", defaultOn: true, source: "cmd/fak/guard.go"},
		{name: "context-budget-tokens", kind: "Int", defLit: "0", defaultOn: false, source: "cmd/fak/guard.go"},
	}
	ampText := `func formatAmplification(kc kernel.Counters) string {
	if kc.VDSOHits == 0 && kc.Transforms == 0 {
		return "fak guard: floor effect (proxy path: the kernel adjudicates with Decide, so the in-kernel axis does not apply)"
	}
	return fmt.Sprintf("fak guard: amplification %dx", kc.VDSOHits)
}`
	surfaces := map[string]string{
		"internal/vcachescore/score.go": `activeSource := "telemetry" // observed`,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = kpiValueFlagDefaultOn(flags, asOf)
		_ = kpiValueFlagContextParity(flags)
		_ = kpiNoVacuousCounterFold(ampText, AmplificationSurface)
		_ = kpiObservedNotModeledDefault(surfaces)
	}
}

// TestBenchmarkDefaultValueScoreSanity verifies that BenchmarkDefaultValueScore executes cleanly.
func TestBenchmarkDefaultValueScoreSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkDefaultValueScore)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
