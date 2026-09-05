package milestonedoc

import (
	"strings"
	"testing"
)

var (
	benchBlockSink    string
	benchScaffoldSink string
	benchExtractSink  string
	benchSpliceSink   string
	benchFreshSink    bool
)

// BenchmarkBlock measures generating the complete milestone-climb markdown block
// from the live covmatrix grid folded by milestonereport.InterpretMaturity onto
// the closed M0-M7 ladder.
func BenchmarkBlock(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBlockSink = Block()
	}
}

// BenchmarkScaffold measures generating the initial STATUS.md scaffold document
// containing SEO frontmatter, title, orientation text, and empty marker pair.
func BenchmarkScaffold(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchScaffoldSink = Scaffold()
	}
}

// BenchmarkExtract measures marker-bounded block extraction across populated
// documents and negative missing-marker cases.
func BenchmarkExtract(b *testing.B) {
	scaffold := Scaffold()
	doc, err := Splice(scaffold)
	if err != nil {
		b.Fatalf("Splice failed: %v", err)
	}
	const plain = "# Milestone status\n\nNo markers in this document.\n"

	b.Run("populated_doc", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extracted, ok := Extract(doc)
			if !ok {
				b.Fatal("Extract failed to find block")
			}
			benchExtractSink = extracted
		}
	})

	b.Run("missing_markers", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extracted, ok := Extract(plain)
			if ok {
				b.Fatal("Extract unexpectedly found block")
			}
			benchExtractSink = extracted
		}
	})
}

// BenchmarkSplice measures replacing the marker-bounded block in both newly
// scaffolded docs and existing populated docs.
func BenchmarkSplice(b *testing.B) {
	scaffold := Scaffold()
	populated, err := Splice(scaffold)
	if err != nil {
		b.Fatalf("Splice failed: %v", err)
	}

	b.Run("into_scaffold", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			spliced, err := Splice(scaffold)
			if err != nil {
				b.Fatalf("Splice into scaffold: %v", err)
			}
			benchSpliceSink = spliced
		}
	})

	b.Run("into_existing", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			spliced, err := Splice(populated)
			if err != nil {
				b.Fatalf("Splice into existing: %v", err)
			}
			benchSpliceSink = spliced
		}
	})
}

// BenchmarkFresh measures the freshness gate predicate across clean matching,
// stale drifted, and missing marker documents.
func BenchmarkFresh(b *testing.B) {
	freshDoc, err := Splice(Scaffold())
	if err != nil {
		b.Fatalf("Splice failed: %v", err)
	}

	staleDoc := freshDoc
	for _, swap := range []struct{ from, to string }{
		{" | 0 |", " | 99 |"},
		{" | 1 |", " | 99 |"},
		{" | 2 |", " | 99 |"},
	} {
		if strings.Contains(staleDoc, swap.from) {
			staleDoc = strings.Replace(staleDoc, swap.from, swap.to, 1)
			break
		}
	}
	if staleDoc == freshDoc {
		b.Fatal("could not create stale doc fixture")
	}

	const bareDoc = "# Milestone status\n\nNo markers here.\n"

	b.Run("fresh_matching", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFreshSink = Fresh(freshDoc)
		}
	})

	b.Run("stale_drifted", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFreshSink = Fresh(staleDoc)
		}
	})

	b.Run("missing_markers", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFreshSink = Fresh(bareDoc)
		}
	})
}

// TestBenchmarkSanity verifies that all benchmark functions execute cleanly
// and complete at least one iteration.
func TestBenchmarkSanity(t *testing.T) {
	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"BenchmarkBlock", BenchmarkBlock},
		{"BenchmarkScaffold", BenchmarkScaffold},
		{"BenchmarkExtract", BenchmarkExtract},
		{"BenchmarkSplice", BenchmarkSplice},
		{"BenchmarkFresh", BenchmarkFresh},
	}

	for _, bm := range benchmarks {
		t.Run(bm.name, func(t *testing.T) {
			res := testing.Benchmark(bm.fn)
			if res.N <= 0 {
				t.Fatalf("%s failed to execute iterations: %d", bm.name, res.N)
			}
		})
	}
}
