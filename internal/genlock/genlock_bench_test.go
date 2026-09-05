package genlock_test

import (
	"bytes"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/genlock"
)

var (
	benchOutcomeSink genlock.Outcome
	benchStringSink  string
	benchBytesSink   []byte
	benchBoolSink    bool
	benchStringsSink []string
)

func BenchmarkSync(b *testing.B) {
	b.Run("Skipped", func(b *testing.B) {
		root := b.TempDir()
		l, err := genlock.Open(root, "marketing-aeo")
		if err != nil {
			b.Fatal(err)
		}
		const art = "docs/marketing/updates.json"
		input := []byte("ships: 1.0.0, 1.1.0, 1.2.0")
		renderBody := []byte("rendered updates json content\n")

		if _, err := l.Sync(art, input, func() ([]byte, error) {
			return renderBody, nil
		}); err != nil {
			b.Fatal(err)
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outcome, err := l.Sync(art, input, func() ([]byte, error) {
				return renderBody, nil
			})
			if err != nil || outcome != genlock.Skipped {
				b.Fatalf("Sync = %v, %v; want Skipped", outcome, err)
			}
			benchOutcomeSink = outcome
		}
	})

	b.Run("Wrote", func(b *testing.B) {
		root := b.TempDir()
		l, err := genlock.Open(root, "bench-tool")
		if err != nil {
			b.Fatal(err)
		}
		const art = "docs/marketing/updates.json"
		renderBody := []byte("rendered content\n")

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			in := []byte("input-variation-" + strconv.Itoa(i))
			outcome, err := l.Sync(art, in, func() ([]byte, error) {
				return renderBody, nil
			})
			if err != nil || outcome != genlock.Wrote {
				b.Fatalf("Sync = %v, %v; want Wrote", outcome, err)
			}
			benchOutcomeSink = outcome
		}
	})
}

func BenchmarkLockPath(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = genlock.PathFor("marketing-aeo")
	}
}

func BenchmarkSum(b *testing.B) {
	smallText := []byte("updated: 2026-09-05T12:00:00Z\nterms: a,b,c\nstatus: active\n")
	crlfText := []byte("updated: 2026-09-05T12:00:00Z\r\nterms: a,b,c\r\nstatus: active\r\n")
	mediumText := bytes.Repeat([]byte("line item with some markdown content and links to docs\n"), 50)
	binaryData := append([]byte{0x00, 0x1f, 0x8b, 0x08}, bytes.Repeat([]byte{0x01, 0x02, 0x03, 0x04}, 100)...)

	b.Run("TextSmall", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = genlock.Sum(smallText)
		}
	})

	b.Run("TextCRLF", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = genlock.Sum(crlfText)
		}
	})

	b.Run("TextMedium", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = genlock.Sum(mediumText)
		}
	})

	b.Run("Binary", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = genlock.Sum(binaryData)
		}
	})
}

func BenchmarkCanonical(b *testing.B) {
	part1 := []byte("commit-sha-7a8b9c0d1e2f")
	part2 := []byte("clock-free-terms-feed-v1")
	part3 := []byte("config-flags: deterministic, no-clock")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBytesSink = genlock.Canonical(part1, part2, part3)
	}
}

func BenchmarkCurrent(b *testing.B) {
	root := b.TempDir()
	l, err := genlock.Open(root, "marketing-aeo")
	if err != nil {
		b.Fatal(err)
	}
	const art = "docs/marketing/updates.json"
	input := []byte("ships: 1.0.0, 1.1.0, 1.2.0")
	if _, err := l.Sync(art, input, func() ([]byte, error) {
		return []byte("rendered updates json content\n"), nil
	}); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = l.Current(art, input)
	}
}

func BenchmarkStale(b *testing.B) {
	root := b.TempDir()
	l, err := genlock.Open(root, "marketing-aeo")
	if err != nil {
		b.Fatal(err)
	}
	inputs := map[string][]byte{
		"docs/marketing/updates.json":              []byte("input-1"),
		"docs/marketing/disambiguation-terms.json": []byte("input-2"),
		"llms-updates.txt":                         []byte("input-3"),
		"llms-terms.txt":                           []byte("input-4"),
	}
	for art, in := range inputs {
		if _, err := l.Sync(art, in, func() ([]byte, error) {
			return []byte("body for " + art), nil
		}); err != nil {
			b.Fatal(err)
		}
	}

	b.Run("Clean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringsSink = l.Stale(inputs)
		}
	})

	driftedInputs := map[string][]byte{
		"docs/marketing/updates.json":              []byte("input-1"),
		"docs/marketing/disambiguation-terms.json": []byte("input-2-drifted"),
		"llms-updates.txt":                         []byte("input-3"),
		"llms-terms.txt":                           []byte("input-4"),
	}
	b.Run("Drifted", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringsSink = l.Stale(driftedInputs)
		}
	})
}

func BenchmarkSave(b *testing.B) {
	root := b.TempDir()
	l, err := genlock.Open(root, "marketing-aeo")
	if err != nil {
		b.Fatal(err)
	}
	const art = "docs/marketing/updates.json"
	input := []byte("ships: 1.0.0, 1.1.0, 1.2.0")
	if _, err := l.Sync(art, input, func() ([]byte, error) {
		return []byte("body"), nil
	}); err != nil {
		b.Fatal(err)
	}
	if err := l.Save(); err != nil {
		b.Fatal(err)
	}

	b.Run("Unchanged", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := l.Save(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func TestBenchmarkSanity(t *testing.T) {
	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"BenchmarkSync", BenchmarkSync},
		{"BenchmarkLockPath", BenchmarkLockPath},
		{"BenchmarkSum", BenchmarkSum},
		{"BenchmarkCanonical", BenchmarkCanonical},
		{"BenchmarkCurrent", BenchmarkCurrent},
		{"BenchmarkStale", BenchmarkStale},
		{"BenchmarkSave", BenchmarkSave},
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
