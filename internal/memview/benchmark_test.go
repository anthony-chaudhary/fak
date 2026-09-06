package memview

import (
	"bytes"
	"fmt"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Global sinks to prevent compiler dead-code elimination.
var (
	benchSinkRecord   MemoryViewRecord
	benchSinkVerdict  abi.Verdict
	benchSinkBytes    []byte
	benchSinkBool     bool
	benchSinkErr      error
	benchSinkTimeline Timeline
	benchSinkManifest SkillManifest
)

type benchPage struct {
	bytes  []byte
	digest string
	taint  abi.TaintLabel
	role   string
}

func (p benchPage) Digest() string         { return p.digest }
func (p benchPage) Bytes() ([]byte, error) { return p.bytes, nil }
func (p benchPage) Role() string           { return p.role }
func (p benchPage) Taint() abi.TaintLabel  { return p.taint }

func newBenchPage(size int, taint abi.TaintLabel) benchPage {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte((i % 26) + 'a')
	}
	return benchPage{
		bytes:  data,
		digest: Digest(data),
		taint:  taint,
		role:   "bench_tool",
	}
}

// BenchmarkMaterializeSnippet measures end-to-end view creation, slice projection,
// digest binding, and taint-aware admission across small, medium, and large source spans.
func BenchmarkMaterializeSnippet(b *testing.B) {
	sizes := []struct {
		name    string
		srcSize int
		offset  int
		length  int
	}{
		{name: "Small_64B", srcSize: 1024, offset: 64, length: 64},
		{name: "Medium_1KB", srcSize: 4096, offset: 512, length: 1024},
		{name: "Large_64KB", srcSize: 128 * 1024, offset: 1024, length: 64 * 1024},
	}

	m := Materializer{Producer: "bench@v1", Epoch: 42}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			page := newBenchPage(tc.srcSize, abi.TaintTrusted)
			b.SetBytes(int64(tc.length))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rec, v, err := m.MaterializeSnippet("view-bench", page, tc.offset, tc.length)
				benchSinkRecord = rec
				benchSinkVerdict = v
				benchSinkErr = err
			}
		})
	}
}

// BenchmarkBoundsChecking measures the memory bounds verification logic
// on both valid and invalid source span projections.
func BenchmarkBoundsChecking(b *testing.B) {
	page := newBenchPage(1024, abi.TaintTrusted)
	m := Materializer{Producer: "bench@v1", Epoch: 1}

	b.Run("ValidSpan", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec, v, err := m.MaterializeSnippet("v", page, 100, 200)
			benchSinkRecord = rec
			benchSinkVerdict = v
			benchSinkErr = err
		}
	})

	b.Run("OutOfRange_OffsetPlusLength", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec, v, err := m.MaterializeSnippet("v", page, 900, 200)
			benchSinkRecord = rec
			benchSinkVerdict = v
			benchSinkErr = err
		}
	})

	b.Run("OutOfRange_NegativeOffset", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec, v, err := m.MaterializeSnippet("v", page, -1, 50)
			benchSinkRecord = rec
			benchSinkVerdict = v
			benchSinkErr = err
		}
	})

	b.Run("EmptySpan_ZeroLength", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec, v, err := m.MaterializeSnippet("v", page, 10, 0)
			benchSinkRecord = rec
			benchSinkVerdict = v
			benchSinkErr = err
		}
	})
}

// BenchmarkRecordIsValid measures cache-line invalidation checking against
// matching, mismatched, and empty digests.
func BenchmarkRecordIsValid(b *testing.B) {
	page := newBenchPage(256, abi.TaintTrusted)
	m := Materializer{Producer: "bench@v1", Epoch: 1}
	rec, _, err := m.MaterializeSnippet("v", page, 10, 50)
	if err != nil {
		b.Fatalf("setup failed: %v", err)
	}
	altDigest := Digest([]byte("mutated page content"))

	b.Run("ValidDigest", func(b *testing.B) {
		digest := page.Digest()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = rec.IsValid(digest)
		}
	})

	b.Run("InvalidDigest", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = rec.IsValid(altDigest)
		}
	})

	b.Run("EmptyDigest", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = rec.IsValid("")
		}
	})
}

// BenchmarkVerdictFor measures the admission gate check on raw memory pages.
func BenchmarkVerdictFor(b *testing.B) {
	pageTrusted := newBenchPage(64, abi.TaintTrusted)
	pageTainted := newBenchPage(64, abi.TaintTainted)

	b.Run("Trusted", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict = VerdictFor(pageTrusted)
		}
	})

	b.Run("Tainted", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict = VerdictFor(pageTainted)
		}
	})
}

// BenchmarkDigest measures SHA-256 hex content-address hashing across memory blocks.
func BenchmarkDigest(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"64B", 64},
		{"1KB", 1024},
		{"64KB", 64 * 1024},
	}

	for _, sc := range sizes {
		b.Run(sc.name, func(b *testing.B) {
			data := bytes.Repeat([]byte("a"), sc.size)
			b.SetBytes(int64(sc.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = Digest(data)
			}
		})
	}
}

// BenchmarkSurfaceEncode measures output format serialization for tabular views
// under markdown, JSON, and TOON encodings.
func BenchmarkSurfaceEncode(b *testing.B) {
	fields := []string{"id", "role", "durability", "digest", "status"}
	rows := make([]Row, 50)
	for i := range rows {
		rows[i] = Row{
			strconv.Itoa(i),
			"tool_execution",
			"session",
			"sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
			"admitted",
		}
	}
	s, err := NewSurface("memview_benchmark_surface", fields, rows)
	if err != nil {
		b.Fatalf("NewSurface failed: %v", err)
	}

	formats := []Format{FormatMarkdown, FormatJSON, FormatTOON}
	for _, f := range formats {
		b.Run(string(f), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := Encode(f, s)
				benchSinkBytes = out
				benchSinkErr = err
			}
		})
	}

	b.Run("SweepFormats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			metrics, err := SweepFormats(s, formats)
			if err != nil {
				b.Fatalf("SweepFormats failed: %v", err)
			}
			if len(metrics) != len(formats) {
				b.Fatalf("unexpected metrics len: %d", len(metrics))
			}
		}
	})
}

// BenchmarkTimeline measures timeline assembly and text rendering.
func BenchmarkTimeline(b *testing.B) {
	events := make([]ProvenanceEvent, 20)
	for i := range events {
		events[i] = ProvenanceEvent{
			Seq:        20 - i, // reverse order to exercise sorting
			Step:       i,
			Kind:       EventPromotion,
			Durability: "session",
			Producer:   "summarizer@v1",
			Consent:    "explicit",
			Digest:     "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
			Descriptor: "page-summary",
			Reason:     "admitted by confidence threshold",
		}
	}

	b.Run("BuildTimeline", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkTimeline = BuildTimeline("cell-101", events)
		}
	})

	tl := BuildTimeline("cell-101", events)
	b.Run("RenderTimeline", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tl.Render()
		}
	})
}

// BenchmarkSkillManifest measures skill manifest view synthesis with budget enforcement.
func BenchmarkSkillManifest(b *testing.B) {
	entries := make([]SkillManifestEntry, 30)
	for i := range entries {
		entries[i] = SkillManifestEntry{
			Name:       fmt.Sprintf("skill-%02d", i),
			Version:    "v1.0.0",
			Provenance: "registry@v1",
			Value:      float64(i * 10),
			Active:     true,
			Witnessed:  true,
			Admitted:   true,
		}
	}

	b.Run("BuildSkillManifest_TOON", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sm, err := BuildSkillManifest(entries, FormatTOON, 1024)
			benchSinkManifest = sm
			benchSinkErr = err
		}
	})

	b.Run("BuildSkillManifest_JSON", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sm, err := BuildSkillManifest(entries, FormatJSON, 2048)
			benchSinkManifest = sm
			benchSinkErr = err
		}
	})
}
