package toolbound

import (
	"os"
	"strings"
	"testing"
)

func BenchmarkToolBound(b *testing.B) {
	dir := b.TempDir()
	bounder := New(Options{
		MaxLines: 10,
		MaxBytes: 256,
		SpillDir: dir,
	})
	payload := strings.Repeat("benchmark tool output line data with content\n", 50)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out, err := bounder.Bound(payload)
		if err != nil {
			b.Fatalf("unexpected bounding error: %v", err)
		}
		if !out.Truncated {
			b.Fatalf("expected truncated output")
		}
		if out.CompletePath != "" {
			_ = os.Remove(out.CompletePath)
		}
	}
}
