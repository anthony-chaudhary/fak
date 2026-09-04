package customizationindex

import (
	"strings"
	"testing"
	"time"
)

func BenchmarkCustomizationIndex(b *testing.B) {
	asOf := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index, err := Read(strings.NewReader(fixture))
		if err != nil {
			b.Fatal(err)
		}
		report := Check(index, asOf)
		if !report.Valid {
			b.Fatal("expected report to be valid")
		}
	}
}
