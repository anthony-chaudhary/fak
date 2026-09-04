package fleetverify

import (
	"testing"
)

// BenchmarkFleetVerify measures performance of brief report collection over a bounded root.
func BenchmarkFleetVerify(b *testing.B) {
	root := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := collectFleetBriefReport(root)
		if err != nil {
			b.Fatalf("collectFleetBriefReport: %v", err)
		}
		if len(report.Skipped) == 0 {
			b.Fatal("collectFleetBriefReport: expected skipped ledgers on empty root")
		}
	}
}
