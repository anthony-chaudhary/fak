package fleetcompare

import "testing"

// Invariant: fleet comparison slicing is fail-closed and deterministic.
// Guard: benchmark exercises SliceFixed over realistic multi-node metric columns.

func BenchmarkFleetCompare(b *testing.B) {
	cols := fixtureCols()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SliceFixed(cols, "agents", 50)
	}
}
