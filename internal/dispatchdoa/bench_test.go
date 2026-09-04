package dispatchdoa

import "testing"

// BenchmarkDispatchDOA exercises DOA classification in a loop.
func BenchmarkDispatchDOA(b *testing.B) {
	const log = SpawnHeaderPrefix + "20260728-213124 issue=2419 lane=gateway backend=claude argv0=fak.exe\n" +
		"flag provided but not defined: -compact-solvency-floor\n" +
		"usage: fak guard [flags] -- <agent command...>\n"
	size := int64(len(log))

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := Classify(log, size)
		if !v.DOA {
			b.Fatal("expected DOA verdict")
		}
	}
}
