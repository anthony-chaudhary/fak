package ultracodedogfood

import (
	"os"
	"testing"
)

// BenchmarkUltraCodeDogfood exercises the dogfood session lifecycle evaluation in a loop.
func BenchmarkUltraCodeDogfood(b *testing.B) {
	raw, err := os.ReadFile("testdata/issue8678-lifecycle-session.json")
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := EvaluateLifecycleSession(raw)
		if err != nil {
			b.Fatal(err)
		}
		if report.Verdict != "PASS" {
			b.Fatalf("unexpected verdict %q", report.Verdict)
		}
	}
}
