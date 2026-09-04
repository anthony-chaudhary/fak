package workerenvelope

import (
	"testing"
)

// BenchmarkWorkerEnvelope exercises parsing and validating representative worker envelopes in a loop.
func BenchmarkWorkerEnvelope(b *testing.B) {
	fixture := []byte(`{
		"status": "shipped",
		"issue": 1795,
		"commit_sha": "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90",
		"tests_run": ["go test ./internal/workerenvelope/"],
		"witness": "commit c99f5c02"
	}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Parse(fixture)
		if err != nil {
			b.Fatalf("unexpected parse error: %v", err)
		}
		if err := res.Validate(); err != nil {
			b.Fatalf("unexpected validation error: %v", err)
		}
	}
}
