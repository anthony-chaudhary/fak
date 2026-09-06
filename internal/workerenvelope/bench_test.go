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

// BenchmarkValidate exercises Validate on a pre-constructed valid Result.
func BenchmarkValidate(b *testing.B) {
	res := Result{
		Status:    StatusShipped,
		Issue:     1795,
		CommitSHA: "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90",
		TestsRun:  []string{"go test ./internal/workerenvelope/"},
		Witness:   "commit c99f5c02",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := res.Validate(); err != nil {
			b.Fatalf("unexpected validation error: %v", err)
		}
	}
}

// BenchmarkLooksLikeSHA exercises commit SHA validation across varying length strings.
func BenchmarkLooksLikeSHA(b *testing.B) {
	const shortSHA = "c99f5c0"
	const fullSHA = "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !looksLikeSHA(shortSHA) || !looksLikeSHA(fullSHA) {
			b.Fatal("unexpected false from looksLikeSHA")
		}
	}
}
