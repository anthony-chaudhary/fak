package quantmatrix

import (
	"testing"
)

// BenchmarkQuantMatrix exercises capability lookup and request adjudication
// in a loop to measure matrix adjudication throughput.
func BenchmarkQuantMatrix(b *testing.B) {
	req := Request{
		ID:              EntryGGUFQ4KCPU,
		ArtifactVersion: "gguf-v3",
		Runtime:         "fak-native-cpu",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		entry, ok := Lookup(req.ID)
		if !ok || entry.ID != req.ID {
			b.Fatalf("lookup failed for %s", req.ID)
		}
		decision := Adjudicate(req)
		if decision.Outcome != OutcomeAllow {
			b.Fatalf("unexpected outcome: %v", decision.Outcome)
		}
	}
}

// BenchmarkQuantMatrixLookup measures throughput of capability lookups.
func BenchmarkQuantMatrixLookup(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		entry, ok := Lookup(EntryGGUFQ4KCPU)
		if !ok || entry.ID != EntryGGUFQ4KCPU {
			b.Fatalf("lookup failed for %s", EntryGGUFQ4KCPU)
		}
	}
}

// BenchmarkQuantMatrixAdjudicate measures throughput of request adjudication.
func BenchmarkQuantMatrixAdjudicate(b *testing.B) {
	req := Request{
		ID:              EntryGGUFQ4KCPU,
		ArtifactVersion: "gguf-v3",
		Runtime:         "fak-native-cpu",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision := Adjudicate(req)
		if decision.Outcome != OutcomeAllow {
			b.Fatalf("unexpected outcome: %v", decision.Outcome)
		}
	}
}
