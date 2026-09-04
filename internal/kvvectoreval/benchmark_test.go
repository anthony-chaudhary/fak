package kvvectoreval

import (
	"testing"
)

func BenchmarkEvaluateSupported(b *testing.B) {
	req := validRequest()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

func BenchmarkEvaluateMalformed(b *testing.B) {
	req := Request{}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != OutcomeRefused {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

func BenchmarkEvaluateDelegate(b *testing.B) {
	req := validRequest()
	req.RuntimeAvailable = false
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != OutcomeDelegate {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

func BenchmarkVerifyArtifact(b *testing.B) {
	data := make([]byte, 1024)
	sumID := pinnedArtifacts()[0].ID
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = VerifyArtifact(sumID, data)
	}
}

func BenchmarkVerifyArtifactMismatch(b *testing.B) {
	data := []byte("arbitrary non-matching artifact payload")
	sumID := pinnedArtifacts()[0].ID
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = VerifyArtifact(sumID, data)
	}
}

func BenchmarkResearchLedger(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = researchLedger()
	}
}
