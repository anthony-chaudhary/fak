package eveparity_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/eveparity"
)

// TestBenchmarkEvaluateSanity ensures the benchmarked Evaluate code path
// executes correctly across the canonical fixture suite cases.
func TestBenchmarkEvaluateSanity(t *testing.T) {
	suite := eveparity.FixtureSuite()
	transcripts := []eveparity.Transcript{
		{CaseID: "succeeded-and-tool", Succeeded: true, ToolCalls: []string{"search"}, SessionID: "s1", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "content-exact", Succeeded: true, FinalText: "The answer is 42.", SessionID: "s2", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "soft-quality", Succeeded: true, FinalText: "a thorough, well-structured summary", SessionID: "s3", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "deliberate-gate-fail", Succeeded: true, FinalText: "refused", SessionID: "s4", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "soft-strict-fail", Succeeded: true, FinalText: "terse.", SessionID: "s5", PromptTokens: 11, CompletionTokens: 7},
	}

	for idx, c := range suite.Cases {
		res := eveparity.Evaluate(c, transcripts[idx], false)
		if res.CaseID != c.ID {
			t.Fatalf("case %d: got CaseID %q, want %q", idx, res.CaseID, c.ID)
		}
	}
}

// BenchmarkEVEParityEvaluate measures deterministic evaluation throughput
// across all fixture cases under varying strictness configurations.
func BenchmarkEVEParityEvaluate(b *testing.B) {
	suite := eveparity.FixtureSuite()
	transcripts := []eveparity.Transcript{
		{CaseID: "succeeded-and-tool", Succeeded: true, ToolCalls: []string{"search"}, SessionID: "s1", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "content-exact", Succeeded: true, FinalText: "The answer is 42.", SessionID: "s2", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "soft-quality", Succeeded: true, FinalText: "a thorough, well-structured summary", SessionID: "s3", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "deliberate-gate-fail", Succeeded: true, FinalText: "refused", SessionID: "s4", PromptTokens: 11, CompletionTokens: 7},
		{CaseID: "soft-strict-fail", Succeeded: true, FinalText: "terse.", SessionID: "s5", PromptTokens: 11, CompletionTokens: 7},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		strict := (i & 1) == 1
		for idx, c := range suite.Cases {
			_ = eveparity.Evaluate(c, transcripts[idx], strict)
		}
	}
}
