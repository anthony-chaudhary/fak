package dropin

import (
	"testing"
)

// BenchmarkDropin benchmarks provider detection and plan generation across
// representative agent command names and gateway targets.
func BenchmarkDropin(b *testing.B) {
	agents := []string{
		"claude",
		"codex",
		"gemini",
		"opencode",
		"aider",
		"unknown-agent",
	}
	const gw = "http://127.0.0.1:8137"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		agent := agents[i%len(agents)]
		_, _ = DetectProvider(agent)
		_ = PlanFor(agent, "", "", gw)
	}
}

func TestBenchmarkDropinSanity(t *testing.T) {
	const gw = "http://127.0.0.1:8137"
	plan := PlanFor("claude", "", "", gw)
	if plan.Provider != "anthropic" {
		t.Fatalf("unexpected provider: %s", plan.Provider)
	}
}
