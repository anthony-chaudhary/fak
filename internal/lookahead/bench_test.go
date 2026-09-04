package lookahead

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// BenchmarkLookahead measures throughput of lesson distillation and witness gating in a loop.
func BenchmarkLookahead(b *testing.B) {
	ev := RolloutEvidence{
		ForkSessionID: "fork-bench",
		BaseSHA:       "abcdef123456",
		Turns:         5,
		Rung:          trajctl.W3,
	}
	distill := func(string) (Proposal, bool) {
		return Proposal{Claim: "witnessed fact", Kind: KindFact}, true
	}
	transcript := longTranscript

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DistillLesson(ev, transcript, "bench-expire-sha", distill)
	}
}
