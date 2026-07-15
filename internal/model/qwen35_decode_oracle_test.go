package model

import (
	"testing"
)

// TestOptionalOrnithOracleGreedyDecodeMatchesHF closes the prefill-only hole in
// TestOptionalOrnithOracleForwardMatchesHF: it proves that recurrent Gated-DeltaNet
// state carried from Prefill into token-by-token Step produces the same greedy
// continuation as the independently generated Hugging Face oracle. This is the
// exact boundary exercised by the Qwen3.6 long-prompt degeneration in #4273.
func TestOptionalOrnithOracleGreedyDecodeMatchesHF(t *testing.T) {
	dir := ornithFixtureDir()
	m, oracle := loadFixtureDir(t, dir, true)
	for pi, p := range oracle.Prompts {
		s := m.NewSession()
		logits := s.Prefill(p.Ids)
		for step, want := range p.GreedyIds {
			got := argmax(logits)
			if got != want {
				t.Fatalf("prompt %d step %d: greedy token=%d want HF=%d", pi, step, got, want)
			}
			if step+1 < len(p.GreedyIds) {
				logits = s.Step(got)
			}
		}
	}
}
