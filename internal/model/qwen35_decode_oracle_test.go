package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestOptionalOrnithOracleGreedyDecodeMatchesHF closes the prefill-only hole in
// TestOptionalOrnithOracleForwardMatchesHF: it proves that recurrent Gated-DeltaNet
// state carried from Prefill into token-by-token Step produces the same greedy
// continuation as the independently generated Hugging Face oracle. This is the
// exact boundary exercised by the Qwen3.6 long-prompt degeneration in #4273.
func TestOptionalOrnithOracleGreedyDecodeMatchesHF(t *testing.T) {
	dir := ornithFixtureDir()
	if dir == "" {
		t.Skip("set FAK_ORNITH_FIXTURE to run optional oracle")
	}
	blob, err := os.ReadFile(filepath.Join(dir, "oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Prompts []struct {
			IDs       []int `json:"ids"`
			GreedyIDs []int `json:"greedy_ids"`
		} `json:"prompts"`
	}
	if err := json.Unmarshal(blob, &oracle); err != nil {
		t.Fatal(err)
	}
	m, _ := loadFixtureDir(t, dir, true)
	for pi, p := range oracle.Prompts {
		s := m.NewSession()
		logits := s.Prefill(p.IDs)
		for step, want := range p.GreedyIDs {
			got := argmax(logits)
			if got != want {
				t.Fatalf("prompt %d step %d: greedy token=%d want HF=%d", pi, step, got, want)
			}
			if step+1 < len(p.GreedyIDs) {
				logits = s.Step(got)
			}
		}
	}
}
