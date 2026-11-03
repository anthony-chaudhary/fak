package rawdecode

import (
	"math"
	"testing"
)

// BenchmarkGreedySelection measures greedy top-1/top-2 selection plus the
// shift-invariant selected-token logprob (-log(sum exp(v-max))) over a
// realistic vocabulary-sized logit vector: the hot inner loop of raw decode.
func BenchmarkGreedySelection(b *testing.B) {
	const vocab = 151936
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = -float32(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		token, _, _, logprob, err := greedySelection(logits)
		if err != nil {
			b.Fatal(err)
		}
		if token != 0 || math.IsNaN(logprob) || logprob > 0 {
			b.Fatalf("unexpected selection: token=%d logprob=%v", token, logprob)
		}
	}
}
