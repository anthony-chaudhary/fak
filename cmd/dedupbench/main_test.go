package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestDedupBenchFusionCoalescesAndMatchesFullPrefill pins the two properties the
// benchmark's headline depends on: the fusion arm REALLY coalesces (exactly one
// leader, N-1 followers) and every arm's decode logits are bit-identical to a
// fresh full prefill. Without the first, the ratio would be measuring N independent
// leaders, not fusion; without the second, a fast-but-wrong arm could post a win.
//
// fak-test:runtime fast est=3000ms
func TestDedupBenchFusionCoalescesAndMatchesFullPrefill(t *testing.T) {
	const n, prefix, suffix = 6, 96, 24
	m := model.NewSynthetic(model.Config{
		HiddenSize:        128,
		NumLayers:         4,
		NumHeads:          8,
		NumKVHeads:        4,
		HeadDim:           16,
		IntermediateSize:  256,
		VocabSize:         512,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	})
	vocab := 512
	shared := lcg(prefix, vocab, 7)
	prompts := make([][]int, n)
	for i := range prompts {
		prompts[i] = append(append([]int(nil), shared...), lcg(suffix, vocab, 100+uint64(i))...)
	}
	oracle := make([][]float32, n)
	for i, p := range prompts {
		s := m.NewSession()
		oracle[i] = s.Prefill(p)
	}

	fusion, ok := armFusion(m, prompts, oracle, prefix)
	if !ok {
		t.Fatal("fusion arm failed its correctness/integrity gate")
	}
	if fusion.CoalescedLeaders != 1 || fusion.CoalescedFollowers != int64(n-1) {
		t.Fatalf("fusion coalescing = leaders %d followers %d, want 1/%d",
			fusion.CoalescedLeaders, fusion.CoalescedFollowers, n-1)
	}
	if !fusion.Correctness {
		t.Fatal("fusion arm not correctness-gated")
	}

	nba, ok := armWarmCacheNBA(m, prompts, oracle)
	if !ok || !nba.Correctness {
		t.Fatal("warm-cache NBA arm failed its correctness gate")
	}
	if nba.CoalescedLeaders != 0 || nba.CoalescedFollowers != 0 {
		t.Fatalf("warm-cache NBA reported coalescing counters %d/%d, want 0/0",
			nba.CoalescedLeaders, nba.CoalescedFollowers)
	}

	naive, ok := armNaive(m, prompts, oracle)
	if !ok || !naive.Correctness {
		t.Fatal("naive arm failed its correctness gate")
	}

	// Sanity on the token accounting: fusion and warm-cache do the SAME total prefill
	// tokens (one shared prefix + N suffixes), which is why the ratio is a wall-clock,
	// not a token-count, comparison.
	fusionWant := prefix + (n-1)*(prefix+suffix-prefix)
	if fusion.PrefillTok != fusionWant {
		t.Fatalf("fusion prefill tokens = %d, want %d", fusion.PrefillTok, fusionWant)
	}
	if naive.PrefillTok != n*(prefix+suffix) {
		t.Fatalf("naive prefill tokens = %d, want %d", naive.PrefillTok, n*(prefix+suffix))
	}
}

// TestDedupBenchBitEqualRejectsDivergence makes the correctness gate non-vacuous.
func TestDedupBenchBitEqualRejectsDivergence(t *testing.T) {
	a := []float32{1, 2, 3}
	if bitEqual(a, []float32{1, 2, 3}) != true {
		t.Fatal("bitEqual rejected identical slices")
	}
	if bitEqual(a, []float32{1, 2, 4}) {
		t.Fatal("bitEqual accepted a divergent slice")
	}
	if bitEqual(a, []float32{1, 2}) {
		t.Fatal("bitEqual accepted a length mismatch")
	}
}
