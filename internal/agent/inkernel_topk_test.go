package agent

import (
	"math/rand"
	"testing"
)

// inkernel_topk_test.go proves the in-kernel sampler honors per-request TopK
// (top-k truncation) — the sibling of the TopP nucleus gap. Like checkStop and the
// nucleus path, sampleLogits is pure, so top-k truncation is witnessed over
// synthetic logits with a fixed seed, no weighted model. The invariant that
// matters: with a tight k, only the k highest-logit tokens are reachable; a token
// outside the top-k is unreachable regardless of the RNG.

// drawHistogramK runs n seeded draws with a top-k cutoff (top-p disabled) and
// returns the count per token id.
func drawHistogramK(logits []float32, temp float64, topK, n int) map[int]int {
	h := map[int]int{}
	for seed := 0; seed < n; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		h[sampleLogits(logits, temp, 0, topK, rng)]++
	}
	return h
}

func TestTopKExcludesTokensOutsideTheK(t *testing.T) {
	// Four roughly-comparable logits; top_k=2 keeps only the two highest (ids 0 and 1,
	// the two largest logits), so ids 2 and 3 must be unreachable across every seed.
	logits := []float32{4, 3, 2, 1}
	h := drawHistogramK(logits, 1.0, 2, 400)
	if h[2] != 0 || h[3] != 0 {
		t.Fatalf("top_k=2 must make ids 2,3 unreachable, got %v", h)
	}
	if h[0] == 0 || h[1] == 0 {
		t.Fatalf("top_k=2 must keep ids 0,1 reachable on a stochastic draw, got %v", h)
	}
}

func TestTopKOneIsArgmaxOnTheStochasticPath(t *testing.T) {
	// top_k=1 collapses the candidate set to the single highest-logit token, so even
	// with temp>0 every draw must be that token — top-k can pin a stochastic draw to
	// the head without going greedy (temp<=0).
	logits := []float32{0.2, 9, 0.3, 0.1}
	h := drawHistogramK(logits, 1.0, 1, 200)
	if h[1] != 200 {
		t.Fatalf("top_k=1 over a peaked dist must draw only the head token id 1, got %v", h)
	}
}

func TestTopKDisabledKeepsFullDistribution(t *testing.T) {
	// top_k<=0 and top_k>=len(logits) both disable truncation — the draw is the full
	// temperature softmax, byte-for-byte the pre-seam sampleLogits behavior. Over a
	// flat distribution every token must be reachable.
	logits := []float32{1, 1, 1, 1}
	for _, topK := range []int{0, 4, 5} {
		h := drawHistogramK(logits, 1.0, topK, 600)
		for id := 0; id < 4; id++ {
			if h[id] == 0 {
				t.Fatalf("top_k=%d must keep every token reachable on a flat dist, id %d never drawn (%v)", topK, id, h)
			}
		}
	}
}

func TestTopKGreedyUnaffected(t *testing.T) {
	// temp<=0 is argmax regardless of top_k — top-k only shapes the stochastic path.
	// The head token wins every time.
	logits := []float32{0.1, 5, 0.2}
	rng := rand.New(rand.NewSource(1))
	if got := sampleLogits(logits, 0, 0, 1, rng); got != 1 {
		t.Fatalf("greedy (temp=0) must pick argmax id 1 regardless of top_k, got %d", got)
	}
}

func TestTopKComposesWithTopP(t *testing.T) {
	// top-k applies BEFORE top-p: top_k=3 first keeps ids {0,1,2} (the three highest
	// logits), then top_p=0.5 over that nucleus keeps only the head — so every draw is
	// id 0, and the top-k-excluded id 3 is doubly unreachable. Witnesses the documented
	// top-k → top-p pipeline order.
	logits := []float32{10, 1, 1, 1}
	h := map[int]int{}
	for seed := 0; seed < 200; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		h[sampleLogits(logits, 1.0, 0.5, 3, rng)]++
	}
	if h[0] != 200 {
		t.Fatalf("top_k=3 then top_p=0.5 over a peaked dist must draw only the head token, got %v", h)
	}
}
