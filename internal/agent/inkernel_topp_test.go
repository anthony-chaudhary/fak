package agent

import (
	"math"
	"math/rand"
	"testing"
)

// inkernel_topp_test.go proves the in-kernel sampler honors per-request TopP
// (nucleus sampling) — the second half of the "TopP/Stop accepted but not honored"
// gap. Like checkStop, sampleLogits is pure, so nucleus truncation is witnessed
// over synthetic logits with a fixed seed, no weighted model. The invariant that
// matters: with a tight nucleus, a high-probability token is the ONLY reachable
// draw; a low-probability tail token is unreachable regardless of the RNG.

// drawHistogram runs n seeded draws and returns the count per token id. top-k is
// disabled (0) so these nucleus tests exercise the top-p path alone.
func drawHistogram(logits []float32, temp, topP float64, n int) map[int]int {
	h := map[int]int{}
	for seed := 0; seed < n; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		h[sampleLogits(logits, temp, topP, 0, rng)]++
	}
	return h
}

func TestTopPNucleusExcludesTail(t *testing.T) {
	// Token 0 carries almost all the mass; tokens 1..3 are a long tail. A nucleus of
	// 0.5 keeps only token 0, so EVERY draw must be token 0 — the tail is unreachable.
	logits := []float32{10, 1, 1, 1}
	h := drawHistogram(logits, 1.0, 0.5, 200)
	if h[0] != 200 {
		t.Fatalf("top_p=0.5 over a peaked dist must draw only the head token, got %v", h)
	}
}

func TestTopPDisabledKeepsFullDistribution(t *testing.T) {
	// top_p<=0 and top_p>=1 both disable nucleus truncation — the draw is the full
	// temperature softmax, byte-for-byte the pre-seam sampleLogits behavior. Over a
	// flat distribution every token must be reachable.
	logits := []float32{1, 1, 1, 1}
	for _, topP := range []float64{0, 1} {
		h := drawHistogram(logits, 1.0, topP, 400)
		for id := 0; id < 4; id++ {
			if h[id] == 0 {
				t.Fatalf("top_p=%v must keep every token reachable on a flat dist, id %d never drawn (%v)", topP, id, h)
			}
		}
	}
}

func TestTopPGreedyUnaffectedByNucleus(t *testing.T) {
	// temp<=0 is argmax regardless of top_p — nucleus only shapes the stochastic
	// path. The head token wins every time.
	logits := []float32{0.1, 5, 0.2}
	rng := rand.New(rand.NewSource(1))
	if got := sampleLogits(logits, 0, 0.9, 0, rng); got != 1 {
		t.Fatalf("greedy (temp=0) must pick argmax id 1 regardless of top_p, got %d", got)
	}
}

func TestTopPKeepsAtLeastTheHead(t *testing.T) {
	// A pathologically small nucleus (top_p just above 0) must still keep at least the
	// single most-probable token — never an empty nucleus that can't draw anything.
	logits := []float32{3, 2, 1}
	rng := rand.New(rand.NewSource(7))
	got := sampleLogits(logits, 1.0, 1e-9, 0, rng)
	if got != 0 {
		t.Fatalf("a near-zero nucleus must still keep the head token (id 0), got %d", got)
	}
	if math.IsNaN(float64(logits[got])) {
		t.Fatalf("sampler returned an invalid index")
	}
}
