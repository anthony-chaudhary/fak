package agent

import (
	"math"
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=300ms lane=default
func TestNucleusTruncatePrefixMatchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	var prefixTaken, cases int
	var w samplerWorkspace
	for _, vocab := range []int{1, 2, 3, 5, 17, 100, 1000, 5000} {
		for _, sigma := range []float64{0.25, 1, 3, 8} {
			for _, topK := range []int{0, 1, 2, 20} {
				for _, topP := range []float64{0.05, 0.5, 0.9, 0.95, 0.999999} {
					for trial := 0; trial < 4; trial++ {
						probs := make([]float64, vocab)
						for i := range probs {
							logit := rng.NormFloat64() * sigma
							if trial == 1 {
								logit = math.Round(logit) // force equal-probability ties
							}
							probs[i] = math.Exp(logit)
							if trial == 3 {
								// Exact powers of two land on the pmax*16^-k thresholds.
								probs[i] = math.Ldexp(1, -4*rng.Intn(6))
							}
						}
						if trial == 2 && vocab > 3 {
							probs[rng.Intn(vocab)] = 0
						}
						if trial == 3 {
							probs[rng.Intn(vocab)] = 1
						}
						var sum float64
						for _, p := range probs {
							sum += p
						}
						if topK > 0 && topK < vocab {
							sum = w.topKTruncate(probs, sum, topK)
						}
						want := append([]float64(nil), probs...)
						wantSum := (&samplerWorkspace{}).nucleusTruncateFull(want, sum, topP)
						probe := append([]float64(nil), probs...)
						if _, ok := w.nucleusTruncatePrefix(probe, topP*sum); ok {
							prefixTaken++
						}
						cases++
						got := append([]float64(nil), probs...)
						gotSum := w.nucleusTruncate(got, sum, topP)
						if math.Float64bits(gotSum) != math.Float64bits(wantSum) {
							t.Fatalf("vocab=%d sigma=%v topK=%d topP=%v trial=%d: sum %v, full sort %v", vocab, sigma, topK, topP, trial, gotSum, wantSum)
						}
						for i := range got {
							if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
								t.Fatalf("vocab=%d sigma=%v topK=%d topP=%v trial=%d: token %d = %v, full sort %v", vocab, sigma, topK, topP, trial, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
	if prefixTaken*2 < cases {
		t.Fatalf("prefix path taken in %d of %d cases; the parity check is too weak", prefixTaken, cases)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestNucleusTruncatePrefixRoundingShortfallDeclines(t *testing.T) {
	// 0.1+0.2+0.3 is 0.6000000000000001 in index order but 0.6 in descending
	// order, so the head passes the threshold scan yet falls short in the walk;
	// the full walk keeps a fourth token, so the prefix path must decline.
	probs := []float64{0.1, 0.2, 0.3, 0.001, 0.001, 0.001, 0.001}
	target := probs[0] + probs[1] + probs[2] // runtime float64 sum, not an exact constant
	got := append([]float64(nil), probs...)
	var w samplerWorkspace
	if _, ok := w.nucleusTruncatePrefix(got, target); ok {
		t.Fatalf("prefix path accepted a head whose sorted mass falls short: %v", got)
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(probs[i]) {
			t.Fatalf("declined prefix path mutated token %d", i)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNucleusTruncatePrefixDeclinesUnsafeRows(t *testing.T) {
	var w samplerWorkspace
	for name, probs := range map[string][]float64{
		"nan":      {0.5, math.NaN(), 0.2, 0.1},
		"all-zero": {0, 0, 0, 0},
		"flat":     {1, 1, 1, 1, 1, 1, 1, 1},
	} {
		row := append([]float64(nil), probs...)
		if _, ok := w.nucleusTruncatePrefix(row, 0.9*4); ok {
			t.Fatalf("%s: prefix path accepted an unsafe row", name)
		}
		for i := range row {
			if math.Float64bits(row[i]) != math.Float64bits(probs[i]) {
				t.Fatalf("%s: declined prefix path mutated token %d", name, i)
			}
		}
	}
}
