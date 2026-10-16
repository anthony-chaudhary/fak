package agent

import (
	"math/rand"
	"testing"
)

// inkernel_sampling_test.go is the #13873 witness: the in-kernel sampler must be
// allocation-free across tokens once its per-session workspace is warm, and the
// partial-selection rewrite must draw the identical token as the un-workspaced path
// for a fixed seed across every option combination.

const samplingWitnessVocab = 129000 // DeepSeek V4.1-class LM-head width.

// sampleOptionsCase is one (temp, topP, topK) combination exercised by the parity
// witness. The set spans the full grid: pure argmax, full softmax (both cutoffs off),
// top-k only, top-p only, and the composed top-k -> top-p pipeline.
type sampleOptionsCase struct {
	name string
	temp float64
	topP float64
	topK int
}

func sampleOptionsCases() []sampleOptionsCase {
	return []sampleOptionsCase{
		{"argmax", 0, 0, 0},
		{"full_softmax", 1.0, 0, 0},
		{"full_softmax_topP1", 1.0, 1, 0},
		{"topk_small", 1.0, 0, 50},
		{"topk_boundary", 1.0, 0, 64},
		{"topk_large", 1.0, 0, 100},
		{"topp", 1.0, 0.9, 0},
		{"topp_tight", 0.7, 0.5, 0},
		{"composed", 1.0, 0.9, 50},
		{"composed_large_k", 0.8, 0.95, 100},
	}
}

func samplingWitnessLogits(vocab int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 2.0)
	}
	return logits
}

// TestInkernelSampleWorkspaceZeroAllocs proves the reusable workspace removes every
// per-token allocation: after one warm-up call sizes its buffers, subsequent samples
// at the V4.1 vocabulary width allocate nothing.
func TestInkernelSampleWorkspaceZeroAllocs(t *testing.T) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	var ws samplerWorkspace
	// Warm-up: the first draw sizes probs/order/eff, so the measured runs are steady state.
	drain := rand.New(rand.NewSource(1))
	_ = sampleLogitsWithPenaltyWS(logits, 1.0, 0.9, 50, nil, 0, 0, nil, drain, &ws)

	for _, tc := range sampleOptionsCases() {
		t.Run(tc.name, func(t *testing.T) {
			var counts []int32
			if tc.name == "composed" {
				counts = make([]int32, len(logits))
				counts[7] = 3
			}
			allocs := testing.AllocsPerRun(50, func() {
				_ = sampleLogitsWithPenaltyWS(logits, tc.temp, tc.topP, tc.topK, nil, 0, 0, counts, drain, &ws)
			})
			if allocs > 0 {
				t.Fatalf("sampleLogitsWithPenaltyWS(%s) allocated %v per call, want 0 after warm-up", tc.name, allocs)
			}
		})
	}
}

// TestInkernelSampleWorkspaceSeededParity proves the workspaced path draws the exact
// same token as the historical per-call path for a fixed seed over every option
// combination — the semantic contract the rewrite must not change.
func TestInkernelSampleWorkspaceSeededParity(t *testing.T) {
	logits := samplingWitnessLogits(4096, 999)
	for _, tc := range sampleOptionsCases() {
		t.Run(tc.name, func(t *testing.T) {
			var ws samplerWorkspace
			for seed := int64(0); seed < 200; seed++ {
				want := sampleLogitsWithPenalty(logits, tc.temp, tc.topP, tc.topK, nil, 0, 0, nil, rand.New(rand.NewSource(seed)))
				got := sampleLogitsWithPenaltyWS(logits, tc.temp, tc.topP, tc.topK, nil, 0, 0, nil, rand.New(rand.NewSource(seed)), &ws)
				if got != want {
					t.Fatalf("%s seed %d: workspaced draw %d != reference %d", tc.name, seed, got, want)
				}
			}
		})
	}
}

// TestInkernelSampleWorkspaceGrowsAcrossVocabularies proves a workspace sized for a
// narrow vocabulary keeps parity when a later, wider call reuses it — the served lane
// holds one workspace for a session whose widest LM head may arrive after warm-up.
func TestInkernelSampleWorkspaceGrowsAcrossVocabularies(t *testing.T) {
	var ws samplerWorkspace
	for _, vocab := range []int{64, 4096, samplingWitnessVocab} {
		logits := samplingWitnessLogits(vocab, int64(vocab))
		for seed := int64(0); seed < 40; seed++ {
			want := sampleLogits(logits, 1.0, 0.9, 50, rand.New(rand.NewSource(seed)))
			got := sampleLogitsWS(logits, 1.0, 0.9, 50, rand.New(rand.NewSource(seed)), &ws)
			if got != want {
				t.Fatalf("vocab %d seed %d: got %d want %d", vocab, seed, got, want)
			}
		}
	}
}

// BenchmarkInkernelSample is the issue's named witness: the sampler at production
// vocabulary width with allocation reporting, so the ns/op delta from the
// partial-selection rewrite is recorded alongside the zero-alloc steady state.
func BenchmarkInkernelSample(b *testing.B) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	var ws samplerWorkspace
	rng := rand.New(rand.NewSource(42))
	_ = sampleLogitsWithPenaltyWS(logits, 1.0, 0.9, 50, nil, 0, 0, nil, rng, &ws)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleLogitsWithPenaltyWS(logits, 1.0, 0.9, 50, nil, 0, 0, nil, rng, &ws)
	}
}

// BenchmarkInkernelSampleOneShot is the pre-#13873 shape (a fresh workspace per call)
// kept as the named baseline for the recorded delta.
func BenchmarkInkernelSampleOneShot(b *testing.B) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	rng := rand.New(rand.NewSource(42))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleLogitsWithPenalty(logits, 1.0, 0.9, 50, nil, 0, 0, nil, rng)
	}
}
