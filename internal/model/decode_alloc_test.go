package model

// decode_alloc_test.go — hot-path allocation regression guard for the f32 decode step.
//
// The per-token decode attention used to do `scores := make([]float32, context)` per head
// per layer EVERY step, so a generation to length N allocated Θ(N²) transient score bytes —
// real GC pressure on the core decode loop. The fix reuses a single Session.decodeScores
// scratch (grown geometrically, fully overwritten per head, bit-identical). The regression
// guard is the reusable leakcheck.AllocScaling primitive: steady-state per-step allocation
// must be INDEPENDENT of context length (pre-fix it tracked the context ratio, ~8x deeper).

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leakcheck"
)

func allocTestModel() *Model {
	return NewSynthetic(Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        64,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       63,
	})
}

func pickNext(logits []float32) int {
	bi, bv := 0, logits[0]
	for i, x := range logits {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi%60 + 1 // keep within vocab, avoid EOS(63)
}

// TestDecodeStepAllocationStaysBounded proves per-step decode allocation does not scale with
// context length. setup(size) builds a session prefilled to `size` tokens and returns a
// closure that decodes one more token; leakcheck.AllocScaling measures bytes/op at a shallow
// (32) vs deep (4096) context (warming each past the measured window so no scratch realloc
// lands in it) and fails if the ratio exceeds 2.0. With the scratch reused the ratio is ≈1;
// with the pre-fix per-head make() it was ≈8 — a clean, machine-independent separation.
func TestDecodeStepAllocationStaysBounded(t *testing.T) {
	m := allocTestModel()
	setup := func(size int) func() {
		s := m.NewSession()
		prompt := make([]int, size)
		for i := range prompt {
			prompt[i] = i%60 + 1
		}
		tok := pickNext(s.Prefill(prompt))
		return func() { tok = pickNext(s.Step(tok)) }
	}
	ratio := leakcheck.AllocScaling(t, leakcheck.ScalingOpts{
		Small: 32, Large: 4096, Iters: 64, Warmup: 64, MaxRatio: 2.0,
	}, setup)
	t.Logf("decode per-step alloc scaling (deep/shallow) ratio = %.2f (want ≤ 2.0)", ratio)
}

// TestDecodeScoreScratchReused is the white-box companion: after decoding, the session's
// reusable scratch is exactly the high-water context width (one buffer), not a fresh
// allocation per step. It also guards that the scratch actually grew (so the reuse path,
// not a no-op, is what's exercised).
func TestDecodeScoreScratchReused(t *testing.T) {
	m := allocTestModel()
	s := m.NewSession()
	tok := pickNext(s.Prefill([]int{1, 2, 3, 4}))
	const steps = 200
	for i := 0; i < steps; i++ {
		tok = pickNext(s.Step(tok))
	}
	// Full causal attention: the scratch must cover the whole decoded context in one buffer.
	wantAtLeast := s.Cache.Len() - 1
	if len(s.decodeScores) < 1 || cap(s.decodeScores) < wantAtLeast {
		t.Fatalf("decodeScores scratch not reused/grown: len=%d cap=%d, want cap ≥ %d",
			len(s.decodeScores), cap(s.decodeScores), wantAtLeast)
	}
}
