package model

// v41_decode_alloc_test.go — hot-path allocation regression guard for the V4.1
// Flash incremental decode step (hot-path copy sweep TICKET-03, issue #13877).
//
// The dense decode step reused a Session-scoped decodeScores scratch and was
// pinned by TestDecodeStepAllocationStaysBounded. The V4.1 Flash compressed
// step keeps the same class of guarantee but on its own state surface: the
// per-token compressor pool, indexer-tail scratch, cached-key rows and RoPE
// tables must not introduce a per-op allocation that grows with the context.
//
// The witness is the reusable leakcheck.AllocScaling primitive: steady-state
// per-step allocation must be INDEPENDENT of context length (pre-fix a
// context-proportional per-token buffer makes the ratio track Large/Small).

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leakcheck"
)

// TestV41DecodeStepAllocationStaysBounded proves per-step V4.1 decode
// allocation does not scale with context length. setup(size) builds a V4.1
// session prefilled to `size` tokens and returns a closure that decodes one
// more token; leakcheck.AllocScaling measures bytes/op at a shallow (32) vs
// deep (4096) context and fails if the ratio exceeds 2.0.
func TestV41DecodeStepAllocationStaysBounded(t *testing.T) {
	m := v41ReducedModel(t)
	vocab := m.Cfg.VocabSize
	if vocab <= 0 {
		vocab = 8
	}
	setup := func(size int) func() {
		s := m.NewSession()
		prompt := make([]int, size)
		for i := range prompt {
			prompt[i] = i % vocab
		}
		tok := pickNextInRange(s.Prefill(prompt), vocab)
		return func() { tok = pickNextInRange(s.Step(tok), vocab) }
	}
	ratio := leakcheck.AllocScaling(t, leakcheck.ScalingOpts{
		Small: 32, Large: 4096, Iters: 64, Warmup: 64, MaxRatio: 2.0,
	}, setup)
	t.Logf("V4.1 decode per-step alloc scaling (deep/shallow) ratio = %.2f (want ≤ 2.0)", ratio)
}

// pickNextInRange draws a deterministic in-vocab next token from logits.
func pickNextInRange(logits []float32, vocab int) int {
	bi, bv := 0, logits[0]
	for i, x := range logits {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi % vocab
}
