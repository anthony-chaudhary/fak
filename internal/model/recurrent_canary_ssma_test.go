package model

import (
	"math"
	"testing"
)

// witnessArgmax returns the greedy token id for one position's logits. Local to this
// file so it never collides with the oracle-suite helper of the same intent.
func witnessArgmax(logits []float32) int {
	best, bi := float32(math.Inf(-1)), 0
	for i, v := range logits {
		if v > best {
			best, bi = v, i
		}
	}
	return bi
}

// decodePositions runs a cacheless Forward over ids and returns the per-position
// greedy argmax id stream — the generated-token-id trace a bucket compares.
func decodePositions(m *Model, ids []int) []int {
	out := m.Forward(ids)
	got := make([]int, len(ids))
	for p := range ids {
		got[p] = witnessArgmax(out.Logits[p])
	}
	return got
}

// witnessPrompt builds a deterministic pseudo-random token stream of length n in
// [1, vocab-2] (never the 0/EOS edges), matching the long-horizon fixture's LCG.
func witnessPrompt(n, vocab int) []int {
	ids := make([]int, n)
	x := uint32(0x9e3779b9)
	for i := range ids {
		x = x*1664525 + 1013904223
		ids[i] = int(x%uint32(vocab-2)) + 1
	}
	return ids
}

// mutateSSMAToIdentity rewrites every linear-attn A_log tensor from the correctly
// inverse-transformed value (A_log, what fak's loader recovers via A_log=log(-decay))
// to the GGUF-stored decay value itself (-exp(A_log)) — i.e. the loader mutation that
// SKIPS the inverse transform and feeds the stored decay straight through as A_log.
// This is the exact "identity ssm_a loader mutation" the #4745 witness must reject.
func mutateSSMAToIdentity(m *Model, cfg Config) {
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		aLog := m.tensor(layerName(l, "linear_attn.A_log"))
		for i, v := range aLog {
			aLog[i] = float32(-math.Exp(float64(v))) // stored decay = -exp(A_log); identity-mutation uses it AS A_log
		}
	}
}

// TestSSMAIdentityMutationRejectedByMediumLongBuckets is the #4745 witness. It drives
// the real synthetic hybrid forward. The corrected inverse transform (an untouched
// second build) reproduces the reference bit-for-bit and PASSES every bucket; the
// identity ssm_a loader mutation drifts the recurrent decay and is REJECTED (argmax
// parity below the promotion floor) at the medium and long buckets — exactly where
// #4273's accumulated recurrent-state error lived, and where short sanity did not
// catch it.
func TestSSMAIdentityMutationRejectedByMediumLongBuckets(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	floor := DefaultPromotionThresholds().MinArgmaxAgreement

	// Reference arm: the correct loader (A_log recovered via the inverse transform).
	ref := NewSynthetic(cfg)
	// Corrected fak arm: an independent identical build — the corrected inverse passes.
	corrected := NewSynthetic(cfg)
	// Broken fak arm: same build with the identity ssm_a loader mutation applied.
	mutated := NewSynthetic(cfg)
	mutateSSMAToIdentity(mutated, cfg)

	type bucket struct {
		kind   BucketKind
		n      int
		reject bool // must the mutation be rejected at this horizon?
	}
	buckets := []bucket{
		{BucketShort, 8, false},   // sanity floor — not required to catch the mutation
		{BucketMedium, 256, true}, // first meaningful recurrent horizon — must reject
		{BucketLong, 1300, true},  // long production horizon — must reject
	}

	for _, b := range buckets {
		ids := witnessPrompt(b.n, cfg.VocabSize)
		refIDs := decodePositions(ref, ids)
		corrIDs := decodePositions(corrected, ids)
		mutIDs := decodePositions(mutated, ids)

		// The corrected inverse transform PASSES: bit-identical build -> parity 1.
		if p := ArgmaxAgreement(corrIDs, refIDs); p < 1.0 {
			t.Fatalf("%s bucket: corrected inverse transform must pass (parity 1), got %.4f", b.kind, p)
		}

		// The identity mutation: rejected at medium/long, per the witness.
		pm := ArgmaxAgreement(mutIDs, refIDs)
		if b.reject {
			if pm >= floor {
				t.Fatalf("%s bucket (n=%d): identity ssm_a mutation NOT rejected — parity %.4f >= floor %.4f; the medium/long canary failed to catch the loader mutation",
					b.kind, b.n, pm, floor)
			}
		}
		t.Logf("%s bucket n=%d: corrected parity=1.0000 (pass), identity-mutation parity=%.4f (reject=%v, floor=%.4f)", b.kind, b.n, pm, b.reject, floor)
	}
}
